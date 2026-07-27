"""Q1_0_g128 -- 1-bit weights with FP16 per-group scales.

Each weight is one sign bit; every ``group_size`` consecutive weights along
the input dimension share one FP16 scale::

    w_i = s_g * (2 * b_i - 1),   b_i in {0, 1},   s_g = max |w| over the group

Storage is ``1 + 16 / group_size`` bits per weight (1.125 at g=128).

Three forward paths
-------------------
``train``
    Full-precision master weights with a straight-through estimator, so the
    forward pass sees exactly the values the quantised model will use.
``dequant`` (default after :meth:`OneBitLinear.quantize`)
    Unpack bits, rebuild the FP32 matrix, ``F.linear``. This is the reference
    path for a frozen checkpoint. It is *not* bit-exact with the training
    forward: ``quantize`` stores the group scales as FP16, which moves the
    layer output by ~2e-4 relative. ``tests/test_quantization.py`` pins that
    gap.

    Measured end to end, the perturbation does **not** compound: a frozen
    0.5B checkpoint scores 282.2098 against the training forward's 282.2077,
    +8e-6 nats. The reason is that FP16 rounding perturbs group *magnitudes*
    and flips no sign bits, so it never crosses the discontinuity that drives
    the depth-compounding divergence measured across backends. Still measure
    it rather than assume it -- that number had never been checked, and the
    two paths are not the same computation.
``int8``
    Grouped INT8 GEMM. Opt-in, and asserted equivalent to ``dequant`` in
    ``tests/test_quantization.py``.

A note on the INT8 path
-----------------------
The scale depends on both the output row *and* the input group, so it cannot
be folded into a single GEMM. An earlier version averaged the per-group
scales down to one scale per row, which discards the entire point of
group-wise quantisation and inflated perplexity by six orders of magnitude.
The implementation below accumulates one INT8 GEMM per group instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ste_sign",
    "fake_quantize",
    "pack_bits",
    "unpack_bits",
    "OneBitLinear",
    "quantize_model_weights",
]


# -- Straight-through estimator ------------------------------------------


class _STESign(torch.autograd.Function):
    """sign() forward, clipped identity backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # sign(0) = 0 would decode to -1 after packing; map it to +1 so the
        # training forward and the packed weights agree everywhere.
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).to(grad.dtype)


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return _STESign.apply(x)


def fake_quantize(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Round-trip a weight matrix through Q1_0 while staying differentiable."""
    out_features, in_features = weight.shape
    w = weight.float().view(out_features, in_features // group_size, group_size)
    scales = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # w / scales lands in [-1, 1], so the STE clip mask never fires here; it
    # exists to keep the estimator well behaved if scales are ever frozen.
    return (ste_sign(w / scales) * scales).view(out_features, in_features)


# -- Bit packing ----------------------------------------------------------
# uint8 throughout. Packing with an int8 shift table overflows on bit 7
# (1 << 7 == 128 wraps to -128) and silently corrupts every eighth weight.


def pack_bits(signs: torch.Tensor) -> torch.Tensor:
    """[out, in] of {0,1} -> uint8 [out, ceil(in/8)], LSB-first."""
    out_features, in_features = signs.shape
    n_bytes = (in_features + 7) // 8
    pad = n_bytes * 8 - in_features
    if pad:
        signs = F.pad(signs, (0, pad))
    shifts = torch.arange(8, dtype=torch.uint8, device=signs.device)
    grouped = signs.to(torch.uint8).view(out_features, n_bytes, 8)
    return (grouped << shifts).sum(dim=-1, dtype=torch.uint8)


def unpack_bits(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """uint8 [out, n_bytes] -> float32 [out, in_features] of {0.0, 1.0}."""
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :in_features].float()


# -- Grouped INT8 GEMM ----------------------------------------------------


def _quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token symmetric INT8 quantisation."""
    flat = x.float().reshape(-1, x.shape[-1])
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return (flat / scale).round().clamp(-127, 127).to(torch.int8), scale


def _grouped_int8_gemm(
    x_int8: torch.Tensor,      # [tokens, in]  int8
    w_int8: torch.Tensor,      # [out, in]     int8, values in {-1, +1}
    x_scale: torch.Tensor,     # [tokens, 1]   float32
    w_scales: torch.Tensor,    # [out, groups] float32
    group_size: int,
) -> torch.Tensor:
    """sum over groups of (x_g @ w_g^T) * scale_g -- exact, not averaged."""
    tokens = x_int8.shape[0]
    out_features, groups = w_scales.shape
    acc = torch.zeros(tokens, out_features, dtype=torch.float32, device=x_int8.device)

    for g in range(groups):
        lo, hi = g * group_size, (g + 1) * group_size
        xg = x_int8[:, lo:hi].contiguous()
        wg = w_int8[:, lo:hi].t().contiguous()
        try:
            partial = torch._int_mm(xg, wg)
        except (RuntimeError, AttributeError):
            # _int_mm is CPU/CUDA-only and has minimum-size constraints.
            partial = xg.float() @ wg.float()
        acc += partial.float() * w_scales[:, g].unsqueeze(0)

    return acc * x_scale


# -- 1-bit linear layer ---------------------------------------------------


class OneBitLinear(nn.Module):
    """``nn.Linear`` replacement backed by Q1_0_g128 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        int8_inference: bool = False,
    ) -> None:
        super().__init__()
        if in_features % group_size:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.int8_inference = int8_inference

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("weight_bits", None)
        self.register_buffer("weight_scales", None)
        # Persisted so a reloaded checkpoint knows it is already frozen; an
        # unmarked quantised checkpoint would silently run the training path
        # over uninitialised master weights.
        self.register_buffer("quantized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_int8_cache", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def is_quantized(self) -> bool:
        return bool(self.quantized.item())

    # -- forward paths ----------------------------------------------------

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize(self.weight, self.group_size)
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), w, bias).to(x.dtype)

    def dequantized_weight(self) -> torch.Tensor:
        bits = unpack_bits(self.weight_bits, self.in_features)
        signs = (2.0 * bits - 1.0).view(
            self.out_features, self.num_groups, self.group_size
        )
        scales = self.weight_scales.float().unsqueeze(-1)
        return (signs * scales).view(self.out_features, self.in_features)

    def _forward_dequant(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.dequantized_weight(), bias).to(x.dtype)

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        if self._int8_cache is None:
            bits = unpack_bits(self.weight_bits, self.in_features)
            self._int8_cache = (2 * bits - 1).to(torch.int8)
        x_int8, x_scale = _quantize_activations(x)
        out = _grouped_int8_gemm(
            x_int8,
            self._int8_cache,
            x_scale,
            self.weight_scales.float(),
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias.float()
        return out.view(*x.shape[:-1], self.out_features).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_quantized:
            return self._forward_train(x)
        if self.int8_inference:
            return self._forward_int8(x)
        return self._forward_dequant(x)

    # -- freezing ---------------------------------------------------------

    @torch.no_grad()
    def quantize(self) -> None:
        """Freeze master weights into packed bits + FP16 scales."""
        if self.is_quantized:
            return
        w = self.weight.float().view(
            self.out_features, self.num_groups, self.group_size
        )
        scales = w.abs().amax(dim=-1).clamp(min=1e-8)          # [out, groups]
        signs = (w >= 0).view(self.out_features, self.in_features)

        self.weight_bits = pack_bits(signs)
        self.weight_scales = scales.to(torch.float16)
        self.quantized.fill_(True)

        del self._parameters["weight"]
        self.register_parameter("weight", None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A quantised checkpoint has no `weight` and carries buffers whose
        # shapes differ from the freshly constructed placeholders.
        flag = state_dict.get(prefix + "quantized")
        if flag is not None and bool(flag.item()):
            if self.weight is not None:
                del self._parameters["weight"]
                self.register_parameter("weight", None)
            for name in ("weight_bits", "weight_scales"):
                key = prefix + name
                if key in state_dict:
                    setattr(self, name, torch.empty_like(state_dict[key]))
                else:
                    # These buffers are registered as None until a checkpoint
                    # fills them, and PyTorch drops None buffers from its
                    # missing-key accounting. So a file claiming `quantized`
                    # while shipping no bits loads without a word and dies at
                    # the first forward. Report it as missing here, which is
                    # the whole point of refusing a partial map.
                    missing_keys.append(key)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def storage_bytes(self) -> int:
        """Bytes this layer occupies on disk once frozen."""
        if self.is_quantized:
            n = self.weight_bits.numel() + 2 * self.weight_scales.numel()
        else:
            n = 4 * self.weight.numel()
        return n + (4 * self.bias.numel() if self.bias is not None else 0)

    def extra_repr(self) -> str:
        if not self.is_quantized:
            mode = "train"
        else:
            mode = "int8" if self.int8_inference else "dequant"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, mode={mode}"
        )


def quantize_model_weights(model: nn.Module) -> nn.Module:
    """Freeze every :class:`OneBitLinear` in ``model`` in place."""
    for module in model.modules():
        if isinstance(module, OneBitLinear):
            module.quantize()
    return model
