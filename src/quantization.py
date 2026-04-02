"""
1-bit linear layer — Q1_0_g128 format.

Storage layout
--------------
  weights  : torch.int8 packed bits  [out, in//8]   (8 weights per byte)
  scales   : torch.float16            [out, in//G]   (one FP16 scale per group)

Forward (training)
------------------
  Uses the Straight-Through Estimator (STE) so gradients pass through the
  sign operation. The effective weight seen by the matmul is:

      W_eff[i] = scale_g * sign(W[i])     (sign maps 0->-1, pos->+1)

  where sign is approximated as tanh(β·w) with β→∞ during training, or
  simply torch.sign with STE grad clipping.

Forward (inference / after quantize_model_weights)
---------------------------------------------------
  Bits are unpacked and dequantized inline:
      w_i = s_g * (2*b_i - 1),   b_i ∈ {0, 1}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── STE sign ──────────────────────────────────────────────────────────────────

class _STESign(torch.autograd.Function):
    """Sign with Straight-Through gradient (|w| ≤ 1 pass-through)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).float()


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return _STESign.apply(x)


# ── 1-bit Linear ──────────────────────────────────────────────────────────────

class OneBitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with 1-bit Q1_0_g128 weights.

    During training  : real-valued parameters W are kept; forward uses
                       STE-sign(W) * per-group scale so gradients flow.
    After quantize() : W is discarded; packed bits + FP16 scales are used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size

        # Full-precision weight for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantised buffers (populated by quantize())
        self.register_buffer("weight_bits", None)   # int8 [out, in//8]
        self.register_buffer("weight_scales", None) # fp16 [out, num_groups]
        self._quantized = False

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ── per-group scale ────────────────────────────────────────────────────

    def _compute_scales(self, w: torch.Tensor) -> torch.Tensor:
        """Return FP16 per-group L-inf scale  [out, num_groups]."""
        out, inp = w.shape
        w_grouped = w.view(out, self.num_groups, self.group_size)
        # scale = max(|w|) per group  →  mirrors the Bonsai approach
        scales = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)
        return scales.to(torch.float16)

    # ── training forward ───────────────────────────────────────────────────

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.float()
        out, inp = w.shape
        w_grouped = w.view(out, self.num_groups, self.group_size)

        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [O, G, 1]
        w_norm = w_grouped / scales                                            # [-1, 1]
        w_sign = ste_sign(w_norm)                                             # ±1 via STE
        w_eff = (w_sign * scales).view(out, inp)                              # dequantised

        out_x = F.linear(x.float(), w_eff, self.bias.float() if self.bias is not None else None)
        return out_x.to(x.dtype)

    # ── quantised forward ──────────────────────────────────────────────────

    def _forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack bits: int8 [O, in//8] → float {0,1} [O, in]
        bits_packed = self.weight_bits           # int8 [O, in//8]
        scales = self.weight_scales.float()      # [O, num_groups]

        # Unpack 8 consecutive bits per byte (LSB first)
        # bits_packed: [O, in//8]; after unpack: [O, in//8, 8] → view [O, in]
        O, n_bytes = bits_packed.shape
        bits_3d = torch.zeros(O, n_bytes, 8, device=bits_packed.device, dtype=torch.float32)
        for i in range(8):
            bits_3d[:, :, i] = ((bits_packed >> i) & 1).float()
        bits = bits_3d.view(O, n_bytes * 8)[:, :self.in_features]

        # w_i = s_g * (2*b_i - 1)
        w = (2.0 * bits - 1.0)                                    # [O, in] in {-1,+1}
        w = w.view(self.out_features, self.num_groups, self.group_size)
        w = w * scales.unsqueeze(-1)                               # broadcast scale
        w = w.view(self.out_features, self.in_features)

        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), w, bias).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantized:
            return self._forward_quantized(x)
        return self._forward_train(x)

    # ── quantize ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def quantize(self) -> None:
        """Freeze weights to packed bits + FP16 scales; free the float tensor."""
        if self._quantized:
            return
        w = self.weight.float()
        out, inp = w.shape
        w_grouped = w.view(out, self.num_groups, self.group_size)

        scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [O, G, 1]
        w_norm = w_grouped / scales
        bits = (w_norm.sign() + 1) / 2    # map -1→0, +1→1  (zero→1 by convention)
        bits = bits.view(out, inp).to(torch.uint8)  # {0, 1}

        # Pack 8 consecutive bits per byte (LSB first)
        # bits: [out, in]; reshape to [out, in//8, 8]
        n_bytes = (inp + 7) // 8
        bits_3d = bits.view(out, n_bytes, 8).to(torch.int8)
        packed = torch.zeros(out, n_bytes, dtype=torch.int8, device=w.device)
        for i in range(8):
            packed |= (bits_3d[:, :, i] << i)

        self.weight_bits = packed
        self.weight_scales = scales.squeeze(-1).to(torch.float16)  # [O, G]
        del self.weight
        self._quantized = True

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"group_size={self.group_size}, quantized={self._quantized}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def quantize_model_weights(model: nn.Module) -> nn.Module:
    """Call .quantize() on every OneBitLinear in the model."""
    for module in model.modules():
        if isinstance(module, OneBitLinear):
            module.quantize()
    return model


def replace_linear_with_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_names: tuple[str, ...] = (),
) -> nn.Module:
    """
    Recursively replace nn.Linear layers with OneBitLinear.
    Layers whose name contains any entry in skip_names are left as-is.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            if any(s in name for s in skip_names):
                continue
            new_layer = OneBitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                group_size=group_size,
            )
            # Copy existing weights if available
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            setattr(model, name, new_layer)
        else:
            replace_linear_with_1bit(child, group_size, skip_names)
    return model
