"""MLX port of the ResABit model -- the training backend on Apple Silicon.

Measured against the PyTorch reference on an M5: 1140 ms vs 1959 ms per
fake-quantised fwd+bwd step at batch 2 x seq 512 (1.72x). Over a four-arm,
multi-seed sweep that is the difference between an afternoon and a day.

The tradeoff is a second implementation that can drift from the reference.
``tests/test_mlx_parity.py`` pins them together: same weights in, logits
agreeing to 1e-3. That test is not optional -- MLX carries the numbers that
go in the paper, and PyTorch is what proves the numbers are Qwen's.

MLX peak memory is much higher than PyTorch's (13.4 GB vs 2.0 GB at batch 2)
because lazy evaluation holds the graph until ``mx.eval``. Batch size is
capped accordingly and effective batch comes from gradient accumulation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig

__all__ = ["MLXResABit", "fake_quantize"]


def fake_quantize(w: mx.array, group_size: int) -> mx.array:
    """Q1_0 round-trip with a straight-through gradient.

    ``stop_gradient`` gives forward = sign(w), backward = identity, which is
    the same estimator as the PyTorch autograd Function. The clip mask is
    omitted deliberately: the argument is w/scale with scale = max|w| over
    the group, so it is already inside [-1, 1] and the mask never fires.
    """
    out_features, in_features = w.shape
    g = w.reshape(out_features, in_features // group_size, group_size)
    scales = mx.maximum(mx.abs(g).max(axis=-1, keepdims=True), 1e-8)
    normed = g / scales
    # sign(0) = 0 in MLX; match the reference, which decodes a zero bit to +1.
    signs = mx.where(normed >= 0, 1.0, -1.0)
    quantized = normed + mx.stop_gradient(signs - normed)
    return (quantized * scales).reshape(out_features, in_features)


class OneBitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
        super().__init__()
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ fake_quantize(self.weight, self.group_size).T
        return y + self.bias if "bias" in self else y


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        return y + self.bias if "bias" in self else y


def _build_linear(cfg: ModelConfig, in_f: int, out_f: int, bias: bool):
    if cfg.quantize_linear:
        return OneBitLinear(in_f, out_f, bias, cfg.quant_group_size)
    return Linear(in_f, out_f, bias)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        self.causal = not cfg.diffusion
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv * self.head_dim
        self.q_proj = _build_linear(cfg, cfg.hidden_size, q_out, cfg.attention_bias)
        self.k_proj = _build_linear(cfg, cfg.hidden_size, kv_out, cfg.attention_bias)
        self.v_proj = _build_linear(cfg, cfg.hidden_size, kv_out, cfg.attention_bias)
        self.o_proj = _build_linear(cfg, q_out, cfg.hidden_size, False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = _build_linear(cfg, h, i, False)
        self.up_proj = _build_linear(cfg, h, i, False)
        self.down_proj = _build_linear(cfg, i, h, False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm Qwen2 block, optionally carrying the attention residual.

    The gate is stored pre-multiplied: the effective alpha is
    ``ALPHA_GAIN * attn_residual_scale``. AdamW takes steps of roughly the
    learning rate regardless of gradient magnitude, so 24 fresh scalars
    starting at zero would barely move inside a short budget and the AR arm
    would be a no-op by construction rather than by evidence. Folding the
    gain into the forward gives those scalars an effective learning rate
    ``ALPHA_GAIN`` times higher without a second optimizer.
    """

    ALPHA_GAIN = 10.0

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = SwiGLUMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.use_ar = cfg.use_attention_residuals
        if self.use_ar:
            self.attn_residual_scale = mx.array(
                float(cfg.attn_residual_init) / self.ALPHA_GAIN
            )

    def __call__(self, x, acc, cos, sin, mask):
        attn = self.self_attn(self.input_layernorm(x), cos, sin, mask)
        if self.use_ar:
            if acc is None:
                acc = mx.zeros_like(attn)
            else:
                x = x + (self.ALPHA_GAIN * self.attn_residual_scale) * acc
            acc = acc + attn
        x = x + attn
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, acc


class MLXResABit(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self._rope_cache: tuple[int, mx.array, mx.array] | None = None

    def _rope(self, seq_len: int):
        if self._rope_cache is None or self._rope_cache[0] < seq_len:
            d = self.cfg.head_dim
            inv = 1.0 / (self.cfg.rope_theta ** (mx.arange(0, d, 2).astype(mx.float32) / d))
            t = mx.arange(seq_len).astype(mx.float32)
            freqs = mx.outer(t, inv)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            self._rope_cache = (seq_len, mx.cos(emb), mx.sin(emb))
        _, cos, sin = self._rope_cache
        return cos[:seq_len][None, None], sin[:seq_len][None, None]

    def __call__(self, input_ids: mx.array) -> mx.array:
        _, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self._rope(T)
        # None means every position attends everywhere, which is what a
        # denoiser needs and what a causal LM must never get.
        mask = "causal" if (not self.cfg.diffusion and T > 1) else None

        acc = None
        for layer in self.layers:
            x, acc = layer(x, acc, cos, sin, mask)

        # Weights are tied, so the head reuses the embedding matrix.
        return self.norm(x) @ self.embed_tokens.weight.T

    def loss(self, input_ids: mx.array, labels: mx.array) -> mx.array:
        """Mean cross-entropy over positions not masked with -100."""
        logits = self(input_ids).astype(mx.float32)
        valid = labels != -100
        safe = mx.where(valid, labels, 0)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), safe.reshape(-1), reduction="none"
        )
        losses = losses * valid.reshape(-1)
        return losses.sum() / mx.maximum(valid.sum(), 1)

    def diffusion_loss(
        self, input_ids: mx.array, rates: mx.array, mask: mx.array
    ) -> mx.array:
        """The 1/t-weighted NELBO estimator; see ``src/diffusion.py``.

        Same shape of contract as the PyTorch twin: corruption arrives as an
        argument. Two backends cannot agree on a random draw, and the parity
        test needs them scoring the identical corruption.
        """
        B, L = input_ids.shape
        corrupted = mx.where(mask, self.cfg.mask_token_id, input_ids)
        logits = self(corrupted).astype(mx.float32)

        per_token = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            input_ids.reshape(-1),
            reduction="none",
        ).reshape(B, L)

        masked_sum = (per_token * mask.astype(mx.float32)).sum(axis=1)
        return (masked_sum / (rates.reshape(B) * L)).mean()

    def alpha_values(self) -> list[float]:
        """Effective per-layer gates, with ``ALPHA_GAIN`` folded back in."""
        return [
            float(layer.attn_residual_scale.item()) * DecoderLayer.ALPHA_GAIN
            for layer in self.layers
            if layer.use_ar
        ]
