"""ResABit reference model (PyTorch).

A faithful reimplementation of ``Qwen2ForCausalLM`` with two switches layered
on top: Q1_0_g128 weights on the block projections, and the cross-layer
attention residual of arXiv 2603.15031.

With ``quantize_linear=False`` and ``use_attention_residuals=False`` this
module is numerically equivalent to HuggingFace Qwen1.5-0.5B-Chat -- asserted
to 1e-4 in ``tests/test_parity.py``. That equivalence is what licenses every
downstream number: an ablation on top of an unvalidated reimplementation
measures the reimplementation's bugs, not the intervention.

This module is the correctness reference. Training runs use the MLX port in
``src/mlx_backend/``, which is checked against this file layer by layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .quantization import OneBitLinear

__all__ = ["ResABitForCausalLM", "RMSNorm", "build_linear"]


def build_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    config: ModelConfig,
) -> nn.Module:
    """A block projection: 1-bit when quantising, plain ``nn.Linear`` otherwise."""
    if config.quantize_linear:
        return OneBitLinear(
            in_features, out_features, bias=bias, group_size=config.quant_group_size
        )
    return nn.Linear(in_features, out_features, bias=bias)


# -- Normalisation --------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


# -- Rotary embeddings ----------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(min(max_seq_len, 4096))

    def _build_cache(self, seq_len: int) -> None:
        self._cached_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        need = seq_len + offset
        if need > self._cached_len:
            self._build_cache(min(max(need, 2 * self._cached_len), self.max_seq_len))
        sl = slice(offset, offset + seq_len)
        return self.cos_cached[sl], self.sin_cached[sl]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# -- Attention ------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        b = config.attention_bias
        self.q_proj = build_linear(config.hidden_size, q_out, b, config)
        self.k_proj = build_linear(config.hidden_size, kv_out, b, config)
        self.v_proj = build_linear(config.hidden_size, kv_out, b, config)
        self.o_proj = build_linear(q_out, config.hidden_size, False, config)

        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, _ = hidden.shape

        q = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self.rotary(T, offset)
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # A single new token attends to the whole cache, so the causal mask
        # only applies when we are processing more than one position.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), present


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = build_linear(h, i, False, config)
        self.up_proj = build_linear(h, i, False, config)
        self.down_proj = build_linear(i, h, False, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# -- Decoder layer --------------------------------------------------------


class DecoderLayer(nn.Module):
    """Pre-norm Qwen2 block, optionally carrying the attention residual.

    With residuals enabled, ``R`` accumulates every attention output produced
    so far and is injected through a learnable per-layer gate::

        A_l = attn(norm1(h))
        h   = h + A_l + alpha_l * R_{l-1}
        R_l = R_{l-1} + A_l
        h   = h + mlp(norm2(h))

    ``alpha`` starts at zero, so an AR arm and its non-AR twin are identical
    at step 0 and diverge only through training.

    The stored parameter is pre-divided by :attr:`ALPHA_GAIN`; the effective
    gate is ``ALPHA_GAIN * attn_residual_scale``. See the MLX twin for why.
    Both backends must agree on this or a trained checkpoint evaluates with
    the wrong gate strength.
    """

    ALPHA_GAIN = 10.0

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.use_attention_residuals:
            self.attn_residual_scale = nn.Parameter(
                torch.tensor(float(config.attn_residual_init) / self.ALPHA_GAIN)
            )
        else:
            self.register_parameter("attn_residual_scale", None)

    def forward(
        self,
        hidden: torch.Tensor,
        residual_acc: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden), past_kv, use_cache
        )

        if self.attn_residual_scale is not None:
            if residual_acc is None:
                residual_acc = torch.zeros_like(attn_out)
            else:
                gate = self.ALPHA_GAIN * self.attn_residual_scale
                hidden = hidden + gate * residual_acc
            residual_acc = residual_acc + attn_out

        hidden = hidden + attn_out
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, residual_acc, present


# -- Full model -----------------------------------------------------------


class ResABitForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings and the LM head stay in full precision. Binarising a
        # 152k-row embedding table is a different and far more destructive
        # intervention; keeping it out means `quantize_linear` isolates the
        # effect on the block projections alone.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> dict:
        hidden = self.embed_tokens(input_ids)
        residual_acc = None
        past_key_values = past_key_values or [None] * len(self.layers)
        present: list = [] if use_cache else []

        for layer, past in zip(self.layers, past_key_values, strict=True):
            hidden, residual_acc, kv = layer(hidden, residual_acc, past, use_cache)
            if use_cache:
                present.append(kv)

        logits = self.lm_head(self.norm(hidden))

        loss = None
        if labels is not None:
            # `labels` is already shifted by the data pipeline; positions to
            # skip are marked -100 rather than trimmed, so padding never
            # contributes to the reported loss.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present if use_cache else None,
        }

    # -- utilities --------------------------------------------------------

    def quantized_modules(self) -> list[OneBitLinear]:
        return [m for m in self.modules() if isinstance(m, OneBitLinear)]

    def alpha_values(self) -> list[float]:
        """Effective per-layer gates, with ``ALPHA_GAIN`` folded back in."""
        return [
            float(layer.attn_residual_scale.item()) * DecoderLayer.ALPHA_GAIN
            for layer in self.layers
            if layer.attn_residual_scale is not None
        ]

    def num_parameters(self, trainable_only: bool = False) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:            # tied embeddings must not double count
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        past, cur = None, input_ids

        for _ in range(max_new_tokens):
            out = self.forward(cur, past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            logits = out["logits"][:, -1, :].float()

            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = logits.softmax(-1)
                order = probs.argsort(dim=-1, descending=True)
                sorted_probs = probs.gather(-1, order)
                cutoff = (sorted_probs.cumsum(-1) - sorted_probs) > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                nxt = order.gather(-1, torch.multinomial(sorted_probs, 1))

            input_ids = torch.cat([input_ids, nxt], dim=-1)
            cur = nxt
            if (nxt == eos).all():
                break

        return input_ids
