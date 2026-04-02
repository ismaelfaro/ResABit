"""
1-bit + Attention Residual Language Model.

Architecture
------------
  Base    : Qwen1.5-0.5B-Chat (decoder-only causal transformer)
  Weights : Q1_0_g128 — 1-bit sign + FP16 per-group scale (g=128)
  Extra   : Attention Residuals (arXiv 2603.15031)

Attention Residual forward pass (per DecoderLayer)
---------------------------------------------------
  R_0 = 0  (zeros, same shape as hidden state, accumulated across layers)

  For layer l = 0 … L-1:
    normed    = rms_norm1(h)
    A_l       = self_attn(normed, ...)             # attention output
    R_l       = R_{l-1} + A_l                      # running attention residual
    h         = h + A_l + alpha * R_{l-1}          # standard + cross-layer residual
    h         = h + mlp(rms_norm2(h))

  alpha is a per-layer learnable scalar initialised to 0 so training starts
  as a standard transformer and gradually activates the residual pathway.

All major linear projections are OneBitLinear (Q1_0_g128).
RMSNorm scales and RoPE cos/sin buffers remain in FP32/FP16.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .quantization import OneBitLinear


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ── Rotary Position Embedding ─────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads

        G = config.quant_group_size

        self.q_proj = OneBitLinear(config.hidden_size, self.num_heads * self.head_dim, group_size=G)
        self.k_proj = OneBitLinear(config.hidden_size, self.num_kv_heads * self.head_dim, group_size=G)
        self.v_proj = OneBitLinear(config.hidden_size, self.num_kv_heads * self.head_dim, group_size=G)
        self.o_proj = OneBitLinear(self.num_heads * self.head_dim, config.hidden_size, group_size=G)

        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden.shape

        Q = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self.rotary(T + offset)
        cos = cos[:, :, offset:offset + T, :]
        sin = sin[:, :, offset:offset + T, :]
        Q, K = apply_rotary(Q, K, cos, sin)

        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)
        new_kv = (K, V) if use_cache else None

        # Expand KV heads to match Q heads (GQA)
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        S = K.shape[2]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :T, :S]
        else:
            causal_mask = torch.full(
                (T, S), float("-inf"), device=hidden.device, dtype=hidden.dtype
            ).triu(1 + offset)
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(Q.dtype)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), new_kv


# ── SwiGLU MLP ───────────────────────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        G = config.quant_group_size
        self.gate_proj = OneBitLinear(config.hidden_size, config.intermediate_size, group_size=G)
        self.up_proj   = OneBitLinear(config.hidden_size, config.intermediate_size, group_size=G)
        self.down_proj  = OneBitLinear(config.intermediate_size, config.hidden_size, group_size=G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Decoder Layer with Attention Residual ─────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Standard pre-norm transformer block augmented with Attention Residuals.

    The cross-layer attention residual R carries the cumulative sum of all
    prior attention outputs and is added with a learnable scale alpha.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Learnable gate for the accumulated attention residual (alpha)
        # Initialised at 0 → starts as a vanilla transformer.
        if config.use_attention_residuals:
            self.attn_residual_scale = nn.Parameter(
                torch.tensor(config.attn_residual_init)
            )
        else:
            self.register_parameter("attn_residual_scale", None)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_residual_acc: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple]]:
        # ── Attention ──────────────────────────────────────────────────────
        normed = self.input_layernorm(hidden)
        attn_out, new_kv = self.self_attn(normed, attention_mask, past_kv, use_cache)

        # ── Attention Residual update ──────────────────────────────────────
        if self.attn_residual_scale is not None and attn_residual_acc is not None:
            # h = h + A_l  +  alpha * R_{l-1}
            hidden = hidden + attn_out + self.attn_residual_scale * attn_residual_acc
            # R_l = R_{l-1} + A_l
            new_acc = attn_residual_acc + attn_out
        else:
            hidden = hidden + attn_out
            new_acc = attn_residual_acc

        # ── MLP ───────────────────────────────────────────────────────────
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))

        return hidden, new_acc, new_kv


# ── Full Model ────────────────────────────────────────────────────────────────

class OneBitResidualLM(nn.Module):
    """
    1-bit + Attention Residual causal language model.

    Compatible with Qwen1.5-0.5B-Chat weights (after conversion via convert.py).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        G = config.quant_group_size
        if config.quantize_embeddings:
            self.embed_tokens = OneBitLinear(config.vocab_size, config.hidden_size, group_size=G)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.quantize_embeddings:
            self.lm_head = OneBitLinear(config.hidden_size, config.vocab_size, group_size=G)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Share weight between embed and lm_head (standard for Qwen)
            if isinstance(self.embed_tokens, nn.Embedding):
                self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, OneBitLinear)):
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[Optional[tuple]]] = None,
        use_cache: bool = False,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device

        # Embedding
        if isinstance(self.embed_tokens, OneBitLinear):
            # Use one-hot lookup through OneBitLinear (treat as embedding)
            hidden = F.embedding(input_ids, self._get_embedding_weight())
        else:
            hidden = self.embed_tokens(input_ids)
        hidden = hidden.to(next(self.parameters()).dtype)

        # Initialise attention residual accumulator
        if self.config.use_attention_residuals:
            attn_res_acc = torch.zeros_like(hidden)
        else:
            attn_res_acc = None

        past_key_values = past_key_values or [None] * len(self.layers)
        new_past = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            hidden, attn_res_acc, new_kv = layer(
                hidden,
                attn_res_acc,
                attention_mask=attention_mask,
                past_kv=past_key_values[i],
                use_cache=use_cache,
            )
            if use_cache:
                new_past.append(new_kv)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_past,
        }

    def _get_embedding_weight(self) -> torch.Tensor:
        """Dequantise the 1-bit embedding weight on the fly."""
        layer: OneBitLinear = self.embed_tokens  # type: ignore
        if layer._quantized:
            # Inline dequant: w = s_g * (2b - 1)
            packed = layer.weight_bits
            scales = layer.weight_scales.float()
            bits = torch.zeros(
                packed.shape[0], packed.shape[1] * 8,
                device=packed.device, dtype=torch.float32
            )
            for i in range(8):
                bits[:, i::8] = ((packed >> i) & 1).float()
            w = (2.0 * bits - 1.0).view(
                layer.out_features, layer.num_groups, layer.group_size
            ) * scales.unsqueeze(-1)
            # embed_tokens is [vocab, hidden] but OneBitLinear is [out, in]
            # For embedding we want [vocab, hidden] = [out_features, in_features]
            return w.view(layer.out_features, layer.in_features)
        else:
            return layer.weight.detach()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos = eos_token_id or self.config.eos_token_id
        past = None
        cur = input_ids

        for _ in range(max_new_tokens):
            out = self.forward(cur, past_key_values=past, use_cache=True)
            logits = out["logits"][:, -1, :]
            past = out["past_key_values"]

            if temperature != 1.0:
                logits = logits / temperature

            # Top-p nucleus sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = sorted_logits.softmax(-1).cumsum(-1)
            sorted_logits[cum_probs - sorted_logits.softmax(-1) > top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

            next_tok = torch.multinomial(logits.softmax(-1), num_samples=1)
            cur = next_tok
            input_ids = torch.cat([input_ids, next_tok], dim=-1)

            if (next_tok == eos).all():
                break

        return input_ids
