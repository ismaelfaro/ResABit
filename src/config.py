"""
Configuration for 1-bit + Attention Residual model.

Base architecture: Qwen/Qwen1.5-0.5B-Chat
  - 24 transformer layers
  - hidden_size = 1024
  - num_attention_heads = 16
  - num_key_value_heads = 16
  - intermediate_size = 2816  (SwiGLU)
  - vocab_size = 151936
  - max_position_embeddings = 32768
  - rope_theta = 1_000_000
  - rms_norm_eps = 1e-6

1-bit quantization (Q1_0_g128):
  w_i = s_g * (2*b_i - 1),  b_i in {0, 1}
  s_g: FP16 scale per group of GROUP_SIZE weights
  Effective bits = 1 + 16/GROUP_SIZE = 1.125 bits/weight

Attention Residuals (from "Attention Residuals" paper, arXiv 2603.15031):
  R_0 = 0
  For each layer l:
    A_l = Attention(RMSNorm(h_{l-1}))
    R_l = R_{l-1} + A_l          (running sum of all attention outputs)
    h_l = h_{l-1} + A_l + alpha * R_{l-1} + MLP(RMSNorm(h_{l-1} + A_l))
  The accumulated residual R is gated by a learnable scalar alpha.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ── Architecture (Qwen1.5-0.5B-Chat) ───────────────────────────────────
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16       # full MHA for 0.5B
    head_dim: int = 64                  # hidden_size // num_attention_heads
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    # ── 1-bit Quantization (Q1_0_g128) ────────────────────────────────────
    quant_group_size: int = 128         # weights per scale group
    quantize_embeddings: bool = True    # 1-bit embed + lm_head
    quantize_attention: bool = True     # 1-bit Q,K,V,O projections
    quantize_mlp: bool = True           # 1-bit gate,up,down projections

    # ── Attention Residuals ────────────────────────────────────────────────
    use_attention_residuals: bool = True
    # Learnable scale for accumulated attention residual (initialised to 0
    # so the model starts as a standard transformer and gradually learns to
    # exploit the cross-layer attention signal).
    attn_residual_init: float = 0.0

    # ── Training defaults ─────────────────────────────────────────────────
    dropout: float = 0.0
