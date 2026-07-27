"""Model configuration for ResABit.

Base architecture is Qwen1.5-0.5B-Chat (``Qwen2ForCausalLM``):
24 layers, hidden 1024, 16 heads (full MHA), SwiGLU intermediate 2816,
vocab 151936, RoPE theta 1e6, tied input/output embeddings.

Two orthogonal switches drive the ablation:

``quantize_linear``
    Binarise the seven projections in every block (q, k, v, o, gate, up,
    down) with Q1_0_g128. Embeddings and the LM head are deliberately left
    in full precision -- see ``bits_per_weight`` below for why that matters
    when quoting a compression ratio.

``use_attention_residuals``
    Add the cross-layer attention accumulator from arXiv 2603.15031.

Setting both to False reproduces stock Qwen1.5-0.5B exactly, which is what
``tests/test_parity.py`` asserts against HuggingFace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    # -- Architecture (Qwen1.5-0.5B-Chat) --------------------------------
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = True   # Qwen2 carries bias on q/k/v, not on o
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    # -- Q1_0_g128 quantization ------------------------------------------
    quantize_linear: bool = True
    quant_group_size: int = 128

    # -- Attention residuals ---------------------------------------------
    use_attention_residuals: bool = False
    # alpha starts at 0 so the residual pathway is initially a no-op and the
    # arm is numerically identical to its no-AR twin at step 0.
    attn_residual_init: float = 0.0

    # -- Discrete diffusion -----------------------------------------------
    # Absorbing-state masked diffusion, LLaDA's formulation: corrupt by
    # replacing tokens with [MASK] at a sampled rate, predict the originals,
    # generate by unmasking iteratively. Turning this on makes attention
    # bidirectional, which is the whole point -- a denoiser sees the entire
    # sequence, including the tokens after the one it is filling in.
    diffusion: bool = False
    # Qwen1.5 ships 151936 embedding rows for a tokenizer that only reaches
    # 151646, so rows 151646..151935 exist, are never emitted, and carry
    # pretrained-but-unused vectors. Taking one for [MASK] avoids resizing
    # the embedding, avoids breaking the tie to the readout, and keeps the
    # HuggingFace parity test meaningful. It is not free: the row is not a
    # trained mask representation, it is whatever initialisation Qwen left
    # there, so the adaptation has to learn it from scratch.
    mask_token_id: int = 151646

    def __post_init__(self) -> None:
        if self.diffusion and not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(
                f"mask_token_id {self.mask_token_id} is outside the embedding "
                f"table (vocab_size {self.vocab_size})"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size % self.quant_group_size:
            raise ValueError("hidden_size must be divisible by quant_group_size")
        if self.intermediate_size % self.quant_group_size:
            raise ValueError("intermediate_size must be divisible by quant_group_size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # -- Parameter accounting ---------------------------------------------
    # Quoting "1.125 bits/weight" for the whole model would be false: only the
    # block projections are binarised. These helpers keep the README and the
    # results table honest about which denominator is in play.

    @property
    def num_quantized_params(self) -> int:
        """Weights actually stored as sign bits + FP16 group scales."""
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * self.head_dim
        per_layer = h * h + 2 * (h * kv) + h * h + 2 * (i * h) + h * i
        return per_layer * self.num_hidden_layers

    @property
    def num_full_precision_params(self) -> int:
        """Embeddings, LM head (if untied), norms and attention biases."""
        h, v = self.hidden_size, self.vocab_size
        embed = v * h * (1 if self.tie_word_embeddings else 2)
        norms = 2 * h * self.num_hidden_layers + h
        bias = 2 * self.num_key_value_heads * self.head_dim + h
        biases = bias * self.num_hidden_layers if self.attention_bias else 0
        alphas = self.num_hidden_layers if self.use_attention_residuals else 0
        return embed + norms + biases + alphas

    @property
    def bits_per_quantized_weight(self) -> float:
        """1 sign bit + one FP16 scale amortised over the group."""
        return 1.0 + 16.0 / self.quant_group_size

    @property
    def effective_bits_per_weight(self) -> float:
        """Model-wide average, counting the full-precision remainder."""
        q, f = self.num_quantized_params, self.num_full_precision_params
        if not self.quantize_linear:
            return 32.0
        return (q * self.bits_per_quantized_weight + f * 32.0) / (q + f)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})
