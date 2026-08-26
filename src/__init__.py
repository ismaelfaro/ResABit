"""TriDi -- 1-bit weights + attention residuals on Qwen1.5-0.5B."""

from .config import ModelConfig
from .model import TriDiForCausalLM
from .quantization import LowBitLinear, quantize_model_weights

__all__ = [
    "ModelConfig",
    "TriDiForCausalLM",
    "LowBitLinear",
    "quantize_model_weights",
]
