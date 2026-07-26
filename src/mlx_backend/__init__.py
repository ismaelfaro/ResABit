"""MLX training backend for ResABit (Apple Silicon)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import ModelConfig
from .model import MLXResABit, fake_quantize

__all__ = ["MLXResABit", "fake_quantize", "load_mlx_pretrained", "torch_state_to_mlx"]


def torch_state_to_mlx(state: dict) -> dict:
    """Flat HF-style torch tensors -> the nested tree MLX expects."""
    tree: dict = {"layers": {}}
    for key, tensor in state.items():
        arr = mx.array(tensor.detach().float().numpy().astype(np.float32))
        parts = key.split(".")
        if parts[0] == "layers":
            layer = tree["layers"].setdefault(int(parts[1]), {})
            node = layer
            for p in parts[2:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = arr
        else:
            node = tree
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = arr

    n_layers = max(tree["layers"]) + 1
    tree["layers"] = [tree["layers"][i] for i in range(n_layers)]
    return tree


def load_mlx_pretrained(
    config: ModelConfig | None = None,
    hf_state: dict | None = None,
    model_id: str | None = None,
) -> MLXResABit:
    """Build an :class:`MLXResABit` and fill it with pretrained Qwen2 weights."""
    from ..loader import HF_MODEL_ID, load_hf_state_dict, remap_qwen2_keys

    config = config or ModelConfig()
    state = remap_qwen2_keys(hf_state or load_hf_state_dict(model_id or HF_MODEL_ID))

    model = MLXResABit(config)
    model.update(torch_state_to_mlx(state))
    mx.eval(model.parameters())
    return model
