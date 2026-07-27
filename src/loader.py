"""Load HuggingFace Qwen2 weights into :class:`ResABitForCausalLM`.

The original converter called ``load_state_dict(..., strict=False)``, which
silently discarded every key it failed to match -- including all 72 attention
bias tensors. This module refuses to load unless every parameter is
accounted for, and reports exactly what moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import ResABitForCausalLM

__all__ = [
    "load_hf_state_dict",
    "remap_qwen2_keys",
    "load_pretrained",
    "load_checkpoint",
    "HF_MODEL_ID",
]

HF_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


def load_hf_state_dict(model_id: str = HF_MODEL_ID) -> dict[str, torch.Tensor]:
    """Fetch (or reuse the cached) safetensors weights for ``model_id``."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
    )
    state: dict[str, torch.Tensor] = {}
    for path in sorted(Path(local_dir).glob("*.safetensors")):
        state.update(load_file(str(path)))
    if not state:
        raise FileNotFoundError(f"no safetensors found for {model_id} in {local_dir}")
    return state


def remap_qwen2_keys(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``model.layers.0.self_attn.q_proj.weight`` -> ``layers.0.self_attn.q_proj.weight``.

    Qwen1.5 ties its embeddings but still ships a materialised ``lm_head``;
    we drop it and let the tie provide the head.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in hf_state.items():
        if key == "lm_head.weight":
            continue
        out[key[len("model."):] if key.startswith("model.") else key] = value
    return out


def load_pretrained(
    config: ModelConfig | None = None,
    model_id: str = HF_MODEL_ID,
    hf_state: dict[str, torch.Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> ResABitForCausalLM:
    """Build a model from ``config`` and fill it with pretrained Qwen2 weights."""
    config = config or ModelConfig()
    model = ResABitForCausalLM(config)

    state = remap_qwen2_keys(hf_state or load_hf_state_dict(model_id))
    state = {k: v.to(dtype) for k, v in state.items()}

    target = dict(model.named_parameters())
    # `alpha` is new architecture with no pretrained counterpart; it is
    # initialised to zero on purpose and must not count as a load failure.
    expected_new = {n for n in target if n.endswith("attn_residual_scale")}
    if config.tie_word_embeddings:
        target.pop("lm_head.weight", None)

    missing = sorted(set(target) - set(state) - expected_new)
    unexpected = sorted(set(state) - set(target))
    mismatched = [
        (k, tuple(target[k].shape), tuple(state[k].shape))
        for k in sorted(set(target) & set(state))
        if target[k].shape != state[k].shape
    ]

    if missing or unexpected or mismatched:
        raise RuntimeError(
            "refusing to load a partially mapped checkpoint\n"
            f"  missing    : {missing}\n"
            f"  unexpected : {unexpected}\n"
            f"  shape clash: {mismatched}"
        )

    with torch.no_grad():
        for name, param in target.items():
            if name in expected_new:      # alpha keeps its zero init
                continue
            param.copy_(state[name])

    if verbose:
        q = len(model.quantized_modules())
        print(
            f"loaded {len(target)} tensors from {model_id}\n"
            f"  quantized projections : {q}\n"
            f"  attention residuals   : {config.use_attention_residuals}\n"
            f"  parameters            : {model.num_parameters()/1e6:.1f}M"
        )
    return model


def load_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[ResABitForCausalLM, dict]:
    """Load an exported checkpoint directory: safetensors weights + manifest.

    Accepts either the manifest written by ``export_checkpoint.py`` (model
    config nested under ``model_config``, alongside the metrics the card
    quotes) or a bare ``ModelConfig`` dump from ``convert.py``.

    ``lm_head.weight`` is deliberately absent from the file -- it aliases the
    embedding table and safetensors will not store aliased memory. The tie is
    rebuilt from the config here, so its absence is expected and any *other*
    missing key is still an error.
    """
    from safetensors.torch import load_file

    directory = Path(path)
    if directory.is_file():                      # tolerate .../model.safetensors
        directory = directory.parent

    manifest = json.loads((directory / "config.json").read_text())
    config = ModelConfig.from_dict(manifest.get("model_config", manifest))

    model = ResABitForCausalLM(config)
    state = load_file(str(directory / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)

    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            "refusing to load a partially mapped checkpoint\n"
            f"  missing    : {sorted(set(missing) - allowed_missing)}\n"
            f"  unexpected : {sorted(unexpected)}"
        )

    model.eval()
    return (model.to(device) if device is not None else model), manifest
