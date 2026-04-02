"""
convert.py — Load Qwen/Qwen1.5-0.5B-Chat weights and convert to 1-bit format.

Usage
-----
    python convert.py --output ./checkpoints/qwen0.5b-1bit

The script:
  1. Downloads Qwen1.5-0.5B-Chat from HuggingFace (GGUF or safetensors).
  2. Maps its weights into the OneBitResidualLM architecture.
  3. Optionally quantises all OneBitLinear layers to packed bits.
  4. Saves a PyTorch checkpoint ready for inference.py.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from pathlib import Path

from src.config import ModelConfig
from src.model import OneBitResidualLM
from src.quantization import quantize_model_weights


# ── Weight name mapping: Qwen1.5 → OneBitResidualLM ──────────────────────────

def _map_state_dict(hf_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remap HuggingFace Qwen1.5-0.5B key names to our model's naming scheme.

    HF names (examples):
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        model.layers.0.self_attn.o_proj.weight
        model.layers.0.mlp.gate_proj.weight
        model.layers.0.mlp.up_proj.weight
        model.layers.0.mlp.down_proj.weight
        model.layers.0.input_layernorm.weight
        model.layers.0.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight

    Our names:
        embed_tokens.weight  (or embed_tokens.weight if nn.Embedding)
        layers.0.self_attn.q_proj.weight
        layers.0.self_attn.k_proj.weight
        ...
        norm.weight
        lm_head.weight
    """
    mapping: dict[str, torch.Tensor] = {}

    prefix = "model."
    for key, val in hf_sd.items():
        new_key = key
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
        mapping[new_key] = val

    return mapping


def load_hf_model(model_name: str) -> dict[str, torch.Tensor]:
    """Download and return the HF safetensors state dict."""
    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("pip install huggingface-hub safetensors")

    print(f"Downloading {model_name} …")
    local_dir = snapshot_download(
        repo_id=model_name,
        ignore_patterns=["*.gguf", "*.bin", "*.ot", "flax_model*", "tf_model*"],
    )
    sd: dict[str, torch.Tensor] = {}
    for p in Path(local_dir).glob("*.safetensors"):
        sd.update(load_file(str(p)))
    if not sd:
        # Fall back to .bin
        import torch
        for p in Path(local_dir).glob("pytorch_model*.bin"):
            sd.update(torch.load(str(p), map_location="cpu"))
    return sd


def convert(
    model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
    output_dir: str = "./checkpoints/qwen0.5b-1bit",
    quantize: bool = True,
    device: str = "cpu",
) -> None:
    config = ModelConfig()
    model = OneBitResidualLM(config).to(device)

    print("Loading HuggingFace weights …")
    hf_sd = load_hf_model(model_name)
    our_sd = _map_state_dict(hf_sd)

    # Load with strict=False so the new attn_residual_scale params are ignored
    missing, unexpected = model.load_state_dict(our_sd, strict=False)
    print(f"  Missing  keys : {missing}")
    print(f"  Unexpected keys: {unexpected}")

    if quantize:
        print("Quantising to 1-bit (Q1_0_g128) …")
        quantize_model_weights(model)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "model.pt")
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved → {ckpt_path}")

    # Save human-readable config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Report size
    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"Checkpoint size: {size_mb:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Qwen1.5-0.5B-Chat to 1-bit")
    p.add_argument("--model", default="Qwen/Qwen1.5-0.5B-Chat")
    p.add_argument("--output", default="./checkpoints/qwen0.5b-1bit")
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    convert(args.model, args.output, not args.no_quantize, args.device)


if __name__ == "__main__":
    main()
