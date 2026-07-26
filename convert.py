"""Convert Qwen1.5-0.5B-Chat into a ResABit checkpoint.

    python convert.py --output checkpoints/qwen0.5b-1bit
    python convert.py --no-quantize --output checkpoints/qwen0.5b-fp32

Freezing to packed bits is one-way: the master weights are dropped and only
sign bits plus FP16 group scales remain. Quantise a model you have already
fine-tuned, not one you still intend to train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import ModelConfig
from src.loader import HF_MODEL_ID, load_pretrained
from src.quantization import OneBitLinear, quantize_model_weights


def checkpoint_bytes(model) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, OneBitLinear):
            total += module.storage_bytes()
    seen = set()
    for name, param in model.named_parameters():
        if "OneBitLinear" in type(param).__name__ or id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel() * param.element_size()
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=HF_MODEL_ID)
    p.add_argument("--output", default="checkpoints/qwen0.5b-1bit")
    p.add_argument("--no-quantize", action="store_true",
                   help="keep FP32 master weights (for the baseline arm)")
    p.add_argument("--attention-residuals", action="store_true")
    args = p.parse_args()

    config = ModelConfig(
        quantize_linear=not args.no_quantize,
        use_attention_residuals=args.attention_residuals,
    )
    model = load_pretrained(config, model_id=args.model)

    if config.quantize_linear:
        print("freezing to Q1_0_g128 ...")
        quantize_model_weights(model)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config.to_dict(), "state_dict": model.state_dict()},
        out / "model.pt",
    )
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    on_disk = (out / "model.pt").stat().st_size / 1e6
    print(
        f"saved -> {out/'model.pt'}\n"
        f"  file size            : {on_disk:.1f} MB\n"
        f"  quantized params     : {config.num_quantized_params/1e6:.1f}M "
        f"@ {config.bits_per_quantized_weight:.3f} bits\n"
        f"  full-precision params: {config.num_full_precision_params/1e6:.1f}M @ 32 bits\n"
        f"  effective average    : {config.effective_bits_per_weight:.2f} bits/weight"
    )


if __name__ == "__main__":
    main()
