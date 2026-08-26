"""Generate text from a TriDi checkpoint.

    python inference.py --checkpoint checkpoints/qwen0.5b-1bit/model.pt \
        --prompt "The future of AI is"
"""

from __future__ import annotations

import argparse

import torch

from src.config import ModelConfig
from src.loader import HF_MODEL_ID
from src.model import TriDiForCausalLM


def load_checkpoint(path: str, device: torch.device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(blob["config"])
    model = TriDiForCausalLM(config)
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.to(device).eval(), config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, _ = load_checkpoint(args.checkpoint, device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
