"""
inference.py — Run text generation with a 1-bit + Attention Residual model.

Usage
-----
    # From a converted checkpoint:
    python inference.py --checkpoint ./checkpoints/qwen0.5b-1bit/model.pt \
        --prompt "The future of AI is"

    # Greedy (temperature=0):
    python inference.py --checkpoint ./checkpoints/qwen0.5b-1bit/model.pt \
        --prompt "Once upon a time" --temperature 0
"""

import argparse
import torch
from src.config import ModelConfig
from src.model import OneBitResidualLM


def load_model(checkpoint_path: str, device: str) -> tuple[OneBitResidualLM, ModelConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig()
    config.__dict__.update(ckpt["config"])
    model = OneBitResidualLM(config)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model, config


def generate(
    model: OneBitResidualLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1e-8,
            top_p=top_p,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description="1-bit Attention Residual LM inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen1.5-0.5B-Chat",
                   help="Tokenizer source")
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params | device: {args.device}")
    print(f"Prompt: {args.prompt!r}\n{'─'*60}")

    response = generate(
        model, tokenizer, args.prompt,
        args.max_new_tokens, args.temperature, args.top_p, args.device
    )
    print(response)


if __name__ == "__main__":
    main()
