"""
train.py — Fine-tune / train from scratch with 1-bit + Attention Residuals.

Example
-------
    python train.py \
        --dataset wikitext \
        --dataset_config wikitext-2-raw-v1 \
        --output_dir ./checkpoints/trained \
        --epochs 3 \
        --lr 1e-4 \
        --batch_size 4 \
        --seq_len 512

The training loop keeps full-precision weights (for stable gradients) and
uses the STE-based OneBitLinear forward to simulate 1-bit inference.
After training, call convert.py --no-quantize first then quantize_model_weights()
to freeze into packed bits.
"""

import argparse
import os
import math
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_tokenizer(model_name: str = "Qwen/Qwen1.5-0.5B-Chat"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_dataset(name: str, config: str, split: str, tokenizer, seq_len: int):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, trust_remote_code=True)

    class TextDataset(Dataset):
        def __init__(self, data, tokenizer, seq_len):
            text = "\n\n".join(d["text"] for d in data if d.get("text"))
            enc = tokenizer.encode(text, add_special_tokens=False)
            self.tokens = torch.tensor(enc, dtype=torch.long)
            self.seq_len = seq_len

        def __len__(self):
            return max(1, (len(self.tokens) - 1) // self.seq_len)

        def __getitem__(self, idx):
            start = idx * self.seq_len
            chunk = self.tokens[start: start + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                pad = torch.full((self.seq_len + 1 - len(chunk),), 0, dtype=torch.long)
                chunk = torch.cat([chunk, pad])
            return {"input_ids": chunk[:-1], "labels": chunk[1:]}

    return TextDataset(ds, tokenizer, seq_len)


def cosine_schedule(step: int, warmup: int, total: int, min_lr: float, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> None:
    from src.config import ModelConfig
    from src.model import OneBitResidualLM

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    tokenizer = get_tokenizer(args.model_name)

    config = ModelConfig()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        config.__dict__.update(ckpt["config"])
        model = OneBitResidualLM(config)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Resumed from {args.checkpoint}")
    else:
        model = OneBitResidualLM(config)

    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count/1e6:.1f}M")

    train_ds = get_dataset(args.dataset, args.dataset_config, "train", tokenizer, args.seq_len)
    val_ds   = get_dataset(args.dataset, args.dataset_config, "validation", tokenizer, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    # Only optimise float parameters (scales + norms + attn_residual_scale)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 20)
    step = 0

    log = []
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            lr = cosine_schedule(step, warmup_steps, total_steps, args.lr * 0.1, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % 100 == 0:
                ppl = math.exp(min(epoch_loss / step, 20))
                print(f"  step {step:5d}  loss={loss.item():.4f}  ppl={ppl:.1f}  lr={lr:.2e}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                out = model(input_ids=input_ids, labels=labels)
                val_loss += out["loss"].item()

        val_loss /= max(1, len(val_loader))
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}  "
              f"({elapsed:.0f}s)")
        log.append({"epoch": epoch + 1, "val_loss": val_loss, "val_ppl": val_ppl})

        # Checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, ckpt_path)

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Training complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Train 1-bit + Attention Residual LM")
    p.add_argument("--model_name", default="Qwen/Qwen1.5-0.5B-Chat")
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint .pt")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--output_dir", default="./checkpoints/trained")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
