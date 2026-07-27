"""Does the diffusion adaptation get anywhere at this budget?

    python run_diffusion_check.py --steps 300

The check that has to come before the grid. An autoregressive model adapted
to masked diffusion on 1.23M tokens may land at or near the uniform floor,
and quantization damage cannot be measured on top of a model that has already
floored. This repository has that failure once already: two 1-bit arms were
indistinguishable on ARC-Easy because both sat at chance.

So: train the FP32 diffusion arm alone, and ask whether its NELBO is
meaningfully below log(vocab_size). If it is not, the budget is the finding
and the four-cell grid is not worth running.

The unadapted model is scored first. It starts *worse* than uniform -- it has
never seen the mask token, and 290 unused embedding rows are not a mask
representation -- so "below uniform" is a real bar and not a formality.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from run_ablation import build_torch_model, git_commit
from src.config import ModelConfig
from src.data import load_wikitext_tokens, make_training_windows
from src.evaluate import evaluate_diffusion_nelbo
from src.loader import HF_MODEL_ID, load_pretrained
from src.mlx_backend.train import TrainConfig, mlx_to_torch_state, run_qat

LEDGER = Path("results/diffusion_ledger.jsonl")


class _Arm:
    """Shape ``build_torch_model`` expects, without the ablation's registry."""

    def __init__(self, config: ModelConfig, name: str) -> None:
        self._config, self.name = config, name
        self.quantize_linear = config.quantize_linear
        self.use_attention_residuals = config.use_attention_residuals

    def model_config(self) -> ModelConfig:
        return self._config


def report(label: str, result) -> None:
    verdict = (
        "BELOW the floor — adaptation is learning"
        if result.headroom > 0
        else "AT OR ABOVE the floor — has learned nothing usable"
    )
    print(
        f"  {label:<12} nelbo {result.nelbo:8.4f}  floor {result.uniform_bound:.4f}  "
        f"headroom {result.headroom:+.4f}  mask-acc {result.mask_accuracy:.4f}"
    )
    print(f"  {'':<12} {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-blocks", type=int, default=48,
                   help="validation blocks scored; each is --seq-len tokens")
    p.add_argument("--eval-samples", type=int, default=4,
                   help="corruptions drawn per block for the Monte Carlo bound")
    p.add_argument("--quantize", action="store_true",
                   help="1-bit arm instead of FP32 (not the point of this check)")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    config = ModelConfig(
        quantize_linear=args.quantize,
        use_attention_residuals=False,
        diffusion=True,
    )
    arm = _Arm(config, "diffusion_1bit" if args.quantize else "diffusion_fp32")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"arm           : {arm.name}  seed {args.seed}")
    print(f"host          : {platform.machine()} / {platform.platform()}")
    print(f"budget        : {train_cfg.steps} steps x {train_cfg.tokens_per_step} "
          f"tok = {train_cfg.total_tokens/1e6:.2f}M tokens")
    print(f"mask token    : {config.mask_token_id}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    val_tokens = load_wikitext_tokens(tokenizer, "validation")
    windows = np.array(make_training_windows(train_tokens, train_cfg.seq_len))
    print(f"train windows : {len(windows)}")

    def score(model, label):
        result = evaluate_diffusion_nelbo(
            model, val_tokens, device,
            block_size=args.seq_len,
            num_samples=args.eval_samples,
            max_blocks=args.eval_blocks,
            progress=False,
        )
        report(label, result)
        return result

    started = time.time()

    # The starting point, before any adaptation. Without it "below uniform"
    # has no baseline to be an improvement over.
    print("\nscoring the unadapted model ...")
    base = load_pretrained(config, verbose=False).to(device).eval()
    before = score(base, "unadapted")
    del base

    print(f"\ntraining {train_cfg.steps} steps ...")
    model, train_result = run_qat(config, train_cfg, windows)
    state = mlx_to_torch_state(model)
    del model
    mx.clear_cache()

    print("\nscoring the adapted model ...")
    adapted = build_torch_model(arm, state, device)
    after = score(adapted, "adapted")
    del adapted

    gained = before.nelbo - after.nelbo
    print(f"\nadaptation moved NELBO by {gained:+.4f} nats "
          f"({before.nelbo:.4f} -> {after.nelbo:.4f})")
    if after.headroom <= 0:
        print("VERDICT: the adaptation did not clear the uniform floor. "
              "The budget is the finding; the grid is not worth running yet.")
    else:
        print(f"VERDICT: {after.headroom:.4f} nats of headroom below the floor. "
              "Quantization damage is measurable on top of this.")

    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "arm": arm.name,
            "seed": args.seed,
            "commit": git_commit(),
            "wall_seconds": round(time.time() - started, 1),
            "model_config": config.to_dict(),
            "train_config": asdict(train_cfg),
            "nelbo_before": before.as_dict(),
            "nelbo_after": after.as_dict(),
            "gain_nats": gained,
            "train": train_result.as_dict(),
        }) + "\n")
    print(f"appended to {LEDGER}")


if __name__ == "__main__":
    main()
