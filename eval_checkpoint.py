"""Evaluate an exported checkpoint without retraining it.

The sweep evaluated each arm inside the training process and discarded the
model, so adding a benchmark meant paying for 300 optimizer steps again --
per arm, per benchmark. That is why PIQA sat implemented but unrun. Once
`export_checkpoint.py` has written a checkpoint, this script scores it as
many times as needed for the cost of the forward passes alone.

The numbers it writes are *frozen-path* numbers: packed sign bits and FP16
group scales, which is what a downloaded checkpoint computes. They are not
interchangeable with the ledger's training-forward numbers and are kept in a
separate file for that reason.

Usage
-----
    python eval_checkpoint.py checkpoints/resabit-qwen1.5-0.5b-1bit
    python eval_checkpoint.py --base            # shipped Qwen, no training
    python eval_checkpoint.py <dir> --tasks piqa --no-kl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import ModelConfig
from src.data import AVAILABLE_TASKS, load_multiple_choice, load_wikitext_tokens
from src.evaluate import (
    evaluate_multiple_choice,
    evaluate_perplexity,
    evaluate_teacher_divergence,
)
from src.loader import HF_MODEL_ID, load_checkpoint, load_pretrained

RESULTS = Path("results/checkpoint_evals.jsonl")

# Full validation splits except HellaSwag, which is large and whose standard
# error at 2000 items (~0.011) is already far below every gap in this table.
DEFAULT_LIMITS = {
    "arc_easy": None,
    "hellaswag": 2000,
    "piqa": None,
    "lambada": 1000,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="directory written by export_checkpoint.py")
    p.add_argument("--base", action="store_true",
                   help="score the shipped Qwen instead, as the reference row")
    p.add_argument("--tasks", nargs="+", default=list(AVAILABLE_TASKS),
                   choices=list(AVAILABLE_TASKS))
    p.add_argument("--eval-tokens", type=int, default=131072)
    p.add_argument("--no-kl", action="store_true",
                   help="skip teacher divergence (halves peak memory)")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()

    if not args.base and not args.checkpoint:
        raise SystemExit("pass a checkpoint directory, or --base")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    val_tokens = load_wikitext_tokens(tokenizer, "validation")[: args.eval_tokens]

    if args.base:
        label, manifest = "shipped_qwen", {}
        model = load_pretrained(
            ModelConfig(quantize_linear=False, use_attention_residuals=False),
            verbose=False,
        ).to(device).eval()
    else:
        label = Path(args.checkpoint).name
        model, manifest = load_checkpoint(args.checkpoint, device)
        print(
            f"loaded {label}: arm={manifest.get('arm')} seed={manifest.get('seed')} "
            f"frozen={manifest.get('frozen')}"
        )

    started = time.time()
    ppl = evaluate_perplexity(model, val_tokens, device, progress=False)
    print(f"wikitext-2 ppl {ppl.perplexity:.4f}  nll {ppl.nll:.4f}  "
          f"top1 {ppl.top1_accuracy:.4f}")

    zero_shot = {}
    for task_name in args.tasks:
        task = load_multiple_choice(task_name, limit=DEFAULT_LIMITS.get(task_name))
        result = evaluate_multiple_choice(model, tokenizer, task, device, progress=False)
        zero_shot[task_name] = result
        # Chance is the number that decides whether a cell carries any signal,
        # so print it next to the score rather than leaving it to the reader.
        chance = 1.0 / len(task.choices[0]) if len(task.choices[0]) > 1 else None
        chance_note = f"  (chance {chance:.3f})" if chance else ""
        print(
            f"  {task_name:<10} {result['primary_metric']} {result['primary']:.4f} "
            f"+/- {result['stderr']:.4f}  n={result['num_items']}{chance_note}"
        )

    divergence = None
    if not args.no_kl and not args.base:
        teacher = load_pretrained(
            ModelConfig(quantize_linear=False, use_attention_residuals=False),
            verbose=False,
        ).to(device).eval()
        divergence = evaluate_teacher_divergence(
            model, teacher, val_tokens, device, max_windows=32, progress=False
        )
        print(f"  teacher    KL {divergence['kl_teacher_student']:.4f} nats  "
              f"agree {divergence['top1_agreement']:.4f}")
        del teacher

    record = {
        "checkpoint": label,
        "path": args.checkpoint,
        "arm": manifest.get("arm", "shipped_qwen"),
        "seed": manifest.get("seed"),
        "frozen": manifest.get("frozen", False),
        "commit": manifest.get("commit"),
        "eval_tokens": args.eval_tokens,
        "wall_seconds": round(time.time() - started, 1),
        "perplexity": ppl.as_dict(),
        "zero_shot": zero_shot,
        "divergence": divergence,
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
