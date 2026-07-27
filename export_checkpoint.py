"""Retrain one ablation arm and export it as a frozen, publishable checkpoint.

The sweep in ``run_ablation.py`` evaluated each model and threw it away, so
none of the reported arms exists on disk. This script rebuilds one of them
and writes it out. Because the pipeline is bitwise reproducible (three
identical reruns returned 282.208 to the digit), the rebuild is not an
approximation of the run that produced the published table -- it is that run.
The ledger perplexity is asserted, not assumed: if the rebuild misses, the
export aborts rather than shipping a checkpoint that no table describes.

Two perplexities, and they are not the same number
--------------------------------------------------
The ledger records the *training forward*: FP32 master weights pushed through
``fake_quantize`` on every call. That is the right number for the ablation --
it is what training optimised and what every arm was compared on -- but it is
not what a downloaded checkpoint computes. Freezing stores the group scales as
FP16, which moves each layer's output by ~2e-4 relative.

So the frozen path is evaluated separately and the model card quotes *that*.
It turns out to cost +8e-6 nats on this model, which is nothing -- FP16
rounding perturbs magnitudes without flipping sign bits, so it stays away
from the discontinuity that makes binarised stacks diverge with depth. That
is a measurement, and it was worth making: the number had never been checked,
and publishing the ledger's figure against a file that computes a different
one is the same class of error as quoting 1.125 bits/weight for a model whose
embeddings are FP32.

Usage
-----
    python export_checkpoint.py --arm onebit --seed 0
    python export_checkpoint.py --arm onebit_ar --seed 0
    python export_checkpoint.py --arm onebit --seed 0 --skip-ledger-check
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from run_ablation import ARMS, build_torch_model, load_ledger
from src.data import load_wikitext_tokens, make_training_windows
from src.evaluate import evaluate_perplexity
from src.loader import HF_MODEL_ID
from src.mlx_backend.train import TrainConfig, mlx_to_torch_state, run_qat
from src.quantization import LowBitLinear, quantize_model_weights

CHECKPOINT_DIR = Path("checkpoints")

# The rebuild is reproducible to the digit, so the tolerance guards against
# code drift between the sweep and this export, not against kernel noise. A
# looser bound would let a real regression through; this one fires on any
# change large enough to matter.
LEDGER_TOLERANCE = 1e-6


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def ledger_perplexity(arm: str, seed: int, stage: str = "full") -> float | None:
    """The published training-forward perplexity for this (arm, seed)."""
    for record in load_ledger():
        if (
            record["arm"] == arm
            and record["seed"] == seed
            and record["stage"] == stage
            and record["status"] != "crash"
            and record.get("perplexity")
        ):
            return record["perplexity"]["perplexity"]
    return None


def storage_report(model) -> dict:
    """Bytes on disk, split by what is actually binarised.

    Reported because the interesting compression claim (1.125 bits on the
    projections) and the honest one (the file is dominated by an FP32
    embedding table) are different claims, and only quoting the first is how
    1-bit checkpoints get oversold.
    """
    quantized_bytes = 0
    quantized_params = 0
    for module in model.modules():
        if isinstance(module, LowBitLinear) and module.is_quantized:
            # Weights only. ``storage_bytes`` folds the bias in, and the bias
            # is still a live FP32 parameter -- counting it here and again
            # below would inflate the total and deflate the bits/weight.
            quantized_bytes += module.weight_bits.numel() + 2 * module.weight_scales.numel()
            quantized_params += module.in_features * module.out_features

    # Freezing deletes the master weights, so what ``parameters()`` still
    # yields is exactly the full-precision remainder: embeddings, norms,
    # attention biases, gates.
    other_params = model.num_parameters()
    other_bytes = 4 * other_params
    total_params = quantized_params + other_params

    return {
        "quantized_params": quantized_params,
        "quantized_bytes": quantized_bytes,
        "full_precision_params": other_params,
        "full_precision_bytes": other_bytes,
        "total_params": total_params,
        "total_bytes": quantized_bytes + other_bytes,
        "bits_per_quantized_weight": (
            8 * quantized_bytes / quantized_params if quantized_params else None
        ),
        "bits_per_weight_model_average": 8 * (quantized_bytes + other_bytes) / total_params,
    }


def save_safetensors(model, out_dir: Path) -> None:
    """Write the frozen state dict, dropping the tie rather than duplicating it.

    ``lm_head.weight`` *is* ``embed_tokens.weight``; safetensors refuses
    aliased storage, and writing the 622 MB table twice to satisfy it would
    be worse than refusing. The tie is re-established on load from the config
    flag, exactly as it is when loading from HuggingFace.
    """
    from safetensors.torch import save_file

    state = model.state_dict()
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)

    save_file(
        {k: v.contiguous().cpu() for k, v in state.items()},
        str(out_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="onebit", choices=sorted(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-tokens", type=int, default=131072)
    p.add_argument("--skip-ledger-check", action="store_true",
                   help="export even if the rebuild misses the published number")
    args = p.parse_args()

    # A 30-minute run whose progress lands only when it exits is a run you
    # cannot watch. Python block-buffers stdout the moment it is redirected
    # to a file, and the whole training log is under 3 KB, so it never fills
    # the buffer and never flushes early.
    sys.stdout.reconfigure(line_buffering=True)

    arm = ARMS[args.arm]
    out_dir = Path(args.out) if args.out else CHECKPOINT_DIR / f"resabit-{args.arm}-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        seed=args.seed,
    )

    expected = ledger_perplexity(arm.name, args.seed)
    print(f"arm           : {arm.name}  (seed {args.seed})")
    print(f"out           : {out_dir}")
    print(f"ledger ppl    : {expected if expected is not None else 'not in ledger'}")
    print(f"budget        : {train_cfg.steps} steps x {train_cfg.tokens_per_step} tok")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    val_tokens = load_wikitext_tokens(tokenizer, "validation")[: args.eval_tokens]
    windows = np.array(make_training_windows(train_tokens, train_cfg.seq_len))

    started = time.time()
    model, train_result = run_qat(arm.model_config(), train_cfg, windows)
    state = mlx_to_torch_state(model)
    del model
    mx.clear_cache()

    torch_model = build_torch_model(arm, state, device)

    # -- 1. reproduce the published number --------------------------------
    train_forward = evaluate_perplexity(torch_model, val_tokens, device, progress=False)
    print(f"\ntraining-forward ppl : {train_forward.perplexity:.6f}  "
          f"nll {train_forward.nll:.6f}")
    if expected is not None:
        drift = abs(train_forward.perplexity - expected)
        print(f"ledger drift         : {drift:.2e}")
        if drift > LEDGER_TOLERANCE and not args.skip_ledger_check:
            raise SystemExit(
                f"refusing to export: rebuild returned {train_forward.perplexity:.6f}, "
                f"ledger says {expected:.6f}. The code has moved since the sweep; "
                f"re-run the sweep or pass --skip-ledger-check deliberately."
            )

    # -- 2. freeze and measure what the checkpoint actually computes -------
    frozen_metrics = None
    if arm.quantize_linear:
        quantize_model_weights(torch_model)
        frozen = evaluate_perplexity(torch_model, val_tokens, device, progress=False)
        frozen_metrics = frozen.as_dict()
        print(f"frozen (FP16 scales) : {frozen.perplexity:.6f}  nll {frozen.nll:.6f}")
        print(f"cost of freezing     : {frozen.nll - train_forward.nll:+.6f} nats "
              f"({frozen.perplexity / train_forward.perplexity:.4f}x ppl)")
    else:
        print("frozen               : n/a (no quantized layers in this arm)")

    storage = storage_report(torch_model)
    print(
        f"\nstorage       : {storage['total_bytes']/1e6:.1f} MB "
        f"({storage['quantized_bytes']/1e6:.1f} MB binarised + "
        f"{storage['full_precision_bytes']/1e6:.1f} MB FP32)"
    )
    print(f"bits/weight   : {storage['bits_per_quantized_weight'] or 32:.3f} on the "
          f"projections, {storage['bits_per_weight_model_average']:.1f} model-wide")

    # -- 3. write it out ---------------------------------------------------
    save_safetensors(torch_model, out_dir)
    tokenizer.save_pretrained(out_dir)

    manifest = {
        "arm": arm.name,
        "seed": args.seed,
        "base_model": HF_MODEL_ID,
        "commit": git_commit(),
        "host": platform.platform(),
        "exported_wall_seconds": round(time.time() - started, 1),
        "model_config": arm.model_config().to_dict(),
        "train_config": asdict(train_cfg),
        "quantize_linear": arm.quantize_linear,
        "use_attention_residuals": arm.use_attention_residuals,
        "frozen": bool(arm.quantize_linear),
        "metrics": {
            "wikitext2_val_train_forward": train_forward.as_dict(),
            "wikitext2_val_frozen": frozen_metrics,
            "ledger_perplexity": expected,
            "final_train_loss": train_result.final_train_loss,
            "diverged": train_result.diverged,
        },
        "storage": storage,
        "alphas": torch_model.alpha_values(),
    }
    (out_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_dir}/model.safetensors + config.json + tokenizer")


if __name__ == "__main__":
    main()
