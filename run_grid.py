"""The grid the thesis rests on: {FP32, ternary} x {autoregressive, diffusion}.

    python run_grid.py --seeds 0                 # shape first, ~1h40
    python run_grid.py --seeds 0 1 2 --resume    # continue without redoing seed 0

The question is whether sub-2-bit quantization damages a diffusion language
model more or less than it damages an autoregressive one at the same budget.
That is an interaction, and it needs all four cells:

    interaction = (ternary_diff - fp32_diff) - (ternary_ar - fp32_ar)

The metric problem, and why it is solvable
------------------------------------------
The two architectures do not report the same quantity. Autoregressive arms
give next-token NLL; diffusion arms give a sampled NELBO bound on masked
prediction with bidirectional context, which is an easier task. **The levels
are not comparable and are never subtracted across architectures.**

What is comparable is each architecture's distance from the same floor. A
model that has learned nothing scores ``log(vocab_size)`` under *both*
metrics -- 11.9312 here -- because both reduce to a uniform distribution over
the vocabulary. So each arm has a headroom below that floor, and the fraction
of headroom that quantization destroys is a dimensionless quantity that means
the same thing in both regimes.

The ledger records the raw nats too, so anyone who disagrees with that
normalisation can compute their own.

What this does not do
---------------------
It does not compare a diffusion model's quality to an autoregressive one's.
Nothing here licenses "diffusion is better/worse than autoregressive at 0.5B",
and the numbers must not be read that way.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import ModelConfig
from src.data import load_wikitext_tokens, make_training_windows
from src.diffusion import uniform_bound
from src.evaluate import evaluate_diffusion_nelbo, evaluate_perplexity
from src.loader import HF_MODEL_ID
from src.trainer import TrainConfig

# MLX exists only on Apple Silicon; a CUDA/Colab host trains through the
# pure-PyTorch loop instead. Import lazily so the grid runs on both.
try:
    import mlx.core  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

LEDGER = Path("results/grid_ledger.jsonl")


def git_commit() -> str:
    """Local copy: run_ablation imports MLX at module top, so a CUDA host
    cannot import anything from it."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:                              # noqa: BLE001
        return "unknown"


@dataclass(frozen=True)
class Cell:
    name: str
    quantize_linear: bool
    diffusion: bool

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            quantize_linear=self.quantize_linear,
            quant_scheme="q1_58",
            use_attention_residuals=False,
            diffusion=self.diffusion,
        )

    # ``build_torch_model`` reads these off the arm it is handed.
    @property
    def use_attention_residuals(self) -> bool:
        return False


CELLS = {
    "fp32_ar": Cell("fp32_ar", False, False),
    "ternary_ar": Cell("ternary_ar", True, False),
    "fp32_diff": Cell("fp32_diff", False, True),
    "ternary_diff": Cell("ternary_diff", True, True),
}

# The contested pair runs first at every seed, so a truncated sweep still
# answers the thesis question rather than half of the control.
ORDER = ("fp32_diff", "ternary_diff", "fp32_ar", "ternary_ar")


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_cell(
    cell: Cell,
    seed: int,
    train_cfg: TrainConfig,
    windows: np.ndarray,
    val_tokens: torch.Tensor,
    device: torch.device,
    args,
) -> dict:
    print(f"\n=== {cell.name}  seed {seed} ===")
    config = cell.model_config()
    cfg = TrainConfig(**{**asdict(train_cfg), "seed": seed})
    started = time.time()

    try:
        if args.backend == "mlx":
            import mlx.core as mx

            from run_ablation import build_torch_model
            from src.mlx_backend.train import mlx_to_torch_state, run_qat

            model, train_result = run_qat(config, cfg, windows)
            state = mlx_to_torch_state(model)
            del model
            mx.clear_cache()
            torch_model = build_torch_model(cell, state, device)
        else:
            from src.trainer import run_qat_torch

            torch_model, train_result = run_qat_torch(config, cfg, windows, device)
        if cell.diffusion:
            result = evaluate_diffusion_nelbo(
                torch_model, val_tokens, device,
                block_size=cfg.seq_len,
                num_samples=args.eval_samples,
                max_blocks=args.eval_blocks,
                progress=False,
            )
            loss_nats, metric = result.nelbo, "nelbo"
            extra = result.as_dict()
        else:
            result = evaluate_perplexity(
                torch_model, val_tokens[: args.eval_tokens], device, progress=False
            )
            loss_nats, metric = result.nll, "nll"
            extra = result.as_dict()
        del torch_model

        floor = uniform_bound(config.vocab_size)
        headroom = floor - loss_nats
        print(f"  -> {metric} {loss_nats:.4f}  floor {floor:.4f}  "
              f"headroom {headroom:+.4f}")
        status, error = ("diverged" if train_result.diverged else "ok"), None
    except Exception as exc:                        # noqa: BLE001
        # Crashes stay in the ledger. A cell that is less feasible than its
        # twin is a result, and dropping it hides that.
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        train_result, extra, loss_nats, headroom, metric = None, None, None, None, None
        status, error = "crash", f"{type(exc).__name__}: {exc}"

    record = {
        "cell": cell.name,
        "seed": seed,
        "stage": "smoke" if cfg.steps < 50 else "full",
        "status": status,
        "error": error,
        "commit": git_commit(),
        "backend": getattr(train_result, "backend", "mlx") if train_result else None,
        "wall_seconds": round(time.time() - started, 1),
        "quantize_linear": cell.quantize_linear,
        "diffusion": cell.diffusion,
        "metric": metric,
        "loss_nats": loss_nats,
        "uniform_floor": uniform_bound(config.vocab_size),
        "headroom": headroom,
        "eval": extra,
        "train": train_result.as_dict() if train_result else None,
        "model_config": config.to_dict(),
        "train_config": asdict(cfg),
    }
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  ({record['wall_seconds']:.0f}s)")
    return record


def summarise() -> None:
    """Quantization cost within each architecture, then the interaction."""
    rows = [r for r in load_ledger()
            if r["status"] != "crash" and r["stage"] == "full"]
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)

    print("\n" + "=" * 74)
    print(f"{'cell':<15}{'n':>3}{'metric':>8}{'loss':>10}{'headroom':>11}")
    for name in ORDER:
        entries = by_cell.get(name, [])
        if not entries:
            continue
        loss = np.mean([e["loss_nats"] for e in entries])
        head = np.mean([e["headroom"] for e in entries])
        print(f"{name:<15}{len(entries):>3}{entries[0]['metric']:>8}"
              f"{loss:>10.4f}{head:>11.4f}")

    def paired_cost(fp32: str, quant: str):
        """Quantization cost per seed, in nats and as a share of headroom."""
        a = {e["seed"]: e for e in by_cell.get(fp32, [])}
        b = {e["seed"]: e for e in by_cell.get(quant, [])}
        shared = sorted(set(a) & set(b))
        if not shared:
            return None
        nats = np.array([b[s]["loss_nats"] - a[s]["loss_nats"] for s in shared])
        share = np.array([
            (b[s]["loss_nats"] - a[s]["loss_nats"]) / a[s]["headroom"]
            for s in shared
        ])
        return shared, nats, share

    print()
    costs = {}
    for label, fp32, quant in (
        ("diffusion", "fp32_diff", "ternary_diff"),
        ("autoregressive", "fp32_ar", "ternary_ar"),
    ):
        got = paired_cost(fp32, quant)
        if got is None:
            continue
        shared, nats, share = got
        costs[label] = (nats, share)
        print(f"ternary cost, {label:<15} seeds {shared}")
        print(f"  nats            : {nats.mean():+.4f}"
              + (f"  (per seed {np.round(nats, 4).tolist()})" if len(nats) > 1 else ""))
        print(f"  share of headroom: {share.mean():+.4f}")

    if len(costs) == 2:
        # Only the normalised quantity crosses architectures. The raw nats
        # are two different metrics and subtracting them would be arithmetic
        # on incommensurable numbers.
        diff_share = costs["diffusion"][1]
        ar_share = costs["autoregressive"][1]
        interaction = diff_share.mean() - ar_share.mean()
        print(f"\ninteraction (share of headroom): {interaction:+.4f}")
        if len(diff_share) > 1 and len(ar_share) > 1:
            se = float(np.sqrt(
                diff_share.var(ddof=1) / len(diff_share)
                + ar_share.var(ddof=1) / len(ar_share)
            ))
            verdict = ("within noise" if abs(interaction) < 2 * se
                       else "ternary costs diffusion more" if interaction > 0
                       else "ternary costs diffusion less")
            print(f"  SE {se:.4f} -> {verdict}")
        else:
            print("  one seed per cell: no noise estimate, no verdict")
    print("=" * 74)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-tokens", type=int, default=131072)
    p.add_argument("--eval-blocks", type=int, default=48)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="skip (cell, seed) pairs already completed")
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"],
                   help="training backend; auto picks MLX where it exists")
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    if args.backend == "auto":
        args.backend = "mlx" if HAS_MLX else "torch"
    if args.backend == "mlx" and not HAS_MLX:
        raise SystemExit("--backend mlx requested but MLX is not installed "
                         "(Apple Silicon only); use --backend torch")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        learning_rate=args.lr,
    )

    print(f"host          : {platform.machine()} / {platform.platform()}")
    print(f"base model    : {HF_MODEL_ID}")
    print(f"scheme        : q1_58 ternary, "
          f"{ModelConfig(quant_scheme='q1_58').bits_per_quantized_weight:.3f} bits/weight")
    print(f"budget        : {train_cfg.steps} steps x {train_cfg.tokens_per_step} "
          f"tok = {train_cfg.total_tokens/1e6:.2f}M tokens per cell")
    print(f"uniform floor : {uniform_bound(ModelConfig().vocab_size):.4f} nats")
    print(f"seeds         : {args.seeds}")
    print(f"backend       : {args.backend}  (eval device {device})")
    if args.backend != "mlx":
        # The ledger row records the backend for the same reason it records
        # the commit: this repository measured quantized stacks amplifying
        # backend differences to ~1e-2, so numbers from different backends
        # pair but do not reproduce each other bitwise.
        print("note          : torch-backend runs pair with mlx runs at the "
              "same seed but are not bitwise comparable to them")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    val_tokens = load_wikitext_tokens(tokenizer, "validation")
    windows = np.array(make_training_windows(train_tokens, train_cfg.seq_len))
    print(f"train windows : {len(windows)}")

    plan = [(CELLS[name], seed) for seed in args.seeds for name in ORDER]

    if args.resume:
        done = {
            (r["cell"], r["seed"])
            for r in load_ledger()
            if r["status"] != "crash" and r["stage"] == "full"
        }
        kept = [(c, s) for c, s in plan if (c.name, s) not in done]
        skipped = [f"{c.name}/{s}" for c, s in plan if (c.name, s) in done]
        if skipped:
            print("resuming; already in ledger: " + ", ".join(skipped))
        plan = kept

    for cell, seed in plan:
        run_cell(cell, seed, train_cfg, windows, val_tokens, device, args)

    summarise()


if __name__ == "__main__":
    main()
