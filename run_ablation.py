"""Run the ResABit ablation: 2x2 over {FP32, 1-bit} x {no AR, AR}.

Protocol
--------
Every arm starts from the same pretrained Qwen1.5-0.5B-Chat weights and gets
an identical token budget in an identical order. The FP32 arms are
fine-tuned too -- comparing a fine-tuned 1-bit model against an
un-fine-tuned FP32 one would credit quantization for a domain-adaptation
gap.

Seeds are paired: arm A and arm B see the same seed set, so the per-seed
difference cancels the variance they share (init draw, data order) and the
noise floor is measured rather than assumed. An effect smaller than the
spread across seeds is reported as no effect.

Training runs on MLX (1.72x faster on Apple Silicon); every reported metric
is computed by the PyTorch harness, whose agreement with HuggingFace is
pinned in tests/test_parity.py.

Usage
-----
    python run_ablation.py --steps 300 --seeds 0 1 2
    python run_ablation.py --stage noise-floor      # variance first
    python run_ablation.py --stage final-evals      # held-out suite
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from src.config import ModelConfig
from src.data import load_wikitext_tokens, make_training_windows
from src.evaluate import evaluate_perplexity
from src.loader import HF_MODEL_ID
from src.mlx_backend.train import TrainConfig, mlx_to_torch_state, run_qat

RESULTS_DIR = Path("results")
LEDGER = RESULTS_DIR / "ledger.jsonl"


@dataclass(frozen=True)
class Arm:
    name: str
    quantize_linear: bool
    use_attention_residuals: bool

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            quantize_linear=self.quantize_linear,
            use_attention_residuals=self.use_attention_residuals,
        )


ARMS = {
    "fp32": Arm("fp32", False, False),
    "fp32_ar": Arm("fp32_ar", False, True),
    "onebit": Arm("onebit", True, False),
    "onebit_ar": Arm("onebit_ar", True, True),
}

# The pair whose difference is the actual research question. The FP32 arms
# exist to size the quantization gap that AR is supposed to be closing.
ABLATION_PAIR = ("onebit", "onebit_ar")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def append_ledger(record: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_torch_model(arm: Arm, mlx_state: dict, device: torch.device):
    """Rehydrate a trained MLX model inside the PyTorch eval harness."""
    from src.model import ResABitForCausalLM

    config = arm.model_config()
    model = ResABitForCausalLM(config)
    target = dict(model.named_parameters())
    if config.tie_word_embeddings:
        target.pop("lm_head.weight", None)

    missing = sorted(set(target) - set(mlx_state))
    if missing:
        raise RuntimeError(f"MLX export is missing {missing}")

    with torch.no_grad():
        for name, param in target.items():
            param.copy_(mlx_state[name])
    return model.to(device).eval()


def run_full_suite(
    torch_model,
    tokenizer,
    teacher,
    eval_tokens: torch.Tensor,
    device: torch.device,
    task_limits: dict[str, int | None],
) -> dict:
    """Held-out metrics, run once per arm on the reference seed.

    Zero-shot accuracy is the number readers expect; teacher divergence is
    the number that still resolves once accuracy has floored at chance.
    """
    from src.data import load_multiple_choice
    from src.evaluate import evaluate_multiple_choice, evaluate_teacher_divergence

    out: dict = {"zero_shot": {}}
    for task_name, limit in task_limits.items():
        task = load_multiple_choice(task_name, limit=limit)
        result = evaluate_multiple_choice(
            torch_model, tokenizer, task, device, progress=False
        )
        out["zero_shot"][task_name] = result
        print(
            f"     {task_name:<10} {result['primary_metric']} "
            f"{result['primary']:.4f} +/- {result['stderr']:.4f} "
            f"(n={result['num_items']})"
        )

    out["divergence"] = evaluate_teacher_divergence(
        torch_model, teacher, eval_tokens, device, max_windows=32, progress=False
    )
    print(
        f"     teacher    KL {out['divergence']['kl_teacher_student']:.4f} nats  "
        f"agree {out['divergence']['top1_agreement']:.4f}"
    )
    return out


def run_one(
    arm: Arm,
    seed: int,
    train_cfg: TrainConfig,
    windows: np.ndarray,
    eval_tokens: torch.Tensor,
    device: torch.device,
    stage: str,
    full_suite: bool = False,
    tokenizer=None,
    teacher=None,
    task_limits: dict | None = None,
) -> dict:
    print(f"\n=== {arm.name}  seed {seed}  ({stage}) ===")

    cfg = TrainConfig(**{**asdict(train_cfg), "seed": seed})
    started = time.time()
    suite = None

    try:
        model, train_result = run_qat(arm.model_config(), cfg, windows)
        state = mlx_to_torch_state(model)
        del model
        # MLX and PyTorch draw on the same unified memory and neither sees
        # the other's allocations. Hand back MLX's cache before standing up
        # the evaluation models.
        mx.clear_cache()

        torch_model = build_torch_model(arm, state, device)
        ppl = evaluate_perplexity(torch_model, eval_tokens, device, progress=False)
        alphas = torch_model.alpha_values()
        print(
            f"  -> ppl {ppl.perplexity:.3f}  nll {ppl.nll:.4f}  "
            f"top1 {ppl.top1_accuracy:.4f}"
        )
        if full_suite:
            suite = run_full_suite(
                torch_model, tokenizer, teacher, eval_tokens, device, task_limits
            )
        del torch_model
        status = "diverged" if train_result.diverged else "ok"
        error = None
    except Exception as exc:                      # noqa: BLE001
        # Crashes stay in the ledger. Dropping them would hide that one arm
        # is less feasible than the other, which is itself a result.
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        train_result, ppl, alphas = None, None, []
        status, error = "crash", f"{type(exc).__name__}: {exc}"

    record = {
        "arm": arm.name,
        "seed": seed,
        "stage": stage,
        "status": status,
        "error": error,
        "commit": git_commit(),
        "wall_seconds": round(time.time() - started, 1),
        "train": train_result.as_dict() if train_result else None,
        "perplexity": ppl.as_dict() if ppl else None,
        "suite": suite,
        "alphas": alphas,
        "train_config": asdict(cfg),
        "quantize_linear": arm.quantize_linear,
        "use_attention_residuals": arm.use_attention_residuals,
    }
    append_ledger(record)
    print(f"  ({record['wall_seconds']:.0f}s total)")
    return record


def summarise(records: list[dict]) -> None:
    """Per-arm mean +/- spread, then the paired difference that matters."""
    print("\n" + "=" * 68)
    by_arm: dict[str, list[dict]] = {}
    for r in records:
        if r["status"] != "crash" and r.get("perplexity"):
            by_arm.setdefault(r["arm"], []).append(r)

    print(f"{'arm':<12}{'n':>3}{'ppl mean':>12}{'ppl sd':>10}{'top1':>9}")
    for name in ARMS:
        rows = by_arm.get(name, [])
        if not rows:
            continue
        ppls = [r["perplexity"]["perplexity"] for r in rows]
        top1 = [r["perplexity"]["top1_accuracy"] for r in rows]
        sd = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else float("nan")
        print(f"{name:<12}{len(ppls):>3}{np.mean(ppls):>12.3f}{sd:>10.3f}{np.mean(top1):>9.4f}")

    a, b = ABLATION_PAIR
    seeds_a = {r["seed"]: r["perplexity"]["perplexity"] for r in by_arm.get(a, [])}
    seeds_b = {r["seed"]: r["perplexity"]["perplexity"] for r in by_arm.get(b, [])}
    shared = sorted(set(seeds_a) & set(seeds_b))
    if len(shared) >= 2:
        diffs = np.array([seeds_b[s] - seeds_a[s] for s in shared])
        mean_d = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        print(f"\npaired {b} - {a} over seeds {shared}")
        print(f"  per-seed deltas : {np.round(diffs, 3).tolist()}")
        print(f"  mean delta ppl  : {mean_d:+.3f}  (SE {se:.3f})")
        verdict = (
            "below the noise floor -- no measurable effect"
            if abs(mean_d) < 2 * se
            else ("AR helps" if mean_d < 0 else "AR hurts")
        )
        print(f"  verdict         : {verdict}")
    print("=" * 68)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="full",
                   choices=["smoke", "determinism", "noise-floor", "full"])
    p.add_argument("--replicates", type=int, default=3,
                   help="determinism stage: identical reruns of one arm/seed")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-tokens", type=int, default=131072,
                   help="wikitext validation tokens used for the dev metric")
    p.add_argument("--hellaswag-limit", type=int, default=2000)
    p.add_argument("--lambada-limit", type=int, default=1000)
    p.add_argument("--no-zero-shot", action="store_true",
                   help="skip the held-out suite (perplexity only)")
    p.add_argument("--resume", action="store_true",
                   help="skip (arm, seed) pairs already recorded in the ledger")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        learning_rate=args.lr,
    )

    print(f"host          : {platform.machine()} / {platform.platform()}")
    print(f"base model    : {HF_MODEL_ID}")
    print(f"budget        : {train_cfg.steps} steps x {train_cfg.tokens_per_step} "
          f"tok = {train_cfg.total_tokens/1e6:.2f}M tokens")
    print(f"eval device   : {device}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    val_tokens = load_wikitext_tokens(tokenizer, "validation")[: args.eval_tokens]
    windows = np.array(make_training_windows(train_tokens, train_cfg.seq_len))
    print(f"train tokens  : {train_tokens.numel()/1e6:.2f}M -> {len(windows)} windows")
    print(f"dev tokens    : {val_tokens.numel()}")

    if args.stage == "smoke":
        plan = [(ARMS["onebit_ar"], args.seeds[0])]
        train_cfg = TrainConfig(**{**asdict(train_cfg), "steps": 3, "log_every": 1})
    elif args.stage == "determinism":
        # Identical configuration, repeated. Everything that moves between
        # these runs is backend nondeterminism -- MLX GPU reductions are not
        # bitwise reproducible, and a binarised network amplifies that. This
        # is the floor beneath the seed-to-seed floor, and without it a
        # small paired difference cannot be interpreted at all.
        plan = [(ARMS["onebit"], args.seeds[0])] * args.replicates
    elif args.stage == "noise-floor":
        # Two variance sources, measured separately. Re-running one seed
        # isolates backend nondeterminism (MLX GPU reductions are not
        # bitwise reproducible); varying the seed adds init and data order
        # on top. A difference smaller than the first is not even an
        # experiment.
        plan = [(ARMS["onebit"], s) for s in args.seeds]
        plan.append((ARMS["onebit"], args.seeds[0]))
    else:
        # The contested pair first, on every seed, so a truncated run still
        # answers the research question. Reference arms come after.
        plan = [(ARMS[n], s) for s in args.seeds for n in ABLATION_PAIR]
        plan += [(ARMS[n], args.seeds[0]) for n in ("fp32", "fp32_ar")]

    # The held-out suite runs once per arm, on the reference seed only. It
    # is expensive and it is held out: running it on every seed would turn
    # it into a dev metric that the protocol then has no guard against.
    reference_seed = args.seeds[0]
    task_limits = {
        "arc_easy": None,
        "hellaswag": args.hellaswag_limit,
        "lambada": args.lambada_limit,
    }
    teacher = None
    if not args.no_zero_shot and args.stage in ("full", "final-evals"):
        from src.loader import load_pretrained

        print("loading FP32 reference for teacher divergence ...")
        teacher = load_pretrained(
            ModelConfig(quantize_linear=False, use_attention_residuals=False),
            verbose=False,
        ).to(device).eval()

    # Resume treats the ledger as the record of what exists, so a sweep can
    # be interrupted and continued without recomputing finished arms. Only
    # successful runs count: a crashed pair should be retried, not skipped.
    suite_done: set[str] = set()
    if args.resume:
        # Count completed runs per (arm, seed, stage) rather than testing
        # membership: the determinism stage deliberately repeats one
        # configuration, so "already present" is the wrong question there --
        # "how many of the requested repeats exist" is the right one.
        completed: dict[tuple[str, int, str], int] = {}
        for record in load_ledger():
            if record["status"] == "crash":
                continue
            key = (record["arm"], record["seed"], record["stage"])
            completed[key] = completed.get(key, 0) + 1
            if record.get("suite"):
                suite_done.add(record["arm"])

        remaining, skipped = [], []
        budget = dict(completed)
        for arm, seed in plan:
            key = (arm.name, seed, args.stage)
            if budget.get(key, 0) > 0:
                budget[key] -= 1
                skipped.append(f"{arm.name}/{seed}")
            else:
                remaining.append((arm, seed))
        if skipped:
            print("resuming; already in ledger: " + ", ".join(skipped))
        plan = remaining

    for arm, seed in plan:
        wants_suite = (
            teacher is not None
            and seed == reference_seed
            and arm.name not in suite_done
        )
        if wants_suite:
            suite_done.add(arm.name)
        run_one(
            arm, seed, train_cfg, windows, val_tokens, device, args.stage,
            full_suite=wants_suite,
            tokenizer=tokenizer,
            teacher=teacher,
            task_limits=task_limits,
        )

    summarise(load_ledger())


if __name__ == "__main__":
    main()
