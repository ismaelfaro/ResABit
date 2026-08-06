"""Does discontinuity-crossing noise compound across denoising steps?

    python run_step_compounding.py            # ~1h: retrains both cells, then measures

The question, and why it is the last one
----------------------------------------
Part I measured that a quantized stack amplifies tiny perturbations
monotonically with *depth*: a one-ulp weight difference flips a stored
level, and the error compounds through 24 layers to ~1e-2 on the logits.
It also measured the counterexample — FP16 scale rounding never crosses the
discontinuity and costs 8 micronats end to end. Benign and malignant
perturbations are separated by whether they flip levels.

A diffusion sampler adds a second axis the autoregressive model does not
have: it feeds its own committed tokens back through the stack, up to
dozens of times per block. A flipped level changes logits; changed logits
can change which token the sampler commits; a committed token is discrete
and irreversible within the trajectory, and every later step conditions on
it. Whether that feedback *amplifies* the per-forward divergence or the
sampler's argmax *absorbs* it (most logit perturbations do not change the
argmax) is exactly the kind of question intuition gets wrong here — the
scale-rounding prediction was wrong in the benign direction, and this one
could be wrong in either.

Protocol
--------
Retrain the published fp32_diff and ternary_diff cells (300 steps, seed 0;
the pipeline is bitwise reproducible, so these are the ledger's models, not
approximations). For each, build a twin whose master weights are nudged by
relative Gaussian noise of 1e-6 — one-ulp scale — then freeze both. For the
ternary pair the nudge flips a measured fraction of stored levels; for the
FP32 control it stays continuous. Then:

1. **Per-forward baseline**: logit divergence and masked-argmax disagreement
   between twin models on the identical fully-corrupted input. This is the
   damage one forward pass shows.
2. **Trajectory divergence**: greedy diffusion generation from identical
   prompts at S in {1,2,4,8,16,32} steps; disagreement = fraction of filled
   positions where the twins commit different tokens. Greedy sampling makes
   both trajectories deterministic, so every difference traces to the weight
   nudge and nothing else.

If token disagreement grows with S beyond the per-forward baseline, the
feedback loop amplifies and sub-2-bit diffusion has a damage mode with no
autoregressive analogue. If it stays flat, commitment absorbs the noise and
the §5.4 chaos is a per-forward phenomenon only. Either answer closes the
question.
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

import numpy as np
import torch

from src.config import ModelConfig
from src.data import load_wikitext_tokens, make_training_windows
from src.loader import HF_MODEL_ID
from src.quantization import LowBitLinear, quantize_model_weights
from src.trainer import TrainConfig

LEDGER = Path("results/compounding_ledger.jsonl")


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def train_cell(quantize: bool, cfg: TrainConfig, windows, device):
    """The published grid cell, rebuilt exactly (MLX where available)."""
    config = ModelConfig(
        quantize_linear=quantize, quant_scheme="q1_58",
        use_attention_residuals=False, diffusion=True,
    )
    try:
        import mlx.core as mx

        from run_ablation import build_torch_model
        from src.mlx_backend.train import mlx_to_torch_state, run_qat

        class _Arm:
            name = "compounding"
            quantize_linear = quantize
            use_attention_residuals = False
            model_config = staticmethod(lambda: config)

        model, _ = run_qat(config, cfg, windows)
        state = mlx_to_torch_state(model)
        del model
        mx.clear_cache()
        return build_torch_model(_Arm(), state, device), config
    except ImportError:
        from src.trainer import run_qat_torch

        model, _ = run_qat_torch(config, cfg, windows, device)
        return model, config


def perturbed_twin(model, config, epsilon: float, seed: int, device):
    """Same masters nudged by relative Gaussian noise, then frozen alongside."""
    from src.model import ResABitForCausalLM

    twin = ResABitForCausalLM(config)
    source = dict(model.named_parameters())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in twin.named_parameters():
            if name == "lm_head.weight" and config.tie_word_embeddings:
                continue
            value = source[name]
            noise = torch.randn(value.shape, generator=generator) \
                * epsilon * value.detach().abs().cpu()
            param.copy_(value.detach().cpu() + noise)
    return twin.to(device).eval()


def freeze_and_count_flips(a, b) -> dict:
    """Freeze both models; count stored ternary levels that differ."""
    quantize_model_weights(a)
    quantize_model_weights(b)
    flipped = total = 0
    for ma, mb in zip(a.modules(), b.modules()):
        if isinstance(ma, LowBitLinear) and ma.is_quantized:
            la, lb = ma._levels(), mb._levels()
            flipped += int((la != lb).sum())
            total += la.numel()
    return {"flipped_levels": flipped, "total_levels": total,
            "flip_fraction": flipped / total if total else 0.0}


@torch.no_grad()
def per_forward_divergence(a, b, ids, mask_token_id) -> dict:
    la = a(input_ids=ids)["logits"].float()
    lb = b(input_ids=ids)["logits"].float()
    rel = float((la - lb).abs().max() / la.abs().max())
    masked = ids == mask_token_id
    disagree = float((la.argmax(-1)[masked] != lb.argmax(-1)[masked]).float().mean())
    return {"logit_rel_delta": rel, "argmax_disagreement": disagree}


@torch.no_grad()
def trajectory_disagreement(a, b, prompts, mask, steps: int) -> float:
    """Greedy generation from identical inputs; fraction of filled slots differing."""
    out_a = a.diffusion_generate(prompts.clone(), num_steps=steps, temperature=0.0)
    out_b = b.diffusion_generate(prompts.clone(), num_steps=steps, temperature=0.0)
    return float((out_a[mask] != out_b[mask]).float().mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps-list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--block-len", type=int, default=128)
    p.add_argument("--prefix-len", type=int, default=32,
                   help="unmasked context per block, drawn from wikitext val")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_cfg = TrainConfig(steps=args.train_steps, seed=args.seed)

    print(f"host    : {platform.platform()}  device {device}")
    print(f"epsilon : {args.epsilon:g} relative, seed {args.seed}")
    print(f"blocks  : {args.blocks} x {args.block_len} tok "
          f"({args.prefix_len} unmasked prefix)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    val_tokens = load_wikitext_tokens(tokenizer, "validation")
    windows = np.array(make_training_windows(train_tokens, train_cfg.seq_len))

    # Real text prefixes: a fully masked block is the degenerate case where
    # every position holds the same token, which §7's parity work showed is
    # numerically *easier* than the mixed case. Prefixes make it mixed.
    rng = np.random.default_rng(args.seed)
    offsets = rng.integers(0, val_tokens.numel() - args.prefix_len, args.blocks)

    results = {}
    for label, quantize in (("ternary", True), ("fp32_control", False)):
        print(f"\n=== {label}: retraining the published cell ===")
        started = time.time()
        model, config = train_cell(quantize, train_cfg, windows, device)
        twin = perturbed_twin(model, config, args.epsilon, args.seed + 1, device)

        flips = None
        if quantize:
            flips = freeze_and_count_flips(model, twin)
            print(f"  levels flipped by epsilon: {flips['flipped_levels']} "
                  f"of {flips['total_levels']} ({flips['flip_fraction']:.2e})")

        prompts = torch.full((args.blocks, args.block_len),
                             config.mask_token_id, device=device)
        for i, off in enumerate(offsets):
            prompts[i, :args.prefix_len] = val_tokens[off:off + args.prefix_len]
        fill_mask = prompts == config.mask_token_id

        baseline = per_forward_divergence(model, twin, prompts, config.mask_token_id)
        print(f"  per-forward: logit delta {baseline['logit_rel_delta']:.2e}  "
              f"argmax disagreement {baseline['argmax_disagreement']:.4f}")

        curve = {}
        for steps in args.steps_list:
            d = trajectory_disagreement(model, twin, prompts, fill_mask, steps)
            curve[steps] = d
            print(f"  S={steps:>2}: token disagreement {d:.4f}")

        results[label] = {
            "flips": flips, "per_forward": baseline,
            "disagreement_by_steps": curve,
            "wall_seconds": round(time.time() - started, 1),
        }
        del model, twin
        if device.type == "mps":
            torch.mps.empty_cache()

    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "stage": "smoke" if args.train_steps < 50 else "full",
            "commit": git_commit(), "device": str(device),
            "epsilon": args.epsilon, "seed": args.seed,
            "blocks": args.blocks, "block_len": args.block_len,
            "prefix_len": args.prefix_len,
            "train_config": asdict(train_cfg),
            "results": results,
        }) + "\n")
    print(f"\nappended to {LEDGER}")

    t = results["ternary"]["disagreement_by_steps"]
    lo, hi = t[min(t)], t[max(t)]
    base = results["ternary"]["per_forward"]["argmax_disagreement"]
    print(f"\nternary: per-forward argmax disagreement {base:.4f}; "
          f"trajectory {lo:.4f} at S={min(t)} -> {hi:.4f} at S={max(t)}")
    if hi > 2 * max(lo, 1e-9):
        print("READING: disagreement grows with steps -- the feedback loop "
              "amplifies, a damage mode with no autoregressive analogue.")
    else:
        print("READING: disagreement does not grow with steps -- commitment "
              "absorbs the noise; the chaos is per-forward only.")


if __name__ == "__main__":
    main()
