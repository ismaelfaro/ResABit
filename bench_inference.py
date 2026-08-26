"""Inference speed across the trained model variants.

    python bench_inference.py                    # all variants, MPS
    python bench_inference.py --device cpu --tiny   # mechanics smoke
    python bench_inference.py --variants fp32 q1_58_dequant

What this measures, and what it must not be read as
---------------------------------------------------
It measures *this repository's* forward paths. There are no fused low-bit
kernels here: the ``dequant`` path rebuilds the FP32 matrix on every forward
and the ``int8`` path loops one GEMM per group of 128. The
memory-bandwidth win that motivates sub-2-bit weights is delivered by
kernels this repo does not have (BitNet.cpp, PrismML's Metal kernels), so
**expect the quantized paths to lose to plain FP32 here** — that is a
finding about this implementation, not about the formats.

The structurally meaningful number is the AR/diffusion comparison. An
autoregressive model pays one cached forward per token; a diffusion model
pays a full uncached forward over the whole block per denoising step. At S
steps over a block of L tokens the diffusion model does S full forwards for
L tokens, so its per-token cost scales with S/L, not with L. That tradeoff
is architecture, not implementation, and it is the one this benchmark can
speak to.

Speed only. Weight *values* do not change timings, so variants are built
from the shipped Qwen weights and frozen; quality numbers live in the
ledgers and must never be quoted from here.

Refuses to run while a training process is alive: two GPU jobs on unified
memory starve each other (measured twice in this project), which would
corrupt both the benchmark and the run.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.config import ModelConfig
from src.loader import HF_MODEL_ID, load_pretrained
from src.model import TriDiForCausalLM
from src.quantization import quantize_model_weights

RESULTS = Path("results/inference_bench.jsonl")

TRAINING_ENTRYPOINTS = ("run_grid.py", "run_ablation.py", "run_diffusion_check.py",
                        "export_checkpoint.py")


def training_is_live() -> str | None:
    out = subprocess.run(["ps", "-Ao", "command="], capture_output=True,
                         text=True, check=False)
    for line in out.stdout.splitlines():
        if any(e in line for e in TRAINING_ENTRYPOINTS) and "tmux new-session" not in line:
            return line.strip()[:80]
    return None


# (name, quantize_linear, scheme, int8, diffusion)
VARIANTS = {
    "fp32":          (False, "q1_0", False, False),
    "q1_0_dequant":  (True, "q1_0", False, False),
    "q1_0_int8":     (True, "q1_0", True, False),
    "q1_58_dequant": (True, "q1_58", False, False),
    "q1_58_int8":    (True, "q1_58", True, False),
    "fp32_diff":     (False, "q1_0", False, True),
    "q1_58_diff":    (True, "q1_58", False, True),
}


def build(name: str, device: torch.device, tiny: bool, hf_state=None):
    quantize, scheme, int8, diffusion = VARIANTS[name]
    if tiny:
        config = ModelConfig(
            vocab_size=512, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            quantize_linear=quantize, quant_scheme=scheme, diffusion=diffusion,
            mask_token_id=511,
        )
        model = TriDiForCausalLM(config)
    else:
        config = ModelConfig(quantize_linear=quantize, quant_scheme=scheme,
                             diffusion=diffusion)
        model = load_pretrained(config, hf_state=hf_state, verbose=False)
    if quantize:
        quantize_model_weights(model)
        if int8:
            for module in model.quantized_modules():
                module.int8_inference = True
    return model.to(device).eval(), config


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_ar(model, config, device, prompt_len: int, new_tokens: int,
             repeats: int) -> dict:
    """Greedy decode with KV cache: prefill once, then one forward per token."""
    times = []
    for _ in range(repeats + 1):                     # first is warmup
        ids = torch.randint(0, config.vocab_size - 2, (1, prompt_len),
                            device=device)
        _sync(device)
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=new_tokens, temperature=0.0,
                             eos_token_id=-1)        # never stop early
        _sync(device)
        times.append(time.perf_counter() - t0)
        assert out.shape[1] == prompt_len + new_tokens
    times = times[1:]
    median = statistics.median(times)
    return {
        "mode": "ar_greedy_kv_cache",
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "median_s": round(median, 4),
        "tokens_per_s": round(new_tokens / median, 2),
        "spread_s": round(max(times) - min(times), 4),
    }


@torch.no_grad()
def bench_diffusion(model, config, device, block_len: int, steps: int,
                    repeats: int) -> dict:
    """Fill a fully masked block: `steps` uncached full forwards for `block_len` tokens."""
    times = []
    for _ in range(repeats + 1):
        ids = torch.full((1, block_len), config.mask_token_id, device=device)
        _sync(device)
        t0 = time.perf_counter()
        out = model.diffusion_generate(ids, num_steps=steps)
        _sync(device)
        times.append(time.perf_counter() - t0)
        assert int((out == config.mask_token_id).sum()) == 0
    times = times[1:]
    median = statistics.median(times)
    return {
        "mode": "diffusion_denoise",
        "block_len": block_len,
        "denoise_steps": steps,
        "median_s": round(median, 4),
        "tokens_per_s": round(block_len / median, 2),
        "spread_s": round(max(times) - min(times), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS))
    p.add_argument("--device", default=None)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--block-len", type=int, default=128)
    p.add_argument("--diffusion-steps", type=int, nargs="+",
                   default=[1, 4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--tiny", action="store_true",
                   help="2-layer random model: validates mechanics, numbers meaningless")
    p.add_argument("--force", action="store_true",
                   help="run even while a training job is live (corrupts both)")
    args = p.parse_args()

    if not args.force and (live := training_is_live()):
        raise SystemExit(
            f"a training job is live ({live}); benchmarking now starves both "
            f"on unified memory. Wait for it or pass --force deliberately."
        )

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    sys.stdout.reconfigure(line_buffering=True)
    print(f"device  : {device}  ({platform.machine()}, torch {torch.__version__})")
    print(f"repeats : {args.repeats} (median reported, first run discarded)")
    if args.tiny:
        print("TINY MODE: mechanics only, numbers meaningless\n")

    hf_state = None
    if not args.tiny:
        from src.loader import load_hf_state_dict
        hf_state = load_hf_state_dict()

    records = []
    for name in args.variants:
        model, config = build(name, device, args.tiny, hf_state)
        diffusion = VARIANTS[name][3]
        print(f"== {name} ==")
        if diffusion:
            for steps in args.diffusion_steps:
                r = bench_diffusion(model, config, device, args.block_len,
                                    steps, args.repeats)
                print(f"  block {r['block_len']} @ {steps:>2} steps : "
                      f"{r['tokens_per_s']:>8.1f} tok/s  ({r['median_s']:.3f}s)")
                records.append({"variant": name, **r})
        else:
            r = bench_ar(model, config, device, args.prompt_len,
                         args.new_tokens, args.repeats)
            print(f"  decode {r['new_tokens']} tok      : "
                  f"{r['tokens_per_s']:>8.1f} tok/s  ({r['median_s']:.3f}s)")
            records.append({"variant": name, **r})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    RESULTS.parent.mkdir(exist_ok=True)
    stamp = {
        "device": str(device),
        "torch": torch.__version__,
        "host": platform.platform(),
        "tiny": args.tiny,
        "base_model": None if args.tiny else HF_MODEL_ID,
    }
    with RESULTS.open("a") as f:
        for r in records:
            f.write(json.dumps({**stamp, **r}) + "\n")
    print(f"\nappended {len(records)} rows to {RESULTS}")

    if not args.tiny:
        print(
            "\nreading guide: fp32 beating the quantized paths here is expected\n"
            "-- this repo has no fused low-bit kernels; dequant rebuilds the\n"
            "matrix per forward and int8 loops per group. The transferable\n"
            "number is the AR-vs-diffusion shape: cached per-token decode\n"
            "against S uncached block forwards for L tokens."
        )


if __name__ == "__main__":
    main()
