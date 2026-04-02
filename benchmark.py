"""
benchmark.py — Full comparison: Original Qwen1.5-0.5B-Chat vs 1-bit + Attention Residual.

Pipeline
--------
  1. Load original Qwen1.5-0.5B-Chat via transformers (FP32/BF16)
  2. Convert weights into OneBitResidualLM + quantise to Q1_0_g128
  3. Run both on the same prompts and tasks
  4. Report: size, throughput, perplexity, generation quality

Usage
-----
    python benchmark.py
    python benchmark.py --device mps          # Apple Silicon GPU
    python benchmark.py --seq-len 256 --runs 10
"""

import argparse
import io
import json
import math
import os
import sys
import time
import textwrap
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

# ── benchmark prompts ──────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "Artificial intelligence is transforming",
    "The key advantage of 1-bit neural networks is",
    "To make a good cup of coffee you need",
    "The most important invention of the 20th century was",
]

PERPLEXITY_TEXT = (
    "Machine learning is a branch of artificial intelligence that focuses on building "
    "systems that learn from data. These systems improve their performance over time "
    "without being explicitly programmed for each task. Deep learning, a subset of "
    "machine learning, uses neural networks with many layers to model complex patterns."
)

BENCHMARK_QUESTIONS = [
    ("Knowledge",    "What is the speed of light in a vacuum?"),
    ("Reasoning",    "If all cats are animals and some animals are pets, are all cats pets?"),
    ("Math",         "What is 17 multiplied by 24?"),
    ("Coding",       "Write a Python function to check if a number is prime."),
    ("Instruction",  "List three benefits of exercise in bullet points."),
]

# ── utilities ──────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_throughput(model, input_ids, n_runs=5) -> float:
    model.eval()
    for _ in range(2):                # warm-up
        model(input_ids)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(input_ids)
    elapsed = time.perf_counter() - t0
    return (input_ids.shape[1] * n_runs) / elapsed


@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str, max_len=256) -> float:
    model.eval()
    ids = tokenizer.encode(text, return_tensors="pt")[:, :max_len].to(device)
    if ids.shape[1] < 2:
        return float("nan")

    # Our model
    from src.model import OneBitResidualLM
    if isinstance(model, OneBitResidualLM):
        out = model(input_ids=ids, labels=ids)
        return math.exp(min(out["loss"].item(), 20))

    # HF model
    out = model(input_ids=ids, labels=ids)
    return math.exp(min(out.loss.item(), 20))


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: str,
                  max_new: int = 80) -> str:
    model.eval()
    from src.model import OneBitResidualLM

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if isinstance(model, OneBitResidualLM):
        out_ids = model.generate(ids, max_new_tokens=max_new, temperature=0.01, top_p=1.0)
    else:
        out_ids = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out_ids[0][ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── load models ───────────────────────────────────────────────────────────────

def load_original(device: str):
    """Load Qwen1.5-0.5B-Chat via HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading original {MODEL_NAME} …")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    )
    model = model.to(device).eval()
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    return model, tokenizer


def convert_to_1bit(hf_model, device: str):
    """
    Copy HuggingFace weights into OneBitResidualLM and quantise.
    Uses quantize_embeddings=False so nn.Embedding lookup is exact.
    """
    from src.config import ModelConfig
    from src.model import OneBitResidualLM
    from src.quantization import quantize_model_weights

    config = ModelConfig(
        quantize_embeddings=False,  # keep embed+lm_head in FP32 for correctness
        use_attention_residuals=True,
    )
    our_model = OneBitResidualLM(config).to(device)

    # ── Weight mapping ────────────────────────────────────────────────────────
    hf_sd = {k.replace("model.", "", 1): v for k, v in hf_model.state_dict().items()}

    our_sd = our_model.state_dict()
    loaded, skipped = 0, []
    for key in our_sd:
        if key in hf_sd and our_sd[key].shape == hf_sd[key].shape:
            our_sd[key] = hf_sd[key].to(our_sd[key].dtype)
            loaded += 1
        else:
            skipped.append(key)

    our_model.load_state_dict(our_sd, strict=False)
    print(f"  Weights loaded: {loaded}  |  New/skipped: {len(skipped)}")

    # ── Quantise linear layers ────────────────────────────────────────────────
    print("  Quantising to Q1_0_g128 …")
    quantize_model_weights(our_model)
    our_model.eval()
    return our_model


# ── benchmark sections ────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


def run_benchmark(args: argparse.Namespace) -> None:
    device = args.device

    # ── Load ──────────────────────────────────────────────────────────────────
    section("1. Loading Models")
    original, tokenizer = load_original(device)

    print(f"\nConverting to 1-bit + Attention Residual …")
    t0 = time.perf_counter()
    optimized = convert_to_1bit(original, device)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # ── Size ──────────────────────────────────────────────────────────────────
    section("2. Model Size & Parameters")
    orig_mb  = model_size_mb(original)
    opt_mb   = model_size_mb(optimized)
    orig_p   = param_count(original)
    opt_p    = param_count(optimized)

    w = 36
    print(f"{'Metric':<{w}} {'Original':>12}  {'1-bit+AR':>12}  {'Ratio':>8}")
    print(f"{'─'*70}")
    print(f"{'Parameters (M)':<{w}} {orig_p/1e6:>12.1f}  {opt_p/1e6:>12.1f}  {'—':>8}")
    print(f"{'Checkpoint size (MB)':<{w}} {orig_mb:>12.1f}  {opt_mb:>12.1f}  {orig_mb/opt_mb:>7.1f}×")
    print(f"{'Bits/weight (linear)':<{w}} {'32.0':>12}  {'~1.125':>12}  {'~28×':>8}")

    results = {
        "original_mb": orig_mb, "optimized_mb": opt_mb,
        "size_ratio": orig_mb / opt_mb,
    }

    # ── Throughput ────────────────────────────────────────────────────────────
    section("3. Throughput (tokens/sec)")
    ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)

    # Wrap original forward for throughput (ignore labels)
    class _HFWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    orig_tps = measure_throughput(_HFWrapper(original), ids, args.runs)
    opt_tps  = measure_throughput(optimized, ids, args.runs)

    print(f"{'Metric':<{w}} {'Original':>12}  {'1-bit+AR':>12}  {'Speedup':>8}")
    print(f"{'─'*70}")
    print(f"{'Forward pass (tok/s)':<{w}} {orig_tps:>12.1f}  {opt_tps:>12.1f}  {opt_tps/orig_tps:>7.1f}×")
    print(f"{'Seq length used':<{w}} {args.seq_len:>12}  {args.seq_len:>12}  {'—':>8}")

    results.update({"orig_tps": orig_tps, "opt_tps": opt_tps,
                    "speedup": opt_tps / orig_tps})

    # ── Perplexity ────────────────────────────────────────────────────────────
    section("4. Perplexity (lower = better)")
    orig_ppl = perplexity(original, tokenizer, PERPLEXITY_TEXT, device)
    opt_ppl  = perplexity(optimized, tokenizer, PERPLEXITY_TEXT, device)

    print(f"{'Metric':<{w}} {'Original':>12}  {'1-bit+AR':>12}")
    print(f"{'─'*70}")
    print(f"{'Perplexity':<{w}} {orig_ppl:>12.2f}  {opt_ppl:>12.2f}")
    print(f"{'Log-PPL diff':<{w}} {'—':>12}  {math.log(opt_ppl)-math.log(orig_ppl):>+12.3f}")
    print()
    print("  Note: perplexity increases after 1-bit quantisation is expected —")
    print("  fine-tuning the 1-bit model recovers most of the gap (see train.py).")

    results.update({"orig_ppl": orig_ppl, "opt_ppl": opt_ppl})

    # ── Generation quality ────────────────────────────────────────────────────
    section("5. Text Generation (greedy, 80 new tokens)")
    for prompt in PROMPTS[:3]:
        orig_out = generate_text(original,  tokenizer, prompt, device, max_new=80)
        opt_out  = generate_text(optimized, tokenizer, prompt, device, max_new=80)
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Original : {textwrap.shorten(orig_out, 120)!r}")
        print(f"  1-bit+AR : {textwrap.shorten(opt_out,  120)!r}")

    # ── Task evaluation ───────────────────────────────────────────────────────
    section("6. Qualitative Task Benchmark")
    print(f"{'Category':<18} {'Original':^30}  {'1-bit+AR':^30}")
    print(f"{'─'*82}")
    for cat, q in BENCHMARK_QUESTIONS:
        orig_ans = generate_text(original,  tokenizer, q, device, max_new=60)
        opt_ans  = generate_text(optimized, tokenizer, q, device, max_new=60)
        print(f"  {cat:<16} {textwrap.shorten(orig_ans, 28):<30}  {textwrap.shorten(opt_ans, 28):<30}")

    # ── Intelligence density ──────────────────────────────────────────────────
    section("7. Intelligence Density (Bonsai paper metric)")
    def i_density(score: float, size_gb: float) -> float:
        pe = max(1e-9, 1.0 - score / 100.0)
        return -math.log(pe) / size_gb

    rows = [
        ("Original Qwen1.5-0.5B (FP32)", 45.0, orig_mb / 1e3),
        ("1-bit+AR Qwen1.5-0.5B",        45.0, opt_mb  / 1e3),
        ("1-bit Bonsai 8B (paper)",       70.5, 1.15),
        ("Qwen3 8B FP16 (paper)",         79.3, 16.38),
    ]
    print(f"  {'Model':<36} {'Score':>6}  {'Size(GB)':>9}  {'I-Density':>10}")
    print(f"  {'─'*65}")
    for name, score, gb in rows:
        d = i_density(score, gb)
        print(f"  {name:<36} {score:>6.1f}  {gb:>9.4f}  {d:>10.4f}")
    print()
    print("  * Score = hypothetical; replace with real benchmark results after training.")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    print(f"  Size reduction : {orig_mb/opt_mb:.1f}× smaller ({orig_mb:.0f} MB → {opt_mb:.0f} MB)")
    print(f"  Throughput     : {opt_tps/orig_tps:.1f}× ({'faster' if opt_tps > orig_tps else 'slower on CPU — needs Metal/CUDA kernels'})")
    print(f"  PPL delta      : +{opt_ppl - orig_ppl:.1f} (recover with fine-tuning via train.py)")
    print(f"  Next step      : ./setup_mlx.sh && python inference_mlx.py -p 'Hello!'")
    print()

    # Save JSON
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark original vs 1-bit+AR model")
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | mps")
    p.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length for throughput test")
    p.add_argument("--runs", type=int, default=5,
                   help="Repetitions for throughput measurement")
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_benchmark(args)


if __name__ == "__main__":
    main()
