"""
compare.py — Compare original Qwen1.5-0.5B-Chat vs 1-bit + Attention Residual model.

Measures:
  - Parameter count & model size (MB)
  - Bits per weight (effective)
  - Forward pass speed (tokens/sec)
  - Output distribution similarity (cosine sim on logits)
  - Intelligence density metric (from Bonsai paper)

Usage (no GPU / HF credentials needed — uses random weights):
    python compare.py --random-weights

With real Qwen weights:
    python compare.py --checkpoint ./checkpoints/qwen0.5b-1bit/model.pt
"""

import argparse
import math
import time
import io
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.model import OneBitResidualLM
from src.quantization import quantize_model_weights


# ── Size helpers ──────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def effective_bits_per_weight(model: nn.Module) -> float:
    """Approximate effective bits per weight after 1-bit quantisation."""
    from src.quantization import OneBitLinear
    bit_weights = 0
    total_weights = 0
    float_weights = 0
    for m in model.modules():
        if isinstance(m, OneBitLinear) and m._quantized:
            bit_weights += m.out_features * m.in_features
        elif isinstance(m, (nn.Linear, nn.Embedding)):
            float_weights += m.weight.numel()
        total_weights += (m.out_features * m.in_features
                          if isinstance(m, OneBitLinear) else 0)
    if bit_weights == 0:
        return 16.0
    # 1 sign bit + 16/128 scale bits per weight
    bpw_1bit = 1 + 16 / 128      # = 1.125
    total = bit_weights + float_weights
    return (bit_weights * bpw_1bit + float_weights * 16) / max(1, total)


# ── Throughput ────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_throughput(model: nn.Module, device: str,
                       vocab_size: int, seq_len: int = 128,
                       n_runs: int = 5) -> float:
    model.eval()
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    # Warm up
    for _ in range(2):
        model(ids)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(ids)
    elapsed = time.perf_counter() - t0
    tokens = seq_len * n_runs
    return tokens / elapsed


# ── Logit similarity ──────────────────────────────────────────────────────────

@torch.no_grad()
def logit_cosine_similarity(m1: nn.Module, m2: nn.Module,
                             device: str, vocab_size: int,
                             seq_len: int = 32) -> float:
    """Mean cosine similarity between logits of two models on random inputs."""
    m1.eval(); m2.eval()
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    l1 = m1(ids)["logits"].float()
    l2 = m2(ids)["logits"].float()
    sims = torch.cosine_similarity(l1.view(-1, vocab_size),
                                   l2.view(-1, vocab_size), dim=-1)
    return sims.mean().item()


# ── Intelligence density ──────────────────────────────────────────────────────

def intelligence_density(avg_score: float, size_gb: float) -> float:
    """D = -log(1 - score/100) / size_GB  (from Bonsai paper, eq.1)."""
    pe = max(1e-9, 1.0 - avg_score / 100.0)
    return -math.log(pe) / size_gb


# ── Main comparison ───────────────────────────────────────────────────────────

def compare(args: argparse.Namespace) -> None:
    device = args.device

    config = ModelConfig(num_hidden_layers=4 if args.fast else ModelConfig().num_hidden_layers,
                         quantize_embeddings=False)   # keep embed as nn.Embedding for baseline

    # ── Baseline: standard transformer (no 1-bit, no attn residual) ──────────
    baseline_cfg = ModelConfig(
        num_hidden_layers=config.num_hidden_layers,
        use_attention_residuals=False,
        quantize_embeddings=False,
    )
    baseline = OneBitResidualLM(baseline_cfg).to(device)
    # Replace OneBitLinear back to nn.Linear for a fair baseline
    _replace_1bit_with_linear(baseline)
    baseline.eval()

    # ── 1-bit + Attention Residual model ─────────────────────────────────────
    ours_cfg = ModelConfig(
        num_hidden_layers=config.num_hidden_layers,
        use_attention_residuals=True,
        quantize_embeddings=False,
    )
    ours = OneBitResidualLM(ours_cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ours.load_state_dict(ckpt["state_dict"], strict=False)
    quantize_model_weights(ours)
    ours.eval()

    # ── Metrics ──────────────────────────────────────────────────────────────
    V = config.vocab_size

    b_params = param_count(baseline)
    o_params = param_count(ours)

    b_size = model_size_mb(baseline)
    o_size = model_size_mb(ours)

    b_bpw = 16.0
    o_bpw = effective_bits_per_weight(ours)

    b_tps = measure_throughput(baseline, device, V)
    o_tps = measure_throughput(ours, device, V)

    cosim = logit_cosine_similarity(baseline, ours, device, V)

    # ── Print ─────────────────────────────────────────────────────────────────
    w = 36
    sep = "─" * (w + 30)
    print()
    print("=" * (w + 30))
    print(f"  Comparison: Baseline vs 1-bit + Attention Residual")
    print("=" * (w + 30))
    print(f"{'Metric':<{w}} {'Baseline':>12}  {'1-bit+AR':>12}  {'Ratio':>8}")
    print(sep)
    print(f"{'Parameters (M)':<{w}} {b_params/1e6:>12.1f}  {o_params/1e6:>12.1f}  {'1.00×':>8}")
    print(f"{'Model size (MB)':<{w}} {b_size:>12.1f}  {o_size:>12.1f}  {b_size/max(o_size,1e-6):>7.1f}×")
    print(f"{'Bits per weight':<{w}} {b_bpw:>12.3f}  {o_bpw:>12.3f}  {b_bpw/o_bpw:>7.1f}×")
    print(f"{'Throughput (tok/s)':<{w}} {b_tps:>12.1f}  {o_tps:>12.1f}  {o_tps/max(b_tps,1e-6):>7.1f}×")
    print(f"{'Logit cosine sim':<{w}} {'—':>12}  {cosim:>12.4f}  {'—':>8}")
    print(sep)

    # Intelligence density (illustrative — using hypothetical benchmark scores)
    # Real scores require running the full benchmark suite
    print()
    print("  Intelligence Density (illustrative, using Bonsai paper scores as reference)")
    print(sep)
    print(f"  {'Model':<30} {'Avg Score':>10}  {'Size (GB)':>10}  {'I-Density':>10}")
    print(f"  {'─'*60}")
    rows = [
        ("1-bit Bonsai 8B (paper)",  70.5, 1.15),
        ("Qwen3 8B FP16 (paper)",    79.3, 16.38),
        ("Qwen1.5 0.5B FP16 (est.)", 45.0, b_size / 1e3),
        ("Ours 1-bit (same arch)",   45.0, o_size / 1e3),
    ]
    for name, score, size_gb in rows:
        d = intelligence_density(score, max(size_gb, 1e-6))
        print(f"  {name:<30} {score:>10.1f}  {size_gb:>10.4f}  {d:>10.4f}")
    print()
    print("  Note: scores for 'Ours' require actual benchmark evaluation.")
    print("        Size ratio alone gives a ~{:.0f}× gain in potential density.".format(
        b_size / max(o_size, 1e-6)
    ))


def _replace_1bit_with_linear(model: nn.Module) -> None:
    """Swap OneBitLinear → nn.Linear (for the baseline model)."""
    from src.quantization import OneBitLinear
    for parent_name, module in list(model.named_children()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, OneBitLinear):
                new = nn.Linear(child.in_features, child.out_features,
                                bias=child.bias is not None)
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(module, child_name, new)
            else:
                _replace_1bit_with_linear(child)
        if isinstance(module, OneBitLinear):
            new = nn.Linear(module.in_features, module.out_features,
                            bias=module.bias is not None)
            with torch.no_grad():
                new.weight.copy_(module.weight)
            setattr(model, parent_name, new)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs 1-bit+AR model")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fast", action="store_true",
                   help="Use 4-layer model for quick comparison")
    args = p.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
