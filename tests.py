"""
tests.py — Unit tests for 1-bit + Attention Residual model.

Run:
    python tests.py
"""

import math
import sys
import traceback
import torch
import torch.nn as nn

from src.config import ModelConfig
from src.quantization import OneBitLinear, quantize_model_weights, replace_linear_with_1bit
from src.model import OneBitResidualLM, RMSNorm, SwiGLUMLP, GroupedQueryAttention, DecoderLayer

PASS = "  PASS"
FAIL = "  FAIL"
results: list[bool] = []
_ALL_TESTS: list = []


def test(name: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                print(f"{PASS}  {name}")
                results.append(True)
            except Exception:
                print(f"{FAIL}  {name}")
                traceback.print_exc()
                results.append(False)
        _ALL_TESTS.append(wrapper)
        return wrapper
    return decorator


# ── Quantization tests ────────────────────────────────────────────────────────

@test("OneBitLinear: output shape matches nn.Linear")
def _():
    layer = OneBitLinear(128, 64, group_size=128)
    x = torch.randn(2, 10, 128)
    out = layer(x)
    assert out.shape == (2, 10, 64), f"Expected (2,10,64) got {out.shape}"


@test("OneBitLinear: scales have correct shape")
def _():
    layer = OneBitLinear(256, 32, group_size=128)
    w = layer.weight.float()
    w_grouped = w.view(32, 2, 128)
    scales = w_grouped.abs().amax(dim=-1)
    assert scales.shape == (32, 2)


@test("OneBitLinear: effective weights are ±scale (STE forward)")
def _():
    G = 128
    layer = OneBitLinear(G, 1, group_size=G)
    # Force weight to +1 everywhere so effective weight = +scale
    with torch.no_grad():
        layer.weight.fill_(0.5)
    x = torch.ones(1, 1, G)
    out = layer(x)
    scale = layer.weight.abs().amax()
    expected = scale.item() * G      # sum of +scale over G inputs
    assert abs(out.item() - expected) < 1e-3, f"{out.item()} != {expected}"


@test("OneBitLinear.quantize(): packed bits shape")
def _():
    in_f, out_f, G = 128, 32, 128
    layer = OneBitLinear(in_f, out_f, group_size=G)
    layer.quantize()
    assert layer._quantized
    assert layer.weight_bits.shape == (out_f, in_f // 8)
    assert layer.weight_scales.shape == (out_f, in_f // G)


@test("OneBitLinear.quantize(): dequantised output close to pre-quantise output")
def _():
    torch.manual_seed(0)
    layer = OneBitLinear(128, 32, group_size=128)
    x = torch.randn(1, 1, 128)
    out_before = layer(x).detach()
    layer.quantize()
    out_after = layer(x).detach()
    # Not exact, but should be in the same ballpark
    rel_err = (out_before - out_after).abs().mean() / out_before.abs().mean().clamp(min=1e-6)
    assert rel_err < 1.0, f"Relative error too large: {rel_err:.4f}"


@test("OneBitLinear: STE gradient flows through sign")
def _():
    layer = OneBitLinear(128, 32, group_size=128)
    x = torch.randn(2, 4, 128, requires_grad=True)
    out = layer(x).sum()
    out.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


@test("replace_linear_with_1bit: converts all nn.Linear layers")
def _():
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    replace_linear_with_1bit(model, group_size=64)
    for m in model.modules():
        assert not isinstance(m, nn.Linear), "Found un-replaced nn.Linear"


@test("quantize_model_weights: freezes all OneBitLinear layers")
def _():
    model = nn.Sequential(OneBitLinear(128, 32, group_size=128))
    quantize_model_weights(model)
    for m in model.modules():
        if isinstance(m, OneBitLinear):
            assert m._quantized


# ── Architecture tests ────────────────────────────────────────────────────────

@test("RMSNorm: output shape preserved")
def _():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


@test("RMSNorm: normalises to unit RMS (approx)")
def _():
    norm = RMSNorm(128)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.randn(4, 8, 128)
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().max() < 0.05, f"RMS off: {rms.mean():.4f}"


@test("GroupedQueryAttention: output shape")
def _():
    config = ModelConfig()
    attn = GroupedQueryAttention(config)
    x = torch.randn(2, 8, config.hidden_size)
    out, kv = attn(x, use_cache=True)
    assert out.shape == (2, 8, config.hidden_size)
    assert kv is not None


@test("GroupedQueryAttention: causal mask — future tokens unseen")
def _():
    config = ModelConfig()
    attn = GroupedQueryAttention(config)
    attn.eval()
    x = torch.randn(1, 4, config.hidden_size)
    with torch.no_grad():
        out1, _ = attn(x[:, :3, :])
        out2, _ = attn(x)
    # Output at position 0 should be identical regardless of future tokens
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4), "Causal mask broken"


@test("SwiGLUMLP: output shape")
def _():
    config = ModelConfig()
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 6, config.hidden_size)
    assert mlp(x).shape == x.shape


@test("DecoderLayer: forward shape with attention residual")
def _():
    config = ModelConfig(use_attention_residuals=True)
    layer = DecoderLayer(config)
    h = torch.randn(2, 8, config.hidden_size)
    acc = torch.zeros_like(h)
    h_out, acc_out, _ = layer(h, acc)
    assert h_out.shape == h.shape
    assert acc_out.shape == h.shape
    # acc should have changed (accumulated attention output)
    assert not torch.allclose(acc_out, acc)


@test("DecoderLayer: forward shape without attention residual")
def _():
    config = ModelConfig(use_attention_residuals=False)
    layer = DecoderLayer(config)
    h = torch.randn(2, 8, config.hidden_size)
    h_out, acc_out, _ = layer(h, None)
    assert h_out.shape == h.shape
    assert acc_out is None


# ── Full model tests ──────────────────────────────────────────────────────────

@test("OneBitResidualLM: forward returns logits of correct shape")
def _():
    config = ModelConfig(num_hidden_layers=2, quantize_embeddings=False)
    model = OneBitResidualLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, config.vocab_size)
    assert out["loss"] is None


@test("OneBitResidualLM: loss computed when labels provided")
def _():
    config = ModelConfig(num_hidden_layers=2, quantize_embeddings=False)
    model = OneBitResidualLM(config)
    ids = torch.randint(0, config.vocab_size, (1, 16))
    out = model(ids, labels=ids)
    assert out["loss"] is not None
    assert out["loss"].item() > 0
    assert not out["loss"].isnan()


@test("OneBitResidualLM: KV cache matches non-cached output")
def _():
    config = ModelConfig(num_hidden_layers=2, quantize_embeddings=False)
    model = OneBitResidualLM(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out_no_cache = model(ids)["logits"]
        out_cached = model(ids, use_cache=True)["logits"]
    assert torch.allclose(out_no_cache, out_cached, atol=1e-4)


@test("OneBitResidualLM: attention residual accumulator grows across layers")
def _():
    config = ModelConfig(num_hidden_layers=4, use_attention_residuals=True,
                         quantize_embeddings=False)
    model = OneBitResidualLM(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 4))
    # Patch layers to record accumulators
    accs = []
    original_forwards = []
    for layer in model.layers:
        orig = layer.forward
        original_forwards.append(orig)
        def make_hook(orig_fn):
            def hook(h, acc, **kw):
                out_h, out_acc, kv = orig_fn(h, acc, **kw)
                if out_acc is not None:
                    accs.append(out_acc.detach().norm().item())
                return out_h, out_acc, kv
            return hook
        layer.forward = make_hook(orig)
    with torch.no_grad():
        model(ids)
    # Each layer should have a non-zero accumulator
    assert len(accs) == 4
    assert all(a > 0 for a in accs), f"Zero accumulators: {accs}"


@test("OneBitResidualLM: parameter count roughly matches Qwen1.5-0.5B scale")
def _():
    config = ModelConfig()
    model = OneBitResidualLM(config)
    n = sum(p.numel() for p in model.parameters())
    # Qwen1.5-0.5B backbone ~464M; untied OneBitLinear embed+lm_head adds ~156M extra
    # so expect 400–700M total
    assert 400e6 < n < 700e6, f"Unexpected param count: {n/1e6:.1f}M"


@test("OneBitResidualLM: quantize then forward (no float weight access)")
def _():
    config = ModelConfig(num_hidden_layers=2, quantize_embeddings=False)
    model = OneBitResidualLM(config)
    quantize_model_weights(model)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids)
    assert out["logits"].shape == (1, 8, config.vocab_size)
    assert not out["logits"].isnan().any()


# ── Intelligence density metric test ─────────────────────────────────────────

@test("Intelligence density formula (from Bonsai paper)")
def _():
    # D = -log(1 - score/100) / size_gb
    def density(avg_score: float, size_gb: float) -> float:
        pe = 1.0 - avg_score / 100.0
        return -math.log(pe) / size_gb

    # Verify Bonsai 8B from paper: score=70.5, size=1.15GB → D≈1.060
    d = density(70.5, 1.15)
    assert abs(d - 1.060) < 0.01, f"Expected ~1.060, got {d:.4f}"

    # Qwen3 8B: score=79.3, size=16.38GB → D≈0.096
    d2 = density(79.3, 16.38)
    assert abs(d2 - 0.096) < 0.005, f"Expected ~0.096, got {d2:.4f}"

    # Bonsai should have ~11× higher density
    assert density(70.5, 1.15) / density(79.3, 16.38) > 10


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("1-bit + Attention Residual — Test Suite")
    print("=" * 60)

    for fn in _ALL_TESTS:
        fn()

    passed = sum(results)
    total  = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
