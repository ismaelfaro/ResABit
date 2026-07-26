"""The MLX training backend must agree with the PyTorch reference.

PyTorch parity against HuggingFace (``test_parity.py``) proves the reference
is really Qwen. This file proves the MLX port is really the reference. The
chain has to be unbroken, because MLX produces the numbers that go in the
paper and PyTorch is the only thing anchoring them to a known model.

Both quantised and unquantised paths are checked: the fake-quantise kernels
are written twice, once per framework, and a mismatch there would look
exactly like an effect of quantisation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from src.config import ModelConfig  # noqa: E402
from src.loader import load_hf_state_dict, load_pretrained  # noqa: E402
from src.mlx_backend import load_mlx_pretrained  # noqa: E402

pytestmark = pytest.mark.slow

_IDS = [[151643, 3838, 374, 264, 220, 16, 15257, 4128, 1614, 30, 151645, 198]]


@pytest.fixture(scope="module")
def hf_state():
    return load_hf_state_dict()


def _relative_delta(config: ModelConfig, hf_state, device) -> float:
    torch_model = load_pretrained(config, hf_state=hf_state, verbose=False).eval()
    with torch.no_grad():
        expected = torch_model(input_ids=torch.tensor(_IDS))["logits"].numpy()
    del torch_model

    mlx_model = load_mlx_pretrained(config, hf_state=hf_state)
    with mx.stream(device):
        actual = np.array(mlx_model(mx.array(_IDS)))
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def test_backends_agree_exactly_on_cpu(hf_state):
    """On CPU, MLX and PyTorch are both true FP32 and must agree tightly."""
    delta = _relative_delta(
        ModelConfig(quantize_linear=False, use_attention_residuals=False),
        hf_state,
        mx.cpu,
    )
    assert delta < 1e-5, f"relative delta = {delta:.2e}"


def test_metal_backend_within_documented_precision(hf_state):
    """MLX's Metal matmul is not true FP32; pin how far it may drift.

    Measured against float64 on an M5, a single 1024-wide FP32 matmul lands
    0.098 away on the GPU versus 0.00013 on the CPU -- roughly bf16-grade
    accumulation. Over 24 unquantised layers that settles at ~4e-3 relative
    error on the logits.

    Consequence for the ablation: training runs on Metal, where the error is
    far below the perturbation binarization itself applies, but every
    reported metric is computed by the PyTorch harness.
    """
    delta = _relative_delta(
        ModelConfig(quantize_linear=False, use_attention_residuals=False),
        hf_state,
        mx.gpu,
    )
    assert delta < 1e-2, f"relative delta = {delta:.2e}"


def test_binarized_network_amplifies_perturbations(hf_state):
    """A binarised, un-retrained network is numerically chaotic.

    The same weights on two FP32 backends diverge by ~1e-2 relative once the
    projections are binarised, against ~1e-5 unquantised, and the gap grows
    monotonically with depth (3.5e-3 at layer 0 to 3.2e-1 at layer 23). No
    backend is wrong: sign() sits on a discontinuity, so a one-ulp
    disagreement flips a weight and the error compounds through the stack.

    This is why post-training 1-bit collapses, and why the PTQ row of the
    results table is implementation-sensitive in a way the QAT rows are not.
    Recovery of this conditioning is itself a thing QAT should buy, so the
    bound here is deliberately loose -- it documents the phenomenon rather
    than constraining it.
    """
    quantized = _relative_delta(
        ModelConfig(quantize_linear=True, use_attention_residuals=False),
        hf_state,
        mx.cpu,
    )
    plain = _relative_delta(
        ModelConfig(quantize_linear=False, use_attention_residuals=False),
        hf_state,
        mx.cpu,
    )
    assert quantized > 100 * plain, (
        f"expected binarisation to amplify backend disagreement; "
        f"got {quantized:.2e} quantised vs {plain:.2e} plain"
    )
    assert quantized < 0.1, f"amplification beyond documented range: {quantized:.2e}"


def test_fake_quantize_kernels_agree():
    from src.mlx_backend.model import fake_quantize as mlx_fq
    from src.quantization import fake_quantize as torch_fq

    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 512)).astype(np.float32)

    expected = torch_fq(torch.from_numpy(w), 128).numpy()
    actual = np.array(mlx_fq(mx.array(w), 128))
    assert np.abs(actual - expected).max() < 1e-6


def test_only_three_weight_levels_per_group():
    """A Q1_0 group must collapse to exactly {-s, +s}."""
    from src.quantization import fake_quantize

    w = torch.randn(4, 256)
    q = fake_quantize(w, 128)
    for row in range(4):
        for g in range(2):
            group = q[row, g * 128 : (g + 1) * 128]
            assert group.abs().unique().numel() == 1
            assert group.unique().numel() <= 2
