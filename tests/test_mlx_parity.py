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


def test_diffusion_backends_agree_on_cpu(hf_state):
    """The bidirectional path has to be pinned too, not just the causal one.

    Dropping the causal mask is a one-line change in each backend, made
    twice, in files that do not import each other. If only one of them takes
    effect the model still trains and still reports a falling loss -- it is
    just answering a different question in each framework.
    """
    delta = _relative_delta(
        ModelConfig(quantize_linear=False, use_attention_residuals=False,
                    diffusion=True),
        hf_state,
        mx.cpu,
    )
    assert delta < 1e-5, f"relative delta = {delta:.2e}"


def test_diffusion_and_causal_are_actually_different(hf_state):
    """Guard against `diffusion=True` silently doing nothing.

    Both parity tests above pass if the flag is ignored in both backends at
    once, because the two would agree with each other while agreeing about
    the wrong architecture.
    """
    from src.loader import load_pretrained

    ids = torch.tensor(_IDS)
    outputs = {}
    for flag in (False, True):
        model = load_pretrained(
            ModelConfig(quantize_linear=False, use_attention_residuals=False,
                        diffusion=flag),
            hf_state=hf_state,
            verbose=False,
        ).eval()
        with torch.no_grad():
            outputs[flag] = model(input_ids=ids)["logits"].numpy()
        del model

    # The first position is the sharpest probe: under a causal mask it sees
    # only itself, under a bidirectional one it sees the whole sequence.
    first = np.abs(outputs[False][0, 0] - outputs[True][0, 0]).max()
    assert first > 1e-3, f"causal and diffusion agree at position 0: {first:.2e}"


def test_diffusion_loss_agrees_across_backends(hf_state):
    """Same corruption in, same NELBO out.

    The corruption is built here and handed to both backends precisely so
    this test compares numbers rather than distributions.
    """
    from src.diffusion import corrupt, sample_rates
    from src.mlx_backend import load_mlx_pretrained

    config = ModelConfig(quantize_linear=False, use_attention_residuals=False,
                         diffusion=True)
    ids = torch.tensor(_IDS)
    generator = torch.Generator().manual_seed(0)
    rates = sample_rates(ids.shape[0], generator=generator)
    _, mask = corrupt(ids, rates, config.mask_token_id, generator=generator)

    torch_model = load_pretrained(config, hf_state=hf_state, verbose=False).eval()
    with torch.no_grad():
        expected = float(torch_model.diffusion_loss(ids, rates, mask))
    del torch_model

    mlx_model = load_mlx_pretrained(config, hf_state=hf_state)
    with mx.stream(mx.cpu):
        actual = float(
            mlx_model.diffusion_loss(
                mx.array(ids.numpy()),
                mx.array(rates.numpy()),
                mx.array(mask.numpy()),
            )
        )

    # 1e-3, not the 1e-5 the clean-input logits get, and the looseness is
    # measured rather than assumed. Feeding MLX's logits through PyTorch's own
    # loss reproduces MLX's number to 2e-6, so the two loss definitions agree
    # exactly and the whole gap is in the forward pass on a corrupted input.
    # See test_corruption_conditions_the_forward_pass for why that input is
    # harder than a clean one.
    assert abs(actual - expected) / expected < 1e-3, f"{actual} vs {expected}"


def test_corruption_conditions_the_forward_pass(hf_state):
    """Cross-backend disagreement depends on how much of the input is masked.

    Measured on an M5, CPU, 12 tokens, one sequence::

         0/12 masked   5.9e-06
         4/12 masked   1.7e-05
         8/12 masked   3.4e-05
        10/12 masked   5.9e-04
        12/12 masked   5.1e-06

    Not monotonic, and the shape is the explanation: a fully masked sequence
    is clean again because every position holds the same token, so there are
    no near-ties for a one-ulp difference to break. The excursions live in
    mixed sequences.

    This matters for the two-backend split. On the causal path the margin
    between backend noise (~1e-5) and binarization damage (~1e-2) is three
    orders of magnitude. Here a bad corruption pattern closes it to about
    one and a half. Training on MLX and measuring in PyTorch is still sound,
    with less room than the autoregressive side had.
    """
    config = ModelConfig(quantize_linear=False, use_attention_residuals=False,
                         diffusion=True)
    torch_model = load_pretrained(config, hf_state=hf_state, verbose=False).eval()
    mlx_model = load_mlx_pretrained(config, hf_state=hf_state)

    deltas = {}
    generator = torch.Generator().manual_seed(0)
    for k in (0, 8, 12):
        ids = torch.tensor(_IDS)
        if k:
            ids[0, torch.randperm(len(_IDS[0]), generator=generator)[:k]] = (
                config.mask_token_id
            )
        with torch.no_grad():
            expected = torch_model(input_ids=ids)["logits"].numpy()
        with mx.stream(mx.cpu):
            actual = np.array(mlx_model(mx.array(ids.numpy())))
        deltas[k] = float(np.abs(actual - expected).max() / np.abs(expected).max())

    assert deltas[0] < 1e-5, f"clean input should be tight: {deltas[0]:.2e}"
    assert deltas[12] < 1e-4, f"fully masked should be tight: {deltas[12]:.2e}"
    # The headroom that licenses training on one backend and measuring on the
    # other. Binarization moves the logits by ~1e-2.
    assert max(deltas.values()) < 1e-3, f"corrupted forward drifted: {deltas}"


def test_ternary_fake_quantize_kernels_agree():
    """Written twice, once per framework, and a mismatch looks like an effect.

    The ternary kernel has a rounding boundary the binary one does not: a
    weight at exactly half the group mean sits on the edge between 0 and 1,
    and the two frameworks must break that tie the same way.
    """
    from src.mlx_backend.model import ternary_fake_quantize as mlx_ternary
    from src.quantization import ternary_fake_quantize as torch_ternary

    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 512)).astype(np.float32)

    expected = torch_ternary(torch.from_numpy(w), 128).numpy()
    actual = np.array(mlx_ternary(mx.array(w), 128))
    assert np.abs(actual - expected).max() < 1e-6


def test_ternary_backends_agree_on_a_real_model(hf_state):
    delta = _relative_delta(
        ModelConfig(quantize_linear=True, use_attention_residuals=False,
                    quant_scheme="q1_58"),
        hf_state,
        mx.cpu,
    )
    # Ternary is a discontinuous map like sign(), so the same amplification
    # applies and the bound is loose for the same reason.
    assert delta < 0.1, f"relative delta = {delta:.2e}"


def test_ternary_keeps_more_of_the_model_than_binary(hf_state):
    """1.725 bits should damage an un-retrained model less than 1.125 does.

    Not a claim about the trained arms -- QAT recovers what post-training
    quantization destroys, and this repository's whole point is that the two
    regimes differ. It is a sanity check that the extra level is being used
    at all: a ternary path that silently behaved like the binary one would
    pass every kernel test and show up here.
    """
    from src.evaluate import evaluate_perplexity
    from src.data import load_wikitext_tokens
    from transformers import AutoTokenizer
    from src.loader import HF_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokens = load_wikitext_tokens(tokenizer, "validation")[:8192]
    device = torch.device("cpu")

    scores = {}
    for scheme in ("q1_0", "q1_58"):
        model = load_pretrained(
            ModelConfig(quantize_linear=True, use_attention_residuals=False,
                        quant_scheme=scheme),
            hf_state=hf_state,
            verbose=False,
        ).eval()
        scores[scheme] = evaluate_perplexity(
            model, tokens, device, progress=False
        ).nll
        del model

    assert scores["q1_58"] < scores["q1_0"], (
        f"ternary should be less destructive than binary before training: {scores}"
    )


def test_ternary_ste_backward_diverges_between_backends_as_documented():
    """Forward kernels agree bitwise; the ternary BACKWARD does not.

    The PyTorch reference clips gradients where |w/scale| > 1; the MLX kernel
    that trained every published arm passes them through unclipped. The binary
    path has no such divergence (its clip never fires). Pinned so the gap can
    only close deliberately -- aligning the estimators is an epoch-break
    change (ROADMAP, Future improvements), not a drive-by fix.
    """
    from src.mlx_backend.model import (
        fake_quantize as mlx_binary,
        ternary_fake_quantize as mlx_ternary,
    )
    from src.quantization import (
        fake_quantize as torch_binary,
        ternary_fake_quantize as torch_ternary,
    )

    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 256)).astype(np.float32)

    def torch_grad(fn):
        wt = torch.from_numpy(w.copy()).requires_grad_(True)
        fn(wt, 128).sum().backward()
        return wt.grad.numpy()

    def mlx_grad(fn):
        return np.array(mx.grad(lambda a: fn(a, 128).sum())(mx.array(w)))

    binary_gap = np.abs(torch_grad(torch_binary) - mlx_grad(mlx_binary))
    assert binary_gap.max() < 1e-5, "binary backward parity broke"

    ternary_gap = np.abs(torch_grad(torch_ternary) - mlx_grad(mlx_ternary))
    assert ternary_gap.max() > 0.5 and (ternary_gap > 1e-6).mean() > 0.99, (
        "ternary backward gap closed -- if the estimators were deliberately "
        "aligned, retire this pin and start a new reproducibility epoch"
    )
