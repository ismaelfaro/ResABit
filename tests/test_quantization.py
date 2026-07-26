"""Unit tests for Q1_0_g128.

These cover the three bugs that made the original numbers meaningless: the
int8 shift-table overflow on bit 7, the averaged per-group scales in the
INT8 GEMM, and the sign(0) disagreement between the training forward and the
packed representation.
"""

from __future__ import annotations

import pytest
import torch

from src.quantization import (
    OneBitLinear,
    fake_quantize,
    pack_bits,
    quantize_model_weights,
    unpack_bits,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# -- bit packing ----------------------------------------------------------


def test_pack_unpack_roundtrip():
    signs = (torch.rand(16, 256) > 0.5).float()
    assert torch.equal(unpack_bits(pack_bits(signs), 256), signs)


def test_bit_seven_survives_packing():
    """1 << 7 overflows int8 to -128; the old packer corrupted every 8th weight."""
    signs = torch.zeros(1, 8)
    signs[0, 7] = 1.0
    assert unpack_bits(pack_bits(signs), 8)[0, 7] == 1.0


def test_every_bit_position_independently():
    for position in range(8):
        signs = torch.zeros(1, 8)
        signs[0, position] = 1.0
        got = unpack_bits(pack_bits(signs), 8)
        assert got[0, position] == 1.0, f"bit {position} lost"
        assert got.sum() == 1.0, f"bit {position} leaked into its neighbours"


def test_pack_handles_non_multiple_of_eight():
    signs = (torch.rand(4, 100) > 0.5).float()
    assert torch.equal(unpack_bits(pack_bits(signs), 100), signs)


# -- fake quantization ----------------------------------------------------


def test_fake_quantize_collapses_group_to_two_levels():
    q = fake_quantize(torch.randn(4, 256), group_size=128)
    for row in range(4):
        for g in range(2):
            group = q[row, g * 128 : (g + 1) * 128]
            assert group.abs().unique().numel() == 1
            assert set(group.unique().tolist()) <= set(
                (-group.abs()[0]).unique().tolist() + group.abs()[:1].tolist()
            )


def test_scale_is_group_amax():
    w = torch.randn(2, 128)
    q = fake_quantize(w, group_size=128)
    assert torch.allclose(q.abs()[0, 0], w[0].abs().max())
    assert torch.allclose(q.abs()[1, 0], w[1].abs().max())


def test_sign_is_preserved():
    w = torch.randn(4, 128)
    q = fake_quantize(w, group_size=128)
    assert torch.equal(torch.where(w >= 0, 1.0, -1.0), torch.sign(q))


def test_gradient_flows_through_ste():
    w = torch.randn(2, 128, requires_grad=True)
    fake_quantize(w, 128).sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert (w.grad != 0).any()


def test_zero_weight_decodes_consistently():
    """sign(0) must agree between the training forward and the packed bits."""
    layer = OneBitLinear(128, 8, group_size=128)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:, 0] = 1.0        # keep the scale non-degenerate
    x = torch.randn(1, 128)
    before = layer(x)
    layer.quantize()
    assert torch.allclose(before, layer(x), atol=1e-5)


# -- layer behaviour ------------------------------------------------------


def test_quantize_preserves_forward():
    """Freezing only costs the FP16 rounding of the group scales."""
    layer = OneBitLinear(256, 64, group_size=128)
    x = torch.randn(3, 7, 256)
    before = layer(x)
    layer.quantize()
    relative = (layer(x) - before).abs().max() / before.abs().max()
    assert relative < 1e-3, f"freezing changed the forward by {relative:.2e}"


def test_int8_path_matches_dequant_path():
    """Only per-token activation quantization (1/127) may separate them."""
    layer = OneBitLinear(256, 64, group_size=128)
    layer.quantize()
    x = torch.randn(2, 5, 256)

    layer.int8_inference = False
    reference = layer(x)
    layer.int8_inference = True
    fast = layer(x)

    relative = (fast - reference).abs().max() / reference.abs().max()
    assert relative < 5e-2, f"INT8 path diverges by {relative:.2e} relative"


def test_int8_path_respects_per_group_scales():
    """Regression: the INT8 GEMM must not collapse scales to a row mean.

    Averaging the per-group scales was the direct cause of the 8.2e6
    perplexity previously recorded for the quantised model. A row whose two
    groups differ by 1000x makes the failure unmissable: under averaging,
    the small group's contribution is inflated by ~500x.
    """
    layer = OneBitLinear(256, 8, group_size=128)
    with torch.no_grad():
        layer.weight[:, :128] = 1000.0
        layer.weight[:, 128:] = 1.0
    layer.quantize()

    scales = layer.weight_scales.float()
    assert scales[0, 0] / scales[0, 1] > 500, "fixture did not produce a scale gap"

    # Probe only the small-scale group; the correct output is ~128 * 1.0.
    x = torch.zeros(1, 256)
    x[0, 128:] = 1.0

    layer.int8_inference = False
    reference = layer(x)
    layer.int8_inference = True
    fast = layer(x)

    assert torch.allclose(reference, torch.full((1, 8), 128.0), rtol=1e-3)
    relative = (fast - reference).abs().max() / reference.abs().max()
    assert relative < 5e-2, (
        f"per-group scales are not being applied: {relative:.2e} relative error"
    )


def test_quantize_is_idempotent():
    layer = OneBitLinear(128, 16, group_size=128)
    layer.quantize()
    bits = layer.weight_bits.clone()
    layer.quantize()
    assert torch.equal(bits, layer.weight_bits)


def test_quantized_layer_survives_a_state_dict_roundtrip():
    layer = OneBitLinear(256, 32, group_size=128)
    layer.quantize()
    x = torch.randn(1, 256)
    expected = layer(x)

    restored = OneBitLinear(256, 32, group_size=128)
    restored.load_state_dict(layer.state_dict())
    assert restored.is_quantized, "reloaded layer forgot it was frozen"
    assert torch.allclose(expected, restored(x), atol=1e-5)


def test_bias_is_kept_through_quantization():
    layer = OneBitLinear(128, 8, bias=True, group_size=128)
    with torch.no_grad():
        layer.bias.fill_(2.5)
    layer.quantize()
    assert torch.allclose(layer(torch.zeros(1, 128)), torch.full((1, 8), 2.5))


def test_rejects_indivisible_group_size():
    with pytest.raises(ValueError, match="divisible"):
        OneBitLinear(100, 8, group_size=128)


def test_storage_is_about_one_and_an_eighth_bits():
    layer = OneBitLinear(1024, 1024, group_size=128)
    layer.quantize()
    bits_per_weight = layer.storage_bytes() * 8 / (1024 * 1024)
    assert abs(bits_per_weight - 1.125) < 1e-6


def test_quantize_model_weights_reaches_every_layer():
    model = torch.nn.Sequential(
        OneBitLinear(128, 128), torch.nn.ReLU(), OneBitLinear(128, 128)
    )
    quantize_model_weights(model)
    assert all(
        m.is_quantized for m in model.modules() if isinstance(m, OneBitLinear)
    )
