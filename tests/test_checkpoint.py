"""Export/reload parity for a published checkpoint.

A frozen checkpoint is the one artifact a reader can run without rerunning
the experiment, so the failure that matters here is silent: weights that load
without complaint and compute something other than what was measured. The
tie-breaking case is ``lm_head.weight``, which aliases the embedding table
and is therefore absent from the file -- a loader that accepted every missing
key would accept a checkpoint missing far more than that.
"""

from __future__ import annotations

import json

import pytest
import torch

from export_checkpoint import save_safetensors, storage_report
from src.config import ModelConfig
from src.loader import load_checkpoint
from src.model import TriDiForCausalLM
from src.quantization import quantize_model_weights


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _tiny(**overrides) -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        quant_group_size=128,
        **overrides,
    )


def _export(model, tmp_path):
    save_safetensors(model, tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_config": model.config.to_dict()})
    )
    return tmp_path


@pytest.mark.parametrize("quantize_linear", [True, False])
@pytest.mark.parametrize("use_attention_residuals", [True, False])
def test_export_reload_is_exact(tmp_path, quantize_linear, use_attention_residuals):
    """Reloading must reproduce the exported model's logits bit for bit."""
    config = _tiny(
        quantize_linear=quantize_linear,
        use_attention_residuals=use_attention_residuals,
    )
    model = TriDiForCausalLM(config).eval()
    if use_attention_residuals:
        with torch.no_grad():                     # a zero gate hides gate bugs
            for layer in model.layers:
                layer.attn_residual_scale.fill_(-0.005)
    if quantize_linear:
        quantize_model_weights(model)

    ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        before = model(input_ids=ids)["logits"]

    reloaded, manifest = load_checkpoint(_export(model, tmp_path))
    with torch.no_grad():
        after = reloaded(input_ids=ids)["logits"]

    assert torch.equal(before, after)
    assert manifest["model_config"]["quantize_linear"] is quantize_linear


def test_reloaded_model_stays_frozen(tmp_path):
    """The `quantized` flag must survive the round trip.

    It is what routes the forward to the packed path. A checkpoint that lost
    it would fall back to the training forward over master weights that no
    longer exist.
    """
    model = TriDiForCausalLM(_tiny(quantize_linear=True)).eval()
    quantize_model_weights(model)

    reloaded, _ = load_checkpoint(_export(model, tmp_path))
    layers = reloaded.quantized_modules()
    assert layers and all(m.is_quantized for m in layers)
    assert all(m.weight is None for m in layers)


def test_loader_rejects_a_truncated_checkpoint(tmp_path):
    """Only the tied readout may be missing; anything else is a bad file."""
    from safetensors.torch import load_file, save_file

    model = TriDiForCausalLM(_tiny(quantize_linear=True)).eval()
    quantize_model_weights(model)
    _export(model, tmp_path)

    state = load_file(str(tmp_path / "model.safetensors"))
    state.pop("layers.0.self_attn.q_proj.weight_bits")
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="partially mapped"):
        load_checkpoint(tmp_path)


def test_freezing_costs_only_the_fp16_scales(tmp_path):
    """Quote the frozen number, not the training-forward one.

    `quantize()` rounds the group scales to FP16, so the two paths differ.
    The gap is small per layer and compounds with depth, which is exactly why
    `export_checkpoint.py` measures the frozen path instead of inheriting the
    ledger's perplexity.
    """
    model = TriDiForCausalLM(_tiny(quantize_linear=True)).eval()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        train_forward = model(input_ids=ids)["logits"]
        quantize_model_weights(model)
        frozen = model(input_ids=ids)["logits"]

    relative = (frozen - train_forward).abs().max() / train_forward.abs().max()
    assert 0 < relative < 1e-2, f"freezing moved the logits by {relative:.2e}"


def test_storage_accounting_does_not_double_count():
    """Binarised weights and the FP32 remainder must partition the model."""
    config = _tiny(quantize_linear=True)
    model = TriDiForCausalLM(config).eval()
    quantize_model_weights(model)

    report = storage_report(model)
    assert report["total_params"] == (
        report["quantized_params"] + report["full_precision_params"]
    )
    assert report["bits_per_quantized_weight"] == pytest.approx(
        config.bits_per_quantized_weight, rel=1e-3
    )
    # The headline compression is on the projections; the model-wide average
    # is dragged up by the FP32 embedding table and must never be quoted as
    # the former.
    assert report["bits_per_weight_model_average"] > report["bits_per_quantized_weight"]
