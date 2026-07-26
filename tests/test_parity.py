"""Golden test: our FP32 model must reproduce HuggingFace Qwen1.5-0.5B-Chat.

Every downstream claim rests on this. An ablation run on top of an
unvalidated reimplementation measures the reimplementation's bugs, not the
intervention being studied.

Marked ``slow`` -- needs the ~1.2 GB checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from src.config import ModelConfig
from src.loader import HF_MODEL_ID, load_hf_state_dict, load_pretrained

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def hf_state():
    return load_hf_state_dict()


@pytest.fixture(scope="module")
def reference_logits():
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        out = hf(input_ids=_INPUT_IDS).logits
    del hf
    return out


_INPUT_IDS = torch.tensor(
    [[151643, 3838, 374, 264, 220, 16, 15257, 4128, 1614, 30, 151645, 198]]
)


def test_matches_huggingface_logits(hf_state, reference_logits):
    config = ModelConfig(quantize_linear=False, use_attention_residuals=False)
    model = load_pretrained(config, hf_state=hf_state, verbose=False).eval()

    with torch.no_grad():
        ours = model(input_ids=_INPUT_IDS)["logits"]

    assert ours.shape == reference_logits.shape
    max_abs = (ours - reference_logits).abs().max().item()
    assert max_abs < 1e-3, f"max |delta logit| = {max_abs:.2e}"

    assert torch.equal(ours.argmax(-1), reference_logits.argmax(-1))


def test_attention_residual_is_identity_at_init(hf_state):
    """alpha=0 must make the AR arm bit-identical to its non-AR twin."""
    plain = load_pretrained(
        ModelConfig(quantize_linear=False, use_attention_residuals=False),
        hf_state=hf_state,
        verbose=False,
    ).eval()
    with_ar = load_pretrained(
        ModelConfig(quantize_linear=False, use_attention_residuals=True),
        hf_state=hf_state,
        verbose=False,
    ).eval()

    with torch.no_grad():
        a = plain(input_ids=_INPUT_IDS)["logits"]
        b = with_ar(input_ids=_INPUT_IDS)["logits"]

    assert torch.equal(a, b), "AR arm diverges from baseline before training"


def test_loader_rejects_incomplete_state_dict(hf_state):
    broken = dict(hf_state)
    del broken["model.layers.0.self_attn.q_proj.bias"]
    with pytest.raises(RuntimeError, match="partially mapped"):
        load_pretrained(ModelConfig(), hf_state=broken, verbose=False)
