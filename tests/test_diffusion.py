"""Masked-diffusion path: corruption, loss, bidirectionality, sampler.

The failure this file is built around is a diffusion model that is secretly
still autoregressive. A causal mask left on, or a target left shifted, gives a
model that trains, produces a falling loss curve and a plausible number, and
answers a different question than the one asked. Both are asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.diffusion import MIN_RATE, corrupt, diffusion_loss, sample_rates, uniform_bound
from src.model import ResABitForCausalLM


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
        quantize_linear=False,
        mask_token_id=511,
        **overrides,
    )


# -- corruption -----------------------------------------------------------


def test_corruption_only_touches_masked_positions():
    ids = torch.randint(0, 500, (4, 64))
    rates = sample_rates(4)
    corrupted, mask = corrupt(ids, rates, mask_token_id=511)

    assert torch.equal(corrupted[~mask], ids[~mask])
    assert (corrupted[mask] == 511).all()


def test_corruption_rate_tracks_the_sampled_rate():
    """A rate that does not control the mask density is a silent no-op."""
    ids = torch.zeros(64, 512, dtype=torch.long)
    for target in (0.1, 0.5, 0.9):
        rates = torch.full((64, 1), target)
        _, mask = corrupt(ids, rates, mask_token_id=511)
        assert abs(mask.float().mean().item() - target) < 0.02


def test_rates_stay_off_zero():
    """1/t has infinite variance at the bottom of U(0,1)."""
    rates = sample_rates(4096)
    assert rates.min() >= MIN_RATE
    assert rates.max() <= 1.0


# -- the estimator --------------------------------------------------------


def test_loss_is_an_unbiased_nelbo_estimate():
    """E[(1/t)(1/L) sum_masked CE] is the mean per-token CE.

    The 1/t weight is the whole reason this estimator is usable; if it were
    dropped the loss would still fall during training and would no longer
    mean anything comparable.
    """
    torch.manual_seed(0)
    B, L, V = 256, 64, 32
    ids = torch.randint(0, V, (B, L))
    # A fixed, deliberately imperfect predictor, so every position has the
    # same expected cross-entropy and the target value is computable.
    logits = torch.zeros(B, L, V)
    reference = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), ids.reshape(-1)
    )

    estimates = []
    for _ in range(20):
        rates = sample_rates(B)
        _, mask = corrupt(ids, rates, mask_token_id=V - 1)
        estimates.append(float(diffusion_loss(logits, ids, mask, rates)))

    assert abs(np.mean(estimates) - float(reference)) < 0.05


def test_loss_scores_the_same_index_not_the_next_one():
    """No shift. A shifted target trains an AR model wearing a diffusion loss."""
    B, L, V = 1, 8, 16
    ids = torch.arange(L).unsqueeze(0) % V
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, 3] = True
    rates = torch.full((B, 1), 1.0)

    logits = torch.zeros(B, L, V)
    logits[0, 3, int(ids[0, 3])] = 50.0        # perfect at the masked index
    assert float(diffusion_loss(logits, ids, mask, rates)) < 1e-4

    shifted = torch.zeros(B, L, V)
    shifted[0, 3, int(ids[0, 4])] = 50.0       # perfect at the *next* token
    assert float(diffusion_loss(shifted, ids, mask, rates)) > 1.0


def test_unmasked_positions_do_not_contribute():
    B, L, V = 2, 16, 32
    ids = torch.randint(0, V, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, ::4] = True
    rates = torch.full((B, 1), 0.25)

    logits = torch.randn(B, L, V)
    baseline = float(diffusion_loss(logits, ids, mask, rates))

    wrecked = logits.clone()
    wrecked[~mask] = torch.randn_like(wrecked[~mask]) * 100
    assert float(diffusion_loss(wrecked, ids, mask, rates)) == pytest.approx(baseline)


def test_uniform_bound_is_the_floor_to_clear():
    assert uniform_bound(151936) == pytest.approx(11.93, abs=0.01)


# -- architecture ---------------------------------------------------------


def test_diffusion_attention_is_bidirectional():
    """Changing a later token must move an earlier position's logits.

    Under a causal mask it cannot, which makes this the direct test for the
    failure mode: a "diffusion" model that never dropped its mask.
    """
    model = ResABitForCausalLM(_tiny(diffusion=True)).eval()
    ids = torch.randint(0, 500, (1, 16))

    with torch.no_grad():
        before = model(input_ids=ids)["logits"][0, 0].clone()
        ids[0, -1] = (ids[0, -1] + 1) % 500
        after = model(input_ids=ids)["logits"][0, 0]

    assert not torch.allclose(before, after, atol=1e-6)


def test_causal_model_stays_causal():
    model = ResABitForCausalLM(_tiny(diffusion=False)).eval()
    ids = torch.randint(0, 500, (1, 16))

    with torch.no_grad():
        before = model(input_ids=ids)["logits"][0, 0].clone()
        ids[0, -1] = (ids[0, -1] + 1) % 500
        after = model(input_ids=ids)["logits"][0, 0]

    assert torch.allclose(before, after, atol=1e-6)


def test_diffusion_loss_refuses_a_causal_model():
    model = ResABitForCausalLM(_tiny(diffusion=False))
    ids = torch.randint(0, 500, (2, 8))
    rates = sample_rates(2)
    _, mask = corrupt(ids, rates, mask_token_id=511)

    with pytest.raises(RuntimeError, match="causal"):
        model.diffusion_loss(ids, rates, mask)


def test_mask_token_must_be_inside_the_embedding_table():
    with pytest.raises(ValueError, match="outside the embedding"):
        ModelConfig(diffusion=True, mask_token_id=999_999)


def test_gradients_reach_the_masked_positions():
    model = ResABitForCausalLM(_tiny(diffusion=True))
    ids = torch.randint(0, 500, (2, 16))
    rates = sample_rates(2)
    _, mask = corrupt(ids, rates, mask_token_id=511)

    model.diffusion_loss(ids, rates, mask).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)


# -- sampler --------------------------------------------------------------


def test_generate_fills_every_masked_slot():
    model = ResABitForCausalLM(_tiny(diffusion=True)).eval()
    ids = torch.randint(0, 500, (2, 24))
    ids[:, 8:16] = 511

    out = model.diffusion_generate(ids, num_steps=4)
    assert (out != 511).all(), "sampler left positions masked"


def test_generate_never_emits_the_mask_token():
    """[MASK] is a corruption symbol, not something the model may output.

    An unconverged model predicts it readily. Without suppression the sampler
    marks the slot filled and writes a mask token into it, so the output is
    filled and still masked at once -- which the previous test only catches
    by accident.
    """
    torch.manual_seed(3)
    model = ResABitForCausalLM(_tiny(diffusion=True)).eval()
    with torch.no_grad():                        # make [MASK] the argmax
        model.embed_tokens.weight[511] += 30.0

    ids = torch.randint(0, 500, (2, 16))
    ids[:, 4:12] = 511
    out = model.diffusion_generate(ids, num_steps=2)
    assert (out != 511).all()


def test_generate_leaves_the_given_context_alone():
    model = ResABitForCausalLM(_tiny(diffusion=True)).eval()
    ids = torch.randint(0, 500, (1, 20))
    ids[0, 10:14] = 511
    fixed = ids.clone()

    out = model.diffusion_generate(ids.clone(), num_steps=3)
    keep = fixed[0] != 511
    assert torch.equal(out[0][keep], fixed[0][keep])


def test_generate_on_a_clean_sequence_is_a_no_op():
    model = ResABitForCausalLM(_tiny(diffusion=True)).eval()
    ids = torch.randint(0, 500, (1, 12))
    assert torch.equal(model.diffusion_generate(ids.clone(), num_steps=4), ids)
