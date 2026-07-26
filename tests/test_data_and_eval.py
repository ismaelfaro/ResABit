"""Unit tests for the windowing and metric code.

The perplexity protocol is where a silent mistake is most expensive: it does
not crash, it just produces a number that cannot be compared to anyone
else's. These tests pin the parts that were previously wrong -- padding
counted in the loss, and overlapping windows scored twice.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.data import make_strided_windows, make_training_windows
from src.evaluate import evaluate_perplexity


# -- training windows -----------------------------------------------------


def test_training_windows_are_contiguous_and_shifted():
    tokens = torch.arange(100)
    windows = make_training_windows(tokens, seq_len=10)
    assert windows.shape[1] == 11
    assert torch.equal(windows[0], torch.arange(11))
    assert torch.equal(windows[1], torch.arange(10, 21))


def test_training_windows_never_pad():
    """The old pipeline padded with token id 0 and trained on the padding."""
    tokens = torch.arange(1, 96)          # deliberately not a multiple of 10
    windows = make_training_windows(tokens, seq_len=10)
    assert (windows != 0).all(), "a pad token leaked into the training stream"
    assert windows.shape[0] == 9


# -- strided evaluation windows -------------------------------------------


def test_strided_windows_score_each_token_exactly_once():
    tokens = torch.arange(1000)
    windows = make_strided_windows(tokens, max_length=100, stride=50)

    scored: list[int] = []
    for inputs, labels in windows:
        scored += labels[labels != -100].tolist()

    assert len(scored) == len(set(scored)), "a token was scored twice"
    assert set(scored) == set(range(1, 1000)), "a token was never scored"


def test_strided_windows_keep_context_unscored():
    tokens = torch.arange(1000)
    windows = make_strided_windows(tokens, max_length=100, stride=50)
    later = windows[3]
    labels = later[1]
    assert (labels[:-50] == -100).all(), "context positions are being scored"
    assert (labels[-50:] != -100).all(), "target positions are masked out"


def test_first_window_scores_everything_it_can():
    tokens = torch.arange(1000)
    inputs, labels = make_strided_windows(tokens, max_length=100, stride=50)[0]
    assert (labels != -100).all()


def test_strided_windows_handle_short_input():
    assert make_strided_windows(torch.arange(3), max_length=100, stride=50)
    assert make_strided_windows(torch.arange(1), max_length=100, stride=50) == []


# -- perplexity -----------------------------------------------------------


class _ConstantModel(torch.nn.Module):
    """Emits a fixed distribution, so the true perplexity is known."""

    def __init__(self, vocab: int, favoured: int, logit: float):
        super().__init__()
        self.vocab, self.favoured, self.logit = vocab, favoured, logit

        class _Cfg:
            vocab_size = vocab

        self.config = _Cfg()

    def forward(self, input_ids, **_):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        logits[..., self.favoured] = self.logit
        return {"logits": logits, "loss": None}


def test_uniform_model_has_perplexity_equal_to_vocab_size():
    model = _ConstantModel(vocab=50, favoured=0, logit=0.0)
    tokens = torch.randint(0, 50, (600,))
    result = evaluate_perplexity(
        model, tokens, torch.device("cpu"), max_length=100, stride=50, progress=False
    )
    assert result.perplexity == pytest.approx(50.0, rel=1e-4)
    assert result.nll == pytest.approx(math.log(50), rel=1e-4)


def test_confident_and_correct_model_has_low_perplexity():
    model = _ConstantModel(vocab=50, favoured=7, logit=20.0)
    tokens = torch.full((600,), 7)
    result = evaluate_perplexity(
        model, tokens, torch.device("cpu"), max_length=100, stride=50, progress=False
    )
    assert result.perplexity < 1.01
    assert result.top1_accuracy == pytest.approx(1.0)


def test_token_count_matches_the_scorable_positions():
    model = _ConstantModel(vocab=50, favoured=0, logit=0.0)
    tokens = torch.randint(0, 50, (600,))
    result = evaluate_perplexity(
        model, tokens, torch.device("cpu"), max_length=100, stride=50, progress=False
    )
    assert result.num_tokens == 599


def test_perplexity_reports_infinity_rather_than_overflowing():
    """A collapsed 1-bit model can produce an NLL that exp() cannot hold."""
    # Drive the probability of the one token that actually occurs to ~0.
    model = _ConstantModel(vocab=50, favoured=1, logit=-1e4)
    tokens = torch.full((300,), 1)
    result = evaluate_perplexity(
        model, tokens, torch.device("cpu"), max_length=100, stride=50, progress=False
    )
    assert math.isinf(result.perplexity)
    assert math.isfinite(result.nll)
