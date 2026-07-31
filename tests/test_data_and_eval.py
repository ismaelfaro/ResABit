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
    for _inputs, labels in windows:
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


# -- batched multiple-choice scoring --------------------------------------


class _TinyLM(torch.nn.Module):
    """Small causal LM with real position-dependence, for scoring tests."""

    def __init__(self, vocab: int = 40, dim: int = 16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.head = torch.nn.Linear(dim, vocab)

        class _Cfg:
            vocab_size = vocab

        self.config = _Cfg()

    def forward(self, input_ids, **_):
        h = self.embed(input_ids)
        # Causal running mean, so a token's logits depend on its prefix only
        # and never on anything to its right -- the property that makes
        # right-padding safe.
        h = torch.cumsum(h, dim=1) / torch.arange(
            1, h.shape[1] + 1, device=h.device
        ).view(1, -1, 1)
        return {"logits": self.head(torch.tanh(self.proj(h))), "loss": None}


class _CharTokenizer:
    """Maps each character to an id, so token boundaries are predictable."""

    def __call__(self, text, add_special_tokens=False):
        class _Out:
            input_ids = [ord(c) % 40 for c in text]

        return _Out()


def _mc_task(n_items: int, n_choices: int):
    from src.data import MultipleChoiceTask

    rng = __import__("random").Random(0)
    letters = "abcdefghijklmnop"
    return MultipleChoiceTask(
        name="synthetic",
        contexts=["".join(rng.choices(letters, k=rng.randint(4, 12))) for _ in range(n_items)],
        choices=[
            ["".join(rng.choices(letters, k=rng.randint(1, 6))) for _ in range(n_choices)]
            for _ in range(n_items)
        ],
        answers=[rng.randrange(n_choices) for _ in range(n_items)],
    )


def test_batched_scoring_matches_one_at_a_time():
    """Padding must not change any score, or the suite is not reproducible."""
    from src.evaluate import evaluate_multiple_choice

    model, tokenizer = _TinyLM(), _CharTokenizer()
    task = _mc_task(n_items=12, n_choices=4)
    device = torch.device("cpu")

    single = evaluate_multiple_choice(
        model, tokenizer, task, device, progress=False, batch_size=1
    )
    batched = evaluate_multiple_choice(
        model, tokenizer, task, device, progress=False, batch_size=8
    )
    assert single["acc"] == batched["acc"]
    assert single["acc_norm"] == batched["acc_norm"]


def test_batch_size_never_changes_the_answer():
    from src.evaluate import evaluate_multiple_choice

    model, tokenizer = _TinyLM(), _CharTokenizer()
    task = _mc_task(n_items=10, n_choices=3)
    device = torch.device("cpu")

    accs = {
        bs: evaluate_multiple_choice(
            model, tokenizer, task, device, progress=False, batch_size=bs
        )["acc"]
        for bs in (1, 2, 3, 5, 16, 64)
    }
    assert len(set(accs.values())) == 1, f"accuracy varies with batch size: {accs}"


def test_padding_does_not_change_a_score():
    """The primitive: a sequence must score the same alone and in a batch.

    Right-padding is only safe because the mask is causal. If a model ever
    attended past its own position, every batched score would silently shift
    and the whole suite would drift with batch size.
    """
    from src.evaluate import _score_batch

    model = _TinyLM()
    device = torch.device("cpu")
    requests = [
        ([3, 9, 14, 2], 2),
        ([7, 1], 1),
        ([5, 5, 5, 5, 5, 5, 8, 8], 3),
    ]

    alone = [_score_batch(model, [r], device)[0] for r in requests]
    together = _score_batch(model, requests, device)

    for (solo, _), (batched, _) in zip(alone, together, strict=True):
        assert solo == pytest.approx(batched, abs=1e-5)


def test_items_with_different_choice_counts_stay_separate():
    """Regrouping is by owner index, not by a fixed stride."""
    from src.data import MultipleChoiceTask
    from src.evaluate import evaluate_multiple_choice

    model, tokenizer = _TinyLM(), _CharTokenizer()
    task = MultipleChoiceTask(
        name="ragged",
        contexts=["abc", "defg", "hi", "jklmn"],
        choices=[["a", "bb", "ccc"], ["d", "ee"], ["f", "g", "h", "i"], ["j", "k"]],
        answers=[0, 1, 2, 0],
    )
    single = evaluate_multiple_choice(
        model, tokenizer, task, torch.device("cpu"), progress=False, batch_size=1
    )
    batched = evaluate_multiple_choice(
        model, tokenizer, task, torch.device("cpu"), progress=False, batch_size=5
    )
    assert single["acc"] == batched["acc"]
    assert single["num_items"] == 4
