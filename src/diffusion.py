"""Absorbing-state masked diffusion, in LLaDA's formulation.

Corruption replaces tokens with ``[MASK]`` at a rate ``t`` sampled per
sequence; the model sees the whole corrupted sequence bidirectionally and
predicts the originals at the masked positions. Generation runs the same
forward repeatedly, unmasking a few positions each pass.

The estimator
-------------
For a sequence of length ``L`` corrupted at rate ``t``::

    loss = (1/t) * (1/L) * sum_{i masked} CE(logits_i, x0_i)

``E[#masked] = tL``, so the ``1/t`` factor makes this an unbiased estimate of
the negative ELBO per token, which is the quantity a diffusion LM reports in
place of autoregressive perplexity. The two are not comparable and must never
be put in the same column.

Why the corruption is injected rather than drawn inside the model
-----------------------------------------------------------------
Two backends have to agree on the loss, and they cannot agree on a random
draw. Every function here takes the mask as an argument, so
``tests/`` can hand both implementations the same corruption and compare
numbers rather than distributions. It also makes the paired-seed design work:
two arms at the same seed see identical corruption, so the difference between
them is the intervention.

The rate is clamped away from zero
----------------------------------
``t ~ U(0,1)`` is the formulation, but ``1/t`` has infinite variance at the
bottom of that range and a single unlucky draw dominates a short budget.
``MIN_RATE`` bounds it. This biases the estimator slightly and the bias is
declared rather than hidden: it drops the contribution of corruptions so mild
that almost nothing is masked.
"""

from __future__ import annotations

import torch

__all__ = [
    "MIN_RATE",
    "sample_rates",
    "corrupt",
    "diffusion_loss",
    "uniform_bound",
]

# Below this the 1/t weight dominates every other term in the batch.
MIN_RATE = 1e-3


def sample_rates(
    batch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """One corruption rate per sequence, in ``[MIN_RATE, 1]``."""
    u = torch.rand(batch_size, 1, generator=generator, device=device)
    return MIN_RATE + (1.0 - MIN_RATE) * u


def corrupt(
    input_ids: torch.Tensor,
    rates: torch.Tensor,
    mask_token_id: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace each token with ``[MASK]`` independently with probability ``rate``.

    Returns ``(corrupted, mask)``. Independent per position, not a contiguous
    span: the absorbing-state process has no notion of a span, and masking one
    would make the task infilling rather than denoising.
    """
    noise = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    mask = noise < rates
    corrupted = torch.where(mask, torch.full_like(input_ids, mask_token_id), input_ids)
    return corrupted, mask


def diffusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    rates: torch.Tensor,
) -> torch.Tensor:
    """The ``1/t``-weighted NELBO estimator, averaged over the batch.

    Positions are scored where they were masked, at the same index -- there
    is no shift. A shifted target here would silently train an autoregressive
    model wearing a diffusion loss, and the number it produced would look
    plausible.
    """
    B, L, V = logits.shape
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(),
        targets.reshape(-1),
        reduction="none",
    ).view(B, L)

    masked_sum = (per_token * mask.float()).sum(dim=1)          # [B]
    per_sequence = masked_sum / (rates.squeeze(-1) * L)
    return per_sequence.mean()


def uniform_bound(vocab_size: int) -> float:
    """NELBO per token of a model that has learned nothing.

    The floor the adaptation has to clear before any quantization result on
    top of it means anything. A damaged model sitting here is indistinguishable
    from a differently damaged model sitting here.
    """
    import math

    return math.log(vocab_size)
