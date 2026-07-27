"""Evaluation metrics for the ResABit ablation.

Three families, deliberately:

Perplexity
    Strided-window token perplexity on wikitext-2-raw. Unbounded, so it
    always resolves.
Zero-shot accuracy
    Log-likelihood scored multiple choice. Comparable to published numbers,
    but floors at chance -- a 4-way task cannot tell a broken model from a
    very broken one.
Teacher divergence
    KL to the FP32 model's next-token distribution, plus top-1 agreement.
    This is the metric that carries the ablation. At 1 bit with a small
    recovery budget every arm sits near chance on the accuracy tasks, and
    only divergence-from-teacher still separates them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import MultipleChoiceTask, make_strided_windows

__all__ = [
    "PerplexityResult",
    "DiffusionResult",
    "evaluate_perplexity",
    "evaluate_diffusion_nelbo",
    "evaluate_multiple_choice",
    "evaluate_teacher_divergence",
]


@dataclass
class PerplexityResult:
    perplexity: float
    nll: float
    top1_accuracy: float
    num_tokens: int

    def as_dict(self) -> dict:
        return {
            "perplexity": self.perplexity,
            "nll": self.nll,
            "top1_accuracy": self.top1_accuracy,
            "num_tokens": self.num_tokens,
        }


@dataclass
class DiffusionResult:
    """NELBO per token, and the floor it has to clear to mean anything."""

    nelbo: float
    uniform_bound: float
    mask_accuracy: float
    num_windows: int
    num_samples: int

    @property
    def headroom(self) -> float:
        """Nats below a model that has learned nothing. Negative is worse."""
        return self.uniform_bound - self.nelbo

    def as_dict(self) -> dict:
        return {
            "nelbo": self.nelbo,
            "uniform_bound": self.uniform_bound,
            "headroom": self.headroom,
            "mask_accuracy": self.mask_accuracy,
            "num_windows": self.num_windows,
            "num_samples": self.num_samples,
        }


@torch.no_grad()
def evaluate_diffusion_nelbo(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    block_size: int = 512,
    num_samples: int = 4,
    max_blocks: int | None = None,
    seed: int = 1234,
    progress: bool = True,
) -> DiffusionResult:
    """Monte Carlo NELBO bound for a masked-diffusion model.

    Not perplexity, and not comparable to it. The autoregressive number
    factorises a joint likelihood exactly; this is an upper bound on the
    negative log-likelihood estimated by sampling corruptions. Putting the
    two in one column would be the single most misleading thing this
    repository could print.

    The corruption seed is fixed and independent of the training seed, so
    every arm is scored on identical corruptions and a paired difference
    between arms is the intervention rather than the draw.

    ``mask_accuracy`` is reported alongside because NELBO keeps moving after
    a model has stopped being able to name any token -- the same reason the
    autoregressive side reports top-1 next to perplexity.
    """
    from .diffusion import MIN_RATE, diffusion_loss, uniform_bound

    model.eval()
    usable = (tokens.numel() // block_size) * block_size
    blocks = tokens[:usable].view(-1, block_size)
    if max_blocks is not None:
        blocks = blocks[:max_blocks]

    generator = torch.Generator().manual_seed(seed)
    total_nelbo, total_correct, total_masked = 0.0, 0, 0

    for index in tqdm(range(blocks.shape[0]), desc="nelbo", disable=not progress):
        ids = blocks[index : index + 1].to(device)
        for _ in range(num_samples):
            rate = MIN_RATE + (1.0 - MIN_RATE) * torch.rand(1, 1, generator=generator)
            mask = torch.rand(ids.shape, generator=generator) < rate
            mask = mask.to(device)
            if not bool(mask.any()):
                continue

            corrupted = torch.where(
                mask,
                torch.full_like(ids, model.config.mask_token_id),
                ids,
            )
            logits = model(input_ids=corrupted)["logits"].float()
            total_nelbo += float(
                diffusion_loss(logits, ids, mask, rate.to(device))
            )
            predicted = logits.argmax(-1)
            total_correct += int((predicted[mask] == ids[mask]).sum())
            total_masked += int(mask.sum())

    draws = max(blocks.shape[0] * num_samples, 1)
    return DiffusionResult(
        nelbo=total_nelbo / draws,
        uniform_bound=uniform_bound(model.config.vocab_size),
        mask_accuracy=total_correct / max(total_masked, 1),
        num_windows=int(blocks.shape[0]),
        num_samples=num_samples,
    )


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    max_length: int = 1024,
    stride: int = 512,
    progress: bool = True,
) -> PerplexityResult:
    """Token-level perplexity over sliding windows.

    Returns the exponentiated mean NLL over every scored token, plus the
    greedy next-token accuracy on the same positions -- a finer-grained view
    that keeps moving after perplexity has blown past four digits.
    """
    model.eval()
    windows = make_strided_windows(tokens, max_length, stride)

    total_nll, total_correct, total_tokens = 0.0, 0, 0
    for inputs, labels in tqdm(windows, desc="perplexity", disable=not progress):
        inputs = inputs.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        logits = model(input_ids=inputs)["logits"].float()

        mask = labels != -100
        n = int(mask.sum())
        if n == 0:
            continue

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        nll = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="sum"
        )
        total_nll += float(nll)
        total_correct += int(
            ((flat_logits.argmax(-1) == flat_labels) & mask.reshape(-1)).sum()
        )
        total_tokens += n

    mean_nll = total_nll / max(total_tokens, 1)
    return PerplexityResult(
        # A collapsed 1-bit model can produce an NLL that overflows exp();
        # report the ceiling rather than crashing the sweep.
        perplexity=math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        nll=mean_nll,
        top1_accuracy=total_correct / max(total_tokens, 1),
        num_tokens=total_tokens,
    )


def _encode_request(
    tokenizer, context: str, continuation: str
) -> tuple[list[int], int]:
    """Token ids for ``context + continuation``, and how many are continuation."""
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    full_ids = tokenizer(context + continuation, add_special_tokens=False).input_ids
    return full_ids, len(full_ids) - len(ctx_ids)


@torch.no_grad()
def _score_batch(
    model, requests: list[tuple[list[int], int]], device: torch.device
) -> list[tuple[float, bool]]:
    """Score a batch of (token_ids, continuation_length) pairs in one forward.

    Sequences are right-padded. Under a causal mask a padded tail cannot
    influence any earlier position, so padding is safe here and no attention
    mask is needed -- every position we read sits before the padding.

    Scoring one sequence per forward is roughly 14000 forwards for a single
    arm's suite, which dominates the sweep's wall clock. Batching leaves the
    arithmetic unchanged; ``tests/test_data_and_eval.py`` asserts the batched
    and unbatched paths agree.
    """
    if not requests:
        return []
    lengths = [len(ids) for ids, _ in requests]
    width = max(lengths)
    padded = torch.zeros(len(requests), width, dtype=torch.long, device=device)
    for row, (ids, _) in enumerate(requests):
        padded[row, : len(ids)] = torch.tensor(ids, device=device)

    logits = model(input_ids=padded)["logits"].float()

    out = []
    for row, ((ids, cont_len), length) in enumerate(zip(requests, lengths, strict=True)):
        if cont_len <= 0:
            out.append((float("-inf"), False))
            continue
        # Position i predicts token i+1, so continuation token j is scored
        # by the logits at index len(context) + j - 1.
        start = length - cont_len - 1
        window = logits[row, start : length - 1]
        target = torch.tensor(ids[-cont_len:], device=device)
        logprobs = window.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        out.append((float(logprobs.sum()), bool((window.argmax(-1) == target).all())))
    return out


@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    task: MultipleChoiceTask,
    device: torch.device,
    progress: bool = True,
    batch_size: int = 16,
) -> dict:
    """Zero-shot accuracy by log-likelihood ranking.

    ``acc`` ranks raw summed log-probability; ``acc_norm`` divides by the
    continuation's character count. Single-choice tasks (LAMBADA) are scored
    by exact greedy match instead of ranking.
    """
    model.eval()
    total = len(task)

    # Flatten every (item, choice) pair into one request list, then walk it
    # in batches. Choices within an item vary in length, so batching across
    # items keeps the padding waste roughly uniform.
    requests, owners = [], []
    for i in range(total):
        for choice in task.choices[i]:
            requests.append(_encode_request(tokenizer, task.contexts[i], choice))
            owners.append(i)

    # Batch length-alike requests together. A batch is padded to its longest
    # member, so mixing a 12-token and a 200-token request wastes most of the
    # compute; sorting first cuts the suite's wall clock several-fold. The
    # original order is restored below, so results are unaffected.
    order = sorted(range(len(requests)), key=lambda i: len(requests[i][0]))
    scored: list[tuple[float, bool] | None] = [None] * len(requests)
    for start in tqdm(
        range(0, len(order), batch_size), desc=task.name, disable=not progress
    ):
        chunk = order[start : start + batch_size]
        for index, result in zip(chunk, _score_batch(model, [requests[i] for i in chunk], device), strict=True):
            scored[index] = result

    per_item: list[list[tuple[float, bool]]] = [[] for _ in range(total)]
    for owner, result in zip(owners, scored, strict=True):
        per_item[owner].append(result)

    n_correct = n_correct_norm = 0
    for i, results in enumerate(per_item):
        scores = [s for s, _ in results]
        norms = [
            s / max(len(choice), 1) for (s, _), choice in zip(results, task.choices[i], strict=True)
        ]
        if len(scores) == 1:
            n_correct += int(results[0][1])
            n_correct_norm += int(results[0][1])
        else:
            gold = task.answers[i]
            n_correct += int(max(range(len(scores)), key=scores.__getitem__) == gold)
            n_correct_norm += int(max(range(len(norms)), key=norms.__getitem__) == gold)

    acc, acc_norm = n_correct / total, n_correct_norm / total
    return {
        "task": task.name,
        "acc": acc,
        "acc_norm": acc_norm,
        "primary": acc_norm if task.primary_metric == "acc_norm" else acc,
        "primary_metric": task.primary_metric,
        # Binomial standard error. Without it a 1.5-point gap on 2376 items
        # reads as a result when it is inside the noise.
        "stderr": math.sqrt(max(acc * (1 - acc), 1e-12) / total),
        "num_items": total,
    }


@torch.no_grad()
def evaluate_teacher_divergence(
    student,
    teacher,
    tokens: torch.Tensor,
    device: torch.device,
    max_length: int = 1024,
    stride: int = 512,
    max_windows: int | None = 64,
    progress: bool = True,
) -> dict:
    """How far the student's next-token distribution drifts from the teacher.

    Reports mean KL(teacher || student) in nats and top-1 agreement rate.
    Both stay informative in the regime where downstream accuracy has
    already bottomed out at chance.
    """
    student.eval()
    teacher.eval()
    windows = make_strided_windows(tokens, max_length, stride)
    if max_windows is not None:
        windows = windows[:max_windows]

    total_kl, total_agree, total_tokens = 0.0, 0, 0
    for inputs, labels in tqdm(windows, desc="teacher-KL", disable=not progress):
        inputs = inputs.unsqueeze(0).to(device)
        mask = (labels != -100).reshape(-1).to(device)

        s_logits = student(input_ids=inputs)["logits"].float().reshape(-1, student.config.vocab_size)
        t_logits = teacher(input_ids=inputs)["logits"].float().reshape(-1, teacher.config.vocab_size)
        s_logits, t_logits = s_logits[mask], t_logits[mask]

        t_logprob = t_logits.log_softmax(-1)
        s_logprob = s_logits.log_softmax(-1)
        kl = (t_logprob.exp() * (t_logprob - s_logprob)).sum(-1)

        total_kl += float(kl.sum())
        total_agree += int((s_logits.argmax(-1) == t_logits.argmax(-1)).sum())
        total_tokens += int(mask.sum())

    return {
        "kl_teacher_student": total_kl / max(total_tokens, 1),
        "top1_agreement": total_agree / max(total_tokens, 1),
        "num_tokens": total_tokens,
    }
