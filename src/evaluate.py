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
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import MultipleChoiceTask, make_strided_windows

__all__ = [
    "PerplexityResult",
    "evaluate_perplexity",
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


@torch.no_grad()
def _score_continuation(
    model, tokenizer, context: str, continuation: str, device: torch.device
) -> tuple[float, int, bool]:
    """Summed log-prob of ``continuation`` given ``context``, and greedy match."""
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    full_ids = tokenizer(context + continuation, add_special_tokens=False).input_ids
    cont_len = len(full_ids) - len(ctx_ids)
    if cont_len <= 0:
        return float("-inf"), 0, False

    ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=ids)["logits"].float()[0]

    # Position i predicts token i+1, so continuation token j is scored by
    # logits at index len(ctx) + j - 1.
    target = torch.tensor(full_ids[-cont_len:], device=device)
    window = logits[-cont_len - 1 : -1]
    logprobs = window.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    greedy = bool((window.argmax(-1) == target).all())
    return float(logprobs.sum()), cont_len, greedy


@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    task: MultipleChoiceTask,
    device: torch.device,
    progress: bool = True,
) -> dict:
    """Zero-shot accuracy by log-likelihood ranking.

    ``acc`` ranks raw summed log-probability; ``acc_norm`` divides by the
    continuation's character count. Single-choice tasks (LAMBADA) are scored
    by exact greedy match instead of ranking.
    """
    model.eval()
    n_correct = n_correct_norm = 0
    total = len(task)

    for i in tqdm(range(total), desc=task.name, disable=not progress):
        scores, norms, greedy_hits = [], [], []
        for choice in task.choices[i]:
            logprob, _, greedy = _score_continuation(
                model, tokenizer, task.contexts[i], choice, device
            )
            scores.append(logprob)
            norms.append(logprob / max(len(choice), 1))
            greedy_hits.append(greedy)

        if len(scores) == 1:
            n_correct += int(greedy_hits[0])
            n_correct_norm += int(greedy_hits[0])
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
