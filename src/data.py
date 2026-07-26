"""Dataset plumbing: wikitext token streams and multiple-choice eval sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "load_wikitext_tokens",
    "make_training_windows",
    "make_strided_windows",
    "MultipleChoiceTask",
    "load_multiple_choice",
    "AVAILABLE_TASKS",
]

_WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def load_wikitext_tokens(tokenizer, split: str) -> torch.Tensor:
    """Concatenate a wikitext split into one token stream.

    Joining with a blank line matches the canonical perplexity setup used by
    the GPT-2 paper and lm-evaluation-harness; joining with ``""`` or
    tokenising line by line gives numbers that are not comparable to anyone
    else's.
    """
    from datasets import load_dataset

    rows = load_dataset(*_WIKITEXT, split=split)
    text = "\n\n".join(r for r in rows["text"] if r.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids.to(torch.long)


def make_training_windows(tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Non-overlapping ``[n, seq_len + 1]`` windows; the tail is dropped.

    Dropping the remainder rather than padding it keeps every training token
    real. The previous implementation padded with token id 0 and let those
    positions contribute to the loss.
    """
    usable = (tokens.numel() - 1) // seq_len * seq_len + 1
    body = tokens[:usable]
    n = (usable - 1) // seq_len
    starts = torch.arange(n) * seq_len
    idx = starts[:, None] + torch.arange(seq_len + 1)[None, :]
    return body[idx]


def make_strided_windows(
    tokens: torch.Tensor, max_length: int, stride: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sliding windows for perplexity, with context positions masked out.

    Each window predicts only its final ``stride`` tokens; earlier positions
    are set to -100 so they act purely as conditioning context and no token
    is ever scored twice. Scoring non-overlapping chunks instead makes every
    chunk's first tokens near-unpredictable and inflates perplexity.
    """
    windows = []
    prev_end = 0
    for begin in range(0, tokens.numel(), stride):
        end = min(begin + max_length, tokens.numel())
        n_new = end - prev_end
        if n_new <= 0:
            break
        chunk = tokens[begin:end]
        if chunk.numel() < 2:
            break
        inputs = chunk[:-1]
        labels = chunk[1:].clone()
        labels[: -n_new] = -100
        windows.append((inputs, labels))
        prev_end = end
        if end == tokens.numel():
            break
    return windows


# -- Zero-shot multiple choice -------------------------------------------


@dataclass
class MultipleChoiceTask:
    """One eval set flattened into (context, continuation) scoring requests."""

    name: str
    contexts: list[str]
    choices: list[list[str]]
    answers: list[int]
    # acc_norm divides each choice's log-likelihood by its character length,
    # which stops long-but-wrong continuations from being penalised purely
    # for length. HellaSwag is always reported this way.
    primary_metric: str = "acc_norm"

    def __len__(self) -> int:
        return len(self.contexts)


AVAILABLE_TASKS = ("arc_easy", "hellaswag", "piqa", "lambada")


def load_multiple_choice(name: str, limit: int | None = None) -> MultipleChoiceTask:
    from datasets import load_dataset

    if name == "arc_easy":
        rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        contexts, choices, answers = [], [], []
        for r in rows:
            texts = r["choices"]["text"]
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            contexts.append(f"Question: {r['question']}\nAnswer:")
            choices.append([f" {t}" for t in texts])
            answers.append(labels.index(r["answerKey"]))
        task = MultipleChoiceTask("arc_easy", contexts, choices, answers, "acc")

    elif name == "hellaswag":
        rows = load_dataset("Rowan/hellaswag", split="validation")
        contexts, choices, answers = [], [], []
        for r in rows:
            ctx = f"{r['activity_label']}: {r['ctx_a']} {r['ctx_b'].capitalize()}"
            contexts.append(ctx.strip())
            choices.append([f" {e}" for e in r["endings"]])
            answers.append(int(r["label"]))
        task = MultipleChoiceTask("hellaswag", contexts, choices, answers, "acc_norm")

    elif name == "piqa":
        rows = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        task = MultipleChoiceTask(
            "piqa",
            [f"Question: {r['goal']}\nAnswer:" for r in rows],
            [[f" {r['sol1']}", f" {r['sol2']}"] for r in rows],
            [int(r["label"]) for r in rows],
            "acc_norm",
        )

    elif name == "lambada":
        # Last-word prediction. Unlike the 4-way tasks it has no floor at 25%,
        # so it keeps resolving after a heavily damaged model has collapsed to
        # chance everywhere else.
        rows = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        contexts, choices = [], []
        for r in rows:
            ctx, _, last = r["text"].rpartition(" ")
            contexts.append(ctx)
            choices.append([f" {last}"])
        task = MultipleChoiceTask(
            "lambada", contexts, choices, [0] * len(contexts), "acc"
        )

    else:
        raise ValueError(f"unknown task {name!r}; expected one of {AVAILABLE_TASKS}")

    if limit is not None and limit < len(task):
        task.contexts = task.contexts[:limit]
        task.choices = task.choices[:limit]
        task.answers = task.answers[:limit]
    return task
