"""Turn results/ledger.jsonl into the results table.

    python report.py                 # print
    python report.py --write-readme  # splice into readme.md

The table is generated, never hand-edited, so a number in the README always
traces to a ledger row and a git commit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

LEDGER = Path("results/ledger.jsonl")
README = Path("readme.md")
START, END = "<!-- RESULTS-TABLE-START -->", "<!-- RESULTS-TABLE-END -->"

COLUMNS = [
    ("fp32", "FP32 +FT"),
    ("fp32_ar", "FP32 +FT +AR"),
    ("onebit", "1-bit QAT"),
    ("onebit_ar", "1-bit QAT +AR"),
]


def load(stage: str | None = None) -> dict[str, list[dict]]:
    if not LEDGER.exists():
        return {}
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["status"] != "crash" and r.get("perplexity")]
    if stage:
        rows = [r for r in rows if r["stage"] == stage]
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def _cell(values: list[float], fmt: str = "{:.2f}") -> str:
    """Mean, with spread when more than one seed contributed."""
    if not values:
        return "—"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(np.mean(values))} ± {fmt.format(np.std(values, ddof=1))}"


def _get(rows: list[dict], *path, default=None) -> list[float]:
    out = []
    for r in rows:
        node = r
        for key in path:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        if node is not None:
            out.append(node)
    return out


def build_table(by_arm: dict[str, list[dict]]) -> str:
    present = [(k, label) for k, label in COLUMNS if by_arm.get(k)]
    if not present:
        return "*No completed runs in the ledger yet.*"

    header = "| metric | " + " | ".join(l for _, l in present) + " |"
    sep = "|---" * (len(present) + 1) + "|"
    lines = [header, sep]

    def row(label, getter, fmt="{:.3f}"):
        cells = [_cell(getter(by_arm[k]), fmt) for k, _ in present]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("wikitext-2 ppl (strided)", lambda r: _get(r, "perplexity", "perplexity"))
    row("wikitext-2 NLL", lambda r: _get(r, "perplexity", "nll"), "{:.4f}")
    row("wikitext-2 top-1 acc", lambda r: _get(r, "perplexity", "top1_accuracy"), "{:.4f}")
    row("final train loss", lambda r: _get(r, "train", "final_train_loss"), "{:.4f}")

    # Held-out suite: one reference seed per arm, so no spread to report.
    row("KL to FP32 base (nats)",
        lambda r: _get(r, "suite", "divergence", "kl_teacher_student"), "{:.4f}")
    row("top-1 agreement with base",
        lambda r: _get(r, "suite", "divergence", "top1_agreement"), "{:.4f}")

    for task, label in (
        ("arc_easy", "ARC-Easy acc"),
        ("hellaswag", "HellaSwag acc_norm"),
        ("lambada", "LAMBADA acc"),
    ):
        def getter(rows, _t=task):
            return _get(rows, "suite", "zero_shot", _t, "primary")

        stderrs = []
        for k, _ in present:
            stderrs += _get(by_arm[k], "suite", "zero_shot", task, "stderr")
        suffix = f" (±{np.mean(stderrs):.4f} SE)" if stderrs else ""
        row(label + suffix, getter, "{:.4f}")

    lines.append(
        "| bits/weight (quantised params) | "
        + " | ".join(
            "1.125" if k.startswith("onebit") else "32" for k, _ in present
        )
        + " |"
    )

    alpha_cells = []
    for k, _ in present:
        alphas = [r["alphas"] for r in by_arm[k] if r.get("alphas")]
        if not alphas:
            alpha_cells.append("—")
        else:
            flat = np.array(alphas)
            alpha_cells.append(
                f"{flat.mean():+.4f} / {np.abs(flat).max():.4f}"
            )
    lines.append("| learned alpha, mean / max | " + " | ".join(alpha_cells) + " |")

    row("seeds", lambda r: [len(r)], "{:.0f}")
    row("wall-clock per run (s)", lambda r: _get(r, "wall_seconds"), "{:.0f}")
    return "\n".join(lines)


def paired_delta(by_arm, a: str, b: str) -> str | None:
    """Per-seed difference b - a, which cancels the variance they share."""
    sa = {r["seed"]: r["perplexity"]["perplexity"] for r in by_arm.get(a, [])}
    sb = {r["seed"]: r["perplexity"]["perplexity"] for r in by_arm.get(b, [])}
    shared = sorted(set(sa) & set(sb))
    if len(shared) < 2:
        return None
    d = np.array([sb[s] - sa[s] for s in shared])
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    verdict = (
        "**within the noise floor — no measurable effect**"
        if abs(mean) < 2 * se
        else (f"**{b} is better** by {abs(mean):.3f} ppl"
              if mean < 0 else f"**{b} is worse** by {abs(mean):.3f} ppl")
    )
    return (
        f"`{b} - {a}` over paired seeds {shared}: "
        f"per-seed {np.round(d, 3).tolist()}, "
        f"mean {mean:+.3f} ppl (SE {se:.3f}) — {verdict}"
    )


def interaction(by_arm) -> str | None:
    """(onebit_ar - onebit) - (fp32_ar - fp32): the headline quantity."""
    def mean_ppl(arm):
        rows = by_arm.get(arm, [])
        return np.mean([r["perplexity"]["perplexity"] for r in rows]) if rows else None

    vals = {a: mean_ppl(a) for a, _ in COLUMNS}
    if any(v is None for v in vals.values()):
        return None
    q_gap = vals["onebit"] - vals["fp32"]
    q_gap_ar = vals["onebit_ar"] - vals["fp32_ar"]
    return (
        f"- Quantization gap without AR: **{q_gap:+.3f}** ppl\n"
        f"- Quantization gap with AR: **{q_gap_ar:+.3f}** ppl\n"
        f"- Interaction `(1bit_AR - 1bit) - (fp32_AR - fp32)`: "
        f"**{q_gap_ar - q_gap:+.3f}** ppl "
        f"({'AR helps more under binarization' if q_gap_ar < q_gap else 'AR does not preferentially help binarization'})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-readme", action="store_true")
    p.add_argument("--stage", default=None)
    args = p.parse_args()

    by_arm = load(args.stage)
    parts = [build_table(by_arm)]

    if delta := paired_delta(by_arm, "onebit", "onebit_ar"):
        parts += ["", "### Paired comparison", "", delta]
    if inter := interaction(by_arm):
        parts += ["", "### Derived quantities", "", inter]

    body = "\n".join(parts)
    print(body)

    if args.write_readme:
        text = README.read_text()
        new = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            f"{START}\n{body}\n{END}",
            text,
            flags=re.S,
        )
        README.write_text(new)
        print(f"\n-> spliced into {README}")


if __name__ == "__main__":
    main()
