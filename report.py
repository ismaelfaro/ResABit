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


def _rows(stage: str | None) -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["status"] != "crash" and r.get("perplexity")]
    return [r for r in rows if stage is None or r["stage"] == stage]


def load(stage: str | None = "full") -> dict[str, list[dict]]:
    """Group runs by arm. Defaults to the main stage.

    Determinism replicates repeat one configuration on purpose, so letting
    them into the main table would weight that arm by however many repeats
    happened to run.
    """
    by_arm: dict[str, list[dict]] = {}
    for r in _rows(stage):
        by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def determinism_floor() -> str | None:
    """Spread across identical reruns: the floor beneath the seed floor.

    Anything smaller than this is not an effect, it is the GPU's reduction
    order. Reported before the paired comparison because it is what makes
    that comparison interpretable.
    """
    rows = _rows("determinism")
    if len(rows) < 2:
        return None
    ppls = np.array([r["perplexity"]["perplexity"] for r in rows])
    nlls = np.array([r["perplexity"]["nll"] for r in rows])
    spread = float(ppls.max() - ppls.min())
    return (
        f"{len(rows)} identical reruns of `{rows[0]['arm']}` at seed "
        f"{rows[0]['seed']} (same config, same data order, same code):\n\n"
        f"- perplexity: {np.round(ppls, 3).tolist()} — "
        f"mean {ppls.mean():.3f}, sd {ppls.std(ddof=1):.3f}, "
        f"range {spread:.3f}\n"
        f"- NLL: mean {nlls.mean():.4f}, sd {nlls.std(ddof=1):.4f}\n\n"
        f"**Any paired difference below ~{spread:.1f} ppl is backend "
        f"nondeterminism, not an effect.**"
    )


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

    # Layer 0's gate is structurally inert: R_{-1} = 0, so it is never
    # applied and its gradient is exactly zero. Including it would drag the
    # reported mean toward zero and understate the live gates.
    alpha_cells = []
    for k, _ in present:
        alphas = [r["alphas"][1:] for r in by_arm[k] if r.get("alphas")]
        if not alphas:
            alpha_cells.append("—")
        else:
            flat = np.array(alphas)
            alpha_cells.append(f"{flat.mean():+.4f} / {np.abs(flat).max():.4f}")
    lines.append(
        "| learned alpha, mean / max (layers 1+) | " + " | ".join(alpha_cells) + " |"
    )

    row("seeds", lambda r: [len(r)], "{:.0f}")
    row("wall-clock per run (s)", lambda r: _get(r, "wall_seconds"), "{:.0f}")
    return "\n".join(lines)


_PAIRED_METRICS = (
    # (label, extractor, unit, lower_is_better)
    ("wikitext ppl", lambda r: r["perplexity"]["perplexity"], "ppl", True),
    ("wikitext NLL", lambda r: r["perplexity"]["nll"], "nats", True),
    ("final train loss", lambda r: r["train"]["final_train_loss"], "nats", True),
    ("top-1 acc", lambda r: r["perplexity"]["top1_accuracy"], "", False),
)


def paired_delta(by_arm, a: str, b: str) -> str | None:
    """Per-seed difference b - a, which cancels the variance they share.

    Reported on several metrics rather than one. Perplexity is an
    exponential of the mean NLL, so it exaggerates spread in the damaged
    regime; if the sign of the effect disagrees between ppl and NLL, that is
    a signal the result is being driven by a single bad window rather than
    by a consistent shift.
    """
    lines, shared = [], None
    for label, extract, unit, lower_better in _PAIRED_METRICS:
        try:
            sa = {r["seed"]: extract(r) for r in by_arm.get(a, [])}
            sb = {r["seed"]: extract(r) for r in by_arm.get(b, [])}
        except (KeyError, TypeError):
            continue
        keys = sorted(set(sa) & set(sb))
        if len(keys) < 2:
            continue
        shared = keys
        d = np.array([sb[s] - sa[s] for s in keys])
        mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

        if abs(mean) < 2 * se:
            verdict = "within noise"
        elif (mean < 0) == lower_better:
            verdict = f"**{b} better**"
        else:
            verdict = f"**{b} worse**"
        digits = 4 if unit != "ppl" else 3
        lines.append(
            f"| {label} | {np.round(d, digits).tolist()} | "
            f"{mean:+.{digits}f} | {se:.{digits}f} | {verdict} |"
        )

    if not lines:
        return None
    header = (
        f"`{b}` minus `{a}`, paired over seeds {shared}. "
        f"Decision rule fixed in advance: an effect smaller than 2 SE is "
        f"reported as no effect.\n\n"
        "| metric | per-seed deltas | mean | SE | verdict |\n|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(lines)


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


def alpha_trajectory(by_arm) -> str | None:
    """Test RaBiT's inter-path adaptation prediction on the gate curve.

    RaBiT (ICML 2026) reports that parallel compensation paths co-adapt into
    redundancy under QAT. If that holds here, the gates should rise and then
    fall back toward zero. A monotonic rise means the pathway is being used
    and kept; a flat curve means the gates never moved and the arm is a
    no-op regardless of what perplexity says.
    """
    lines = []
    for arm in ("onebit_ar", "fp32_ar"):
        rows = [r for r in by_arm.get(arm, []) if r.get("train", {}).get("alpha_curve")]
        if not rows:
            continue
        # Layer 0's gate is structurally inert; drop it.
        curves = np.array([np.array(r["train"]["alpha_curve"])[:, 1:] for r in rows])
        magnitude = np.abs(curves).mean(axis=-1)      # [runs, steps]
        mean_curve = magnitude.mean(axis=0)
        peak_step = int(mean_curve.argmax())
        peak, final = float(mean_curve.max()), float(mean_curve[-1])
        collapse = 1.0 - final / peak if peak > 0 else 0.0

        if peak < 1e-4:
            verdict = "**gates never moved** — the arm is effectively a no-op"
        elif collapse > 0.2:
            verdict = (
                f"**gates peaked at step {peak_step} then fell {collapse:.0%}** — "
                "consistent with RaBiT's inter-path adaptation"
            )
        else:
            verdict = "**gates rose and held** — no sign of inter-path collapse"
        lines.append(
            f"- `{arm}`: mean |alpha| {mean_curve[0]:.5f} at start, "
            f"{peak:.5f} peak (step {peak_step}), {final:.5f} final. {verdict}"
        )
    return "\n".join(lines) if lines else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-readme", action="store_true")
    p.add_argument("--stage", default=None)
    args = p.parse_args()

    by_arm = load(args.stage)
    parts = [build_table(by_arm)]

    if floor := determinism_floor():
        parts += ["", "### Noise floor (identical reruns)", "", floor]
    if delta := paired_delta(by_arm, "onebit", "onebit_ar"):
        parts += ["", "### Paired comparison", "", delta]
    if inter := interaction(by_arm):
        parts += ["", "### Derived quantities", "", inter]
    if traj := alpha_trajectory(by_arm):
        parts += ["", "### Gate trajectories", "", traj]

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
