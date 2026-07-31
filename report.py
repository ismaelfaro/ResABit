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
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
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
        f"- NLL: mean {nlls.mean():.4f}, sd {nlls.std(ddof=1):.4f}"
        + (
            "\n\n**The pipeline is bitwise reproducible: this floor is exactly "
            "zero, so every bit of seed-to-seed spread is genuine seed "
            "effect.**"
            if spread == 0
            else f"\n\n**Any paired difference below ~{spread:.1f} ppl is "
            f"backend nondeterminism, not an effect.**"
        )
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

    header = "| metric | " + " | ".join(label for _, label in present) + " |"
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
    # The reference is the *shipped* model, chosen so the measurement does
    # not depend on which arm finished first. Consequence: the FP32 arm's
    # value is pure fine-tuning drift, and the 1-bit arms carry drift plus
    # binarization. KL does not decompose additively, so the FP32 row is the
    # scale of the non-quantization component, not a term to subtract.
    row("KL to shipped Qwen (nats)",
        lambda r: _get(r, "suite", "divergence", "kl_teacher_student"), "{:.4f}")
    row("top-1 agreement with shipped Qwen",
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


def derived_quantities(by_arm) -> dict | None:
    """The headline numbers, as values rather than prose.

    Split out from :func:`interaction` so the model cards and the results
    table cannot disagree: a card that retypes "2.97 nats" is a second source
    of truth, and the two drift the first time a seed lands.
    """
    def mean_nll(arm):
        rows = by_arm.get(arm, [])
        return float(np.mean([r["perplexity"]["nll"] for r in rows])) if rows else None

    vals = {a: mean_nll(a) for a, _ in COLUMNS}
    if any(v is None for v in vals.values()):
        return None

    ar_cost_fp32 = vals["fp32_ar"] - vals["fp32"]
    ar_cost_1bit = vals["onebit_ar"] - vals["onebit"]

    nlls = {
        arm: [r["perplexity"]["nll"] for r in by_arm.get(arm, [])]
        for arm in ("onebit", "onebit_ar")
    }
    paired_se = None
    if len(nlls["onebit"]) == len(nlls["onebit_ar"]) >= 2:
        d = np.array(nlls["onebit_ar"]) - np.array(nlls["onebit"])
        paired_se = float(d.std(ddof=1) / np.sqrt(len(d)))

    return {
        "nll": vals,
        "quantization_gap": vals["onebit"] - vals["fp32"],
        "ar_cost_fp32": ar_cost_fp32,
        "ar_cost_1bit": ar_cost_1bit,
        "interaction": ar_cost_1bit - ar_cost_fp32,
        "paired_se_1bit": paired_se,
        "fp32_perplexity": float(
            np.mean([r["perplexity"]["perplexity"] for r in by_arm["fp32"]])
        ),
        "seeds": {arm: len(by_arm.get(arm, [])) for arm, _ in COLUMNS},
    }


def interaction(by_arm) -> str | None:
    """(onebit_ar - onebit) - (fp32_ar - fp32), the headline quantity.

    Computed in NLL, not perplexity. Perplexity is exp(NLL), so a fixed
    relative change is worth ~2.9 perplexity points at the 1-bit arms' scale
    and ~0.15 at the FP32 arms'. Differencing raw perplexities across a 19x
    scale gap subtracts quantities that are not commensurable and would
    manufacture an interaction out of the scale difference alone.

    In log space the same subtraction is a ratio of ratios, which is what
    "does AR buy more under binarization" actually asks.
    """
    # Judge the interaction against the noise on its own terms, not against
    # zero. Reporting a sign for a quantity many times smaller than the
    # standard error of one of its components is how a null becomes a claim.
    q = derived_quantities(by_arm)
    if q is None:
        return None

    ar_cost_fp32 = q["ar_cost_fp32"]
    ar_cost_1bit = q["ar_cost_1bit"]
    inter = q["interaction"]
    quant_gap = q["quantization_gap"]
    paired_se = q["paired_se_1bit"]

    if paired_se is None:
        verdict = "no noise estimate available yet"
        scale = ""
    elif abs(inter) < 2 * paired_se:
        verdict = "**not distinguishable from zero**"
        # State the ratio in one direction only. An earlier version phrased
        # it as "SE is Nx the interaction", which reads as strong evidence
        # when N is large and as its opposite when N drops below 1 -- and it
        # did drop below 1 once the fifth seed landed. The decision rule is
        # |interaction| < 2 SE either way.
        scale = (
            f"\n\nThe paired standard error on the 1-bit AR term is "
            f"{paired_se:.4f} nats; the interaction is {abs(inter)/paired_se:.1f}x "
            f"that, inside the 2 SE rule fixed in advance. The FP32 arms "
            f"contribute one seed each, so their variance is absent from this "
            f"estimate and the true standard error on the interaction is "
            f"larger than the one quoted. **No evidence that the attention "
            f"residual preferentially repairs binarization damage; the "
            f"measurement does not have the resolution to rule out an effect "
            f"of this size either.**"
        )
    else:
        verdict = (
            "**AR buys more under binarization**"
            if inter < 0
            else "**AR helps less under binarization**"
        )
        scale = f"\n\nPaired standard error on the 1-bit term: {paired_se:.4f} nats."

    return (
        f"All quantities in nats of NLL; perplexity ratios in parentheses.\n\n"
        f"- Quantization cost (no AR): **{quant_gap:+.4f}** nats "
        f"({np.exp(quant_gap):.1f}x perplexity)\n"
        f"- AR cost at FP32: **{ar_cost_fp32:+.4f}** nats "
        f"({np.exp(ar_cost_fp32):.3f}x)\n"
        f"- AR cost at 1-bit: **{ar_cost_1bit:+.4f}** nats "
        f"({np.exp(ar_cost_1bit):.3f}x)\n"
        f"- Interaction: {inter:+.4f} nats — {verdict}"
        + scale
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


def stability_comparison(by_arm, a: str, b: str) -> str | None:
    """Compare seed-to-seed spread, not just the means.

    Two arms can share a mean and differ entirely in how reliably they reach
    it. For a recovery technique that is the more practical question: a
    method whose outcome swings with the seed is one you cannot deploy on a
    single run, whatever its average.
    """
    rows_a = by_arm.get(a, [])
    rows_b = by_arm.get(b, [])
    if len(rows_a) < 3 or len(rows_b) < 3:
        return None

    pa = np.array([r["perplexity"]["perplexity"] for r in rows_a])
    pb = np.array([r["perplexity"]["perplexity"] for r in rows_b])
    sd_a, sd_b = pa.std(ddof=1), pb.std(ddof=1)
    ratio = (sd_b / sd_a) ** 2

    note = ""
    if ratio > 4:
        # A variance estimate from a handful of runs is itself high-variance:
        # this figure moved from 23.2 at three seeds to 24.2 at four and
        # again at five. Quote the ratio with its degrees of freedom and the
        # critical value, never on its own.
        critical = {2: 19.0, 3: 9.28, 4: 6.39, 5: 5.05, 6: 4.28}
        df = len(pa) - 1
        crit = critical.get(df)
        strength = ""
        if crit:
            strength = (
                f" The 95% critical value of F({df},{df}) is {crit}, so this "
                + ("clears it" if ratio > crit else "does not clear it")
                + "."
            )
        note = (
            f"\n\n**`{b}` is {sd_b/sd_a:.1f}x more variable across seeds than "
            f"`{a}` while not being better on average.** Variance ratio "
            f"{ratio:.1f} on {df} and {len(pb)-1} degrees of freedom.{strength} "
            f"A spread estimated from {len(pa)} runs is itself unstable, so "
            f"treat the ratio as an order of magnitude, not a measurement."
        )
    return (
        f"| arm | perplexities | mean | sd |\n|---|---|---|---|\n"
        f"| `{a}` | {np.round(pa, 3).tolist()} | {pa.mean():.3f} | {sd_a:.3f} |\n"
        f"| `{b}` | {np.round(pb, 3).tolist()} | {pb.mean():.3f} | {sd_b:.3f} |"
        + note
    )


def alpha_by_depth(by_arm) -> str | None:
    """Where in the stack the residual pathway actually gets used.

    The accumulator grows with depth by construction, so a uniform gate
    would already mean a deeper effect at deeper layers. What the per-layer
    profile adds is direction: a gate that goes negative is being used to
    damp the residual stream rather than to enrich it, which is the opposite
    of the mechanism's stated purpose.
    """
    blocks = []
    for arm in ("onebit_ar", "fp32_ar"):
        rows = [r for r in by_arm.get(arm, []) if r.get("alphas")]
        if not rows:
            continue
        alphas = np.array([r["alphas"] for r in rows])       # [seeds, layers]
        mean = alphas.mean(axis=0)
        n_layers = len(mean)
        thirds = [
            ("early", mean[1 : n_layers // 3]),               # layer 0 is inert
            ("middle", mean[n_layers // 3 : 2 * n_layers // 3]),
            ("late", mean[2 * n_layers // 3 :]),
        ]
        summary = ", ".join(f"{name} {seg.mean():+.4f}" for name, seg in thirds)
        negative = int((mean[1:] < 0).sum())
        blocks.append(
            f"- `{arm}` ({len(rows)} seed(s)): {summary}. "
            f"{negative}/{n_layers - 1} live gates are negative"
            + (
                " — the pathway is being used to damp the residual stream, "
                "not to enrich it."
                if negative > (n_layers - 1) * 0.6
                else "."
            )
        )
    return "\n".join(blocks) if blocks else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-readme", action="store_true")
    # Not None. The determinism stage repeats one configuration on purpose;
    # pooling it into the main table weights `onebit` by however many repeats
    # happened to run and reports them as distinct seeds. It did: three
    # replicates of seed 0 turned five seeds into "8" and pulled the mean
    # 1.3 ppl toward the one seed that was rerun.
    p.add_argument("--stage", default="full")
    args = p.parse_args()

    by_arm = load(args.stage)
    parts = [build_table(by_arm)]

    if floor := determinism_floor():
        parts += ["", "### Noise floor (identical reruns)", "", floor]
    if delta := paired_delta(by_arm, "onebit", "onebit_ar"):
        parts += ["", "### Paired comparison", "", delta]
    if stab := stability_comparison(by_arm, "onebit", "onebit_ar"):
        parts += ["", "### Run-to-run stability", "", stab]
    if inter := interaction(by_arm):
        parts += ["", "### Derived quantities", "", inter]
    if traj := alpha_trajectory(by_arm):
        parts += ["", "### Gate trajectories", "", traj]
    if depth := alpha_by_depth(by_arm):
        parts += ["", "### Gate profile by depth", "", depth]

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
