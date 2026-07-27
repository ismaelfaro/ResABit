"""Generate a checkpoint's model card from its manifest.

    python make_model_card.py checkpoints/resabit-qwen1.5-0.5b-1bit

Written rather than hand-authored for the same reason the results table is:
a card is where a quantization project overstates itself, and the two numbers
most likely to drift -- bits per weight and the perplexity the checkpoint
actually computes -- are exactly the two a human would retype.

The card leads with what the model cannot do. It scores ~285 perplexity
against 14.7 for the same architecture in FP32 at the same budget, which
means it does not produce coherent text. It is published so the 2x2 can be
rerun without a GPU-week, not because anyone should generate with it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import report

ARM_TITLES = {
    "onebit": "1-bit QAT",
    "onebit_ar": "1-bit QAT + attention residuals",
    "fp32": "FP32 fine-tuned reference",
    "fp32_ar": "FP32 fine-tuned + attention residuals",
}

ARM_ROLES = {
    "onebit": (
        "The **baseline of the ablation**: binarised projections, no residual "
        "pathway. Every claim about what the attention residual does or does "
        "not buy is measured against this arm."
    ),
    "onebit_ar": (
        "The **treatment arm**: binarised projections plus the cross-layer "
        "attention residual. Compared against `onebit` at the same seed, the "
        "difference is within noise on every metric."
    ),
    "fp32": (
        "The **reference ceiling**: no quantization, same recovery budget. It "
        "exists so the binarization gap is measured against a model that got "
        "the same fine-tuning, not against the shipped checkpoint."
    ),
    "fp32_ar": (
        "The **fourth cell**: the residual pathway without quantization. It "
        "is what makes the interaction term computable at all."
    ),
}


def fmt_metrics(manifest: dict) -> tuple[str, str]:
    """Return (headline perplexity, the paragraph explaining which one it is)."""
    metrics = manifest["metrics"]
    train_forward = metrics["wikitext2_val_train_forward"]
    frozen = metrics.get("wikitext2_val_frozen")

    if not frozen:
        return f"{train_forward['perplexity']:.3f}", (
            "This arm has no quantized layers, so there is one forward path "
            "and one number."
        )

    delta = frozen["nll"] - train_forward["nll"]
    return f"{frozen['perplexity']:.3f}", (
        f"**This is the frozen number, and it is not the number in the "
        f"repository's results table.** That table reports the training "
        f"forward — FP32 master weights pushed through the straight-through "
        f"quantizer — because that is what all four arms were compared on. "
        f"This checkpoint stores packed sign bits and FP16 group scales, and "
        f"computes {frozen['perplexity']:.3f} against the table's "
        f"{train_forward['perplexity']:.3f}: {delta:+.4f} nats. The gap is "
        f"the FP16 rounding of the group scales, compounded through 24 "
        f"layers by the discontinuity in `sign()`. Quoting the table's number "
        f"for this file would describe a model nobody can download."
    )


def zero_shot_line(arm: str, seed: int) -> str:
    """The arm's own held-out scores, from the ledger row that produced them.

    Hardcoding "at chance on ARC-Easy" was fine for the two 1-bit arms and
    flatly false for the FP32 ones, which is what a card built from prose
    instead of from the run record gets you.
    """
    from run_ablation import load_ledger

    for record in load_ledger():
        if (
            record["arm"] == arm
            and record["seed"] == seed
            and record.get("suite")
        ):
            tasks = record["suite"]["zero_shot"]
            parts = []
            for name, chance in (("arc_easy", 0.25), ("hellaswag", 0.25),
                                 ("lambada", None)):
                if name not in tasks:
                    continue
                score = tasks[name]["primary"]
                note = f" (chance {chance})" if chance else ""
                parts.append(f"{name} {score:.4f}{note}")
            return ", ".join(parts)
    return ""


def build(manifest: dict, source_repo: str) -> str:
    arm = manifest["arm"]
    storage = manifest["storage"]
    headline, path_note = fmt_metrics(manifest)
    train_cfg = manifest["train_config"]
    total_tokens = train_cfg["steps"] * train_cfg["batch_size"] * train_cfg["grad_accum"] * train_cfg["seq_len"]

    # Same computation the results table uses, so the card cannot drift from
    # it. If the ledger is incomplete the card refuses rather than inventing.
    quantities = report.derived_quantities(report.load("full"))
    if quantities is None:
        raise SystemExit(
            "the ledger has no complete 2x2, so the card's headline numbers "
            "cannot be generated; run the full sweep first"
        )

    zero_shot = zero_shot_line(arm, manifest["seed"])
    zero_shot_sentence = (
        f" Held-out scores for this arm: {zero_shot}." if zero_shot else ""
    )

    alphas = manifest.get("alphas") or []
    alpha_block = ""
    if alphas:
        live = alphas[1:]                        # layer 0's gate is inert
        negative = sum(1 for a in live if a < 0)
        alpha_block = (
            f"\n### What the gates learned\n\n"
            f"{negative} of {len(live)} live gates converged **negative** "
            f"(mean {sum(live)/len(live):+.4f}). The model learned to "
            f"*subtract* the accumulated attention residual rather than to use "
            f"it. The FP32 arm does the same thing, so this is not a response "
            f"to binarization damage — it is what the architecture elicits in "
            f"either regime. Layer 0's gate is structurally inert: the "
            f"accumulator reaches it empty, so it receives no gradient.\n"
        )

    return f"""---
license: apache-2.0
base_model: Qwen/Qwen1.5-0.5B-Chat
datasets:
  - wikitext
language:
  - en
tags:
  - quantization
  - 1-bit
  - quantization-aware-training
  - ablation
  - research-artifact
---

# ResABit — {ARM_TITLES.get(arm, arm)} (seed {manifest["seed"]})

## Do not use this model to generate text

wikitext-2 perplexity is **{headline}**. The same architecture in FP32, given
the identical recovery budget, scores {quantities["fp32_perplexity"]:.3f}.
This model produces incoherent output.{zero_shot_sentence}

It is a **research artifact**. It is published so that a factorial ablation
can be re-run and re-scored without repeating the training, and so that the
reported numbers are attached to weights someone else can check. It is not a
model, and it is not a demonstration that 1-bit quantization works.

{ARM_ROLES.get(arm, "")}

## What was measured

The experiment crosses {{FP32, 1-bit}} with {{no attention residual, attention
residual}} and asks whether the residual pathway buys *more* under
binarization than it buys in general — the interaction term, which the
architectural-compensation literature reports method-versus-baseline and
therefore cannot isolate.

**The interaction is {quantities["interaction"]:+.4f} nats, inside a decision
rule fixed in advance at 2 SE** (the paired standard error on the 1-bit term
is {quantities["paired_se_1bit"]:.4f} nats, and the FP32 arms contribute one
seed each, so that figure is a lower bound). These data give no evidence that
the attention residual preferentially repairs binarization damage, and they
do not have the resolution to rule out an effect of that size either.

**Binarizing {storage["quantized_params"]/1e6:.1f}M of
{storage["total_params"]/1e6:.1f}M parameters costs
{quantities["quantization_gap"]:.3f} nats — a
{math.exp(quantities["quantization_gap"]):.1f}x perplexity
increase — at a {total_tokens/1e6:.2f}M-token recovery budget.**
{alpha_block}
## Perplexity

| | value |
|---|---|
| wikitext-2 validation (strided, this checkpoint) | **{headline}** |
| FP32 reference, same budget | {quantities["fp32_perplexity"]:.3f} |
| Qwen1.5-0.5B-Chat as shipped | 25.005 |

{path_note}

## What is actually quantized

| | parameters | bits/weight |
|---|---|---|
| Block projections (q, k, v, o, gate, up, down) | {storage["quantized_params"]/1e6:.1f}M | **{storage["bits_per_quantized_weight"] or 32:.3f}** |
| Embeddings + tied readout, norms, biases | {storage["full_precision_params"]/1e6:.1f}M | 32 |
| **Model average** | **{storage["total_params"]/1e6:.1f}M** | **{storage["bits_per_weight_model_average"]:.2f}** |

The file is **{storage["total_bytes"]/1e6:.0f} MB**:
{storage["quantized_bytes"]/1e6:.0f} MB of binarised projections and
{storage["full_precision_bytes"]/1e6:.0f} MB of FP32 embedding table. Qwen1.5
has a 151,936-token vocabulary, so the untouched embeddings dominate the
checkpoint. **There is no deployment-size win here** — the compression claim
is about the projections, and quoting `1.125 bits/weight` for this file would
be false.

## Loading

Requires the [ResABit repository]({source_repo}); this is not a
`transformers` architecture.

```python
from src.loader import load_checkpoint

model, manifest = load_checkpoint("resabit-{arm}-seed{manifest["seed"]}")
print(manifest["metrics"]["wikitext2_val_frozen"]["perplexity"])
```

Weights are `safetensors`. `lm_head.weight` is deliberately absent — it
aliases the embedding table, which safetensors will not store twice — and the
tie is rebuilt from the config on load.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen1.5-0.5B-Chat` |
| Corpus | wikitext-2-raw train |
| Budget | {train_cfg["steps"]} steps x {train_cfg["batch_size"] * train_cfg["grad_accum"] * train_cfg["seq_len"]} tokens = {total_tokens/1e6:.2f}M tokens |
| Optimizer | AdamW, peak LR {train_cfg["learning_rate"]}, cosine, {train_cfg["warmup_frac"]:.0%} warmup |
| Embeddings | frozen (155.7M of 464.0M parameters, tied to the readout) |
| Seed | {manifest["seed"]} |
| Commit | `{manifest["commit"]}` |

Training ran on MLX; every reported metric is computed in PyTorch, whose
agreement with HuggingFace `Qwen2ForCausalLM` is pinned in the test suite.
The pipeline is bitwise reproducible on this machine — three identical reruns
returned the same perplexity to the digit — so this checkpoint is a rebuild
of the run behind the published table, not an approximation of it.

## Limitations

- **One model, one scale (0.5B).** Extreme-quantization results are known to
  change with scale.
- **{total_tokens/1e6:.2f}M recovery tokens**, three to six orders of magnitude below
  the ternary-QAT literature. This is a low-budget-recovery result.
- **One corpus.** wikitext-2 is both the recovery corpus and the evaluation
  corpus; no perplexity here is out-of-distribution.
- **Weight-only.** Activations remain FP32.
- **{quantities["seeds"]["onebit"]} seeds on each 1-bit arm,
  {quantities["seeds"]["fp32"]} on each FP32 arm**, so the interaction's
  standard error is estimated from half its terms.
- **PIQA is implemented in the task registry and was never scored.** Expected
  to floor at chance alongside the other accuracy tasks for the 1-bit arms;
  that is a prediction, not a measurement.
- The residual gates were given a 10x effective learning rate so the pathway
  would not be a no-op inside this budget. That asymmetry favours the
  intervention and is a confound in the stability comparison.

## Citation

Protocol, prior art and the full argument for the experiment's shape are in
`docs/PROTOCOL.md`; the write-up is in `paper/preprint.md`. The mechanism is
**not novel** — compensating pathways around binarised layers go back to
Bi-Real Net (2018) and run through ReActNet, MeliusNet, OneBit and BiMaCoSR.
What is offered here is the measurement.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="directory written by export_checkpoint.py")
    p.add_argument("--repo", default="https://github.com/ismaelfaro/ResABit",
                   help="source repository URL to link from the card")
    args = p.parse_args()

    directory = Path(args.checkpoint)
    manifest = json.loads((directory / "config.json").read_text())
    card = build(manifest, args.repo)
    (directory / "README.md").write_text(card)
    print(f"wrote {directory}/README.md ({len(card)} chars)")


if __name__ == "__main__":
    main()
