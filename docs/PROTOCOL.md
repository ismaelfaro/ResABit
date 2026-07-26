# ResABit ablation protocol

The question is narrow on purpose: **does a cross-layer attention residual
reduce the damage that 1-bit weight quantization does to a pretrained
transformer, under a fixed recovery budget?**

Everything below exists to make that question answerable with the compute of
a single Apple Silicon machine, and to make the answer survive a reviewer who
is actively trying to break it.

---

## 1. What the original three-column table could not have shown

The first plan was three columns: FP32 baseline, 1-bit without AR, 1-bit with
AR. That table has a confound severe enough to invalidate it.

The FP32 column would be Qwen1.5-0.5B-Chat as shipped, carrying trillions of
pretraining tokens. The 1-bit columns would be that same model binarised and
then given roughly a million tokens of wikitext to recover. The measured gap
is then mostly "pretraining budget minus recovery budget", and the first
reviewer question is *"your recovery is 0.0001% of pretraining, so what did
you actually learn?"*

Two changes fix it:

**The FP32 arm is fine-tuned too.** Same data, same steps, same order. It is
`FP32 + identical recovery`, not `FP32 as shipped`. Otherwise the 1-bit arms
absorb a domain-adaptation penalty that has nothing to do with quantization.

**A fourth arm: FP32 + AR.** Without it you cannot separate *"AR helps"* from
*"AR helps specifically under binarization"*. Only the second is interesting,
and it is an interaction term, so it needs the full 2x2:

|            | no AR       | with AR        |
|------------|-------------|----------------|
| **FP32**   | `fp32`      | `fp32_ar`      |
| **1-bit**  | `onebit`    | `onebit_ar`    |

The headline quantity is the interaction:

```
interaction = (onebit_ar - onebit) - (fp32_ar - fp32)
```

A negative interaction means AR buys more under binarization than it buys in
general. That is the number nobody has. "AR lowers perplexity" on its own is
not, and would be a weaker claim than the existing literature already
supports.

---

## 2. Prior art, and what is actually novel here

This matters more than the experiment design, because it determines whether
the result is publishable at all.

**The mechanism is not new.** Adding a compensating pathway around a
binarised layer is roughly eight years old and well covered:

| Work | Year / venue | What it adds |
|---|---|---|
| Bi-Real Net | ECCV 2018 | real-valued shortcut around every binary conv |
| ReActNet ([2003.03488](https://arxiv.org/abs/2003.03488)) | ECCV 2020 | learnable per-channel threshold/shift (RSign, RPReLU) |
| MeliusNet ([2001.05936](https://arxiv.org/abs/2001.05936)) | WACV 2021 | alternating capacity/quality compensation blocks |
| OneBit ([2402.11295](https://arxiv.org/abs/2402.11295)) | NeurIPS 2024 | sign matrix + two FP16 value vectors, i.e. rank-1 correction |
| BiMaCoSR ([2502.00333](https://arxiv.org/abs/2502.00333)) | ICML 2025 | parallel sparse + low-rank FP branches on a binary diffusion model |
| HGF ([2602.05269](https://arxiv.org/abs/2602.05269)) | preprint 2026 | ternary backbone + gated low-rank correction |

**RaBiT ([2602.05367](https://arxiv.org/abs/2602.05367), ICML 2026) is the
one to read first, and it is the reviewer's first objection.** It identifies
*inter-path adaptation*: when parallel residual paths are added for error
compensation, QAT lets them co-adapt into redundant features and the
compensation structure dissolves. Its fix is to derive each path
*sequentially* from one shared full-precision weight, enforcing a strict
hierarchy where path k corrects path k-1's error.

ResABit's accumulator `R_l = sum_{i<=l} A_i` is a parallel pathway. RaBiT
predicts it should degrade into redundancy. **Any write-up must engage this
head-on**, and the alpha trajectory is the evidence: if the gates grow and
then collapse toward zero, that is inter-path adaptation happening in front
of us and it should be reported as a replication of RaBiT, not buried.

HGF is the closest twin on the LLM side and reports recovering ~55% of the
FP16-to-ternary gap on TinyStories — but at 5.4M parameters, single author,
unreviewed. It is prior art to cite and a result that is plausibly beatable
at real scale.

**What is defensibly novel here** is not the mechanism. It is:

1. The **interaction measurement** at matched budget with a measured noise
   floor. Most of the works above report "our full method vs baseline", not a
   factorial isolating the pathway's contribution.
2. **Recovery from a pretrained checkpoint on a fixed small budget**, which is
   the regime a practitioner actually faces, versus the from-scratch
   trillion-token regime of BitNet.
3. If the result is null, **a clean negative result with the noise floor
   attached** is worth publishing and is rarer than another positive claim.

Do not claim novelty for "add a residual pathway to survive binarization".
It is taken.

---

## 3. Controls

**Same starting weights.** Every arm loads the same Qwen1.5-0.5B-Chat
tensors, verified complete (`src/loader.py` refuses a partial map; the old
converter silently dropped 72 attention bias tensors).

**Neutral at initialisation.** `alpha = 0`, so an AR arm is bit-identical to
its non-AR twin at step 0 — asserted in `tests/test_parity.py`. Any
difference is what training did with the pathway, not a changed init
distribution.

**Fixed token budget, not wall clock.** 300 optimizer steps x 4096 tokens =
1.23M tokens, same order for every arm. Wall-clock budgeting, which is what
autoresearch uses, converts thermal throttling into treatment variance.

**Frozen embeddings everywhere.** The tied embedding/readout is 155M of 464M
parameters. Leaving it trainable lets a 1.2M-token run rewrite the output
head. All arms share a frozen readout so differences come from the blocks.

**Parameter parity.** AR adds 24 scalars to 464M parameters. This is not a
capacity confound and needs no width adjustment — unlike most of the prior
art above, which adds real parameters and therefore does need one.

**Only 302M of 464M parameters are binarised.** Embeddings and the readout
stay FP32, standard for the BitNet family. So `1.125 bits/weight` is true of
the quantised projections and false of the model. Both numbers are reported
(`ModelConfig.bits_per_quantized_weight` and `effective_bits_per_weight`).
Quoting only the first would be the single easiest thing to catch.

**One deliberate asymmetry, declared.** The residual gates get an effective
learning rate 10x the base, via `DecoderLayer.ALPHA_GAIN`. AdamW takes steps
of roughly the learning rate regardless of gradient magnitude, so 24 fresh
scalars starting at zero barely move in 300 steps — the AR arm would be a
no-op by construction rather than by evidence. This gives the intervention
its best shot; the alpha trajectory is logged so the choice is auditable.

---

## 4. Metrics, and why accuracy alone is not enough

**Perplexity — wikitext-2-raw, strided.** Window 1024, stride 512, scoring
only each window's final 512 tokens. Non-overlapping chunks make every
chunk's first tokens near-unpredictable and inflate the number; padding to a
fixed length and counting pad positions (which the original code did)
inflates it further. Token-level, same tokenizer across arms.

**Top-1 next-token accuracy** on the same positions. Keeps moving after
perplexity has passed four digits.

**Zero-shot suite — ARC-Easy, HellaSwag, LAMBADA.** Log-likelihood ranked, no
sampling noise. Chosen because a 0.5B model is meaningfully above chance on
them; WinoGrande and MMLU are at chance at this scale and would only add
noise. Reported with binomial standard error, because a 1.5-point gap on 2376
items is not a result.

**Teacher divergence — KL to the FP32 model, and top-1 agreement.** This is
the metric that carries the ablation and the reason to expect a usable
answer.

At 1 bit with a 1.2M-token recovery, every arm may sit at or near chance on
the accuracy tasks. A 4-way task cannot distinguish a broken model from a
very broken one — it floors. KL-to-teacher has no floor and stays sensitive
exactly where accuracy stops resolving. It is also the natural measure of
"how much did quantization damage this model", which is the actual question.

Almost nobody reports it. It is the strongest column in the table.

---

## 5. Noise floor before effect

The single most important step, and the one autoresearch omits.

Two variance sources are measured separately:

1. **Backend nondeterminism.** Re-run one seed unchanged. MLX's GPU
   reductions are not bitwise reproducible, and the binarised network
   amplifies that: two three-step runs at the same seed already differed by
   **7% in perplexity**. `sign()` sits on a discontinuity, so a one-ulp
   difference flips a weight and the error compounds with depth
   (`tests/test_mlx_parity.py` pins this: cross-backend disagreement rises
   from ~1e-5 unquantised to ~1e-2 binarised, growing monotonically layer
   over layer).

2. **Seed variance.** Vary init and data order across seeds {0, 1, 2}.

**Seeds are paired.** Both arms see the same seed set, and the statistic is
the per-seed difference `d_s = ppl_AR(s) - ppl_noAR(s)`. Shared seeds cancel
shared variance, which is the cheapest statistical power available on a
budget of hours.

**The decision rule, written down before the runs:**

- `|mean(d)| < 2 * SE(d)` → **no measurable effect at this budget.** Report
  it as such. Do not go looking for a subgroup where it worked.
- Otherwise report the signed effect with its interval.

If the effect is smaller than the noise floor, the honest output is that the
experiment cannot answer the question at this budget — and knowing that after
two hours is worth more than a confident number that does not replicate.

**Dev/held-out separation.** wikitext *validation* perplexity is the dev
metric used while iterating. The zero-shot suite and wikitext *test* are
held out and run **once**, at the end, on the final arms. Autoresearch has no
such guard and its author flagged the resulting overfitting; with ~700
experiments against one split, the accepted chain is a running minimum over
noisy draws and is upward-biased by construction.

---

## 6. Failures are data

Crashes, divergences and OOMs are ledger rows with `status` set, never
dropped. If one arm is less feasible than the other — more likely to diverge,
more memory-hungry — that is a finding about the intervention, and silently
dropping those runs would both hide it and bias the keep rate.

The ledger (`results/ledger.jsonl`) is append-only and stores **raw per-run
samples**, not aggregates, so any statistic can be recomputed later without
re-running anything.

---

## 7. The table to fill

Rows are metrics, columns are arms. Everything marked `+/-` carries spread
across paired seeds.

| | FP32 +FT | FP32 +FT +AR | 1-bit QAT | 1-bit QAT +AR |
|---|---|---|---|---|
| wikitext-2 ppl (strided) | | | | |
| wikitext-2 top-1 acc | | | | |
| KL to FP32 teacher (nats) | | | | |
| top-1 agreement with teacher | | | | |
| ARC-Easy acc | | | | |
| HellaSwag acc_norm | | | | |
| LAMBADA acc | | | | |
| bits/weight, quantised params | 32 | 32 | 1.125 | 1.125 |
| effective bits/weight, model | 32 | 32 | | |
| learned alpha, mean / max | — | | — | |
| tokens seen | | | | |
| wall-clock, M5 | | | | |

Followed by the two derived quantities that are the actual result:

- **Quantization gap**: `onebit - fp32`, and `onebit_ar - fp32_ar`
- **Interaction**: `(onebit_ar - onebit) - (fp32_ar - fp32)`, with its
  interval

---

## 8. Known limitations to state in the write-up

- **One base model, one scale.** 0.5B. Nothing here shows the effect
  survives to 7B, and the extreme-quantization literature is full of results
  that change sign with scale — BitNet b1.58's central claim is explicitly
  that its parity with FP16 only *begins* at 3B.
- **One dataset for recovery.** wikitext-2. Cheap and standard; also narrow.
- **1.2M recovery tokens.** Three to six orders of magnitude below the
  ternary-QAT literature (BitNet b1.58 2B4T used 4T). This is a
  low-budget-recovery result and must be labelled as one.
- **Weight-only quantization.** Activations stay FP32. BitNet a4.8 and
  BitNet v2 are the relevant comparison for activation quantization, and
  activation outliers are the documented blocker.
- **Hardware-specific throughput numbers.** Apple Silicon, MLX. The MLX fork
  of autoresearch found that different Macs converged on different winning
  recipes, so treat any wall-clock claim as machine-local.
- **MLX Metal matmul is not true FP32** (~bf16-grade accumulation). Training
  uses it; every reported metric is computed in PyTorch. This is why the two
  backends exist.

---

## 9. Structural note: only 23 of the 24 gates are live

Layer 0 receives an empty accumulator (`R_{-1} = 0`), so its gate is never
applied and carries no gradient — confirmed in
`tests/test_parity.py::test_first_layer_gate_is_structurally_inert`.

This is correct: there is no prior attention to re-inject at the first
layer. But it means any statistic over `alpha` must exclude index 0, or it
reports a mean dragged toward zero by a parameter that could not have moved.
`report.py` averages over layers 1 and up.

It also slightly weakens the intervention relative to how it is usually
described: the accumulator only begins contributing at layer 1, so the
"every layer sees all previous attention" framing is off by one.
