# Sub-2-bit weights for a discrete diffusion language model: a factorial measurement at 0.5B

**Draft preprint. Every number is generated from a ledger
(`results/ledger.jsonl`, `results/diffusion_ledger.jsonl`,
`results/grid_ledger.jsonl`); none is hand-entered. Quantities not yet
measured are marked `[PENDING]`, and there is no other kind of placeholder.**

---

## Abstract

Every published quantization result for diffusion language models is
post-training and stops at 2 bits; quantization-aware training below 2 bits
exists only for autoregressive models. We target the gap: ternary (1.725
bits/weight stored) weights on a masked discrete diffusion model, asked as a
factorial — does sub-2-bit quantization damage a diffusion LM more or less
than it damages an autoregressive one at the same recovery budget?

The cross-architecture comparison has a metric problem: autoregressive arms
report next-token NLL, diffusion arms report a sampled NELBO bound on an
easier task, and the two levels are not comparable. We resolve it by
normalising against a shared floor. A model that has learned nothing scores
`log(vocab)` under both metrics, so the *fraction of headroom below that
floor destroyed by quantization* is dimensionless and means the same thing in
both regimes. The interaction is computed on that quantity only; raw nats
stay in the ledger.

Measured so far, on Qwen1.5-0.5B-Chat at a 1.23M-token budget: the
diffusion adaptation clears its floor by 8.06 nats (NELBO 10.539 → 3.876,
mask accuracy 0.008 → 0.275), so quantization damage is measurable on top of
it; and ternary with an absmax scale silently zeroes 85% of a Gaussian
matrix, which is why the scale statistic travels with the scheme.

The factorial is complete at three paired seeds, and it resolves: **ternary
quantization costs the diffusion model 25.7% of its headroom against 14.6%
for the autoregressive model — an interaction of +0.111 in headroom share,
56x its paired standard error, sign stable across every seed.** The same
protocol that returned a null 1.2x its SE in Part I returns this at 56x,
which is the strongest available evidence the effect is real rather than
pipeline artifact. At this scale and budget, the masked-diffusion objective
is measurably more fragile under sub-2-bit weights than next-token
prediction — the opposite ordering from the literature's one comparative
PTQ datapoint at 2–4 bits, which makes the regime boundary the next
question.

This work sits on a completed prior measurement: a 2x2 crossing {FP32, 1-bit}
with a cross-layer attention residual on the autoregressive model (§Part I),
whose result is a null — no detectable interaction, with the residual gates
converging *negative* in FP32 as well. Part I is retained in full because it
is what validates the quantizer, the two-backend split, and the measurement
discipline Part II inherits: paired seeds against a measured noise floor
(exactly zero — the pipeline is bitwise reproducible), failures kept in the
ledger, and two methodological findings that transfer. Binarised
transformers are numerically chaotic, with cross-backend disagreement
compounding monotonically with depth — a diffusion model re-runs that forward
up to dozens of times per block, and whether the chaos also compounds across
denoising steps is an open question this protocol can now ask. And accuracy
benchmarks floor at chance while divergence-based metrics keep resolving,
which is why the grid's headline quantity is measured in nats and not
accuracy points.

---

# Part I — The autoregressive 1-bit ablation (completed)

The remainder of this part is the completed measurement, unchanged in
substance: a factorial isolating a cross-layer attention residual's
interaction with 1-bit quantization. Its abstract-level summary: interaction
**−0.0212 nats**, inside a decision rule fixed in advance (2 SE) and of the
same order as the paired standard error on one of its own terms — no
evidence of preferential repair, and no resolution to exclude an effect of
that size either. 22 of 23 live gates converge negative in both the 1-bit
and FP32 arms; the model learns to subtract the accumulated residual, so the
suppression is not a response to quantization damage.

---

## 1. Introduction

Weight binarization is attractive because autoregressive decoding is
memory-bandwidth-bound: dropping from 16 bits to ~1.1 bits per weight cuts
the traffic that dominates per-token latency. It is difficult because
`sign()` discards almost all information in a weight, and a model not trained
for it degenerates.

The standard remedy is quantization-aware training. A second, widely reused
remedy is architectural: give the network a higher-precision route around the
binarised operation. Bi-Real Net [1] added a real-valued shortcut to every
binary convolution; ReActNet [2] added learnable per-channel thresholds and
shifts; OneBit [3] replaced each linear with a sign matrix plus two FP16
vectors; BiMaCoSR [4] attached parallel sparse and low-rank branches to a
binarised diffusion model; SVDQuant [5] absorbed outliers into a 16-bit
low-rank branch at 4 bits.

These works report *method versus baseline*. That comparison conflates two
effects: the pathway may improve the model generally, and it may specifically
repair binarization damage. Only the second justifies the pathway as a
quantization technique, and separating them requires a 2x2, which — to our
knowledge — none of the above reports.

RaBiT [6] gives a concrete reason to expect the separation to matter. It
identifies *inter-path adaptation*: under QAT, parallel compensation paths
co-adapt into redundant features and the intended residual structure
dissolves. Its remedy is to derive paths sequentially from one shared
full-precision weight. If inter-path adaptation is general, then a parallel
accumulator should show a characteristic signature — gates that grow and then
collapse — and a near-zero interaction term.

**Contributions.**

1. A factorial ablation isolating a cross-layer attention residual's
   interaction with 1-bit quantization, at matched budget with paired seeds
   and a measured noise floor (§3, §5).
2. Evidence on whether RaBiT's inter-path adaptation appears for an
   accumulating (rather than parallel-per-layer) residual, read from the
   learned gate trajectory (§5.3).
3. Two protocol results for the low-budget recovery regime: numerical chaos
   under binarization (§5.4), and the failure of accuracy benchmarks to
   resolve differences that KL-to-teacher resolves (§4.3).

We explicitly do **not** claim the mechanism is novel.

---

## 2. Method

### 2.1 Q1_0_g128

Each weight is one sign bit; every 128 consecutive weights along the input
dimension share one FP16 scale:

```
w_i = s_g * (2 b_i - 1),   b_i in {0,1},   s_g = max |w| over group g
```

Storage is `1 + 16/128 = 1.125` bits per quantised weight. Training keeps
FP32 master weights and applies a straight-through estimator; because the
STE argument `w/s_g` lies in `[-1,1]` by construction, the usual clip mask
never activates and the estimator is plain pass-through.

**Scope of quantization.** Only the seven projections per block (q, k, v, o,
gate, up, down) are binarised: 308.3M of 464.0M parameters. Embeddings and
the tied readout remain FP32, following the BitNet family. The model-wide
average is therefore 11.49 bits/weight, not 1.125, and we report both. An
exported checkpoint bears this out: 666 MB on disk, 43 MB of binarised
projections against 623 MB of FP32 embedding table.

**A frozen checkpoint does not compute the number in the table.** Every
perplexity reported here is the training forward — FP32 master weights pushed
through the straight-through quantizer on each call — because that is what
training optimised and what all four arms were compared on. Freezing to the
packed representation stores the group scales in FP16, perturbing each layer
by ~2e-4 relative. Released checkpoints are therefore scored separately on
the frozen path and carry that number, not this one.

We expected that perturbation to compound: §5.4 measures binarised stacks
diverging monotonically with depth, and 24 layers of 2e-4 is not obviously
negligible. **It does not compound.** The frozen `onebit` checkpoint scores
282.2098 against the training forward's 282.2077, a difference of 8e-6 nats.
The distinction is the sign bits: FP16 rounding moves group magnitudes and
flips nothing, so it never crosses the discontinuity that drives the
divergence in §5.4. Perturbations that stay on one side of `sign()` are
benign; the ones that cross it are not. That is a sharper statement of §5.4's
finding than §5.4 makes, and it was only available by measuring both paths.

### 2.2 Attention residuals

Each layer accumulates all prior attention outputs and re-injects them
through a learnable per-layer gate:

```
A_l = Attn(RMSNorm(h_{l-1}))
h_l = h_{l-1} + A_l + alpha_l * R_{l-1}
R_l = R_{l-1} + A_l
h_l = h_l + MLP(RMSNorm(h_l))
```

`alpha` is initialised to 0, so an AR arm is bit-identical to its non-AR twin
at step 0 (asserted in the test suite). The gate is stored pre-divided by a
constant gain of 10, giving those 24 scalars an effective learning rate 10x
the base without a second optimizer — AdamW takes steps of roughly the
learning rate regardless of gradient magnitude, so fresh scalars starting at
zero would otherwise barely move inside a short budget and the arm would be a
no-op by construction rather than by evidence. This asymmetry favours the
intervention and is declared as such.

---

## 3. Experimental protocol

**Arms.** Four, from identical pretrained Qwen1.5-0.5B-Chat weights:
`fp32`, `fp32_ar`, `onebit`, `onebit_ar`.

**The FP32 arms are fine-tuned too**, on the same data for the same steps.
A fine-tuned 1-bit model compared against an un-fine-tuned FP32 model
collects a domain-adaptation bonus the reference never received. On our
runs that understates binarization damage by 0.535 nats: Qwen1.5-0.5B-Chat
scores 25.005 perplexity on wikitext-2 as shipped and 14.653 after the same
recovery budget the 1-bit arms get.

**Budget.** 300 optimizer steps x 4096 tokens = 1.23M tokens of
wikitext-2-raw train, identical order across arms. Fixed tokens rather than
fixed wall clock: wall-clock budgeting converts thermal throttling on a
laptop into treatment variance. AdamW, peak LR 1e-4, cosine decay, 5% warmup,
grad clip 1.0, batch 2 x 4 accumulation, sequence length 512.

**Frozen embeddings in every arm.** The tied embedding/readout is 155M of
464M parameters; leaving it trainable lets a ~1M-token run rewrite the output
head. A shared frozen readout keeps the comparison on the blocks.

**Paired seeds.** Seeds {0,1,2,3,4} on the contested 1-bit pair, identical
across arms; the statistic is the per-seed difference, which cancels shared
init and data-order variance. The FP32 arms carry one seed each, which is a
real asymmetry: it means the interaction's standard error is estimated from
the 1-bit term alone and is quoted as a lower bound.

**Noise floor, measured before the effect.** Two sources are separated: a
same-seed rerun isolates backend nondeterminism (measured at exactly zero on
this machine), and varying the seed adds init and data order. The decision rule is fixed in advance: if
`|mean(d)| < 2 SE(d)`, the reported result is *no measurable effect at this
budget*.

**Dev/held-out separation.** wikitext validation perplexity is the dev metric
during iteration; the zero-shot suite and wikitext test are run once, at the
end, on the final arms.

**Implementation.** Training on MLX (1.72x faster than PyTorch/MPS on the
real QAT step: 1140 ms vs 1959 ms per fwd+bwd, batch 2 x seq 512, Apple M5).
All reported metrics are computed in PyTorch, because MLX's Metal matmul is
not true FP32 — measured against float64 on a 1024-wide matmul it is 0.098
off versus PyTorch's 0.00013, roughly bf16-grade accumulation. The PyTorch
model is validated against HuggingFace `Qwen2ForCausalLM` to 1e-3 on logits;
the MLX port is validated against the PyTorch model to 1e-5 on CPU.

---

## 4. Metrics

### 4.1 Perplexity

wikitext-2-raw, strided windows (length 1024, stride 512), scoring only each
window's final 512 tokens so no token is scored twice and no token is scored
without context. Token-level, one tokenizer across all arms.

### 4.2 Zero-shot accuracy

ARC-Easy, HellaSwag (acc_norm), LAMBADA, log-likelihood ranked, reported with
binomial standard error. WinoGrande and MMLU are omitted: a 0.5B model is at
chance on both, so they contribute noise, not signal.

### 4.3 Teacher divergence

Mean `KL(teacher || student)` on the student's next-token distribution over
wikitext, plus top-1 agreement, against the FP32 fine-tuned arm.

This is the metric that carries the ablation, and the reason is specific.

The accuracy suite is not insensitive in general: it separates the FP32 and
1-bit arms decisively (ARC-Easy 0.542 against 0.262, chance 0.25). What it
cannot do is separate two 1-bit arms from each other, because both have
already fallen to chance and a 4-way task has no room below it. LAMBADA is
starker still: exactly 0.000 for both 1-bit arms, against 0.33 for the base
model.

So the standard suite sizes the quantization gap and stops. Comparing
recovery methods needs a metric with no floor, and KL-to-teacher is the
natural one -- it is a direct measure of how far quantization moved the
model's predictive distribution. We suggest it belongs in
extreme-quantization tables and note it is almost never reported.

---

## 5. Results

The canonical table is generated from the run ledger by `report.py`; the
numbers quoted below are from it. Five paired seeds on the 1-bit arms, one
reference seed on the FP32 arms.

### 5.1 Main table

| | FP32 +FT | FP32 +FT +AR | 1-bit QAT | 1-bit QAT +AR |
|---|---|---|---|---|
| wikitext-2 ppl | 14.653 | 14.827 | 285.669 ± 3.165 | 283.098 ± 8.749 |
| wikitext-2 NLL | 2.6846 | 2.6965 | 5.6548 ± 0.0110 | 5.6454 ± 0.0307 |
| wikitext-2 top-1 | 0.4739 | 0.4717 | 0.2095 ± 0.0010 | 0.2104 ± 0.0030 |
| KL to shipped Qwen (nats) | 1.5953 | 1.5952 | 3.9582 | 3.9792 |
| top-1 agreement | 0.6412 | 0.6432 | 0.2774 | 0.2715 |
| ARC-Easy acc | 0.5417 | 0.5429 | 0.2622 | 0.2639 |
| HellaSwag acc_norm | 0.4560 | 0.4440 | 0.2590 | 0.2630 |
| LAMBADA acc | 0.3360 | 0.3180 | 0.0000 | 0.0000 |

For reference, Qwen1.5-0.5B-Chat as shipped scores 25.005 perplexity, 0.4192
top-1, and 0.33 LAMBADA under the same protocol.

**Binarizing 308.3M of 464.0M parameters costs 2.970 nats — a 19.5x
perplexity increase — at a 1.23M-token recovery budget.** That is the
headline quantization number and it is measured against an FP32 arm given the
identical budget.

A note on how these numbers moved. An earlier revision of this table pooled
three identical reruns of `onebit` at seed 0 — a determinism check, not
seeds — into the seed statistics, reporting the arm as eight seeds and
pulling its mean 1.3 perplexity toward the one configuration that happened to
be repeated. The aggregation now filters by stage. It is worth stating
because the error was invisible in the output: the table simply read `seeds |
8` and every downstream quantity inherited it.

### 5.2 The interaction term

All quantities in nats. Perplexity is exp(NLL), so differencing raw
perplexities across arms 19x apart would subtract non-commensurable
quantities and manufacture an interaction from the scale gap alone.

- Quantization cost, no AR: **+2.9702** nats (19.5x)
- AR cost at FP32: **+0.0118** nats (1.012x)
- AR cost at 1-bit: **−0.0094** nats (0.991x)
- **Interaction: −0.0212 nats**

The decision rule was fixed in advance at 2 SE. The paired standard error on
the 1-bit AR term is 0.0181 nats, so the rule places the interaction inside
the noise band — but only just: it is 1.2x that standard error, not many
times smaller. And 0.0181 is a lower bound on the interaction's true standard
error, because the FP32 arms contribute one seed each and their variance is
absent from it entirely.

**The honest statement is two-sided.** These data give no evidence that the
attention residual preferentially repairs binarization damage. They also lack
the resolution to rule out an effect of the size measured. Reporting the
first half alone would overstate the null; reporting the point estimate's
sign as a finding would be worse.

The paired comparison on the 1-bit arms is within noise on every metric, and
the per-seed deltas change sign three times:

| metric | per-seed deltas | mean | SE | verdict |
|---|---|---|---|---|
| wikitext ppl | +13.117, +5.408, −10.833, −6.150, −14.398 | −2.571 | 5.151 | within noise |
| wikitext NLL | +0.0454, +0.0189, −0.0386, −0.0218, −0.0508 | −0.0094 | 0.0181 | within noise |
| final train loss | +0.0558, +0.0173, −0.0394, −0.0193, −0.0500 | −0.0071 | 0.0195 | within noise |
| top-1 acc | −0.0037, −0.0005, +0.0031, +0.0016, +0.0042 | +0.0009 | 0.0014 | within noise |

Seed 0 alone would have supported "the attention residual costs 13
perplexity". Seed 4 supports the opposite with a similar magnitude. We report
this explicitly because it is the failure the paired design exists to catch,
and because a single-seed ablation at this budget would have produced a
confident number that does not replicate — in either direction, depending on
which seed was run.

### 5.3 Stability, and where the gates go

The means are close; the spreads are not.

| arm | perplexities | mean | sd |
|---|---|---|---|
| `onebit` | 282.208, 283.768, 286.369, 285.441, 290.561 | 285.669 | 3.165 |
| `onebit_ar` | 295.324, 289.176, 275.536, 279.291, 276.164 | 283.098 | 8.749 |

The AR arm is 2.8x more variable across seeds without being better on
average. The variance ratio is 7.6 on 4 and 4 degrees of freedom, against a
95% critical value of 6.39 for F(4,4) — it clears the threshold, and it
clears it narrowly enough that we do not treat the result as established. The
ratio moved from 23.2 at three seeds to 7.6 at five; a variance estimated
from five runs is itself high-variance, and the honest reading is an order of
magnitude, not a measurement. For a recovery technique this is nonetheless
the more practical question than the mean: a method whose outcome swings with
the seed cannot be trusted on a single run.

**This finding carries a confound of our own making.** The gates are given a
10x effective learning rate so the arm would not be a no-op inside a
300-step budget (§2.2). At gain 1 the gates would barely move and the
variance would presumably match. The claim is therefore narrower than
"attention residuals destabilise training": at a gain large enough for the
pathway to be used at all in this budget, the arm becomes markedly less
reproducible without becoming better. A gain sweep is the obvious follow-up
and we do not run it.

**The gate direction is not subject to that confound** — the gain sets how
fast the gates move, not which way. And the direction is the substantive
result:

- 22 of 23 live gates converge **negative**
- profile by depth: early −0.064, middle −0.069, late −0.025
- mean |alpha| rises from 0.00003 to a 0.053 plateau by step ~294 and holds

Note the profile's shape: the gates are pushed hardest in the early and
middle thirds and weakest in the late third. A pathway compensating for
accumulated per-layer damage would be expected to act most strongly where the
damage has accumulated most, which is the opposite ordering.

The model learns to **subtract** the accumulated attention residual. The
accumulator `R_l = sum_{i<=l} A_i` grows with depth by construction, so it
injects increasing variance into the residual stream; what training does
with the pathway is cancel it. It supplies a nuisance to be damped rather
than signal to be used, which is consistent with the arm being no better and
less stable.

**The FP32 arm does the same thing**, which settles what the damping is a
response to. `fp32_ar` also converges 22 of 23 gates negative (early −0.019,
middle −0.015, late −0.009). So the suppression is not a reaction to
binarization damage; it is what the architecture elicits in either regime.

What quantization changes is the magnitude. Final mean |alpha| is 0.0532 in
the 1-bit arm against 0.0154 in FP32 — the damaged model pushes the gates
roughly 3.5x harder negative. This is an independent line of evidence
against the hypothesis: if the pathway were repairing quantization damage we
would expect positive gates growing with damage, and what we observe is
negative gates growing with damage.

**This is not RaBiT's inter-path adaptation.** That failure mode predicts
gates that grow and then collapse toward zero as parallel paths co-adapt
into redundancy. Here they rise and hold at a stable negative value. The
outcome is also unhelpful, but the mechanism is distinct and we name it
separately: the accumulator is not becoming redundant, it is being actively
suppressed.

Note also that layer 0's gate is structurally inert — the accumulator enters
the first layer empty, so it receives no gradient. Only 23 of 24 gates are
live, and the "every layer sees all previous attention" framing is off by
one.

### 5.4 Numerical chaos under binarization

Binarising the projections raises cross-backend disagreement on identical
weights from ~1e-5 to ~1e-2 relative, growing monotonically with depth
(3.5e-3 at layer 0 to 3.2e-1 at layer 23 on a 24-layer stack). `sign()` sits
on a discontinuity, so a one-ulp difference in the master weight flips a
stored bit and the perturbation compounds.

Two consequences. Post-training 1-bit does not merely degrade quality, it
destabilises the computation, which is an additional argument for QAT beyond
the usual accuracy one. And any published PTQ-to-1-bit perplexity is
implementation-dependent unless the backend, the accumulation order and the
seed are stated.

We had expected same-backend nondeterminism to contribute here, and it does
not: three identical reruns of the 1-bit arm returned 282.208 perplexity
each time, to the digit. On this machine the pipeline is bitwise
reproducible, so the entire seed-to-seed spread reported above is genuine
seed effect and not a mixture of seed effect and kernel noise. The chaos is
across implementations, not within one.

---

# Part II — Ternary weights on a discrete diffusion model (in progress)

Part I answered its question and, in doing so, validated the instruments.
Part II is the thesis: sub-2-bit weights on a model that generates by
iterative denoising rather than left to right. Nothing published does this —
quantization of LLaDA/Dream-family models bottoms out at 2-bit post-training
[14, 15], with no QAT and nothing below 2 bits, while sub-2-bit QAT exists
only for autoregressive models. One comparative datapoint favours the
attempt: a diffusion coding model degrades *less* than an autoregressive peer
under 2–4-bit PTQ on coding benchmarks (arXiv:2604.20079). And diffusion LMs
need weight compression more, not less: there is no KV cache to amortise —
every denoising step pays the full weight traffic that dominates decode
latency.

## 6. Q1_58: ternary at 1.725 bits stored

Weights take values in {−1, 0, +1}, scaled per group of 128 by the group's
**absmean**, packed five trits to a byte (3⁵ = 243 ≤ 256): 1.6 bits of
payload against the information-theoretic log2(3) = 1.585, and 1.725
bits/weight with the FP16 group scale amortised.

The scale statistic is part of the scheme, not a tunable, and this is the
part that does not survive being guessed. Q1_0 divides by the group maximum,
correct there because `sign()` needs only the sign. Reusing a maximum with
ternary rounding sends every weight below half the group's largest to zero:
measured on Gaussian weights, 85% of the matrix zeroed at 0.83 relative
reconstruction error, against a 35/31/34 split at 0.44 for absmean. Both
variants train and both produce a falling loss curve; only one represents the
matrix. The split is pinned in the test suite, the scheme identifier is a
persisted buffer (a ternary checkpoint loaded as binary decodes base-3 bytes
as bit fields — it loads clean and is wrong everywhere), and the freeze
reduction is asserted to match the training forward's, because an absmax
freeze against an absmean forward trains normally and collapses on export.

Unlike Part I's binary path, the straight-through clip genuinely fires here:
`w/absmean` exceeds 1 for the roughly one-third of weights above the group
mean.

## 7. The diffusion model, and the check that gated everything

Absorbing-state masked diffusion in LLaDA's formulation: corrupt by replacing
tokens with `[MASK]` at a rate `t ~ U(0,1)` per sequence, predict the
originals bidirectionally, generate by iterative unmasking with
low-confidence remasking. The loss is the `1/t`-weighted NELBO estimator,
with the rate clamped at 1e-3 because `1/t` has infinite variance at the
bottom of the range — a declared, slight bias in place of an undeclared
unstable one.

`[MASK]` takes token id 151646: Qwen1.5 ships 151936 embedding rows for a
tokenizer that stops at 151646, so 290 pretrained-but-unreachable rows exist
(distinct vectors, identical norm 0.3094, the 1.2th percentile of trained
rows), and taking one avoids resizing the embedding, breaking the tie to the
readout, or invalidating the HuggingFace parity test.

Corruption is always injected, never drawn inside the model: two backends
cannot agree on a random draw, so the parity suite hands both the identical
mask and compares numbers rather than distributions — and paired seeds need
both arms of a comparison to see identical corruption for their difference to
be the intervention.

**The gating check.** An AR-pretrained model adapted to diffusion on 1.23M
tokens could plausibly land at the uniform floor, and quantization damage is
not measurable on a model that has already floored — Part I hit exactly this
in its accuracy suite, where two 1-bit arms were indistinguishable because
both sat at chance. So the FP32 diffusion arm ran alone first:

| | NELBO | mask accuracy |
|---|---|---|
| uniform floor, `log(151936)` | 11.9312 | — |
| Qwen1.5-0.5B, unadapted | 10.5390 | 0.0081 |
| after 1.23M tokens | **3.8761** | **0.2747** |

The unadapted model already sits 1.39 nats below the floor — bidirectional
context recovers that much for free, so the floor alone is not the bar. The
adaptation moves the bound 6.66 nats and ends **8.06 below the floor**,
naming 27% of the tokens it cannot see. There is room to measure damage in.
(48 validation blocks of 512 tokens, 4 corruptions each, fixed evaluation
seed. The number reproduced to the digit when the grid re-ran this cell from
a different script.)

**This bound is not comparable to Part I's perplexities.** Masked prediction
with bidirectional context is a different and easier task than next-token
prediction; a lower number here is not a better model, and the two are never
placed in one column.

## 8. The grid, and the shared-floor normalisation

Four cells: {FP32, ternary} × {autoregressive, diffusion}, identical
1.23M-token budget, identical data order, paired seeds, the contested
diffusion pair first at every seed so a truncated sweep still answers the
thesis question. The quantity is the interaction — does ternary cost the
diffusion model a larger share of what it had than it costs the
autoregressive one?

Because the two architectures' metrics differ in level, the interaction is
computed on a normalised quantity: both metrics assign `log(vocab) = 11.9312`
to a model that has learned nothing, so each cell has a headroom below that
shared floor, and the **fraction of headroom destroyed by quantization** is
dimensionless and commensurable across regimes. Raw nats are recorded per
cell for anyone who rejects the normalisation.

### 8.1 Result

Three paired seeds, all cells complete. Means with seed-to-seed sd:

| cell | metric | loss (nats) | headroom |
|---|---|---|---|
| fp32_diff | NELBO | 3.8702 | +8.0610 |
| ternary_diff | NELBO | 5.9408 | +5.9904 |
| fp32_ar | NLL | 2.6909 | +9.2403 |
| ternary_ar | NLL | 4.0350 | +7.8962 |

Ternary cost, as the share of headroom destroyed:

| | per-seed shares | mean | sd |
|---|---|---|---|
| diffusion | 0.2607, 0.2558, 0.2541 | **0.2569** | 0.0034 |
| autoregressive | 0.1523, 0.1405, 0.1436 | **0.1455** | 0.0061 |

Per-seed paired interaction: +0.1084, +0.1153, +0.1105 — the sign never
moves. Mean **+0.1114**, paired SE **0.0020**: 56x the standard error,
against the same 2 SE rule Part I's null sat inside.

**Ternary quantization costs the diffusion model a 1.77x larger share of its
headroom than it costs the autoregressive model, at matched budget, matched
data order, and paired seeds.** In raw nats (not commensurable across the
two metrics, reported for the ledger's sake): +2.071 NELBO on diffusion
against +1.344 NLL autoregressive.

Part I's null and this result were produced by the same protocol, which is
worth a sentence: the instrument distinguishes an effect it cannot detect
(−0.021 nats, 1.2x its SE) from one it can (+0.111, 56x). The contrast is
the strongest available evidence that the effect here is real rather than
an artifact of the pipeline.

Supporting observations. `fp32_ar` reproduced Part I's fine-tuned FP32 NLL
to the digit (2.6846 at seed 0) from a different script. QAT recovered the
ternary diffusion cell from below the uniform floor (smoke headroom −1.14)
to +5.99 — post-training ternary destroys the diffusion model outright and
recovery is what the budget buys. And ternary against Part I's binary at the
same architecture and budget is NLL 4.035 versus 5.655: the 0.6 extra
stored bits per weight buy back 1.62 nats of the quantization gap.

### 8.2 Budget context: where 1.23M tokens sits against BitNet and Bonsai

The recovery budget is the axis on which this work is least comparable to
its neighbours, so the comparison is tabulated rather than implied:

| model | scheme | params | training/recovery tokens | vs ours |
|---|---|---|---|---|
| **ResABit grid (this work)** | ternary QAT recovery | 0.5B | **1.23M** | 1x |
| ResABit ladder rung 2 | ternary QAT recovery | 0.5B | 4.92M | 4x |
| ResABit ladder rung 3 `[PENDING]` | ternary QAT recovery | 0.5B | 19.7M | 16x |
| Bonsai 0.5B (deepgrove) | ternary, trained | 0.5B | 3.8B | ~3,100x |
| BitNet b1.58 (paper) | ternary, trained | 0.7–3.9B | 100B | ~81,000x |
| BitNet b1.58 2B4T | ternary, trained | 2B | 4T | ~3,300,000x |

**Rung 2 is in, and the interaction is budget-stable.** At 4.92M tokens,
seed 0: ternary's cost falls in both regimes as recovery deepens — the
diffusion share drops 0.2607 → 0.1630 and the autoregressive share 0.1523 →
0.0584 — but their *difference* barely moves: interaction +0.1046 against
+0.1084 at the same seed on the 1.23M budget, well inside the ±0.03 rule
fixed in advance. Recovery buys both architectures back at similar absolute
rates and the diffusion penalty persists; as a *ratio* the gap widens
(1.7x → 2.8x) because the AR side approaches zero cost faster. Under the
pre-registered rule this triggered the 19.7M rung, whose cells are
`[PENDING]`. One seed per rung: the ladder measures a trend, and its error
bars are the grid's, not its own.

Two different quantities share this column and the distinction matters:
BitNet and Bonsai train ternary models from scratch (or near it); we
*recover* a pretrained FP32 model after ternarising it. Recovery is the
cheaper regime and the only one this hardware supports — Bonsai's 3.8B
tokens would take ~43 days per cell here. The table is context, not a
claim of parity.

What is actionable at this scale is the *trend*: a budget ladder on the same
four cells (1.23M → 4.92M → 19.7M tokens, seed 0) measures whether the
1.77x interaction grows, shrinks or holds as recovery deepens. If the
diffusion penalty shrinks with budget, the 1.77x is a
low-budget-transition artifact and Bonsai-scale recovery might erase it; if
it holds or grows, the fragility is structural. The first rung beyond
baseline is `[PENDING]` (running); the 19.7M rung is ~21 h of compute and
queued behind its result. The grid's resume key includes the budget, and the
summariser refuses to pool budgets — a 1200-step cell averaged into
300-step statistics would be the determinism-replicates mistake wearing a
different key.

### 8.3 What the result does and does not say

It says: at 0.5B and 1.23M recovery tokens, the masked-diffusion objective
loses proportionally more of what it had than next-token prediction does,
under identical ternary weights. Candidate mechanisms — the denoiser's
bidirectional attention reuses every projection under a wider input
distribution (clean and masked tokens mixed); the `1/t`-weighted objective
concentrates gradient on heavily corrupted batches; the AR-pretrained
initialisation is simply further from a good diffusion solution so the same
damage is harder to route around — are not separated by this design, and we
decline to pick one post hoc.

It does not say diffusion models cannot be quantized (26% of headroom lost
is damaged, not floored), does not compare diffusion quality to
autoregressive quality, and does not automatically survive scale — Part I's
§6 caveat about 0.5B applies with full force. The one comparative PTQ
datapoint in the literature (arXiv:2604.20079, 2–4 bits) found diffusion
*more* robust; at 1.725 bits with QAT we find the opposite ordering, and
reconciling the two regimes is the obvious next measurement.

One shape observation from the 2-step smoke run, reported as shape and
nothing more: before any meaningful recovery, ternary put the diffusion model
*below* the uniform floor (headroom −1.14) while the autoregressive twin kept
+1.52. Post-training ternary destroys the diffusion model outright at this
scale; what the grid measures is what a matched recovery budget buys back in
each regime. Single seeds carry no error bars, and Part I documents at length
what single-seed confidence is worth.

**The open question the protocol can now ask.** Part I measured binarised
stacks amplifying cross-backend perturbations monotonically with *depth* —
and measured the one counterexample: FP16 scale rounding does not compound,
because it moves magnitudes without flipping any sign bit, and only
perturbations that cross the discontinuity amplify. A diffusion model feeds
its own output back through the stack up to dozens of times per block.
Whether discontinuity-crossing noise also compounds across *denoising steps*
is a damage mode with no autoregressive analogue, nobody has measured it, and
the instruments here — bitwise-reproducible pipeline, injected corruption,
two pinned backends — are sufficient to do so.

## 9. Limitations

Shared by both parts:

- **One model, one scale (0.5B).** Extreme-quantization results are known to
  change with scale; BitNet b1.58's parity claim explicitly begins at 3B.
  The named target of Part II's thesis — models like DiffusionGemma-26B-A4B —
  cannot be QAT'd on this hardware at all, so Part II bears the same relation
  to its target that BitNet's small-scale ablations bear to its 3B claim.
- **1.23M recovery tokens**, three to six orders of magnitude below the
  ternary-QAT literature (BitNet b1.58 2B4T: 4T tokens). This is a
  low-budget-recovery result and is labelled as one throughout.
- **One recovery corpus** (wikitext-2), which is narrow — and it is also the
  corpus every reported perplexity is measured on. There is no held-out
  perplexity from a second distribution, so none of the perplexity numbers
  here is out-of-distribution.
- **Weight-only quantization**; activations remain FP32. Activation outliers
  are the documented blocker addressed by BitNet a4.8 and BitNet v2.
- **Five seeds on the 1-bit pair, one on each FP32 arm.** Enough to establish
  a noise floor and to catch a sign-flipping effect, not enough for a tight
  interval on a small one — and the asymmetry means the interaction's
  standard error is estimated from half of its terms.
- **PIQA is implemented and was never scored.** It appears in the task
  registry, which makes the suite read as larger than it is. The loader was
  broken (a script-based Hub dataset against `datasets` 4.x) and scoring a
  new benchmark used to require retraining every arm; both are now fixed and
  the benchmark is still unrun. We expect it to floor at chance alongside
  ARC-Easy and LAMBADA for the 1-bit arms, which is a prediction and is not
  reported as a result.
- **No sweep of the gate gain.** `ALPHA_GAIN = 10` is the declared asymmetry
  that makes the AR arm non-trivial inside this budget, and it is the
  confound in §5.3. No run varies it.
- **Throughput numbers are machine-local** to Apple Silicon and MLX.

Specific to Part II:

- **Three seeds.** Enough for a sign-stable effect 56x its SE; not enough
  for a tight interval on the magnitude, and Part I is the standing reminder
  of what fewer seeds are worth.
- **The candidate mechanisms are not separated** (§8.3). The design
  identifies *that* the diffusion objective loses more, not *why*.
- **The headroom-share normalisation is a modelling choice.** It is the only
  quantity offered as commensurable across architectures, the raw nats are
  ledgered so it can be recomputed differently, and a reader who rejects it
  is left with two within-architecture costs and no interaction.
- **The diffusion evaluation is a bound**, estimated with 4 corruptions per
  block; its Monte Carlo error is not yet characterised, and it shares the
  wikitext-2 single-corpus caveat above.
- **The adaptation is shallow.** 1.23M tokens of continued pretraining on top
  of an autoregressive model, not a diffusion model trained as one — the
  Dream/TESS-2 recipe at a tiny fraction of their budgets. Mask accuracy of
  0.27 clears the measurability bar; it is not a claim of a usable model.
- **Cross-denoising-step error compounding is posed, not measured** (§8).

---

## 10. Related work

**Extreme-quantization LLMs.** BitNet [7] and BitNet b1.58 [8]; BitNet a4.8
and v2 for activation quantization; the 2B4T report [9] for a fully trained
ternary model. Recent work has collapsed the training cost: Tequila [10]
reaches 3B with 10% of Spectra's tokens by repurposing deadzone-trapped
weights as dynamic biases, and CAT-Q [11] achieves ternary via post-training
quantization with 512 calibration samples, scaling to 235B. CAT-Q in
particular changes the economics of the regime studied here.

**Architectural compensation.** Bi-Real Net [1], ReActNet [2], MeliusNet
[12], OneBit [3], BiLLM and STBLLM for residual binarization; SVDQuant [5]
and EfficientDM for low-rank branches at 4 bits; BiMaCoSR [4] and BinaryDM
for binarised diffusion; HGF [13] for a gated low-rank correction on a
ternary LLM backbone, though at 5.4M parameters and unreviewed. RaBiT [6] is
the closest methodological precedent and the source of the inter-path
adaptation prediction we test.

**Adjacent white space.** No sub-2-bit result exists for diffusion language
models: quantization of LLaDA/Dream-family models bottoms out at 2-bit
post-training [14, 15], and a July 2026 survey of efficient dLLM inference
devotes one subsection to quantization citing two papers, with no mention of
1-bit, ternary, or QAT. The dLLM damage modes — iterative error amplification
across denoising steps, bimodal masked/unmasked activation distributions —
differ structurally from the autoregressive case and are not addressed by
existing compensation designs.

---

## References

[1] Liu et al. Bi-Real Net. ECCV 2018.
[2] Liu et al. ReActNet. arXiv:2003.03488, ECCV 2020.
[3] Xu et al. OneBit. arXiv:2402.11295, NeurIPS 2024.
[4] Liu et al. BiMaCoSR. arXiv:2502.00333, ICML 2025.
[5] Li et al. SVDQuant. arXiv:2411.05007, ICLR 2025.
[6] You et al. RaBiT. arXiv:2602.05367, ICML 2026.
[7] Wang et al. BitNet. JMLR 26(24-2050).
[8] Ma et al. The Era of 1-bit LLMs. arXiv:2402.17764.
[9] Ma et al. BitNet b1.58 2B4T. arXiv:2504.12285.
[10] Tequila. arXiv:2509.23809, ICLR 2026.
[11] Wang et al. CAT-Q. arXiv:2606.26650, ICML 2026.
[12] Bethge et al. MeliusNet. arXiv:2001.05936, WACV 2021.
[13] Trejo Pizzo. Hybrid Gated Flow. arXiv:2602.05269 (preprint).
[14] Zhang et al. Quant-dLLM. arXiv:2510.03274, ICLR 2026.
[15] Xu & Yang. DLLMQuant. arXiv:2508.14090.
[16] Attention Residuals. arXiv:2603.15031.

---

## Reproducibility

```bash
python -m pytest tests/          # validate against HuggingFace first

# Part I
python run_ablation.py --stage full --steps 300 --seeds 0 1 2
python report.py

# Part II
python run_diffusion_check.py --steps 300      # the gating check
python run_grid.py --seeds 0 1 2 --resume      # the factorial
```

Every number traces to a ledger row carrying its arm or cell, seed, git
commit and full training configuration: `results/ledger.jsonl` for Part I,
`results/diffusion_ledger.jsonl` and `results/grid_ledger.jsonl` for
Part II. Failed and diverged runs are retained; plumbing runs are tagged
`smoke` so no aggregate can pool them. The pipeline is bitwise reproducible
on this machine — the gating check's NELBO reproduced to the digit when the
grid re-ran the same cell from a different script — so any cell can be
rebuilt exactly from its ledger row.
