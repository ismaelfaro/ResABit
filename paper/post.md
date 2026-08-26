# The 2x2 nobody runs

*A null result about 1-bit quantization, and the eight times the experiment
caught me being wrong.*

---

The first number this project produced was a perplexity of **8,246,580**.

That is not a bad model. A model that has learned nothing at all scores about
151,936 — the vocabulary size, uniform over every token. Eight million means
the code was confidently wrong, which is a different and more interesting
failure. It turned out to be six bugs, and the largest was this: the INT8
inference path averaged the per-group quantization scales down to one scale
per row. Q1_0_g128 exists precisely so that every 128 weights get their own
scale. Averaging them discards the entire point of the format and inflates
perplexity by six orders of magnitude.

The other five: attention biases silently dropped by
`load_state_dict(strict=False)`, an embedding table loaded transposed, a bit
packer whose shift table overflowed on bit 7 (`1 << 7` wraps to `-128` in
int8, corrupting every eighth weight), `sign(0)` disagreeing between the
training forward and the packed representation, and `quantize()` never
marking the layer as quantized. Two config switches — `quantize_linear` and
`use_attention_residuals` — were declared and never read, so there was no way
to build the FP32 control arm at all.

None of that is the story. The story is what happened after the code worked.

---

## The question

Extreme quantization has a standard architectural remedy: give the network a
higher-precision route around the binarised operation. Bi-Real Net added a
real-valued shortcut to every binary convolution in 2018. ReActNet added
learnable thresholds. OneBit replaced each linear layer with a sign matrix
plus two FP16 vectors. BiMaCoSR attached sparse and low-rank branches to a
binarised diffusion model. It is a well-populated design space.

Every one of those papers reports **method versus baseline**. That comparison
answers "is the model with the pathway better than the model without it,"
which conflates two entirely different effects:

1. the pathway makes the model better in general, and
2. the pathway specifically repairs damage that binarization caused.

Only the second justifies calling it a quantization technique. Separating
them takes a 2x2 — the pathway crossed with the quantization — and the
quantity you want is the **interaction**:

```
interaction = (1bit_AR − 1bit) − (fp32_AR − fp32)
```

To my knowledge none of the papers above reports it. So I ran it, on
Qwen1.5-0.5B-Chat, with a cross-layer attention residual as the pathway:
every layer accumulates the sum of all prior attention outputs and re-injects
it through a learnable per-layer gate, initialised to zero.

## The answer

**No detectable interaction.** −0.0212 nats, against a decision rule fixed in
advance at two standard errors.

That is the finding, and it comes with a caveat I want to state before the
result rather than after. The paired standard error on the 1-bit term is
0.0181 nats. The interaction is 1.2x that — inside the rule, but not
comfortably inside it. And 0.0181 is a *lower bound* on the real standard
error, because the FP32 arms have one seed each and their variance is not in
that estimate at all.

The honest statement is two-sided: **these data give no evidence that the
attention residual preferentially repairs binarization damage, and they do
not have the resolution to rule out an effect of that size either.** Reporting
only the first half would overstate the null. Reporting the point estimate's
sign as a finding would be worse.

For scale: binarizing 308.3M of 464.0M parameters costs **2.970 nats**, a
19.5x perplexity increase, at a 1.23M-token recovery budget. The effect being
argued about is a hundred times smaller than the damage it was supposed to
repair.

## The part that did resolve

Perplexity was a null. The gates were not.

**22 of 23 live gates converge negative.** The model does not learn to use
the accumulated attention residual. It learns to *subtract* it.

Which makes mechanical sense once you look at what the accumulator does.
`R_l = R_{l−1} + A_l` grows without normalization; by layer 24 it is the sum
of 24 attention outputs, injecting steadily increasing variance into the
residual stream. Training's response is to cancel it. The pathway supplies a
nuisance to damp, not signal to use.

Two details sharpen this.

**The FP32 arm does the same thing** — also 22 of 23 negative. So the damping
is not a response to binarization damage. It is what the architecture elicits
in either regime, and that single observation does more to answer the
research question than the perplexity table does.

**The profile runs the wrong way.** Gates are pushed hardest in the early and
middle thirds of the stack (−0.064, −0.069) and weakest in the late third
(−0.025). A pathway compensating for accumulated per-layer damage should act
most strongly where damage has accumulated most. This is the opposite
ordering.

Quantization changes the magnitude, not the sign: final mean |alpha| is 0.0532
at 1 bit against 0.0154 in FP32. The more damaged model pushes the gates 3.5x
harder *negative*. If the pathway were repairing quantization damage we would
expect positive gates growing with damage. We observe negative gates growing
with damage.

One structural footnote that took embarrassingly long to notice: layer 0's
gate is inert. The accumulator reaches the first layer empty, so that gate is
never applied and receives no gradient. Only 23 of 24 gates are live, and the
"every layer sees all previous attention outputs" framing is off by one.

---

## Eight times the experiment caught me

This is the part I would most want to read in someone else's write-up, so
here it is in mine.

**1. I had the fine-tuning argument backwards.** I wrote that fine-tuning the
FP32 reference *overstates* binarization damage. It understates it. An
un-fine-tuned reference gives the 1-bit arm a domain-adaptation bonus the
reference never received — on these runs, 0.535 nats of one. Qwen1.5-0.5B-Chat
scores 25.005 on wikitext-2 as shipped and 14.653 after the same recovery
budget the 1-bit arms get. Comparing against 25.005 would have flattered the
1-bit result by hiding a fine-tuning gain inside a quantization number.

**2. I computed the interaction in perplexity.** Perplexity is `exp(NLL)`. The
FP32 arms sit near 14.7 and the 1-bit arms near 285. A fixed *relative* change
is worth ~0.15 perplexity points at one scale and ~2.9 at the other, so
subtracting raw perplexities across a 19x gap manufactures an interaction out
of the scale difference alone. In log space the same subtraction is a ratio of
ratios, which is what the question actually asks.

**3. I reported 7% run-to-run nondeterminism that did not exist.** I had taken
the figure from the MLX fork of `autoresearch` without checking it on this
machine. Three identical reruns returned 282.208 perplexity — to the digit,
every time. The pipeline is bitwise reproducible here. What I had actually
measured was my own `ALPHA_GAIN` constant changing between runs. Retracting
that made the seed-to-seed spread *more* meaningful, not less: with a floor of
exactly zero, all of it is genuine seed effect rather than a mixture of seed
effect and kernel noise.

**4. The results table silently pooled three determinism reruns into the seed
statistics.** The report generator defaulted its stage filter to "all stages",
so three identical repeats of seed 0 — run specifically to measure the noise
floor — were counted as three additional seeds. The table read `seeds | 8` for
a five-seed arm and pulled its mean 1.3 perplexity toward the one
configuration that happened to be repeated. Every derived quantity inherited
it. Nothing in the output looked wrong.

**5. A sentence that reversed meaning when the data changed.** The generator
described the interaction as "the standard error is Nx the interaction
itself," which reads as strong evidence for a null when N is large. With the
fifth seed, N dropped below 1 — the same template then described a *smaller*
standard error in language that still sounded like a null. It now states the
ratio in one direction and applies the decision rule explicitly.

**6. The parameter table had drifted from the code.** The README said 302M
binarised, 162M full precision, ~11.8 bits/weight average. Computed from the
config: 308.3M, 155.7M, 11.49. Nobody typed a lie; a hand-maintained table
simply stopped matching a computed one, which is how this particular claim
usually goes wrong in public.

**7. The checkpoint computes a different number than the table.** Every
perplexity in the results table is the *training forward*: FP32 master weights
pushed through the straight-through quantizer on each call. A frozen
checkpoint stores packed sign bits and FP16 group scales, and FP16 rounding
moves each layer's output by ~2e-4 relative. That number had never been
measured on a real checkpoint, so the released weights are scored separately
on the frozen path and carry their own figure.

This one has a twist, and it went against me in the useful direction. I
predicted the perturbation would compound with depth — 24 layers, and this
project's own numbers show binarised stacks diverging monotonically across
backends. It does not compound. The frozen checkpoint scores 282.2098 against
282.2077: **eight micronats.** The distinction turns out to be the sign bits.
FP16 rounding moves group magnitudes and flips nothing, so it stays on one
side of the discontinuity; the cross-backend divergence flips stored bits and
crosses it. Perturbations that respect `sign()` are benign, and ones that
cross it are not — which is a sharper statement than the chaos result I had
already written, and I only got it by measuring something I was confident I
could predict.

**8. The strict loader was not strict.** The module exists because the
original code used `load_state_dict(strict=False)` and lost 72 bias tensors.
Yet a quantized checkpoint missing its packed weights loaded without a
complaint: those buffers are registered as `None` until a checkpoint fills
them, and PyTorch drops `None` buffers from its missing-key accounting. The
model would have failed at the first forward pass instead of at load. Same
class of bug as the one the file was written to prevent, hiding one layer
down.

Also, quietly: **PIQA was in the task registry and never ran.** Its loader was
broken — `ybisk/piqa` is a script-based Hub dataset and `datasets` 4.x removed
script support, so `trust_remote_code=True` raises rather than degrades. The
repo read as though the suite were larger than it was.

---

## Two things worth taking away, independent of the ablation

**Binarization makes the network numerically chaotic.** The same weights on
two FP32 backends disagree by ~1e-2 relative once quantized, against ~1e-5
unquantized, and the disagreement grows monotonically with depth — 3.5e-3 at
layer 0 to 3.2e-1 at layer 23. `sign()` is a discontinuity, so a one-ulp
difference in a master weight flips a stored bit and the perturbation
compounds. Two consequences: post-training 1-bit does not merely degrade
quality, it destabilises the computation, which is an argument for QAT beyond
the usual accuracy one; and any published PTQ-to-1-bit perplexity is
implementation-dependent unless the backend, accumulation order and seed are
stated. Note this is chaos *across* implementations, not within one — the same
pipeline on the same machine is bitwise reproducible.

**Accuracy benchmarks have a floor, and it is above the interesting region.**
The standard suite separates FP32 from 1-bit decisively: ARC-Easy 0.542
against 0.262, chance 0.25. It cannot separate one 1-bit arm from another,
because both have already fallen to chance and a 4-way task has no room below
it. LAMBADA is starker: exactly 0.000 for both 1-bit arms against 0.336 for
the base model. So the standard table can size a quantization gap and then
stops. Comparing *recovery methods* needs a metric with no floor. KL-to-teacher
is the natural one, it kept resolving where accuracy did not, and it is almost
never reported in extreme-quantization tables.

## And one about small-scale ablations

Five paired seeds. Per-seed deltas on the contested pair:

```
+13.117   +5.408   −10.833   −6.150   −14.398
```

The sign changes three times. Seed 0 alone supports "the attention residual
costs 13 perplexity." Seed 4 supports the opposite at similar magnitude. A
single-seed ablation at this budget produces a confident number that does not
replicate — in whichever direction the seed happened to fall.

There is a second finding hiding in that spread. The AR arm is 2.8x more
variable across seeds than its twin without being better on average (variance
ratio 7.6 on 4 and 4 degrees of freedom, against a 95% critical value of
6.39 — it clears, narrowly). For a recovery technique that is arguably the
more practical question than the mean: a method whose outcome swings with the
seed cannot be trusted on a single run.

**This one carries a confound of my own making**, and it should not be quoted
without it. The gates are given a 10x effective learning rate, because AdamW
takes steps of roughly the learning rate regardless of gradient magnitude, so
24 fresh scalars starting at zero barely move inside a 300-step budget — the
AR arm would have been a no-op by construction rather than by evidence. That
choice is doing work in the instability result. At gain 1 the gates would
barely move and the variance would presumably match. The claim is therefore
narrower than "attention residuals destabilise training": at a gain large
enough for the pathway to be used at all in this budget, the arm becomes
markedly less reproducible without becoming better. The gain sweep that would
settle it is the obvious follow-up and I did not run it.

---

## What this is not

It is not a working 1-bit model. Here is the released `onebit` checkpoint,
greedy decoding, prompted with *"The capital of France is"*:

> The capital of France is the first 1990s . The first 1990s was the first
> time of the

Grammatical shape, no content, immediate collapse into repetition. That is
what 282 perplexity looks like from the inside, and it is the reason the
checkpoints ship as research artifacts — published so the 2x2 can be
re-scored without repeating the training, not because anyone should generate
with them.

It is not a compression result either. The file is 666 MB: 43 MB of binarised
projections and 623 MB of FP32 embedding table. Qwen1.5 has a 151,936-token
vocabulary and the embeddings are untouched, so `1.125 bits/weight` is true of
the projections and false of the model, whose average is 11.49. There is no
deployment-size win here and it would be easy, and wrong, to imply one.

And the mechanism is not novel. Compensating pathways around binarised layers
go back to 2018. What is offered is the measurement: a factorial isolating the
pathway, at a stated budget, against a measured noise floor.

## Limits, stated plainly

One model, one scale (0.5B) — extreme-quantization results are known to change
with scale, and BitNet b1.58's parity claim explicitly *begins* at 3B. 1.23M
recovery tokens, between 10^3.5 and 10^6.5 below the ternary-QAT
literature. One corpus, which is also the evaluation corpus, so no perplexity
here is out-of-distribution. Weight-only quantization; activations stay FP32.
Five seeds on the contested pair, one on each FP32 arm. No sweep of the gate
gain. PIQA implemented and unrun.

The interesting follow-up is not a bigger version of this. It is that **no
sub-2-bit result exists for diffusion language models at all** — quantization
of LLaDA/Dream-family models bottoms out at 2-bit post-training, and a recent
survey of efficient dLLM inference devotes one subsection to quantization
citing two papers, with no mention of 1-bit, ternary, or QAT. The damage modes
there are structurally different: iterative error amplification across
denoising steps, bimodal masked/unmasked activation distributions. Existing
compensation designs do not address them, and nothing in this post suggests
the accumulator would.

---

*Code, ledger, protocol and preprint:
[TriDi](https://github.com/ismaelfaro/TriDi). Every number traces to a row
in `results/ledger.jsonl` carrying its arm, seed, git commit and full training
configuration. Failed and diverged runs stay in the ledger.*
