# What this project is actually for

The thesis is **sub-2-bit weights for a discrete diffusion language model.**
Ternary, in the BitNet b1.58 sense, on a model that generates by iterative
denoising rather than left to right.

Nothing in the repository does that yet. This document says exactly how far
away it is, so the autoregressive work is not mistaken for the goal.

---

## Where the gap is

Two gaps, and they are not the same size.

| | built | thesis |
|---|---|---|
| quantization | 1-bit binary, `sign()`, {−1,+1}, absmax scale per group of 128 | **1.58-bit ternary**, {−1,0,+1} |
| architecture | autoregressive, causal mask, KV cache, next-token cross-entropy | **discrete diffusion**, bidirectional attention, iterative denoising |

Binary to ternary is an afternoon: a different quantizer, a packer that fits
five trits in a byte (3^5 = 243 ≤ 256, so 1.6 bits/weight against the current
1.0), the bit accounting, and the tests that pin them. The straight-through
estimator, the group machinery and the freeze/export path all survive.

Autoregressive to diffusion is not an edit. It is a second model: the causal
mask goes, the loss becomes masked-token prediction under a noise schedule
instead of shifted cross-entropy, generation becomes an iterative denoiser
instead of a KV-cached loop, and the evaluation harness does not transfer at
all — diffusion LMs report a likelihood bound, not autoregressive perplexity.

## Why the target is worth taking

Every published quantization result for diffusion LMs is post-training and
stops at 2 bits: Quant-dLLM (arXiv:2510.03274), DLLMQuant (arXiv:2508.14090),
the PTQ study in arXiv:2508.14896, and arXiv:2604.20079 at 2–4 bits with
GPTQ and a Hessian-aware variant. None does quantization-aware training.
Ternary post-training exists for autoregressive models — TWLA
(arXiv:2606.13054) — and not for diffusion.

Two things make the gap look worth closing rather than merely empty.

**The one comparative datapoint favours diffusion.** arXiv:2604.20079 finds
CoDA degrading *less* than Qwen3-1.7B at reduced precision on HumanEval and
MBPP. If diffusion LMs tolerate quantization better, the regime where they
break may sit below where anyone has looked.

**Diffusion LMs need weight compression more than autoregressive ones do.**
There is no KV cache to amortise: every denoising step is a full bidirectional
forward over the block. DiffusionGemma-26B-A4B runs up to 48 denoising steps
per block. Weight traffic, which dominates decode latency, is paid per step.

## The damage mode nobody has measured

This repository already measured that binarization makes a transformer
numerically chaotic: identical weights on two FP32 backends diverge from
~1e-5 to ~1e-2 relative once quantized, compounding monotonically with depth,
because `sign()` sits on a discontinuity and a one-ulp difference flips a
stored bit.

A diffusion LM runs that forward pass up to 48 times per block, feeding each
output back in as the next input. The open question is whether the same
perturbation compounds across *denoising steps* as well as across depth. If
it does, sub-2-bit diffusion has a failure mode with no autoregressive
analogue. If it does not, that is worth knowing too, and the frozen-checkpoint
measurement in this repository is the precedent: the FP16 scale rounding we
expected to compound with depth turned out not to, because it perturbs
magnitudes without flipping any sign bit.

## Scale, and what this machine can hold

The field is almost entirely ≥7B: LLaDA-8B, Dream-7B, W1-4B, LLaDA2.0 to
100B. DiffusionGemma-26B-A4B is 25.2B total and 3.8B active across 128
experts, 30 layers, 262K vocab, bidirectional decoder, block-autoregressive
sampling, BF16, Apache 2.0.

**None of these can be QAT'd here.** 25.2B in BF16 is ~50 GB of weights
before optimizer state, and QAT needs FP32 masters plus gradients plus Adam
moments over every expert, not merely the active ones. This machine has
32 GB.

So DiffusionGemma is the model the thesis names as its target, not the model
this repository trains — the same relationship BitNet b1.58 has with its own
claim, which explicitly begins at 3B.

The route that fits: **adapt the validated Qwen1.5-0.5B into a masked
diffusion model.** That is how Dream was made from Qwen2.5-7B and TESS-2 from
Mistral, so the recipe is established rather than invented here, and it
reuses the one implementation in this repository that is pinned against
HuggingFace.

## Order of work

1. **Diffusion port of the current architecture.** Bidirectional attention, a
   mask token, the masked-diffusion objective, an iterative sampler, and a
   likelihood-bound evaluator. PyTorch reference first, MLX port pinned
   against it, exactly as the autoregressive side is.
2. **Ternary — done.** `quant_scheme="q1_58"`: {−1, 0, +1} with an absmean
   group scale, five trits to a byte, **1.725 bits/weight** against Q1_0's
   1.125 and an information-theoretic floor of log2(3) = 1.585.

   The scale statistic is the part that does not survive being guessed. Q1_0
   divides by the group maximum, which is right for `sign()`. Reusing a
   maximum with ternary levels rounds every weight below half the group's
   largest to zero — 85% of a Gaussian matrix, at 0.83 relative
   reconstruction error, against 35/31/34 and 0.44 for absmean. Both versions
   train and produce a falling loss. The scheme therefore carries its scale
   rather than exposing it, and the split is pinned in a test.

   Two things the tests exist for: the freeze reduction must match the
   training forward's (an absmax freeze against an absmean forward gives a
   model that trains normally and collapses on export), and the scheme has to
   survive a checkpoint round trip (`scheme` is a plain attribute, so a
   ternary checkpoint loaded into a binary layer decodes base-3 bytes as bit
   fields — it loads clean and every weight is wrong).
3. **The grid — complete, three paired seeds, and it resolves.** Ternary
   costs the diffusion model **25.7%** of its headroom below the uniform
   floor against **14.6%** for the autoregressive model: an interaction of
   **+0.111 in headroom share, 56x its paired SE**, sign stable across every
   seed (per-seed +0.108, +0.115, +0.111). The same protocol that returned
   Part I's null at 1.2x SE returns this at 56x. Raw nats: +2.071 NELBO on
   diffusion, +1.344 NLL autoregressive — not commensurable, ledgered.
   Full table and the mechanisms the design does not separate:
   `paper/preprint.md` §8.

**The check that came before all of it — passed.** An AR-pretrained model
adapted to diffusion on 1.23M tokens might have landed at the uniform floor,
and quantization damage cannot be measured on top of a model that has already
floored; this repository has that failure once already, in accuracy
benchmarks that could not separate two 1-bit arms because both sat at chance.
So the FP32 diffusion arm ran alone first.

| | NELBO | mask accuracy |
|---|---|---|
| uniform floor, `log(151936)` | 11.9312 | — |
| Qwen1.5-0.5B, no adaptation | 10.5390 | 0.0081 |
| after 1.23M tokens | **3.8761** | **0.2747** |

The adaptation moves the bound 6.66 nats and ends 8.06 below the floor. Mask
accuracy goes from under 1% to 27%: the model names better than a quarter of
the tokens it cannot see. Whatever quantization does to this, there is room
to measure it. 48 validation blocks of 512 tokens, 4 corruptions each, fixed
evaluation seed. 1273 s on an M5.

Two things not to read into that number. **3.876 nats is not comparable to
the autoregressive perplexity** anywhere else in this repository — masked
prediction with bidirectional context is a different and easier task than
next-token prediction, so a lower bound here does not mean a better model.
And the training loss stays visibly noisy to the end (last-ten mean 4.567,
individual steps between 3.19 and 5.52), which is the `1/t` weight doing what
it is expected to do rather than instability.

## Gating discipline: cheap signal before expensive confirmation

The order that saves weeks, stated as policy because it was once followed by
accident and once by design:

1. **Mechanism check first** (does the knob move at all?). Part I logged the
   gate trajectory inside the sweep rather than running one arm first — had
   alpha stayed at zero, the full 2x2 would have been spent discovering a
   no-op. That it moved was luck of `ALPHA_GAIN`, not correct ordering. Do
   not repeat this: one arm, one seed, read the mechanism, then decide.
2. **The contested pair at one seed** (is there a signal worth paying for?).
   Part II did this right: the gating check (fp32 diffusion alone, ~1/12 of
   the grid's cost) would have killed the experiment cheaply if the
   adaptation had floored; grid seed 0 then showed +0.108 before seeds 1–2
   were bought.
3. **Full seeds only after signal.** Three seeds bought a 56x-SE verdict
   precisely because the shape was already visible at one.

**Applied to the budget ladder, decision rule fixed in advance:** the 19.7M
rung (~21 h) and any additional ladder seeds are gated on the 4.92M rung at
seed 0. If the interaction's headroom share at 4.92M stays within ±0.03 of
the 1.23M value (+0.111), the effect is treated as budget-stable and the
19.7M rung runs; if it shrinks by more than that, the interesting question
becomes the decay curve and the 19.7M rung runs *with priority*; only if
the 4.92M interaction collapses below +0.02 — smaller than Part I could
even resolve — is the ladder stopped as answered.

## Status of the autoregressive work

**Frozen, unpublished.** The 2x2 is complete and its result is a null: no
detectable interaction between a cross-layer attention residual and 1-bit
quantization, with the gates converging negative in FP32 as well. Two
checkpoints and their cards exist locally and are not being uploaded; the
preprint and the post are not being published.

It stays in the repository as what validates the quantizer, the export path,
and the measurement discipline the diffusion work inherits. It is prior work
for this project, not a result this project is putting its name to.

## Future improvements: unexploited MLX headroom

Audited while rung 3 ran. The execution already uses what matters —
`mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention`, the
per-micro-batch `mx.eval` that keeps grad-accum graphs out of swap, GPU
device confirmed live, 1.72x over PyTorch/MPS measured. Two real gains are
deliberately left on the table:

- **`mx.compile` on the training step** — typically 20–40% on MLX from
  kernel fusion. Not used anywhere.
- **`mx.fast.rope`** — RoPE is hand-rolled (outer + concat + rotate_half).

Both change floating-point accumulation order, and three of this project's
results lean on bitwise reproducibility (zero ledger drift across scripts,
identical replicates, `fp32_ar` reproduced from three entry points).
Flipping either mid-ladder would put old and new rungs in different numeric
regimes — the silent-confound class this repository exists to catch.

**Adoption plan, in order:**

1. After rung 3 completes (GPU free — two concurrent GPU jobs starve on
   this machine, measured), microbenchmark both on the real QAT step:
   batch 2 x seq 512, ternary and FP32 arms, fwd+bwd wall time.
2. If the combined win exceeds ~20%, adopt for *future* experiments only,
   declaring a **reproducibility epoch break** in the ledger: a `numerics`
   field on new rows, old and new numbers comparable through paired seeds,
   never bitwise.
3. Re-pin MLX/PyTorch parity under compile before any new experiment trusts
   it (`tests/test_mlx_parity.py` thresholds re-measured, not assumed).

Not adopted, with reasons already on record: batch 4 x accum 2 (same
tokens, different RNG consumption — breaks seed pairing, see COLAB.md);
bf16 masters (interacts with STE quantization thresholds; a measurement of
its own, not a flag flip).

## What does not carry over

The attention-residual ablation. That question is answered, it is null, and
nothing about the diffusion target depends on it.
