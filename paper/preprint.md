# Does a cross-layer attention residual reduce binarization damage? A factorial measurement at 0.5B

**Draft preprint — numbers marked `[TBD]` are filled from `results/ledger.jsonl` by `report.py`.**

---

## Abstract

Adding a compensating pathway around a binarised layer is a recurring design
in extreme quantization, from Bi-Real Net's real-valued shortcuts to OneBit's
rank-1 value vectors. The pattern is well established; what is rarely
measured is how much of any observed gain is attributable to the pathway
*interacting with* binarization, rather than to the pathway being generically
useful.

We run the missing factorial. On Qwen1.5-0.5B-Chat we cross
{FP32, 1-bit Q1_0_g128} with {no attention residual, attention residual} at a
matched recovery budget of 1.23M tokens, with paired seeds and a measured
noise floor, and report the interaction term
`(1bit_AR - 1bit) - (fp32_AR - fp32)`.

We find `[TBD]`. We also report two methodological results that hold
regardless of the ablation's outcome: binarised transformers are numerically
chaotic, with cross-backend disagreement rising from ~1e-5 to ~1e-2 and
compounding monotonically with depth, which makes post-training 1-bit
perplexity implementation-dependent; and at this budget, log-likelihood
accuracy benchmarks floor at chance while KL-to-teacher continues to resolve,
so accuracy-only tables cannot distinguish recovery methods in the
low-budget regime.

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
gate, up, down) are binarised: 302M of 464M parameters. Embeddings and the
tied readout remain FP32, following the BitNet family. The model-wide
average is therefore ~11.8 bits/weight, not 1.125, and we report both.

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
Comparing a fine-tuned 1-bit model against an un-fine-tuned FP32 model
charges quantization for a domain-adaptation gap.

**Budget.** 300 optimizer steps x 4096 tokens = 1.23M tokens of
wikitext-2-raw train, identical order across arms. Fixed tokens rather than
fixed wall clock: wall-clock budgeting converts thermal throttling on a
laptop into treatment variance. AdamW, peak LR 1e-4, cosine decay, 5% warmup,
grad clip 1.0, batch 2 x 4 accumulation, sequence length 512.

**Frozen embeddings in every arm.** The tied embedding/readout is 155M of
464M parameters; leaving it trainable lets a ~1M-token run rewrite the output
head. A shared frozen readout keeps the comparison on the blocks.

**Paired seeds.** Seeds {0,1,2}, identical across arms; the statistic is the
per-seed difference, which cancels shared init and data-order variance.

**Noise floor, measured before the effect.** Two sources are separated: a
same-seed rerun isolates backend nondeterminism, and varying the seed adds
init and data order. The decision rule is fixed in advance: if
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

This is the metric that carries the ablation. A 4-way task floors at 25%: it
cannot distinguish a damaged model from a destroyed one. In the low-budget
recovery regime every 1-bit arm may sit at chance on accuracy while differing
substantially in distributional fidelity. KL has no floor. We suggest it
should be standard in extreme-quantization tables and note it is almost never
reported.

---

## 5. Results

### 5.1 Main table

`[TBD — generated by report.py]`

### 5.2 The interaction term

`[TBD]`

### 5.3 Gate trajectories and inter-path adaptation

`[TBD — alpha mean/max per layer over training; the RaBiT prediction is
growth followed by collapse]`

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
seed are stated. We observed two same-seed three-step runs differing by 7% in
perplexity from GPU reduction nondeterminism alone.

---

## 6. Limitations

- **One model, one scale (0.5B).** Extreme-quantization results are known to
  change with scale; BitNet b1.58's parity claim explicitly begins at 3B.
- **1.23M recovery tokens**, three to six orders of magnitude below the
  ternary-QAT literature (BitNet b1.58 2B4T: 4T tokens). This is a
  low-budget-recovery result and is labelled as one throughout.
- **One recovery corpus** (wikitext-2), which is narrow.
- **Weight-only quantization**; activations remain FP32. Activation outliers
  are the documented blocker addressed by BitNet a4.8 and BitNet v2.
- **Three seeds.** Enough to establish a noise floor, not enough for a tight
  interval on a small effect.
- **Throughput numbers are machine-local** to Apple Silicon and MLX.

---

## 7. Related work

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
python run_ablation.py --stage full --steps 300 --seeds 0 1 2
python report.py
```

Every number in the results table traces to a row in
`results/ledger.jsonl` carrying its arm, seed, git commit and full training
configuration. Failed and diverged runs are retained in the ledger.
