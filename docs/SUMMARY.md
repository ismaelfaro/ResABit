# ResABit — project summary and final results

One repository, two completed factorial experiments, one thesis answered.
Every number below traces to a ledger row (`results/ledger.jsonl`,
`results/diffusion_ledger.jsonl`, `results/grid_ledger.jsonl`) carrying its
arm or cell, seed, git commit and full training configuration. 50 commits,
83 tests, single Apple M5, base model Qwen1.5-0.5B-Chat throughout.

---

## The headline result

**Ternary (1.725 bits/weight stored) quantization damages a masked discrete
diffusion language model 1.77x more than it damages the same architecture
trained autoregressively, at matched budget, matched data order, and paired
seeds.**

| | share of headroom destroyed, per seed | mean | sd |
|---|---|---|---|
| diffusion | 0.2607, 0.2558, 0.2541 | **25.7%** | 0.0034 |
| autoregressive | 0.1523, 0.1405, 0.1436 | **14.6%** | 0.0061 |

Paired per-seed interaction +0.1084, +0.1153, +0.1105 — mean **+0.1114**,
paired SE **0.0020**, i.e. **56x the standard error**, sign stable across
every seed. Raw nats (not commensurable across metrics, ledgered): +2.071
NELBO on diffusion, +1.344 NLL autoregressive.

Nothing published is comparable: every existing quantization result for
diffusion LMs is post-training and stops at 2 bits; sub-2-bit QAT existed
only for autoregressive models. The one comparative PTQ datapoint
(arXiv:2604.20079, 2–4 bits) found diffusion *more* robust — the opposite
ordering — so the regime boundary between PTQ-above-2-bits and
QAT-below-2-bits is the natural next measurement.

**Why the comparison is legitimate.** Autoregressive arms report next-token
NLL; diffusion arms report a sampled NELBO bound on an easier task. The
levels are never compared. What is compared is the fraction of headroom below
a *shared* floor — `log(151936) = 11.9312` nats, which both metrics assign to
a model that has learned nothing — destroyed by quantization. That fraction
is dimensionless and means the same thing in both regimes.

**What it does not say.** Not that diffusion models cannot be quantized (26%
of headroom lost is damaged, not floored). Not that diffusion is worse than
autoregressive — quality across the two is never compared. Not that this
survives scale: one model, 0.5B, 1.23M recovery tokens. And the design
identifies *that* the diffusion objective loses more, not *why* — three
candidate mechanisms (bidirectional reuse under a mixed clean/masked input
distribution; the `1/t` weight concentrating gradient on heavily corrupted
batches; AR initialisation being further from a good diffusion solution) are
stated and deliberately not adjudicated.

---

## The grid behind it

Four cells, three paired seeds, twelve runs, zero crashes:

| cell | metric | loss (nats), mean | headroom |
|---|---|---|---|
| fp32_diff | NELBO | 3.8702 | +8.0610 |
| ternary_diff | NELBO | 5.9408 | +5.9904 |
| fp32_ar | NLL | 2.6909 | +9.2403 |
| ternary_ar | NLL | 4.0350 | +7.8962 |

Supporting results:

- **The gating check.** Adapting AR-pretrained Qwen1.5-0.5B to masked
  diffusion on 1.23M tokens moves NELBO 10.539 → 3.876 (8.06 nats below the
  uniform floor), mask accuracy 0.008 → 0.275. Run *before* the grid because
  quantization damage is unmeasurable on a floored model. The unadapted model
  already sits 1.39 nats below the floor — bidirectional context is worth
  that much for free.
- **QAT is what makes ternary diffusion exist at all.** Without recovery the
  ternary diffusion cell lands *below* the uniform floor (headroom −1.14);
  300 steps bring it to +5.99.
- **Ternary vs binary**, same AR architecture and budget: NLL 4.035 vs
  5.655. The 0.6 extra stored bits buy back 1.62 nats.

## The budget ladder (seed 0, one seed per rung)

| rung | tokens | diffusion share | AR share | interaction |
|---|---|---|---|---|
| 1 | 1.23M (~0.5 epoch) | 0.2607 | 0.1523 | +0.108 |
| 2 | 4.92M (~2 epochs) | 0.1630 | 0.0584 | +0.105 |
| 3 | 19.7M (~8 epochs) | 0.0884 | −0.0697 ⚠ | not computable |

Rungs 1→2: the interaction is **budget-stable** (inside the ±0.03 rule fixed
in advance) while both absolute costs fall — recovery does not close the
architecture gap, and as a ratio it widens (1.7x → 2.8x).

Rung 3 **breaks the instrument on the AR side, and the break is the
finding**: 8 epochs of a 2.5M-token corpus and the FP32 AR arm memorises it
(train loss 0.0302, validation NLL 4.89 — worse than at 1x budget). The
ternary twin cannot memorise as hard and generalises better, flipping the
"cost" negative — quantization-as-regularisation, reported as a corpus-
exhaustion artifact, not a deployable claim. The diffusion side survives
untouched (train ≈ eval both arms): random masking is data augmentation.
Extending the ladder needs a bigger recovery corpus before more compute.

## Part I — the prior experiment (frozen, unpublished)

A 2x2 crossing {FP32, 1-bit binary} with a cross-layer attention residual
(arXiv 2603.15031), five paired seeds. Result: **null**. Interaction −0.0212
nats, 1.2x its paired SE, inside the 2 SE rule fixed in advance. The residual
does not preferentially repair binarization damage. The sharper finding: 22
of 23 live gates converge *negative* — the model learns to subtract the
accumulated residual — and the FP32 arm does the same, so the suppression is
architectural, not a response to quantization. Binarizing 308.3M of 464.0M
parameters costs 2.97 nats (19.5x perplexity) at this budget.

Per the decision to freeze: checkpoints and model cards exist locally, weights
were not uploaded, preprint Part I and `paper/post.md` were not published.

**The two-part contrast is itself a validation.** Same protocol, same
pipeline: a null at 1.2x SE and an effect at 56x. The instrument demonstrably
reports what it cannot detect.

## Methodological findings that stand on their own

1. **Binarised transformers are numerically chaotic across implementations.**
   Identical weights on two FP32 backends diverge ~1e-5 → ~1e-2 relative once
   binarised, compounding monotonically with depth. Any PTQ-to-1-bit number
   is implementation-dependent unless backend and seed are stated.
2. **But the chaos needs the discontinuity.** FP16 group-scale rounding —
   predicted to compound — costs 8 *micro*nats end to end (282.2077 →
   282.2098), because it moves magnitudes without flipping a sign bit.
   Perturbations that respect `sign()` are benign; ones that cross it are
   not. (Prediction was wrong; measurement replaced it.)
3. **The pipeline is bitwise reproducible.** Three identical reruns: 282.208
   to the digit. Same numbers reproduced from three different scripts. The
   7% nondeterminism initially reported was a config bug, retracted.
4. **Accuracy benchmarks floor; divergence metrics don't.** Two 1-bit arms
   were indistinguishable on ARC-Easy (both at chance, LAMBADA exactly
   0.000) while KL-to-teacher still resolved. Accuracy tables can size a
   quantization gap but cannot compare recovery methods.
5. **Cross-backend agreement depends on corruption density, non-monotonically**
   (5.9e-6 clean → 5.9e-4 at 10/12 masked → 5.1e-6 fully masked): a fully
   masked sequence has no near-ties to break. Narrows the two-backend margin
   from ~3 orders of magnitude to ~1.5; still sound.
6. **The ternary scale statistic is part of the scheme.** Absmax (right for
   binary) with ternary rounding zeroes 85% of a Gaussian matrix at 0.83
   relative error; absmean splits 35/31/34 at 0.44. Both train; only one
   represents the matrix.

## Bugs caught by the discipline (selection)

- Six original correctness bugs, including per-group scales averaged to
  per-row (the `opt_ppl: 8246580`), an int8 shift table overflowing on bit 7,
  and `strict=False` silently dropping 72 bias tensors.
- The results table pooled determinism replicates as seeds (`seeds | 8` for a
  five-seed arm); every derived quantity inherited it.
- The "strict" loader accepted a quantized checkpoint with no packed bits
  (None buffers escape PyTorch's missing-key accounting).
- The diffusion sampler emitted `[MASK]` as an output token.
- PIQA sat in the task registry, never scored, loader broken by datasets 4.x
  (still unrun; documented TODO in PROTOCOL §8).
- A hand-typed parameter table drifted from the code (302M/162M/~11.8 vs the
  computed 308.3M/155.7M/11.49).
- A single-seed interaction would have reported "AR costs 13 perplexity";
  seed 4 supported the opposite. Part I's per-seed deltas changed sign three
  times.

## Artifacts

| | |
|---|---|
| `paper/preprint.md` | The whitepaper: Part I (AR ablation) + Part II (ternary diffusion factorial) |
| `paper/post.md` | Part I as a technical post (unpublished, per freeze) |
| `docs/PROTOCOL.md` | Part I design, prior art, confounds, TODOs |
| `docs/ROADMAP.md` | Thesis, gap analysis, order of work, statuses |
| `results/*.jsonl` | Three ledgers; smoke runs tagged so nothing pools them |
| `checkpoints/` | Two 1-bit AR checkpoints + generated model cards (local only) |
| `run_grid.py` / `run_diffusion_check.py` / `run_ablation.py` | The experiments |
| `export_checkpoint.py` / `eval_checkpoint.py` / `make_model_card.py` / `upload_to_hf.py` | Publication chain (dry-run by default, preflighted) |
| `dashboard.py` | Live run monitor; measures vs infers, labels which is which |

Validation chain: PyTorch model = HuggingFace Qwen to 1e-3 on logits; MLX
port = PyTorch to 1e-5 on CPU; MLX trains (1.72x faster), PyTorch measures.
83 tests.

## Open

- Seeds beyond three on the grid (magnitude interval; sign is settled).
- Mechanism separation for the 1.77x (§8.2 candidates).
- Cross-denoising-step error compounding: posed, instrumented, unmeasured —
  a damage mode with no autoregressive analogue.
- The PTQ/QAT regime boundary against arXiv:2604.20079's opposite ordering.
- PIQA. Gain sweep on `ALPHA_GAIN`. Second-distribution held-out corpus.
- MLX headroom: `mx.compile` (~20–40% typical) and `mx.fast.rope`, both
  deferred because they change accumulation order and would break bitwise
  comparability mid-experiment. Benchmark and epoch-break plan in
  ROADMAP §Future improvements.
- Publication of any of it — everything is frozen-local until you say
  otherwise.
