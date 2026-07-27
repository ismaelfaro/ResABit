# ResABit

**Does a cross-layer attention residual reduce the damage that 1-bit weight
quantization does to a pretrained transformer?**

A controlled 2x2 ablation on [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat),
run on a single Apple Silicon machine, combining:

- **Q1_0_g128** — one sign bit per weight plus an FP16 scale per group of
  128, following the 1-bit Bonsai line of work.
- **Attention Residuals** (arXiv 2603.15031) — every layer accumulates the
  sum of all prior attention outputs and re-injects it through a learnable
  per-layer gate.

**Answer: no measurable interaction.** The residual does not preferentially
repair binarization damage; the gates converge *negative*, and they do so in
FP32 too. Binarizing 308M of 464M parameters costs 2.97 nats — a 19.5x
perplexity increase — at this recovery budget.

Status: **2x2 complete**, 15 runs in `results/ledger.jsonl`, five paired
seeds on the contested pair. Every number below is generated from that ledger
by `report.py` and traces to a row carrying its arm, seed and git commit.
Open items are listed as TODOs in [docs/PROTOCOL.md](docs/PROTOCOL.md) §8
rather than left implied.

---

## The claim, and what it is not

The interesting quantity is not "1-bit is worse than FP32" (it is) or "AR
lowers perplexity" (weaker than what the literature already shows). It is the
**interaction**:

```
interaction = (1bit_AR - 1bit) - (fp32_AR - fp32)
```

Does the residual pathway buy *more* under binarization than it buys in
general? That needs all four cells, which is why the original three-column
plan became a 2x2.

**The mechanism is not novel.** Compensating pathways around binarised layers
go back to Bi-Real Net (2018) and run through ReActNet, MeliusNet, OneBit,
BiMaCoSR and HGF. In particular
[RaBiT](https://arxiv.org/abs/2602.05367) (ICML 2026) shows that parallel
residual paths co-adapt into redundancy under QAT — which is a direct
prediction about this architecture, and which the logged alpha trajectory
either confirms or contradicts. What is defensible here is the *measurement*:
a factorial isolating the pathway, at a stated budget, against a measured
noise floor. See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the full argument.

---

## Results

<!-- RESULTS-TABLE-START -->
| metric | FP32 +FT | FP32 +FT +AR | 1-bit QAT | 1-bit QAT +AR |
|---|---|---|---|---|
| wikitext-2 ppl (strided) | 14.653 | 14.827 | 285.669 ± 3.165 | 283.098 ± 8.749 |
| wikitext-2 NLL | 2.6846 | 2.6965 | 5.6548 ± 0.0110 | 5.6454 ± 0.0307 |
| wikitext-2 top-1 acc | 0.4739 | 0.4717 | 0.2095 ± 0.0010 | 0.2104 ± 0.0030 |
| final train loss | 2.8455 | 2.8653 | 5.7300 ± 0.0317 | 5.7229 ± 0.0432 |
| KL to shipped Qwen (nats) | 1.5953 | 1.5952 | 3.9582 | 3.9792 |
| top-1 agreement with shipped Qwen | 0.6412 | 0.6432 | 0.2774 | 0.2715 |
| ARC-Easy acc (±0.0096 SE) | 0.5417 | 0.5429 | 0.2622 | 0.2639 |
| HellaSwag acc_norm (±0.0146 SE) | 0.4560 | 0.4440 | 0.2590 | 0.2630 |
| LAMBADA acc (±0.0105 SE) | 0.3360 | 0.3180 | 0.0000 | 0.0000 |
| bits/weight (quantised params) | 32 | 32 | 1.125 | 1.125 |
| learned alpha, mean / max (layers 1+) | — | -0.0141 / 0.0345 | — | -0.0522 / 0.0988 |
| seeds | 1 | 1 | 5 | 5 |
| wall-clock per run (s) | 1746 | 1741 | 2299 ± 1273 | 1844 ± 289 |

### Noise floor (identical reruns)

3 identical reruns of `onebit` at seed 0 (same config, same data order, same code):

- perplexity: [282.208, 282.208, 282.208] — mean 282.208, sd 0.000, range 0.000
- NLL: mean 5.6426, sd 0.0000

**The pipeline is bitwise reproducible: this floor is exactly zero, so every bit of seed-to-seed spread is genuine seed effect.**

### Paired comparison

`onebit_ar` minus `onebit`, paired over seeds [0, 1, 2, 3, 4]. Decision rule fixed in advance: an effect smaller than 2 SE is reported as no effect.

| metric | per-seed deltas | mean | SE | verdict |
|---|---|---|---|---|
| wikitext ppl | [13.117, 5.408, -10.833, -6.15, -14.398] | -2.571 | 5.151 | within noise |
| wikitext NLL | [0.0454, 0.0189, -0.0386, -0.0218, -0.0508] | -0.0094 | 0.0181 | within noise |
| final train loss | [0.0558, 0.0173, -0.0394, -0.0193, -0.05] | -0.0071 | 0.0195 | within noise |
| top-1 acc | [-0.0037, -0.0005, 0.0031, 0.0016, 0.0042] | +0.0009 | 0.0014 | within noise |

### Run-to-run stability

| arm | perplexities | mean | sd |
|---|---|---|---|
| `onebit` | [282.208, 283.768, 286.369, 285.441, 290.561] | 285.669 | 3.165 |
| `onebit_ar` | [295.324, 289.176, 275.536, 279.291, 276.164] | 283.098 | 8.749 |

**`onebit_ar` is 2.8x more variable across seeds than `onebit` while not being better on average.** Variance ratio 7.6 on 4 and 4 degrees of freedom. The 95% critical value of F(4,4) is 6.39, so this clears it. A spread estimated from 5 runs is itself unstable, so treat the ratio as an order of magnitude, not a measurement.

### Derived quantities

All quantities in nats of NLL; perplexity ratios in parentheses.

- Quantization cost (no AR): **+2.9702** nats (19.5x perplexity)
- AR cost at FP32: **+0.0118** nats (1.012x)
- AR cost at 1-bit: **-0.0094** nats (0.991x)
- Interaction: -0.0212 nats — **not distinguishable from zero**

The paired standard error on the 1-bit AR term is 0.0181 nats; the interaction is 1.2x that, inside the 2 SE rule fixed in advance. The FP32 arms contribute one seed each, so their variance is absent from this estimate and the true standard error on the interaction is larger than the one quoted. **No evidence that the attention residual preferentially repairs binarization damage; the measurement does not have the resolution to rule out an effect of this size either.**

### Gate trajectories

- `onebit_ar`: mean |alpha| 0.00003 at start, 0.05323 peak (step 294), 0.05320 final. **gates rose and held** — no sign of inter-path collapse
- `fp32_ar`: mean |alpha| 0.00003 at start, 0.01715 peak (step 160), 0.01536 final. **gates rose and held** — no sign of inter-path collapse

### Gate profile by depth

- `onebit_ar` (5 seed(s)): early -0.0637, middle -0.0691, late -0.0252. 22/23 live gates are negative — the pathway is being used to damp the residual stream, not to enrich it.
- `fp32_ar` (1 seed(s)): early -0.0186, middle -0.0150, late -0.0091. 22/23 live gates are negative — the pathway is being used to damp the residual stream, not to enrich it.
<!-- RESULTS-TABLE-END -->

---

## What is quantized, precisely

Quoting "1.125 bits/weight" for the whole model would be false, so both
numbers are reported:

| | parameters | bits/weight |
|---|---|---|
| Block projections (q, k, v, o, gate, up, down) | 308.3M | **1.125** |
| Embeddings + tied readout, norms, attention biases | 155.7M | 32 |
| **Model average** | **464.0M** | **11.49** |

Embeddings and the readout stay full precision, standard for the BitNet
family. The compression story is about the projections; the model-wide
average is what an honest checkpoint-size claim rests on.

These come from `ModelConfig.num_quantized_params` and
`effective_bits_per_weight`, and they are confirmed by what an exported
checkpoint actually weighs: **666 MB, of which 43 MB is the binarised
projections and 623 MB is the FP32 embedding table.** An earlier revision of
this table was hand-typed as 302M / 162M / ~11.8 and had drifted from the
code — the kind of small dishonesty this section exists to prevent.

---

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify the reimplementation before trusting anything it produces — this
checks our FP32 model against HuggingFace Qwen, and the MLX port against the
PyTorch reference:

```bash
python -m pytest tests/ -v
```

Run the ablation:

```bash
python run_ablation.py --stage noise-floor --steps 300 --seeds 0 1 2
```

```bash
python run_ablation.py --stage full --steps 300 --seeds 0 1 2
```

Regenerate the results table from the ledger:

```bash
python report.py
```

---

## Checkpoints

The sweep evaluated each arm inside the training process and threw the model
away, so none of the reported arms existed on disk. Rebuild one:

```bash
python export_checkpoint.py --arm onebit --seed 0
```

The rebuild is not an approximation of the run behind the table — the
pipeline is bitwise reproducible, and the export aborts if it misses the
ledger's perplexity rather than shipping a checkpoint no table describes.

**A frozen checkpoint does not compute the ledger's number.** The ledger
records the training forward: FP32 master weights pushed through
`fake_quantize` on every call. Freezing stores the group scales as FP16,
which moves each layer by ~2e-4 relative. The export measures both and the
model card quotes the frozen one.

Measured, it costs **+8e-6 nats** — 282.2098 against 282.2077. We expected it
to compound with depth and it does not, because FP16 rounding perturbs group
magnitudes without flipping any sign bit, so it never crosses the
discontinuity that makes binarised stacks diverge. Worth measuring anyway:
the two paths are different computations and that number had never been
checked.

Score a checkpoint on the held-out suite without retraining:

```bash
python eval_checkpoint.py checkpoints/resabit-onebit-seed0
```

Watch a run in flight — step progress, loss curve, ETA, and the frozen-path
numbers as each arm lands:

```bash
python dashboard.py --watch
```

It reads the log rather than instrumenting the runs, and labels anything it
infers rather than presenting it as measured. If a run's log has gone silent
it says so instead of extrapolating a healthy-looking bar.

### Publishing one

The card is generated from the checkpoint's own manifest, for the same reason
the results table is generated from the ledger — bits per weight and the
perplexity the file actually computes are exactly the two numbers a human
would retype wrong.

```bash
python make_model_card.py checkpoints/resabit-onebit-seed0
```

Upload is a dry run by default and refuses to publish a checkpoint whose card
does not quote its own frozen perplexity:

```bash
python upload_to_hf.py checkpoints/resabit-onebit-seed0 --repo <user>/resabit-1bit
```

The released weights are research artifacts, and the card leads with that.
Greedy decoding from the `onebit` checkpoint, prompted with *"The capital of
France is"*:

> The capital of France is the first 1990s . The first 1990s was the first
> time of the

Grammatical shape, no content. That is 282 perplexity from the inside.

---

## Layout

```
src/
  config.py            Model config; the two ablation switches; bit accounting
  quantization.py      Q1_0_g128 — STE training, bit packing, grouped INT8 GEMM
  model.py             PyTorch reference model (validated against HuggingFace)
  loader.py            Strict HF weight loading — refuses a partial map
  data.py              wikitext streams, strided eval windows, zero-shot tasks
  evaluate.py          Perplexity, zero-shot accuracy, KL-to-teacher
  mlx_backend/
    model.py           MLX port (1.72x faster training on Apple Silicon)
    train.py           QAT loop

run_ablation.py        The 2x2 sweep: paired seeds, JSONL ledger
report.py              Ledger -> results table
export_checkpoint.py   Retrain one arm -> frozen, publishable checkpoint
eval_checkpoint.py     Score a checkpoint without retraining it
make_model_card.py     Checkpoint manifest -> model card
upload_to_hf.py        Preflight + push to the Hub (dry run by default)
dashboard.py           Live progress view over the run log
convert.py             HF weights -> frozen 1-bit checkpoint
inference.py           Generation from a checkpoint
tests/                 Parity against HuggingFace and across backends
docs/PROTOCOL.md       Why the experiment is shaped this way
paper/preprint.md      The write-up
paper/post.md          The same result for a general technical audience
```

---

## Two backends, on purpose

Training runs on MLX; **every reported metric is computed in PyTorch.**

MLX is 1.72x faster on the real QAT step (1140 ms vs 1959 ms per
fake-quantised fwd+bwd at batch 2 x seq 512 on an M5). But MLX's Metal
matmul is not true FP32 — measured against float64 on a 1024-wide matmul it
lands 0.098 away, versus 0.00013 for PyTorch, roughly bf16-grade
accumulation. That is negligible next to the perturbation binarization
applies, and decisive next to an FP32 baseline perplexity.

So: MLX optimises, PyTorch measures, and `tests/test_mlx_parity.py` keeps
them pinned together.

---

## A finding that fell out of the plumbing

Binarising the projections makes the network **numerically chaotic**. The
same weights on two FP32 backends diverge by ~1e-2 relative once quantised,
against ~1e-5 unquantised, and the gap grows monotonically with depth
(3.5e-3 at layer 0, 3.2e-1 at layer 23). `sign()` sits on a discontinuity, so
a one-ulp disagreement flips a weight and the error compounds.

Two consequences. Post-training 1-bit does not just degrade, it destabilises
— which is why QAT is not optional. And the PTQ row of any 1-bit results
table is implementation-sensitive in a way the QAT rows are not, so a PTQ
number quoted without a backend and a seed is not reproducible.

---

## References

- **BitNet b1.58** — Ma et al., [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet b1.58 2B4T** — [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)
- **RaBiT** — [arXiv:2602.05367](https://arxiv.org/abs/2602.05367), ICML 2026
- **OneBit** — [arXiv:2402.11295](https://arxiv.org/abs/2402.11295), NeurIPS 2024
- **ReActNet** — [arXiv:2003.03488](https://arxiv.org/abs/2003.03488), ECCV 2020
- **Tequila** — [arXiv:2509.23809](https://arxiv.org/abs/2509.23809), ICLR 2026
- **CAT-Q** — [arXiv:2606.26650](https://arxiv.org/abs/2606.26650), ICML 2026
- **Attention Residuals** — arXiv 2603.15031
- **Qwen1.5** — Qwen Team

Experiment discipline (paired seeds, measured noise floor, failures as
ledger rows, frozen eval harness) is adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the
`rigor.py` addition in its MLX fork.

## License

Apache 2.0 — see [LICENSE](LICENSE).
