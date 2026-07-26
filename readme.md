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

Status: **experiment running.** Results land in `results/ledger.jsonl`; the
table below is filled from it and is empty until it is. There is no
"recoverable" placeholder in this README where a number belongs.

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
| wikitext-2 ppl (strided) | 14.653 | 14.827 | 284.115 ± 2.102 | 286.679 ± 10.127 |
| wikitext-2 NLL | 2.6846 | 2.6965 | 5.6494 ± 0.0074 | 5.6579 ± 0.0356 |
| wikitext-2 top-1 acc | 0.4739 | 0.4717 | 0.2091 ± 0.0011 | 0.2087 ± 0.0025 |
| final train loss | 2.8455 | 2.8653 | 5.7349 ± 0.0243 | 5.7461 ± 0.0356 |
| KL to shipped Qwen (nats) | 1.5953 | 1.5952 | 3.9582 | 3.9792 |
| top-1 agreement with shipped Qwen | 0.6412 | 0.6432 | 0.2774 | 0.2715 |
| ARC-Easy acc (±0.0096 SE) | 0.5417 | 0.5429 | 0.2622 | 0.2639 |
| HellaSwag acc_norm (±0.0146 SE) | 0.4560 | 0.4440 | 0.2590 | 0.2630 |
| LAMBADA acc (±0.0105 SE) | 0.3360 | 0.3180 | 0.0000 | 0.0000 |
| bits/weight (quantised params) | 32 | 32 | 1.125 | 1.125 |
| learned alpha, mean / max (layers 1+) | — | -0.0141 / 0.0345 | — | -0.0520 / 0.0943 |
| seeds | 1 | 1 | 3 | 3 |
| wall-clock per run (s) | 1746 | 1741 | 2690 ± 1634 | 1961 ± 338 |

### Paired comparison

`onebit_ar` minus `onebit`, paired over seeds [0, 1, 2]. Decision rule fixed in advance: an effect smaller than 2 SE is reported as no effect.

| metric | per-seed deltas | mean | SE | verdict |
|---|---|---|---|---|
| wikitext ppl | [13.117, 5.408, -10.833] | +2.564 | 7.058 | within noise |
| wikitext NLL | [0.0454, 0.0189, -0.0386] | +0.0086 | 0.0248 | within noise |
| final train loss | [0.0558, 0.0173, -0.0394] | +0.0113 | 0.0276 | within noise |
| top-1 acc | [-0.0037, -0.0005, 0.0031] | -0.0004 | 0.0020 | within noise |

### Run-to-run stability

| arm | perplexities | mean | sd |
|---|---|---|---|
| `onebit` | [282.208, 283.768, 286.369] | 284.115 | 2.102 |
| `onebit_ar` | [295.324, 289.176, 275.536] | 286.679 | 10.127 |

**`onebit_ar` is 4.8x more variable across seeds than `onebit` while not being better on average.** With 3 seeds this is a variance ratio of 23.2 on 2 and 2 degrees of freedom — suggestive, not conclusive, and worth more seeds before it is claimed.

### Derived quantities

All quantities in nats of NLL; perplexity ratios in parentheses.

- Quantization cost (no AR): **+2.9647** nats (19.4x perplexity)
- AR cost at FP32: **+0.0118** nats (1.012x)
- AR cost at 1-bit: **+0.0086** nats (1.009x)
- Interaction: -0.0032 nats — **not distinguishable from zero**

The paired standard error on the 1-bit AR term alone is 0.0248 nats, 7.6x the interaction itself. **The attention residual does not preferentially repair binarization damage.**

### Gate trajectories

- `onebit_ar`: mean |alpha| 0.00003 at start, 0.05326 peak (step 284), 0.05317 final. **gates rose and held** — no sign of inter-path collapse
- `fp32_ar`: mean |alpha| 0.00003 at start, 0.01715 peak (step 160), 0.01536 final. **gates rose and held** — no sign of inter-path collapse

### Gate profile by depth

- `onebit_ar` (3 seed(s)): early -0.0614, middle -0.0702, late -0.0255. 22/23 live gates are negative — the pathway is being used to damp the residual stream, not to enrich it.
- `fp32_ar` (1 seed(s)): early -0.0186, middle -0.0150, late -0.0091. 22/23 live gates are negative — the pathway is being used to damp the residual stream, not to enrich it.
<!-- RESULTS-TABLE-END -->

---

## What is quantized, precisely

Quoting "1.125 bits/weight" for the whole model would be false, so both
numbers are reported:

| | parameters | bits/weight |
|---|---|---|
| Block projections (q, k, v, o, gate, up, down) | 302M | **1.125** |
| Embeddings + tied readout, norms, attention biases | 162M | 32 |
| **Model average** | **464M** | **~11.8** |

Embeddings and the readout stay full precision, standard for the BitNet
family. The compression story is about the projections; the model-wide
average is what an honest checkpoint-size claim rests on.

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
convert.py             HF weights -> frozen 1-bit checkpoint
inference.py           Generation from a checkpoint
tests/                 Parity against HuggingFace and across backends
docs/PROTOCOL.md       Why the experiment is shaped this way
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
