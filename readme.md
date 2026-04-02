# 1Bits + Residual

A research reimplementation combining **1-bit weight quantization** (Q1_0_g128, from the [1-bit Bonsai 8B whitepaper](https://prismml.com)) with **Attention Residuals** (arXiv 2603.15031), using [Qwen/Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) as the base architecture.

```
2.8×           ~28×         ~8×
smaller        bits saved   faster* on Apple Silicon
vs FP32        vs FP32
```
*With PrismML Metal kernels via `setup_mlx.sh`

---

## What is this?

This project combines two ideas:

### 1. 1-bit Weights (Q1_0_g128)
Every linear weight in the model — attention projections, MLP projections — is stored as a **single sign bit** plus one shared **FP16 scale** per 128 weights:

```
w_i = s_g × (2b_i − 1),   b_i ∈ {0, 1},   s_g ∈ FP16
```

Effective storage: **1.125 bits/weight** (vs 16 bits for FP16, 32 bits for FP32).

### 2. Attention Residuals
Each transformer layer accumulates a running sum of all prior attention outputs (`R_l`) and adds it back as a learnable skip connection:

```
A_l = Attention(RMSNorm(h_{l-1}))
R_l = R_{l-1} + A_l                      # running attention residual
h_l = h_{l-1} + A_l + α·R_{l-1}          # α: learnable gate (init=0)
h_l = h_l + MLP(RMSNorm(h_l))
```

This improves gradient flow across layers and allows later layers to directly access earlier attention patterns.

---

## Benchmark Results (Qwen1.5-0.5B-Chat)

| Metric | Original (FP32) | 1-bit + AR |
|--------|----------------|------------|
| Parameters | 464M | 156M* |
| Checkpoint size | 1,856 MB | 666 MB |
| Size ratio | — | **2.8×** |
| Bits/weight (linear) | 32 | ~1.125 |
| Throughput (CPU) | baseline | 0.3× (no custom kernel) |
| Throughput (Metal)** | baseline | ~5–8× |
| PPL (post-quant, no fine-tune) | 5.4 | high |
| PPL (after QAT fine-tuning) | — | recoverable |

\* packed bits + scales; embeddings remain FP32
\** requires `./setup_mlx.sh` (PrismML Metal kernels)

> **Important**: Post-training quantization to 1-bit without fine-tuning collapses model quality.
> The Bonsai paper trains with Quantization-Aware Training (QAT). Use `train.py` to recover quality.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all unit tests (22 tests)
python tests.py

# Download Qwen1.5-0.5B-Chat and run full benchmark
python benchmark.py --device mps   # Apple Silicon
python benchmark.py                 # CPU fallback

# Apple Silicon MLX (fast 1-bit inference via PrismML kernels)
./setup_mlx.sh 1.7B                 # ~250MB model
source .venv-mlx/bin/activate
python inference_mlx.py -p "What is a 1-bit LLM?"
```

---

## File Structure

```
1Bits+residual/
├── src/
│   ├── config.py          # Model config (Qwen1.5-0.5B + extensions)
│   ├── quantization.py    # OneBitLinear — Q1_0_g128 with STE training
│   └── model.py           # Full model: GQA + SwiGLU + RoPE + Attention Residuals
│
├── benchmark.py           # Compare original vs 1-bit+AR (size, speed, PPL, quality)
├── convert.py             # Convert HF weights → 1-bit checkpoint
├── train.py               # Quantization-aware fine-tuning
├── inference.py           # PyTorch inference (CPU/GPU/MPS)
├── inference_mlx.py       # MLX streaming inference for Apple Silicon
├── compare.py             # Quick side-by-side metric comparison
├── tests.py               # 22 unit tests
├── setup_mlx.sh           # Install PrismML MLX fork + download Bonsai models
└── requirements.txt
```

---

## Training (recover quality after quantization)

```bash
python train.py \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./checkpoints/trained \
    --epochs 3 \
    --lr 1e-4 \
    --device mps
```

The training loop keeps full-precision weights for stable gradients and uses the **Straight-Through Estimator (STE)** to simulate 1-bit inference end-to-end during training.

---

## MLX Inference (Apple Silicon)

Uses the [PrismML Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo/) approach with custom Metal kernels for 1-bit weight decoding inline:

```bash
./setup_mlx.sh 8B   # download Bonsai-8B (~1.28 GB)
source .venv-mlx/bin/activate

# Stream generation
python inference_mlx.py -p "Explain 1-bit quantization" --model prism-ml/Bonsai-8B-mlx-1bit

# Custom parameters
python inference_mlx.py -p "Write a haiku" -n 128 --temp 0.7
```

| Model | Size | tok/s (M4 Pro) |
|-------|------|----------------|
| `prism-ml/Bonsai-1.7B-mlx-1bit` | ~250 MB | ~290 |
| `prism-ml/Bonsai-4B-mlx-1bit`   | ~566 MB | ~132 |
| `prism-ml/Bonsai-8B-mlx-1bit`   | 1.28 GB | ~131 |

---

## Key Concepts

**Why 1-bit?** LLM token generation is memory-bandwidth-bound: each token requires reading all model weights from memory. Reducing from 16 bits to 1.125 bits per weight cuts memory traffic by ~14×, directly translating to faster generation on memory-constrained hardware (phones, laptops, edge devices).

**Why Attention Residuals?** Standard residual connections carry information only one layer forward. The Attention Residual accumulator `R_l = Σ_{i≤l} A_i` gives every layer a direct view of all previous attention outputs, improving gradient flow in deep networks.

**Why not just post-training quantize?** At 1-bit precision, the sign operation is so aggressive that a model not trained for it degenerates. The fix is Quantization-Aware Training (QAT) with the Straight-Through Estimator, which trains the full-precision weights while simulating 1-bit inference.

---

## References

- **1-bit Bonsai 8B** — PrismML, March 2026 ([whitepaper](.docs/1-bit-bonsai-8b-whitepaper.pdf))
- **Attention Residuals** — arXiv 2603.15031 ([paper](.docs/2603.15031v1.pdf))
- **BitNet** — Wang et al., arXiv 2310.11453
- **1.58-bit LLMs** — Ma et al., arXiv 2402.17764
- **Qwen1.5** — Qwen Team, HuggingFace

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
