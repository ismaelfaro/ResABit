#!/usr/bin/env bash
# setup_mlx.sh — Install PrismML MLX fork + mlx-lm for 1-bit Apple Silicon inference
#
# Usage:
#   ./setup_mlx.sh            # install deps + download Bonsai-1.7B
#   ./setup_mlx.sh 8B         # download Bonsai-8B instead
#   ./setup_mlx.sh --no-model # skip model download
#
# After setup:
#   python inference_mlx.py -p "Your question"

set -euo pipefail

MODEL_SIZE="${1:-1.7B}"
SKIP_MODEL=false
if [[ "${1:-}" == "--no-model" ]]; then
    SKIP_MODEL=true
    MODEL_SIZE="1.7B"
fi

echo "============================================"
echo " 1-bit Bonsai MLX Setup for Apple Silicon"
echo "============================================"

# ── Platform check ────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS (Apple Silicon) only."
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "WARNING: Not running on arm64. MLX performance may be limited."
fi

echo "Platform: $(uname -s) $(uname -m)"
echo "Python:   $(python3 --version 2>/dev/null || echo 'not found')"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_DIR=".venv-mlx"
if [[ ! -d "$VENV_DIR" ]]; then
    echo ""
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── Install PrismML MLX fork (1-bit kernel support) ──────────────────────────
echo ""
echo "Installing PrismML MLX fork (1-bit Q1_0_g128 kernels) …"
pip install --quiet --upgrade pip

# PrismML fork adds Metal kernels for 1-bit weight inference
pip install --quiet \
    "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism" \
    || {
        echo "PrismML fork failed — falling back to stock MLX (no 1-bit acceleration)"
        pip install --quiet mlx
    }

pip install --quiet mlx-lm huggingface-hub

echo "MLX version: $(python -c 'import mlx; print(mlx.__version__)' 2>/dev/null || echo 'unknown')"

# ── Download Bonsai model ─────────────────────────────────────────────────────
if [[ "$SKIP_MODEL" == "false" ]]; then
    MODEL_REPO="prism-ml/Bonsai-${MODEL_SIZE}-mlx-1bit"
    MODEL_DIR="./models/Bonsai-${MODEL_SIZE}-mlx"

    echo ""
    echo "Downloading ${MODEL_REPO} → ${MODEL_DIR}"
    python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_REPO}",
    local_dir="${MODEL_DIR}",
    ignore_patterns=["*.gguf"],
)
print("Done.")
EOF
    echo "Model path: $MODEL_DIR"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run inference:"
if [[ "$SKIP_MODEL" == "false" ]]; then
    MODEL_DIR="./models/Bonsai-${MODEL_SIZE}-mlx"
    echo "  python inference_mlx.py --model $MODEL_DIR -p 'What is a 1-bit LLM?'"
else
    echo "  python inference_mlx.py --model prism-ml/Bonsai-1.7B-mlx-1bit -p 'Hello!'"
fi
echo ""
echo "Or use the Bonsai models directly from HuggingFace:"
echo "  python inference_mlx.py --model prism-ml/Bonsai-8B-mlx-1bit -p 'Your question'"
