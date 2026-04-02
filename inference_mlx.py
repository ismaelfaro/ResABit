"""
inference_mlx.py — Streaming MLX inference on Apple Silicon.

Uses the PrismML MLX fork (with 1-bit Q1_0_g128 kernel support) to run
the Bonsai 1-bit models or any mlx_lm-compatible model.

Setup (one-time):
    ./setup_mlx.sh

Usage:
    # Run Bonsai 1.7B (smallest, ~250 MB)
    python inference_mlx.py -p "What is quantum computing?"

    # Run Bonsai 8B
    python inference_mlx.py --model prism-ml/Bonsai-8B-mlx-1bit -p "Explain 1-bit LLMs"

    # Run Qwen1.5-0.5B (standard MLX, not 1-bit)
    python inference_mlx.py --model Qwen/Qwen1.5-0.5B-Chat -p "Hello!"

    # With custom parameters
    python inference_mlx.py -p "Write a haiku" -n 128 --temp 0.7

References:
    Bonsai demo: https://github.com/PrismML-Eng/Bonsai-demo/
    PrismML MLX: https://github.com/PrismML-Eng/mlx
"""

import argparse
import sys
import time
import platform


def _check_apple_silicon() -> None:
    if platform.system() != "Darwin":
        print("ERROR: MLX inference requires macOS (Apple Silicon).", file=sys.stderr)
        sys.exit(1)
    if platform.machine() != "arm64":
        print("WARNING: MLX is optimised for Apple Silicon (arm64).", file=sys.stderr)


def _check_mlx() -> None:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print(
            "ERROR: mlx not installed.\n"
            "Run:  ./setup_mlx.sh\n"
            "  or: pip install mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism\n"
            "      pip install mlx-lm",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("ERROR: mlx-lm not installed.\nRun: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)


# ── Core generation ───────────────────────────────────────────────────────────

DEFAULT_MODEL = "prism-ml/Bonsai-1.7B-mlx-1bit"

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise AI assistant. "
    "Answer clearly and directly."
)


def build_prompt(tokenizer, user_message: str, system: str = SYSTEM_PROMPT) -> str:
    """Apply the model's chat template to format the conversation."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without a chat template
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.5,
    top_p: float = 0.85,
    verbose: bool = True,
) -> str:
    from mlx_lm import stream_generate as mlx_stream

    full_text = ""
    t0 = time.perf_counter()
    n_tokens = 0

    if verbose:
        print(flush=True)

    for response in mlx_stream(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
    ):
        token_text = response.text
        full_text += token_text
        n_tokens += 1
        if verbose:
            print(token_text, end="", flush=True)

    elapsed = time.perf_counter() - t0
    tps = n_tokens / max(elapsed, 1e-6)

    if verbose:
        print(f"\n\n{'─'*50}")
        print(f"  {n_tokens} tokens  |  {tps:.1f} tok/s  |  {elapsed:.2f}s")

    return full_text


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.5,
    top_p: float = 0.85,
) -> str:
    """Non-streaming generation (returns complete string)."""
    from mlx_lm import generate as mlx_generate
    return mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        verbose=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_apple_silicon()
    _check_mlx()

    p = argparse.ArgumentParser(
        description="1-bit MLX inference on Apple Silicon (Bonsai family)"
    )
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--prompt", "-p", required=True, help="User prompt")
    p.add_argument("--system", "-s", default=SYSTEM_PROMPT, help="System prompt")
    p.add_argument("--max-tokens", "-n", type=int, default=256)
    p.add_argument("--temp",    type=float, default=0.5)
    p.add_argument("--top-p",   type=float, default=0.85)
    p.add_argument("--no-stream", action="store_true", help="Non-streaming mode")
    args = p.parse_args()

    from mlx_lm import load

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")

    # Try to show memory usage
    try:
        import mlx.core as mx
        mem_gb = mx.metal.get_active_memory() / 1e9
        print(f"Active memory: {mem_gb:.2f} GB")
    except Exception:
        pass

    prompt = build_prompt(tokenizer, args.prompt, args.system)

    print(f"\nPrompt: {args.prompt!r}")
    print("─" * 50)

    if args.no_stream:
        result = generate_once(model, tokenizer, prompt, args.max_tokens, args.temp, args.top_p)
        print(result)
    else:
        stream_generate(model, tokenizer, prompt, args.max_tokens, args.temp, args.top_p)


if __name__ == "__main__":
    main()
