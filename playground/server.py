"""
playground/server.py — Local inference server for the ResAbit playground.

Loads (or converts) the 1-bit + Attention Residual model and exposes:
  GET  /status          — model load status
  POST /generate        — streaming SSE token generation
  POST /generate/full   — non-streaming, returns full response

Run:
    cd /path/to/1Bits+residual
    python playground/server.py [--checkpoint path] [--device mps|cpu]

Then open http://localhost:8000 (or the playground at http://localhost:8080).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── path so we can import src/ from playground/ ────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.model import OneBitResidualLM
from src.quantization import quantize_model_weights

# ── globals (populated at startup) ─────────────────────────────────────────
_model: OneBitResidualLM | None = None
_tokenizer = None
_device: str = "cpu"
_status: dict = {"state": "loading", "message": "Starting…"}

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="ResAbit Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request schemas ─────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    info = dict(_status)
    if _model is not None:
        total = sum(p.numel() for p in _model.parameters())
        info["params_m"] = round(total / 1e6, 1)
        info["device"] = _device
    return JSONResponse(info)


@app.post("/generate")
async def generate_stream(req: GenerateRequest):
    """Server-Sent Events streaming generation."""
    if _model is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    async def token_stream() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_generation():
            try:
                ids = _tokenizer.encode(req.prompt, return_tensors="pt").to(_device)
                prompt_len = ids.shape[1]

                temp = req.temperature if req.temperature > 0 else 1e-8
                generated = ids.clone()
                past_kv = None
                t0 = time.perf_counter()

                with torch.no_grad():
                    for step in range(req.max_new_tokens):
                        input_ids = generated if past_kv is None else generated[:, -1:]
                        logits, past_kv = _model(input_ids, past_key_values=past_kv,
                                                  use_cache=True)
                        logits = logits[:, -1, :] / temp

                        # top-p nucleus sampling
                        if req.top_p < 1.0:
                            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > req.top_p
                            sorted_logits[remove] = float("-inf")
                            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, 1)
                        generated = torch.cat([generated, next_id], dim=1)

                        token_text = _tokenizer.decode(
                            next_id[0].tolist(), skip_special_tokens=True
                        )

                        elapsed = time.perf_counter() - t0
                        tps = (step + 1) / elapsed if elapsed > 0 else 0

                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            json.dumps({
                                "token": token_text,
                                "step": step + 1,
                                "tps": round(tps, 1),
                            })
                        )

                        eos = _tokenizer.eos_token_id
                        if eos is not None and next_id.item() == eos:
                            break

                loop.call_soon_threadsafe(queue.put_nowait, "[DONE]")
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"error": str(e)}))
                loop.call_soon_threadsafe(queue.put_nowait, "[DONE]")

        import threading
        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item == "[DONE]":
                yield "data: [DONE]\n\n"
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/generate/full")
def generate_full(req: GenerateRequest):
    """Non-streaming: return complete response at once."""
    if _model is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    ids = _tokenizer.encode(req.prompt, return_tensors="pt").to(_device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = _model.generate(
            ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature if req.temperature > 0 else 1e-8,
            top_p=req.top_p,
        )
    elapsed = time.perf_counter() - t0
    new_tokens = out.shape[1] - ids.shape[1]
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return {
        "text": text,
        "new_tokens": new_tokens,
        "elapsed_s": round(elapsed, 2),
        "tps": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
    }


# ── model loading ────────────────────────────────────────────────────────────

def _load_or_convert(checkpoint: str | None, device: str) -> None:
    global _model, _tokenizer, _device, _status
    _device = device

    from transformers import AutoTokenizer

    _status = {"state": "loading", "message": "Loading tokenizer…"}
    _tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True
    )

    # ── try loading existing checkpoint ─────────────────────────────────────
    if checkpoint and Path(checkpoint).exists():
        _status = {"state": "loading", "message": f"Loading checkpoint {checkpoint}…"}
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ModelConfig()
        cfg.__dict__.update(ckpt["config"])
        _model = OneBitResidualLM(cfg).to(device)
        _model.load_state_dict(ckpt["state_dict"], strict=False)
        _model.eval()
        _status = {"state": "ready", "message": "Loaded from checkpoint"}
        print(f"  Model loaded from {checkpoint}")
        return

    # ── convert from HuggingFace on the fly ─────────────────────────────────
    _status = {"state": "loading", "message": "Downloading Qwen/Qwen1.5-0.5B-Chat…"}
    print("  No checkpoint found — converting from Qwen/Qwen1.5-0.5B-Chat")

    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", torch_dtype=torch.float32, trust_remote_code=True
    )

    _status = {"state": "loading", "message": "Converting to 1-bit + Attention Residuals…"}
    from convert import _map_state_dict
    cfg = ModelConfig()
    _model = OneBitResidualLM(cfg)
    our_sd = _map_state_dict(hf.state_dict())
    _model.load_state_dict(our_sd, strict=False)
    quantize_model_weights(_model)
    _model = _model.to(device)
    _model.eval()

    # optionally save for next time
    if checkpoint:
        _status = {"state": "loading", "message": f"Saving checkpoint to {checkpoint}…"}
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": cfg.__dict__, "state_dict": _model.state_dict()}, checkpoint)
        print(f"  Saved to {checkpoint}")

    _status = {"state": "ready", "message": "Converted from Qwen/Qwen1.5-0.5B-Chat"}
    print("  Model ready")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ResAbit playground server")
    parser.add_argument("--checkpoint", default="./checkpoints/qwen0.5b-1bit/model.pt",
                        help="Path to .pt checkpoint (created on first run if missing)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"ResAbit Inference Server")
    print(f"  device    : {args.device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  server    : http://{args.host}:{args.port}")
    print()

    import threading
    thread = threading.Thread(
        target=_load_or_convert,
        args=(args.checkpoint, args.device),
        daemon=True,
    )
    thread.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
