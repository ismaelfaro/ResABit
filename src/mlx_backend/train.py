"""Quantization-aware fine-tuning on MLX.

The loop keeps full-precision master weights and pushes them through
:func:`fake_quantize` on every forward, so gradients are computed against
exactly the weights the frozen model will use. Updates land on the master
copy via the straight-through estimator.

Design choices that the ablation depends on
-------------------------------------------
Fixed token budget, not fixed wall clock
    Wall-clock budgeting converts thermal throttling into treatment
    variance. Every arm sees the same number of tokens in the same order.

Frozen embeddings in every arm
    The embedding table is 155M of the model's 464M parameters and is tied
    to the readout. Leaving it trainable would let a 1.2M-token run rewrite
    the output head, which both overfits and confounds the comparison. All
    arms share a frozen readout, so differences come from the blocks.

Effective learning-rate gain on the residual gates
    alpha starts at 0 and there are only 24 of them. At the shared rate they
    barely move inside a short budget, which would make the AR arm a no-op
    by construction rather than by evidence. The gain is folded into the
    forward pass (see ``DecoderLayer.ALPHA_GAIN``) rather than into a second
    optimizer, and the resulting trajectory is logged so the choice is
    auditable.
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from ..config import ModelConfig
from ..trainer import TrainConfig, TrainResult, batch_order, cosine_lr
from .model import MLXResABit

# TrainConfig and TrainResult live in ``src.trainer`` so the pure-PyTorch
# loop (Colab/CUDA, where MLX does not exist) shares them; re-exported here
# because every entry point imports them from this module.
__all__ = ["TrainConfig", "TrainResult", "run_qat"]


def _batches(windows: mx.array, cfg: TrainConfig):
    """Deterministic shuffled stream; the seed fixes the data order.

    Both arms of a paired comparison therefore see identical batches, which
    removes data-order variance from the difference between them.
    """
    rng = np.random.default_rng(cfg.seed)
    order = batch_order(windows.shape[0], cfg, rng)
    need = len(order)
    for i in range(0, need, cfg.batch_size):
        chunk = windows[order[i : i + cfg.batch_size].tolist()]
        yield chunk[:, :-1], chunk[:, 1:]


def _diffusion_batches(windows: mx.array, cfg: TrainConfig):
    """The same stream, plus a corruption drawn from the same seed.

    The whole window is both input and target -- there is no shift, because
    a denoiser predicts the token at the position it is looking at. The
    corruption comes from the seeded numpy generator rather than from MLX so
    that a PyTorch run at the same seed sees the identical masks, which is
    what makes the two backends comparable and the seeds paired. The shared
    generator discipline (one `batch_order` sweep, then per-batch rate and
    mask draws in this exact order) is the contract `src.trainer` documents.
    """
    from ..diffusion import MIN_RATE

    rng = np.random.default_rng(cfg.seed)
    order = batch_order(windows.shape[0], cfg, rng)
    need = len(order)
    for i in range(0, need, cfg.batch_size):
        chunk = windows[order[i : i + cfg.batch_size].tolist()]
        B, L = chunk.shape
        rates = MIN_RATE + (1.0 - MIN_RATE) * rng.random((B, 1))
        mask = rng.random((B, L)) < rates
        yield chunk, mx.array(rates.astype(np.float32)), mx.array(mask)


def run_qat(
    model_config: ModelConfig,
    train_config: TrainConfig,
    windows: np.ndarray,
    log=print,
) -> tuple[MLXResABit, TrainResult]:
    """Fine-tune ``model_config`` for ``train_config.steps`` optimizer steps."""
    from . import load_mlx_pretrained

    mx.random.seed(train_config.seed)
    model = load_mlx_pretrained(model_config)

    if train_config.freeze_embeddings:
        model.embed_tokens.freeze()

    optimizer = optim.AdamW(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=[0.9, 0.95],
    )

    diffusion = model_config.diffusion
    if diffusion:
        def loss_fn(m, inputs, rates, mask):
            return m.diffusion_loss(inputs, rates, mask)

        stream = _diffusion_batches(mx.array(windows), train_config)
    else:
        def loss_fn(m, inputs, labels):
            return m.loss(inputs, labels)

        stream = _batches(mx.array(windows), train_config)

    grad_fn = nn.value_and_grad(model, loss_fn)

    result = TrainResult(final_train_loss=float("nan"))
    t_start = time.perf_counter()

    for step in range(train_config.steps):
        lr = cosine_lr(step, train_config)
        optimizer.learning_rate = lr

        accumulated = None
        step_loss = 0.0
        for _ in range(train_config.grad_accum):
            loss, grads = grad_fn(model, *next(stream))
            accumulated = (
                grads
                if accumulated is None
                else tree_map(lambda a, b: a + b, accumulated, grads)
            )
            # Force the graph after every micro-batch. Without this, MLX's
            # lazy evaluation keeps all grad_accum graphs alive and the step
            # spills to swap -- 35s per step instead of 5s.
            mx.eval(accumulated, loss)
            step_loss += float(loss) / train_config.grad_accum

        accumulated = tree_map(lambda g: g / train_config.grad_accum, accumulated)
        accumulated, _ = optim.clip_grad_norm(accumulated, train_config.grad_clip)
        optimizer.update(model, accumulated)
        mx.eval(model.trainable_parameters(), optimizer.state)

        result.loss_curve.append(step_loss)
        if model_config.use_attention_residuals:
            result.alpha_curve.append(model.alpha_values())

        # A diverged QAT run must be recorded, not silently dropped -- the
        # keep-rate is only meaningful if failures stay in the denominator.
        if not np.isfinite(step_loss) or step_loss > 30.0:
            result.diverged = True
            log(f"  diverged at step {step}: loss={step_loss}")
            break

        if step % train_config.log_every == 0 or step == train_config.steps - 1:
            elapsed = time.perf_counter() - t_start
            msg = f"  step {step:4d}/{train_config.steps}  loss {step_loss:.4f}  lr {lr:.2e}  {elapsed:.0f}s"
            if model_config.use_attention_residuals:
                alphas = model.alpha_values()
                msg += f"  alpha[mean|max] {np.mean(alphas):+.4f}|{np.max(np.abs(alphas)):.4f}"
            log(msg)

    result.wall_seconds = time.perf_counter() - t_start
    result.tokens_seen = len(result.loss_curve) * train_config.tokens_per_step
    result.final_train_loss = (
        float(np.mean(result.loss_curve[-10:])) if result.loss_curve else float("nan")
    )
    return model, result


def mlx_to_torch_state(model: MLXResABit) -> dict:
    """Export MLX parameters as a torch state_dict for the eval harness."""
    import torch

    state = {}
    for path, array in tree_flatten(model.parameters()):
        state[path] = torch.from_numpy(np.array(array, copy=True).astype(np.float32))
    return state
