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
from dataclasses import dataclass, field, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from ..config import ModelConfig
from .model import MLXResABit

__all__ = ["TrainConfig", "TrainResult", "run_qat"]


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 2
    grad_accum: int = 4
    seq_len: int = 512
    learning_rate: float = 1e-4
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    freeze_embeddings: bool = True
    seed: int = 0
    log_every: int = 25

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len

    @property
    def total_tokens(self) -> int:
        return self.steps * self.tokens_per_step


@dataclass
class TrainResult:
    final_train_loss: float
    loss_curve: list[float] = field(default_factory=list)
    alpha_curve: list[list[float]] = field(default_factory=list)
    wall_seconds: float = 0.0
    tokens_seen: int = 0
    diverged: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _cosine_lr(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_frac * cfg.steps))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    floor = cfg.learning_rate * cfg.min_lr_frac
    return floor + 0.5 * (cfg.learning_rate - floor) * (1 + np.cos(np.pi * progress))


def _batches(windows: mx.array, cfg: TrainConfig):
    """Deterministic shuffled stream; the seed fixes the data order.

    Both arms of a paired comparison therefore see identical batches, which
    removes data-order variance from the difference between them.
    """
    rng = np.random.default_rng(cfg.seed)
    n = windows.shape[0]
    need = cfg.steps * cfg.grad_accum * cfg.batch_size
    order = np.concatenate(
        [rng.permutation(n) for _ in range(need // n + 1)]
    )[:need]
    for i in range(0, need, cfg.batch_size):
        chunk = windows[order[i : i + cfg.batch_size].tolist()]
        yield chunk[:, :-1], chunk[:, 1:]


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

    def loss_fn(m, inputs, labels):
        return m.loss(inputs, labels)

    grad_fn = nn.value_and_grad(model, loss_fn)
    stream = _batches(mx.array(windows), train_config)

    result = TrainResult(final_train_loss=float("nan"))
    t_start = time.perf_counter()

    for step in range(train_config.steps):
        lr = _cosine_lr(step, train_config)
        optimizer.learning_rate = lr

        accumulated = None
        step_loss = 0.0
        for _ in range(train_config.grad_accum):
            inputs, labels = next(stream)
            loss, grads = grad_fn(model, inputs, labels)
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
