"""Backend-neutral training: the config, and a pure-PyTorch QAT loop.

The MLX loop in ``src/mlx_backend/train.py`` is the fast path on Apple
Silicon and does not exist anywhere else -- MLX has no Linux/CUDA build, so a
Colab or CUDA box could previously not train at all. This module is the
portable twin: same config, same seeded data order, same corruption draws.

The RNG contract, stated because it is load-bearing
---------------------------------------------------
Both backends must consume the *same numpy generator calls in the same
order*: one ``permutation`` sweep for the batch order, then per batch one
``random((B,1))`` for rates and one ``random((B,L))`` for the mask. A torch
run at seed 0 therefore sees the identical batches and identical corruptions
the MLX run at seed 0 saw, which is what lets a Colab cell pair with a local
one. What does NOT transfer is bitwise reproducibility of the result: CUDA,
Metal and CPU accumulate in different orders, and this repository measured
binarised stacks amplifying exactly such differences to ~1e-2. Cross-backend
runs are paired, not identical, and must be labelled with their backend.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .config import ModelConfig

__all__ = ["TrainConfig", "TrainResult", "run_qat_torch"]


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
    backend: str = "mlx"

    def as_dict(self) -> dict:
        return asdict(self)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_frac * cfg.steps))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    floor = cfg.learning_rate * cfg.min_lr_frac
    return floor + 0.5 * (cfg.learning_rate - floor) * (1 + np.cos(np.pi * progress))


def batch_order(n_windows: int, cfg: TrainConfig, rng: np.random.Generator) -> np.ndarray:
    """The shared shuffle. One permutation stream, consumed identically."""
    need = cfg.steps * cfg.grad_accum * cfg.batch_size
    return np.concatenate(
        [rng.permutation(n_windows) for _ in range(need // n_windows + 1)]
    )[:need]


def run_qat_torch(
    model_config: ModelConfig,
    train_config: TrainConfig,
    windows: np.ndarray,
    device: torch.device | str | None = None,
    log=print,
):
    """Fine-tune on plain PyTorch: CUDA, MPS or CPU.

    Returns ``(model, TrainResult)`` with the model already on ``device`` in
    eval mode -- unlike the MLX path there is no state-dict hop, because the
    training model *is* the evaluation model.
    """
    from .diffusion import MIN_RATE
    from .loader import load_pretrained

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device)

    torch.manual_seed(train_config.seed)
    model = load_pretrained(model_config, verbose=False).to(device)
    model.train()

    if train_config.freeze_embeddings:
        # The readout is tied to this table, so freezing it freezes both --
        # which is the point: a 1.2M-token run must not rewrite the head.
        model.embed_tokens.weight.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    rng = np.random.default_rng(train_config.seed)
    order = batch_order(len(windows), train_config, rng)
    diffusion = model_config.diffusion

    result = TrainResult(final_train_loss=float("nan"), backend=f"torch/{device.type}")
    t_start = time.perf_counter()
    cursor = 0

    for step in range(train_config.steps):
        lr = cosine_lr(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(train_config.grad_accum):
            idx = order[cursor : cursor + train_config.batch_size]
            cursor += train_config.batch_size
            chunk = torch.from_numpy(windows[idx].astype(np.int64))

            if diffusion:
                # Same generator, same call shapes, same order as the MLX
                # stream -- this is the cross-backend pairing contract.
                B, L = chunk.shape
                rates_np = MIN_RATE + (1.0 - MIN_RATE) * rng.random((B, 1))
                mask_np = rng.random((B, L)) < rates_np
                loss = model.diffusion_loss(
                    chunk.to(device),
                    torch.from_numpy(rates_np.astype(np.float32)).to(device),
                    torch.from_numpy(mask_np).to(device),
                )
            else:
                inputs = chunk[:, :-1].to(device)
                labels = chunk[:, 1:].to(device)
                loss = model(input_ids=inputs, labels=labels)["loss"]

            (loss / train_config.grad_accum).backward()
            step_loss += float(loss.detach()) / train_config.grad_accum

        torch.nn.utils.clip_grad_norm_(params, train_config.grad_clip)
        optimizer.step()

        result.loss_curve.append(step_loss)
        if model_config.use_attention_residuals:
            result.alpha_curve.append(model.alpha_values())

        if not np.isfinite(step_loss) or step_loss > 30.0:
            result.diverged = True
            log(f"  diverged at step {step}: loss={step_loss}")
            break

        if step % train_config.log_every == 0 or step == train_config.steps - 1:
            elapsed = time.perf_counter() - t_start
            log(f"  step {step:4d}/{train_config.steps}  loss {step_loss:.4f}  "
                f"lr {lr:.2e}  {elapsed:.0f}s  [{result.backend}]")

    result.wall_seconds = time.perf_counter() - t_start
    result.tokens_seen = len(result.loss_curve) * train_config.tokens_per_step
    result.final_train_loss = (
        float(np.mean(result.loss_curve[-10:])) if result.loss_curve else float("nan")
    )
    return model.eval(), result
