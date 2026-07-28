# Running ResABit on Google Colab (and from VS Code)

MLX — the training backend every local number was produced on — is Apple
Silicon only. Colab is Linux/CUDA, so it trains through the pure-PyTorch
loop in `src/trainer.py`, selected automatically:

```bash
python run_grid.py --seeds 0 1 2 --resume        # --backend auto -> torch
```

**Pairing, not reproduction.** A torch run at seed N consumes the identical
batch order and corruption masks the MLX run at seed N consumed (the RNG
contract is documented in `src/trainer.py`). But this repository measured
quantized stacks amplifying backend accumulation differences to ~1e-2, so a
Colab run will not reproduce the ledger's digits — it pairs with local runs
and is labelled with its backend in every ledger row. Do not mix backends
inside one paired comparison.

## Route 1 — the notebook (recommended)

`colab/ResABit_colab.ipynb`: GPU check → clone → install → test suite →
smoke grid → full grid → ledger to Drive. Open it in Colab directly once the
repo is on GitHub:

```
https://colab.research.google.com/github/<user>/ResABit/blob/<branch>/colab/ResABit_colab.ipynb
```

The branch must be pushed first (`git push -u origin <branch>`). Until then,
upload a zip of the repo through Colab's file panel and skip the clone cell.

## Route 2 — VS Code connected to the Colab machine

Two workable options, both with the same caveat: Colab sessions are
ephemeral (hours), the filesystem dies with them, and Google's terms frown
on pure-SSH usage that never touches a notebook. Keep the ledger synced to
Drive or push it out regularly.

### 2a. VS Code Remote-Tunnels (simplest, official VS Code path)

In a Colab cell:

```python
!curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz
!tar -xf vscode_cli.tar.gz
!./code tunnel --accept-server-license-terms
```

The cell prints a `github.com/login/device` code. Authenticate, then in
local VS Code: **Remote Explorer → Tunnels** (or `Ctrl/Cmd-Shift-P → Remote
Tunnels: Connect to Tunnel`). You get a full VS Code window on the Colab
machine — terminal, debugger, the repo checked out where the notebook left
it. The tunnel dies when the Colab runtime does.

### 2b. SSH via cloudflared (`colab-ssh`)

```python
!pip install -q colab-ssh
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="pick-something")
```

It prints a hostname; follow its VS Code instructions (install
`cloudflared` locally, add the printed block to `~/.ssh/config`, connect
with Remote-SSH). More moving parts than 2a; use it only if Tunnels is
blocked.

### After connecting, either way

```bash
git clone <repo-url> && cd ResABit
pip install -r requirements.txt        # mlx line self-skips on Linux
python -m pytest tests/ -q             # mlx parity tests self-skip
python run_grid.py --seeds 0 --steps 2 --eval-blocks 2 --eval-samples 1 --eval-tokens 4096   # smoke
python run_grid.py --seeds 0 1 2 --resume &> results/grid.log &
python dashboard.py --log results/grid.log --watch
```

`dashboard.py` works unchanged — it reads the log, not the platform. The
CPU-rate verdicts were calibrated on Apple Silicon (GPU-bound arms idle
their CPU); on CUDA the data-loader keeps CPU busier, so read the step
cadence, not the verdict, until it is recalibrated.

## What to expect

- T4: several hours for the 12-cell grid; A100: well under two.
- Batch size can go up on an A100 (`--batch-size 8 --grad-accum 1` keeps
  tokens/step identical). **That changes the batch-order RNG consumption**,
  so such runs no longer pair with the published seeds — they answer scale
  questions, not pairing questions. Keep `2×4` when you want pairing.
- Nothing here uploads anything. The publication chain
  (`upload_to_hf.py`) stays dry-run-by-default and local.
