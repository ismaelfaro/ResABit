#!/usr/bin/env bash
# Launch a grid rung in a tmux session, so it survives the shell that started it.
#
#   ./run_ladder.sh 4800            # 19.7M-token rung, seed 0
#   ./run_ladder.sh 4800 "0 1 2"    # ...across three seeds
#
# Why this exists: a rung is 10-20 hours and the 19.7M one was killed twice
# mid-cell by its parent shell going away, losing ~3h of compute each time.
# `nohup ... &` is not enough -- it blocks SIGHUP but leaves the run in the
# caller's session, so a terminal or agent teardown still takes it out. macOS
# has no `setsid`, so tmux is the portable detach here: the run gets its own
# session and outlives anything that started it.
#
# `--resume` bounds the loss anyway -- the ledger is the checkpoint, at cell
# granularity -- but not losing the cell at all is better.
#
# `caffeinate -i -w <pid>` blocks idle sleep for exactly as long as the run
# lives. It does NOT block lid-close sleep: for an overnight rung, leave the
# lid open on power, or run clamshell with an external display. Three
# "STARVED" readings in this project's dashboard were this and nothing else.
set -euo pipefail

STEPS="${1:?usage: run_ladder.sh <steps> [seeds]}"
SEEDS="${2:-0}"
LOG="results/grid_ladder.log"
SESSION="resabit-rung-${STEPS}"

cd "$(dirname "$0")"
mkdir -p results

if ! command -v tmux >/dev/null; then
    echo "tmux not found; install it (brew install tmux) or run run_grid.py in a terminal you keep open" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session $SESSION already running -- attach with: tmux attach -t $SESSION" >&2
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    ".venv/bin/python run_grid.py --seeds $SEEDS --steps $STEPS --resume >> $LOG 2>&1"

# Keep the machine awake for exactly as long as the run lives. Started inside
# its own tmux session for the same reason the run is.
tmux new-session -d -s "${SESSION}-awake" \
    "while tmux has-session -t $SESSION 2>/dev/null; do caffeinate -i -t 300; done"

TOKENS=$(.venv/bin/python -c "print(f'{$STEPS*4096/1e6:.2f}')")
echo "rung    : $STEPS steps = ${TOKENS}M tokens, seeds [$SEEDS]"
echo "session : $SESSION  (detached; survives this shell)"
echo "log     : $LOG"
echo "watch   : python dashboard.py --log $LOG --watch"
echo "attach  : tmux attach -t $SESSION"
echo "stop    : tmux kill-session -t $SESSION"
