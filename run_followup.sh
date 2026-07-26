#!/usr/bin/env bash
# Queue the two follow-up stages behind the running sweep.
#
# Order matters. Extra seeds come first: the paired standard error is what
# decides the verdict, and at three seeds it rests on two degrees of freedom.
# The determinism replicates are a scientifically interesting floor but they
# cannot change the verdict, so they run second.
set -u
cd "$(dirname "$0")"

PID_FILE=results/sweep.pid
LOG=results/sweep.log

if [[ -f $PID_FILE ]]; then
    PID=$(tr -dc '0-9' < $PID_FILE)
    echo "waiting for sweep pid $PID ..."
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    echo "sweep finished at $(date '+%H:%M:%S')"
fi

echo "=== extra seeds on the contested pair ===" >> $LOG
.venv/bin/python -u run_ablation.py --stage full --steps 300 \
    --seeds 0 1 2 3 4 --no-zero-shot --resume >> $LOG 2>&1

echo "=== determinism replicates ===" >> $LOG
.venv/bin/python -u run_ablation.py --stage determinism --steps 300 \
    --seeds 0 --replicates 3 --no-zero-shot --resume >> $LOG 2>&1

echo "all stages complete at $(date '+%H:%M:%S')" >> $LOG
