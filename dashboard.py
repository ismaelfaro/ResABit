"""Live view of a running export or sweep.

    python dashboard.py                       # one snapshot
    python dashboard.py --watch               # refresh every 10s
    python dashboard.py --log results/sweep.log

Reads the log rather than instrumenting the runs, so it works on a job that
is already in flight and on one that finished yesterday.

It reports what it knows and labels what it is guessing. A run whose log has
gone quiet is the normal case here -- Python block-buffers stdout behind a
redirect, and until that was fixed a 25-minute arm emitted nothing until it
exited. When the log is silent but the process is alive, progress is inferred
from elapsed time against the arms that already finished, and it says so.
Silently extrapolating would make a stalled run look healthy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG = Path("results/export.log")
CHECKPOINTS = Path("checkpoints")

STEP = re.compile(r"^\s*step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+lr\s+(\S+)\s+(\d+)s")
ARM = re.compile(r"^arm\s+:\s+(\S+)\s+\(seed (\d+)\)")
OUT = re.compile(r"^out\s+:\s+(\S+)")
LEDGER = re.compile(r"^ledger ppl\s+:\s+(\S+)")
DRIFT = re.compile(r"^ledger drift\s+:\s+(\S+)")
TRAIN_PPL = re.compile(r"^training-forward ppl\s+:\s+([\d.]+)\s+nll\s+([\d.]+)")
FROZEN_PPL = re.compile(r"^frozen \(FP16 scales\)\s+:\s+([\d.]+)\s+nll\s+([\d.]+)")
FREEZE_COST = re.compile(r"^cost of freezing\s+:\s+(\S+) nats")
STORAGE = re.compile(r"^storage\s+:\s+(.+)$")
BITS = re.compile(r"^bits/weight\s+:\s+(.+)$")
WROTE = re.compile(r"^wrote (\S+)")
DIVERGED = re.compile(r"^\s*diverged at step (\d+)")
FAILED = re.compile(r"^\s*FAILED: (.+)$")
REFUSED = re.compile(r"^refusing to export: (.+)$")

BAR_WIDTH = 32
SPARK = "▁▂▃▄▅▆▇█"


@dataclass
class Arm:
    name: str = "?"
    seed: int | None = None
    out: str | None = None
    ledger_ppl: float | None = None
    steps_done: int = 0
    steps_total: int = 0
    last_loss: float | None = None
    elapsed_s: int = 0
    losses: list[float] = field(default_factory=list)
    train_ppl: float | None = None
    frozen_ppl: float | None = None
    freeze_cost: str | None = None
    drift: str | None = None
    storage: str | None = None
    bits: str | None = None
    wrote: str | None = None
    error: str | None = None
    diverged_at: int | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "failed"
        if self.wrote:
            return "done"
        if self.diverged_at is not None:
            return "diverged"
        if self.train_ppl is not None:
            return "freezing"
        if self.steps_total and self.steps_done >= self.steps_total - 1:
            return "evaluating"
        if self.steps_done:
            return "training"
        return "starting"


def parse(log: Path) -> list[Arm]:
    if not log.exists():
        return []
    arms: list[Arm] = []
    current: Arm | None = None

    for raw in log.read_text(errors="replace").splitlines():
        line = raw.rstrip()
        if m := ARM.match(line):
            current = Arm(name=m.group(1), seed=int(m.group(2)))
            arms.append(current)
            continue
        if current is None:
            continue

        if m := OUT.match(line):
            current.out = m.group(1)
        elif m := LEDGER.match(line):
            try:
                current.ledger_ppl = float(m.group(1))
            except ValueError:
                current.ledger_ppl = None
        elif m := STEP.match(line):
            current.steps_done = int(m.group(1)) + 1
            current.steps_total = int(m.group(2))
            current.last_loss = float(m.group(3))
            current.elapsed_s = int(m.group(5))
            current.losses.append(float(m.group(3)))
        elif m := TRAIN_PPL.match(line):
            current.train_ppl = float(m.group(1))
        elif m := FROZEN_PPL.match(line):
            current.frozen_ppl = float(m.group(1))
        elif m := FREEZE_COST.match(line):
            current.freeze_cost = m.group(1)
        elif m := DRIFT.match(line):
            current.drift = m.group(1)
        elif m := STORAGE.match(line):
            current.storage = m.group(1).strip()
        elif m := BITS.match(line):
            current.bits = m.group(1).strip()
        elif m := WROTE.match(line):
            current.wrote = m.group(1)
        elif m := DIVERGED.match(line):
            current.diverged_at = int(m.group(1))
        elif m := FAILED.match(line):
            current.error = m.group(1)
        elif m := REFUSED.match(line):
            current.error = "refused: " + m.group(1)

    return arms


def _cpu_seconds(cputime: str) -> float:
    """ps CPU time: [hh:]mm:ss[.ff]."""
    parts = cputime.split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def live_processes(sample_seconds: float = 0.0) -> list[dict]:
    """Running exports, with elapsed and CPU time straight from ps.

    With ``sample_seconds``, read the CPU clock twice. Elapsed time alone
    cannot tell a slow run from a wedged one, and on this machine that is not
    hypothetical: MLX and PyTorch draw on the same unified memory and neither
    sees the other's allocations, so a second job started alongside a sweep
    can leave the first alive, resident, and making no progress at all.
    """
    out = subprocess.run(
        ["ps", "-Ao", "pid=,etime=,time=,rss=,command="],
        capture_output=True, text=True, check=False,
    )
    found = []
    for line in out.stdout.splitlines():
        if "export_checkpoint.py" not in line and "run_ablation.py" not in line:
            continue
        if "/bin/zsh" in line or "dashboard.py" in line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, etime, cputime, rss, command = parts
        found.append({
            "pid": pid,
            "etime": etime,
            "cpu_time": cputime,
            "cpu_seconds": _cpu_seconds(cputime),
            "rss_mb": int(rss) / 1024,
            "cpu_rate": None,
            "arm": next(
                (t for t in command.split() if t in
                 ("onebit", "onebit_ar", "fp32", "fp32_ar")),
                "?",
            ),
        })

    if sample_seconds and found:
        time.sleep(sample_seconds)
        again = subprocess.run(
            ["ps", "-o", "pid=,time=", "-p", ",".join(p["pid"] for p in found)],
            capture_output=True, text=True, check=False,
        )
        later = {}
        for line in again.stdout.splitlines():
            bits = line.split()
            if len(bits) == 2:
                later[bits[0]] = _cpu_seconds(bits[1])
        for proc in found:
            if proc["pid"] in later:
                delta = later[proc["pid"]] - proc["cpu_seconds"]
                proc["cpu_rate"] = delta / sample_seconds

    return found


def _etime_seconds(etime: str) -> int:
    """ps elapsed time: [[dd-]hh:]mm:ss."""
    days = 0
    if "-" in etime:
        d, etime = etime.split("-", 1)
        days = int(d)
    bits = [int(x) for x in etime.split(":")]
    while len(bits) < 3:
        bits.insert(0, 0)
    return days * 86400 + bits[0] * 3600 + bits[1] * 60 + bits[2]


def bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    if not total:
        return "─" * width
    filled = min(width, int(width * done / total))
    return "█" * filled + "░" * (width - filled)


def sparkline(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return SPARK[0] * len(values)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in values)


def mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def checkpoint_rows() -> list[str]:
    if not CHECKPOINTS.exists():
        return []
    rows = []
    for directory in sorted(p for p in CHECKPOINTS.iterdir() if p.is_dir()):
        weights = directory / "model.safetensors"
        size = f"{weights.stat().st_size/1e6:.1f} MB" if weights.exists() else "—"
        card = "yes" if (directory / "README.md").exists() else "NO — run make_model_card.py"
        ppl = "—"
        manifest = directory / "config.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                frozen = (data.get("metrics") or {}).get("wikitext2_val_frozen")
                if frozen:
                    ppl = f"{frozen['perplexity']:.3f}"
            except (json.JSONDecodeError, KeyError, TypeError):
                ppl = "unreadable"
        rows.append(f"  {directory.name:<34} {size:>10}  ppl {ppl:>9}  card {card}")
    return rows


def render(log: Path, sample_seconds: float = 0.0) -> str:
    arms = parse(log)
    running = live_processes(sample_seconds)
    by_arm = {p["arm"]: p for p in running}

    lines = [
        "ResABit — export dashboard",
        "=" * 78,
    ]
    if log.exists():
        age = time.time() - log.stat().st_mtime
        lines.append(
            f"log      {log}  ({log.stat().st_size} B, written {mmss(age)} ago)"
        )
    else:
        lines.append(f"log      {log}  (does not exist yet)")
    lines.append(f"running  {len(running)} process(es)")
    lines.append("")

    if not arms:
        lines.append("  no arms in the log yet")

    # Median duration of the arms that finished, used only to label a guess.
    completed = [a.elapsed_s for a in arms if a.state == "done" and a.elapsed_s]
    reference = sorted(completed)[len(completed) // 2] if completed else None

    for arm in arms:
        label = f"{arm.name} seed {arm.seed}"
        lines.append(f"  {label:<22} {arm.state.upper()}")

        proc = by_arm.get(arm.name) if arm.state not in ("done", "failed") else None
        stale = proc is not None and arm.steps_done == 0
        if stale:
            # "STARTING" is wrong for something 25 minutes in. The log says
            # nothing; the process table says otherwise, and the process
            # table is the one that is not buffered.
            lines[-1] = f"  {label:<22} RUNNING (log silent)"

        if arm.steps_total:
            pct = 100 * arm.steps_done / arm.steps_total
            lines.append(
                f"    [{bar(arm.steps_done, arm.steps_total)}] "
                f"{arm.steps_done}/{arm.steps_total}  {pct:5.1f}%"
            )
            if arm.last_loss is not None:
                rate = arm.elapsed_s / max(arm.steps_done, 1)
                remaining = rate * (arm.steps_total - arm.steps_done)
                eta = f"  eta {mmss(remaining)}" if arm.state == "training" else ""
                lines.append(
                    f"    loss {arm.last_loss:.4f}  {sparkline(arm.losses)}  "
                    f"{arm.elapsed_s}s{eta}"
                )
        elif stale:
            elapsed = _etime_seconds(proc["etime"])
            if not reference:
                lines.append(
                    "    no finished arm to compare against — elapsed "
                    f"{mmss(elapsed)}, no estimate offered"
                )
            elif elapsed >= reference:
                # Past the training duration of a completed arm. Do not keep
                # inching a bar toward 100%: the arm has moved into the two
                # perplexity evaluations, which this log cannot see either.
                lines.append(
                    f"    [{bar(1, 1)}] past the training duration of a finished "
                    f"arm ({mmss(elapsed)} vs {mmss(reference)})"
                )
                if elapsed > 2 * reference:
                    # Elapsed time alone cannot separate "slow" from "wedged",
                    # and reading a silent log as either is guesswork. Say
                    # which one the CPU clock supports.
                    lines.append(
                        "    WELL OVER BUDGET — check the CPU rate below before "
                        "concluding anything;"
                    )
                    lines.append(
                        "    elapsed time cannot tell a swap-bound run from a "
                        "wedged one."
                    )
            else:
                lines.append(
                    f"    [{bar(elapsed, reference)}] "
                    f"~{int(100 * elapsed / reference)}% of the training phase "
                    f"by elapsed time  (INFERRED)"
                )
            lines.append(
                "    log silent: this arm started before stdout was line-buffered, "
                "so its"
            )
            lines.append(
                "    step lines land in one burst when the process exits."
            )

        if arm.drift is not None:
            ok = "" if arm.drift.startswith("0.00") else "   <-- MISSED THE LEDGER"
            lines.append(f"    ledger drift  {arm.drift}{ok}")
        if arm.train_ppl is not None:
            lines.append(f"    training fwd  {arm.train_ppl:.6f}")
        if arm.frozen_ppl is not None:
            lines.append(
                f"    frozen        {arm.frozen_ppl:.6f}   freezing cost "
                f"{arm.freeze_cost} nats"
            )
        if arm.storage:
            lines.append(f"    storage       {arm.storage}")
        if arm.bits:
            lines.append(f"    bits/weight   {arm.bits}")
        if arm.diverged_at is not None:
            lines.append(f"    DIVERGED at step {arm.diverged_at}")
        if arm.error:
            lines.append(f"    ERROR         {arm.error}")

        if proc:
            lines.append(
                f"    pid {proc['pid']}  elapsed {proc['etime']}  "
                f"cpu {proc['cpu_time']}  rss {proc['rss_mb']:.0f} MB"
            )
            # Two rates, because they disagree and the disagreement is the
            # diagnosis. A run starved for memory runs in bursts: sample it
            # during a burst and the instantaneous rate looks healthy while
            # the lifetime average says it has done a tenth of the work.
            # Reporting only the instantaneous figure once read "normal" for
            # an arm running at a tenth speed.
            lifetime = proc["cpu_seconds"] / max(_etime_seconds(proc["etime"]), 1)
            lines.append(f"    cpu, lifetime average {lifetime*100:.1f}%")
            rate = proc.get("cpu_rate")
            if rate is not None:
                lines.append(f"    cpu, sampled just now {rate*100:.1f}%")

            # These runs are GPU-bound; a healthy arm averages roughly 10%.
            if rate is not None and rate < 0.005 and lifetime < 0.02:
                verdict = "STALLED — the CPU clock is not advancing"
            elif reference and lifetime < 0.04:
                verdict = (
                    "STARVED — running at a fraction of the rate a finished arm "
                    "managed; check memory pressure, and consider killing and "
                    "rerunning alone (the pipeline is deterministic, so nothing "
                    "is lost)"
                )
            else:
                verdict = "healthy for a GPU-bound arm"
            lines.append(f"    verdict {verdict}")
        lines.append("")

    rows = checkpoint_rows()
    if rows:
        lines.append("checkpoints/")
        lines += rows
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--watch", action="store_true", help="refresh until interrupted")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--sample", type=float, default=3.0,
                   help="seconds to sample the CPU clock over (0 to skip)")
    args = p.parse_args()

    log = Path(args.log)
    if not args.watch:
        print(render(log, args.sample))
        return

    try:
        while True:
            print("\033[2J\033[H" + render(log, args.sample), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
