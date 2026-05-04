"""OOM watchdog for Phase 5 sweep — auto-restart on GPU memory creep or stalled progress.

Triggers a clean restart when ANY of:
  (1) GPU memory >= threshold (default 12 GB / 16 GB on RTX 4080)
  (2) Launcher PID died unexpectedly
  (3) cell count hasn't increased for stall_threshold seconds (default 25 min)
        — guards against the v7 bug where launcher silently re-ran existing cells
          because --skip-completed read a stale incomplete CSV.

Each restart:
  - SIGTERM launcher (SIGKILL fallback at 30s)
  - Wait for GPU memory to release (< 1 GB)
  - Regenerate _phase_summary_complete_<ts>.csv from ALL summary.json files
    so --skip-completed always sees the true latest state.
  - Re-spawn launcher with --skip-completed --continue-on-cell-failure

The watchdog itself logs to /tmp/oom_watchdog.log and stdout. Exit cleanly
on SIGTERM (does NOT kill the launcher on its own exit).

Usage:
    nohup /home/thc1006/dev/fl-oran-tmc/.venv/bin/python \\
        /home/thc1006/dev/fl-oran-tmc/scripts/oom_watchdog.py \\
        > /tmp/oom_watchdog.stdout.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/thc1006/dev/fl-oran-tmc")
ART = ROOT / "artifacts" / "v7_stage2_full"
SPEC = ROOT / "experiments" / "specs" / "stage2_full.yaml"
PARQ = Path(
    "/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet"
)
LOG_DIR = ROOT / "logs"
WATCHDOG_LOG = Path("/tmp/oom_watchdog.log")
PYTHON_VENV = ROOT / ".venv" / "bin" / "python"

DEFAULT_GPU_THRESHOLD_MB = 12 * 1024
DEFAULT_INTERVAL_S = 30
DEFAULT_STALL_S = 25 * 60          # 25 min without new cell → assume stalled
KILL_GRACEFUL_S = 30
GPU_CLEAR_TIMEOUT_S = 60
RESTART_GRACE_S = 30
TOTAL_CELLS = 900

_should_exit = False


def _on_term(signum, frame):
    """Exit cleanly on SIGTERM/SIGINT — do NOT kill the launcher."""
    global _should_exit
    _log(f"received signal {signum}, exiting watchdog (launcher untouched)")
    _should_exit = True


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with WATCHDOG_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _get_launcher_pid() -> int | None:
    """Return PID of run_v7_phase_sweep.py (python child, not bash wrapper)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_v7_phase_sweep.py"],
            stderr=subprocess.DEVNULL,
        )
        pids = [int(p) for p in out.decode().strip().split("\n") if p.strip()]
        if not pids:
            return None
        # Prefer python child (longer cmdline); pgrep -f matches bash wrapper too
        for pid in pids:
            try:
                with open(f"/proc/{pid}/comm") as f:
                    if f.read().strip() == "python":
                        return pid
            except (FileNotFoundError, ProcessLookupError):
                continue
        return pids[0]
    except subprocess.CalledProcessError:
        return None


def _detect_threshold_mb() -> int:
    """Auto-detect GPU memory threshold as 75% of total card capacity.

    Per GH#7: the previous hardcoded 12 GB / 16 GB default was tied to
    RTX 4080 and would silently fail on V100 (32 GB) or A100 (40/80 GB)
    — the threshold would never trigger because the absolute creep
    level scales with capacity. Detect via ``nvidia-smi
    --query-gpu=memory.total`` and use 75% as a hardware-portable
    default.

    Falls back to ``DEFAULT_GPU_THRESHOLD_MB`` (12 GB) if nvidia-smi is
    unavailable or errors, so the watchdog still runs rather than
    crashing on startup.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        total_mb = int(out.decode().strip().split("\n")[0])
        return int(total_mb * 0.75)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return DEFAULT_GPU_THRESHOLD_MB


def _get_gpu_mem_mb() -> int:
    """GPU 0 memory.used in MiB; -1 on error."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception as e:
        _log(f"GPU query failed: {e}")
        return -1


def _count_completed_cells() -> int:
    return sum(1 for _ in ART.glob("v7_*/summary.json"))


def _kill_launcher(pid: int) -> None:
    _log(f"SIGTERM → PID {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _log(f"PID {pid} already gone")
        return
    for i in range(KILL_GRACEFUL_S):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _log(f"PID {pid} exited after {i + 1}s")
            return
    _log(f"PID {pid} did not exit in {KILL_GRACEFUL_S}s, SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(3)
    except ProcessLookupError:
        pass


def _wait_gpu_clear(target_mb: int = 1024) -> bool:
    _log(f"waiting for GPU < {target_mb} MB")
    for i in range(GPU_CLEAR_TIMEOUT_S):
        mem = _get_gpu_mem_mb()
        if 0 <= mem < target_mb:
            _log(f"GPU cleared to {mem} MB after {i + 1}s")
            return True
        time.sleep(1)
    _log(f"GPU did not clear after {GPU_CLEAR_TIMEOUT_S}s (now {_get_gpu_mem_mb()} MB)")
    return False


def _regenerate_csv() -> Path:
    """Build a fresh _phase_summary_complete_<ts>.csv from every summary.json
    so --skip-completed picks it up as MOST RECENT."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ART / f"_phase_summary_complete_{ts}.csv"
    header = [
        "name", "arch", "algorithm", "partition_mode", "alpha", "n_clients",
        "seed", "lr", "lr_warmup_rounds", "test_auc", "test_acc", "test_f1",
        "best_val_auc", "duration_s", "status",
    ]
    rows = []
    for d in sorted(ART.glob("v7_*")):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            s = json.loads(sj.read_text())
        except Exception as e:
            _log(f"skip malformed summary {d.name}: {e}")
            continue
        cfg = s.get("config", {}) or {}
        test = s.get("test", {}) or {}
        pt = s.get("phase_timings_s", {}) or {}
        rows.append({
            "name": d.name,
            "arch": cfg.get("arch", ""),
            "algorithm": cfg.get("algorithm", ""),
            "partition_mode": cfg.get("partition_mode", ""),
            "alpha": (cfg.get("alpha", "") if cfg.get("partition_mode") == "dirichlet"
                      else ""),
            "n_clients": cfg.get("n_clients", 7),
            "seed": cfg.get("seed", 0),
            "lr": cfg.get("lr", 5e-4),
            "lr_warmup_rounds": cfg.get("lr_warmup_rounds", 3),
            "test_auc": test.get("auc", 0),
            "test_acc": test.get("accuracy", 0),
            "test_f1": test.get("f1", 0),
            "best_val_auc": s.get("best_val_auc", 0),
            "duration_s": round(pt.get("TOTAL", 0), 2),
            "status": "ok",
        })
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    _log(f"regenerated CSV: {out_csv.name} ({len(rows)} cells)")
    return out_csv


def _start_launcher() -> int | None:
    """Spawn launcher in background with nohup, return python PID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase5_watchdog_{ts}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd_str = (
        f"source {ROOT}/.venv/bin/activate && "
        f"python {ROOT}/experiments/run_v7_phase_sweep.py "
        f"--spec {SPEC} "
        f"--output-dir {ART} "
        f"--unified-parquet {PARQ} "
        f"--skip-completed "
        f"--continue-on-cell-failure"
    )
    with log_path.open("w") as logf:
        # start_new_session=True already creates a new session+process group;
        # preexec_fn=os.setpgrp on top can raise SubprocessError. Use one only.
        proc = subprocess.Popen(
            ["bash", "-c", cmd_str],
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _log(f"started launcher (bash PID {proc.pid}) → {log_path.name}")
    # Wait a few seconds then find python child
    time.sleep(8)
    pid = _get_launcher_pid()
    if pid is None:
        _log("ERROR: launcher python child not found after 8s")
        return None
    _log(f"launcher python PID = {pid}")
    return pid


def _restart_cycle(pid: int | None, reason: str) -> int | None:
    _log(f"=== RESTART triggered: {reason} ===")
    if pid is not None:
        _kill_launcher(pid)
    _wait_gpu_clear()
    _regenerate_csv()
    new_pid = _start_launcher()
    if new_pid is None:
        return None
    time.sleep(RESTART_GRACE_S)
    return new_pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-mb", type=int, default=None,
                    help="GPU memory threshold (MB) to trigger restart. "
                         "Default: 75%% of detected total via nvidia-smi "
                         "(hardware-portable per GH#7).")
    ap.add_argument("--interval-s", type=int, default=DEFAULT_INTERVAL_S)
    ap.add_argument("--stall-s", type=int, default=DEFAULT_STALL_S)
    args = ap.parse_args()
    if args.threshold_mb is None:
        args.threshold_mb = _detect_threshold_mb()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    _log(f"=== watchdog start | threshold={args.threshold_mb}MB "
         f"interval={args.interval_s}s stall={args.stall_s}s ===")

    pid = _get_launcher_pid()
    if pid is None:
        _log("no launcher running; starting one")
        pid = _restart_cycle(None, "no launcher at watchdog start")
        if pid is None:
            _log("FATAL: cannot start launcher")
            return 1
    else:
        _log(f"attached to existing launcher PID {pid}")

    last_cell_count = _count_completed_cells()
    last_progress_at = time.time()
    restart_count = 0

    while not _should_exit:
        time.sleep(args.interval_s)
        now = time.time()

        # Check launcher liveness
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid = _restart_cycle(None, f"launcher PID {pid} died")
            if pid is None:
                return 1
            restart_count += 1
            last_cell_count = _count_completed_cells()
            last_progress_at = time.time()
            continue

        # Check sweep complete
        cells = _count_completed_cells()
        if cells >= TOTAL_CELLS:
            _log(f"=== SWEEP COMPLETE ({cells} cells) ===")
            return 0

        # Update progress tracker
        if cells > last_cell_count:
            last_cell_count = cells
            last_progress_at = now

        # Check stall
        stalled_s = now - last_progress_at
        if stalled_s >= args.stall_s:
            pid = _restart_cycle(
                pid,
                f"stall detected: cells={cells} unchanged for {stalled_s:.0f}s"
            )
            if pid is None:
                return 1
            restart_count += 1
            last_cell_count = _count_completed_cells()
            last_progress_at = time.time()
            continue

        # Check GPU memory
        mem = _get_gpu_mem_mb()
        if mem < 0:
            continue
        if mem >= args.threshold_mb:
            pid = _restart_cycle(
                pid, f"GPU {mem} MB >= {args.threshold_mb} MB"
            )
            if pid is None:
                return 1
            restart_count += 1
            last_cell_count = _count_completed_cells()
            last_progress_at = time.time()
            continue

        # Heartbeat (sparse)
        if cells % 10 == 0 and cells != last_cell_count:
            _log(f"heartbeat: cells={cells} GPU={mem}MB stall={stalled_s:.0f}s "
                 f"restarts={restart_count}")

    _log(f"=== watchdog exiting cleanly | restarts done = {restart_count} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
