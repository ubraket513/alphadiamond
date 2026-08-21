"""Lightweight process-tree and GPU sampler for one benchmark point.

Samples at a fixed interval and writes one CSV row per tick:

  * parent process CPU%          (the az_train loop: coordinator + bridge threads)
  * worker CPU% distribution     (mean / p90 / max / sum across children)
  * total process-tree CPU%
  * system-wide CPU%
  * GPU utilisation, VRAM, power, SM clock
  * parent + tree RSS, live child count, parent thread count
  * context switches (voluntary / involuntary) for the parent

Why a sampler rather than ``pidstat``: this image ships neither sysstat nor a
reliable ``pidstat``, and the previous GPU pass established that a *per-second*
sampler is enough to perturb a latency-bound run.  Default interval is therefore
2 s, and every reading comes from one ``psutil`` pass over the tree.

The GPU is read by one long-lived ``nvidia-smi`` query subprocess rather than a
fork per tick, so the sampler's own cost stays flat as worker counts rise.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; ``None`` when there is nothing to rank."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


class GpuSampler:
    """One persistent ``nvidia-smi -l`` reader, polled non-destructively.

    Holds the most recent line so a tick never blocks on the GPU query, and a
    stalled ``nvidia-smi`` degrades to stale values instead of stalling the
    sampler.
    """

    FIELDS = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"

    def __init__(self, interval_s: int) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={self.FIELDS}",
                "--format=csv,noheader,nounits",
                "-l",
                str(max(1, interval_s)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.latest: list[str] = ["", "", "", "", ""]

    def poll(self) -> list[str]:
        """Drain whatever nvidia-smi has produced, keeping only the last line."""
        import os
        import select

        stream = self.process.stdout
        if stream is None:
            return self.latest
        while True:
            ready, _, _ = select.select([stream], [], [], 0)
            if not ready:
                break
            line = stream.readline()
            if not line:
                break
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                self.latest = parts
        return self.latest

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


COLUMNS = [
    "t_s",
    "parent_cpu",
    "worker_cpu_mean",
    "worker_cpu_p90",
    "worker_cpu_max",
    "worker_cpu_sum",
    "tree_cpu",
    "system_cpu",
    "children",
    "parent_threads",
    "parent_rss_mb",
    "tree_rss_mb",
    "mem_available_mb",
    "parent_ctx_vol",
    "parent_ctx_invol",
    "gpu_util",
    "gpu_mem_util",
    "gpu_mem_used_mb",
    "gpu_power_w",
    "gpu_clock_sm",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="az_train parent PID.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    try:
        parent = psutil.Process(args.pid)
    except psutil.NoSuchProcess:
        print(f"[sampler] pid {args.pid} is already gone", file=sys.stderr)
        return 1

    gpu = GpuSampler(int(args.interval))
    # Prime the CPU-percent counters; the first reading of each is always 0.0.
    parent.cpu_percent(None)
    psutil.cpu_percent(None)
    known: dict[int, psutil.Process] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        while True:
            time.sleep(args.interval)
            if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                break
            try:
                # Children are primed on first sight, so a worker that spawns
                # mid-run reports 0.0 once rather than a bogus lifetime average.
                for child in parent.children(recursive=True):
                    if child.pid not in known:
                        try:
                            child.cpu_percent(None)
                        except psutil.Error:
                            continue
                        known[child.pid] = child

                worker_cpu: list[float] = []
                tree_rss = 0
                for pid, child in list(known.items()):
                    try:
                        worker_cpu.append(child.cpu_percent(None))
                        tree_rss += child.memory_info().rss
                    except psutil.Error:
                        known.pop(pid, None)

                parent_cpu = parent.cpu_percent(None)
                parent_rss = parent.memory_info().rss
                context = parent.num_ctx_switches()
                threads = parent.num_threads()
            except psutil.Error:
                break

            gpu_row = gpu.poll()
            writer.writerow(
                [
                    round(time.monotonic() - started, 1),
                    round(parent_cpu, 1),
                    round(sum(worker_cpu) / len(worker_cpu), 2) if worker_cpu else "",
                    round(percentile(worker_cpu, 0.9) or 0.0, 2) if worker_cpu else "",
                    round(max(worker_cpu), 2) if worker_cpu else "",
                    round(sum(worker_cpu), 1) if worker_cpu else "",
                    round(parent_cpu + sum(worker_cpu), 1),
                    round(psutil.cpu_percent(None), 1),
                    len(worker_cpu),
                    threads,
                    round(parent_rss / 1024**2, 1),
                    round((parent_rss + tree_rss) / 1024**2, 1),
                    round(psutil.virtual_memory().available / 1024**2, 1),
                    context.voluntary,
                    context.involuntary,
                    *gpu_row,
                ]
            )
            handle.flush()

    gpu.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
