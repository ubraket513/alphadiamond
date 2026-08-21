"""Join self-play ledgers with sampler CSVs into one worker-scaling table.

``tools/az_report.py`` answers "was this configuration better"; this answers
"and what saturated".  It pairs each run's ledger record with the process-tree
and GPU samples taken during it, and reports throughput, batching, latency and
utilisation side by side, plus the per-worker and speedup columns needed to find
the knee rather than the maximum.

Samples are trimmed to the self-play window: every point runs exactly one
iteration, so the trailing optimizer steps and checkpoint writes would otherwise
drag GPU and parent CPU means toward a phase the sweep is not measuring.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ledger_record(run_dir: Path) -> dict | None:
    """The single iteration record for this point, or None if it never finished."""
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "iteration":
            records.append(record)
    return records[-1] if records else None


def sample_summary(run_dir: Path, *, selfplay_s: float | None) -> dict:
    """Utilisation over the self-play phase only."""
    path = run_dir / "samples.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {}

    def column(name: str, limit: float | None) -> list[float]:
        values = []
        for row in rows:
            raw = (row.get(name) or "").strip()
            if not raw:
                continue
            try:
                elapsed = float(row["t_s"])
            except (KeyError, ValueError):
                continue
            if limit is not None and elapsed > limit:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return values

    # Spawn and checkpoint load precede self-play; the iteration's own timer
    # starts at process start, so the window is [0, selfplay_s] plus slack.
    limit = None if selfplay_s is None else selfplay_s * 1.05

    parent = column("parent_cpu", limit)
    total = column("tree_cpu", limit)
    gpu = column("gpu_util", limit)
    worker_mean = column("worker_cpu_mean", limit)
    worker_max = column("worker_cpu_max", limit)
    return {
        "parent_cpu_mean": mean(parent),
        "parent_cpu_p90": percentile(parent, 0.9),
        "parent_cpu_max": max(parent) if parent else None,
        "worker_cpu_mean": mean(worker_mean),
        "worker_cpu_max": max(worker_max) if worker_max else None,
        "tree_cpu_mean": mean(total),
        "tree_cpu_p90": percentile(total, 0.9),
        "system_cpu_mean": mean(column("system_cpu", limit)),
        "gpu_util_mean": mean(gpu),
        "gpu_util_p50": percentile(gpu, 0.5),
        "gpu_util_p90": percentile(gpu, 0.9),
        "gpu_mem_used_mb": max(column("gpu_mem_used_mb", limit), default=None),
        "gpu_power_mean": mean(column("gpu_power_w", limit)),
        "tree_rss_mb_max": max(column("tree_rss_mb", limit), default=None),
        "mem_available_mb_min": min(column("mem_available_mb", limit), default=None),
        "parent_threads_max": max(column("parent_threads", limit), default=None),
        "children_max": max(column("children", limit), default=None),
        "ctx_vol": max(column("parent_ctx_vol", limit), default=None),
        "ctx_invol": max(column("parent_ctx_invol", limit), default=None),
    }


def summarize(run_dir: Path) -> dict | None:
    record = ledger_record(run_dir)
    if record is None:
        return None
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    inference = record.get("inference") or {}
    selfplay_s = record.get("selfplay_s")
    throughput = record.get("throughput") or {}

    # samples/hour over the self-play window, not the iteration wall: the
    # optimizer steps are held at the same ratio and are not under test.
    samples_per_hour = (
        record["samples_new"] * 3600.0 / selfplay_s if selfplay_s else None
    )
    return {
        "run": run_dir.name,
        "workers": record.get("worker_count"),
        "games": record.get("attempted"),
        "completed": record.get("completed"),
        "aborted": record.get("aborted"),
        "abort_reasons": record.get("abort_reasons") or {},
        "selfplay_s": selfplay_s,
        "train_s": record.get("train_s"),
        "samples_new": record.get("samples_new"),
        "samples_per_hour": samples_per_hour,
        "completed_games_per_hour": (
            record["completed"] * 3600.0 / selfplay_s if selfplay_s else None
        ),
        "median_moves": record.get("median_moves"),
        "p90_moves": record.get("p90_moves"),
        "max_wait_ms": (config.get("inference") or {}).get("max_wait_ms"),
        "max_batch_size_cfg": (config.get("inference") or {}).get("max_batch_size"),
        "simulations": record.get("simulations"),
        "requests": inference.get("requests_completed"),
        "batches": inference.get("batches_completed"),
        "mean_batch": inference.get("mean_batch_size"),
        "median_batch": inference.get("median_batch_size"),
        "p90_batch": inference.get("p90_batch_size"),
        "max_batch": inference.get("max_batch_size"),
        "qd_p50_ms": inference.get("queue_to_dispatch_p50_ms"),
        "qd_p90_ms": inference.get("queue_to_dispatch_p90_ms"),
        "inf_p50_ms": inference.get("inference_p50_ms"),
        "inf_p90_ms": inference.get("inference_p90_ms"),
        "resp_p50_ms": inference.get("response_p50_ms"),
        "resp_p90_ms": inference.get("response_p90_ms"),
        "evals_per_second": inference.get("evaluations_per_second"),
        "batches_per_second": inference.get("batches_per_second"),
        "throughput_samples_per_hour": throughput.get("samples_per_hour"),
        **sample_summary(run_dir, selfplay_s=selfplay_s),
    }


def fmt(value, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--baseline",
        default=None,
        help="run name whose samples/hour is the speedup denominator.",
    )
    args = parser.parse_args()

    summaries = [s for s in (summarize(path) for path in args.runs) if s]
    if not summaries:
        print("no completed iterations found")
        return 1

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0

    baseline = None
    if args.baseline:
        baseline = next((s for s in summaries if s["run"] == args.baseline), None)
    if baseline is None:
        baseline = min(summaries, key=lambda s: s["workers"] or 0)
    base_rate = baseline.get("samples_per_hour")

    width = max(10, *(len(s["run"]) for s in summaries))
    header = (
        f"{'run':<{width}} {'wrk':>4} {'gms':>4} {'wall':>7} {'samp/h':>8} {'eval/s':>7} "
        f"{'meanB':>6} {'p90B':>5} {'maxB':>5} {'qd50':>6} {'qd90':>7} {'rsp50':>6} "
        f"{'GPU%':>5} {'par%':>5} {'tot%':>6} {'done':>7}"
    )
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda s: (s["workers"] or 0, s["run"])):
        print(
            f"{s['run']:<{width}} "
            f"{fmt(s['workers'], '>4')} {fmt(s['games'], '>4')} "
            f"{fmt(s['selfplay_s'], '>7.1f')} "
            f"{fmt(s['samples_per_hour'], '>8.0f')} "
            f"{fmt(s['evals_per_second'], '>7.0f')} "
            f"{fmt(s['mean_batch'], '>6.2f')} "
            f"{fmt(s['p90_batch'], '>5.0f')} "
            f"{fmt(s['max_batch'], '>5')} "
            f"{fmt(s['qd_p50_ms'], '>6.2f')} "
            f"{fmt(s['qd_p90_ms'], '>7.2f')} "
            f"{fmt(s['resp_p50_ms'], '>6.2f')} "
            f"{fmt(s['gpu_util_mean'], '>5.1f')} "
            f"{fmt(s['parent_cpu_mean'], '>5.0f')} "
            f"{fmt(s['tree_cpu_mean'], '>6.0f')} "
            f"{s['completed']}/{s['games']:<3}"
        )

    print()
    scaling = (
        f"{'run':<{width}} {'wrk':>4} {'samp/h':>8} {'samp/h/wrk':>11} "
        f"{'eval/s/wrk':>11} {'speedup':>8} {'par.eff':>8} {'p90mv':>6} aborts"
    )
    print(scaling)
    print("-" * len(scaling))
    for s in sorted(summaries, key=lambda s: (s["workers"] or 0, s["run"])):
        workers = s["workers"] or 1
        rate = s.get("samples_per_hour")
        speedup = rate / base_rate if rate and base_rate else None
        # Efficiency relative to the baseline's per-worker rate, so a row that
        # only kept up by adding processes is visibly distinguishable from one
        # that genuinely scaled.
        efficiency = (
            speedup / (workers / (baseline["workers"] or 1)) if speedup else None
        )
        print(
            f"{s['run']:<{width}} {fmt(workers, '>4')} "
            f"{fmt(rate, '>8.0f')} "
            f"{fmt(rate / workers if rate else None, '>11.0f')} "
            f"{fmt(s['evals_per_second'] / workers if s['evals_per_second'] else None, '>11.1f')} "
            f"{fmt(speedup, '>8.2f')} "
            f"{fmt(efficiency, '>8.2f')} "
            f"{fmt(s['p90_moves'], '>6')} "
            f"{s['abort_reasons'] or '{}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
