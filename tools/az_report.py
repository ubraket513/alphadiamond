"""Summarize AlphaZero training ledgers for CPU-vs-GPU comparison.

Reads one or more run directories and prints the throughput, game-quality and
inference-batching numbers that decide whether a configuration is actually
better, rather than merely faster per step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iterations(run_root: Path) -> list[dict]:
    ledger = run_root / "ledger.jsonl"
    if not ledger.exists():
        return []
    records = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "iteration":
            records.append(record)
    return records


def summarize(run_root: Path) -> dict | None:
    records = iterations(run_root)
    if not records:
        return None
    attempted = sum(r["attempted"] for r in records)
    completed = sum(r["completed"] for r in records)
    samples = sum(r["samples_new"] for r in records)
    selfplay_s = sum(r["selfplay_s"] for r in records)
    train_s = sum(r.get("train_s", 0.0) for r in records)

    aborts: dict[str, int] = {}
    for record in records:
        for reason, count in (record.get("abort_reasons") or {}).items():
            aborts[reason] = aborts.get(reason, 0) + count

    # Inference metrics only exist on GPU-era records; average across iterations.
    inference = [r["inference"] for r in records if r.get("inference")]

    def mean(key: str) -> float | None:
        values = [i[key] for i in inference if i.get(key) is not None]
        return sum(values) / len(values) if values else None

    medians = [r["median_moves"] for r in records if r.get("median_moves") is not None]
    p90s = [r["p90_moves"] for r in records if r.get("p90_moves") is not None]

    return {
        "run": run_root.name,
        "iterations": len(records),
        "attempted": attempted,
        "completed": completed,
        "aborted": attempted - completed,
        "abort_reasons": aborts,
        "sec_per_game": selfplay_s / attempted if attempted else None,
        "sec_per_completed_game": selfplay_s / completed if completed else None,
        "completed_games_per_hour": completed * 3600.0 / selfplay_s if selfplay_s else None,
        "samples_per_hour": samples * 3600.0 / selfplay_s if selfplay_s else None,
        "selfplay_s": selfplay_s,
        "train_s": train_s,
        "median_moves": sorted(medians)[len(medians) // 2] if medians else None,
        "p90_moves": max(p90s) if p90s else None,
        "training_step": records[-1]["training_step"],
        "simulations": records[-1].get("simulations"),
        "workers": records[-1].get("worker_count"),
        "loss": (records[-1].get("metrics") or {}).get("total_loss"),
        "mean_batch_size": mean("mean_batch_size"),
        "max_batch_size": max((i["max_batch_size"] for i in inference), default=None),
        "inference_p50_ms": mean("inference_p50_ms"),
        "queue_p50_ms": mean("queue_to_dispatch_p50_ms"),
        "evals_per_second": mean("evaluations_per_second"),
        "requests": sum(i["requests_completed"] for i in inference) or None,
    }


def _format(value, spec: str = "") -> str:
    if value is None:
        return "-"
    return format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories.")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead.")
    args = parser.parse_args()

    summaries = [s for s in (summarize(path) for path in args.runs) if s]
    if not summaries:
        print("no completed iterations found")
        return 1

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0

    width = max(12, *(len(s["run"]) for s in summaries))
    header = (
        f"{'run':<{width}} {'sims':>4} {'wrk':>4} {'games':>9} {'s/game':>8} "
        f"{'compl/h':>8} {'samp/h':>9} {'p90mv':>6} {'batch':>6} "
        f"{'inf_ms':>7} {'evals/s':>8}  aborts"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        games = f"{s['completed']}/{s['attempted']}"
        print(
            f"{s['run']:<{width}} "
            f"{_format(s['simulations'], '>4')} "
            f"{_format(s['workers'], '>4')} "
            f"{games:>9} "
            f"{_format(s['sec_per_game'], '>8.1f')} "
            f"{_format(s['completed_games_per_hour'], '>8.1f')} "
            f"{_format(s['samples_per_hour'], '>9.0f')} "
            f"{_format(s['p90_moves'], '>6')} "
            f"{_format(s['mean_batch_size'], '>6.2f')} "
            f"{_format(s['inference_p50_ms'], '>7.2f')} "
            f"{_format(s['evals_per_second'], '>8.0f')}"
            f"  {s['abort_reasons'] or '{}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
