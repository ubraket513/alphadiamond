"""Summarize one CPU training run's ledger into the blueprint report format."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no ledger at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe", type=Path, default=None, help="Optional OFF-probe JSON.")
    args = parser.parse_args()

    records = load_ledger(args.run_dir / "ledger.jsonl")
    starts = [r for r in records if r["event"] == "run_start"]
    iterations = [r for r in records if r["event"] == "iteration"]
    state_path = args.run_dir / "loop_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    if not starts:
        raise SystemExit("ledger has no run_start record")
    first = starts[0]
    model = first["model"]
    phases = sorted({r.get("phase", "B0") for r in iterations}) or [first.get("phase", "B0")]

    attempted = sum(r["attempted"] for r in iterations)
    completed = sum(r["completed"] for r in iterations)
    aborted = sum(r["aborted"] for r in iterations)
    moves = [int(m) for m in state.get("move_counts", [])]
    elapsed = max((r["elapsed_s"] for r in iterations), default=0.0)
    last_metrics = next(
        (r["metrics"] for r in reversed(iterations) if r.get("metrics")), None
    )
    last_iteration = iterations[-1] if iterations else {}

    print(f"MODEL\n    {model}\n")
    print(f"MODE\n    {' -> '.join(phases)}\n")
    print(f"CPU TIME\n    {elapsed / 3600:.2f} h ({elapsed:.0f} s)\n")
    print("SEARCH")
    print(f"    simulations   {first['simulations']}")
    print(f"    worker count  {first['worker_count']}\n")
    print("SELF-PLAY")
    print(f"    attempted     {attempted}")
    print(f"    completed     {completed}")
    print(f"    aborted       {aborted}")
    print(f"    completion    {completed / attempted:.1%}" if attempted else "    completion    n/a")
    print(f"    median moves  {statistics.median(moves) if moves else None}")
    print(f"    p90 moves     {percentile(moves, 0.9)}")
    print(f"    abort reasons {state.get('abort_reasons') or '{}'}\n")
    print("REPLAY")
    print(f"    final samples     {last_iteration.get('replay_size')}")
    print(f"    samples generated {state.get('samples_generated')}\n")
    print("TRAINING")
    print(f"    iterations    {len(iterations)}")
    print(f"    training step {last_iteration.get('training_step')}")
    if last_metrics:
        print(f"    policy loss   {last_metrics['policy_loss']:.4f}")
        print(f"    value loss    {last_metrics['value_loss']:.4f}")
        print(f"    total loss    {last_metrics['total_loss']:.4f}")
    else:
        print("    losses        none (no training step executed)")
    print()
    print("CHECKPOINT")
    print(f"    latest        {args.run_dir / 'latest.pt'}")
    print(f"    archive       {last_iteration.get('checkpoint')}")
    print(f"    training step {last_iteration.get('training_step')}\n")

    print("HEURISTIC-OFF PROBE")
    if args.probe and args.probe.exists():
        for row in json.loads(args.probe.read_text(encoding="utf-8")):
            print(
                f"    sims={row['simulations']} "
                f"{row['completed']}/{row['episodes']} complete "
                f"({row['completion_rate']:.0%}) median={row['median_moves']} "
                f"aborts={row['abort_reasons'] or '{}'} => {row['verdict']}"
            )
    else:
        print("    not run")
    print()
    print("FINAL STATE")
    print(f"    heuristic {'OFF' if last_iteration.get('bootstrap_prior') == 'none' else 'ON'}"
          f" ({last_iteration.get('bootstrap_prior')})")
    print(f"    promotion/rating: {first.get('promotion_and_rating')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
