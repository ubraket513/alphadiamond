"""Short CPU search-budget calibration for the 8-hour bootstrap session.

Measures wall-clock cost and data yield of fixed-seed bootstrap self-play at a
few MCTS simulation budgets, using the real Torch evaluator so the numbers
reflect actual training-time cost.  Reports only what the repository already
exposes; asserts no thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diamond.alphazero.bootstrap.probe import run_probe
from diamond.alphazero.config import (
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
    NetworkConfig,
)
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.identity import MIN_MODEL_NAME, SOO_MODEL_NAME
from diamond.alphazero.network import MinModel, SooModel


def build_evaluator(model_name: str, width: int, blocks: int, seed: int) -> TorchEvaluator:
    torch.manual_seed(seed)
    config = NetworkConfig(width=width, residual_blocks=blocks)
    if model_name == SOO_MODEL_NAME:
        return TorchEvaluator(SooModel(config), value_size=1, device="cpu")
    return TorchEvaluator(MinModel(config), value_size=3, device="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=f"{SOO_MODEL_NAME},{MIN_MODEL_NAME}")
    parser.add_argument("--simulations", default="1,8,16")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=2000)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--prior", default=CANONICAL_TARGET_VACANCY_DISTANCE_V2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rows: list[dict[str, object]] = []

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        evaluator = build_evaluator(model_name, args.width, args.blocks, seed=7)
        for simulations in [int(s) for s in args.simulations.split(",") if s.strip()]:
            start = time.perf_counter()
            report = run_probe(
                model_name=model_name,
                bootstrap_prior=args.prior,
                episodes=args.episodes,
                simulations=simulations,
                max_moves=args.max_moves,
                base_seed=args.base_seed,
                base_evaluator=evaluator,
            )
            elapsed = time.perf_counter() - start
            moves = sum(report.move_counts)
            row = {
                "model": model_name,
                "simulations": simulations,
                "prior": args.prior,
                "episodes": report.episodes,
                "completed": report.completed,
                "completion_rate": round(report.completion_rate, 4),
                "median_moves": report.median_moves,
                "samples": report.samples,
                "samples_per_episode": round(report.samples_per_episode, 2),
                "abort_reasons": report.abort_reasons,
                "elapsed_s": round(elapsed, 2),
                "sec_per_game": round(elapsed / report.episodes, 2),
                # One MCTS search runs per move; each search issues about
                # `simulations` evaluations.  Only completed games contribute moves.
                "evals_per_s": round(moves * simulations / elapsed, 1) if elapsed else None,
                "threads": args.threads,
            }
            rows.append(row)
            print(
                f"{model_name:4s} sims={simulations:<4d} "
                f"completion={report.completion_rate:6.1%} "
                f"median_moves={report.median_moves!s:>7s} "
                f"sec/game={row['sec_per_game']:>8.2f} "
                f"samples/ep={report.samples_per_episode:8.1f} "
                f"evals/s={row['evals_per_s']!s:>7s} "
                f"aborts={report.abort_reasons or '{}'}",
                flush=True,
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
