"""The heuristics-off (A0) readiness gate, run on the production engine.

Answers one question: **can this network sustain production self-play without
the bootstrap prior?**  Not "can it finish a game if it plays its best move
every time" -- that is a different and much easier question, and answering it
instead is what caused the failed A0 switch at step 14,250.

Two deliberate choices, both paid for.

**It runs the native backend, not the Python self-play runner.**  The gate exists
to clear a network for the native path, so it runs the native path.  Clearing
one implementation by measuring another reintroduces exactly the class of
mismatch this gate was written in response to -- and as a side effect a 40-game
gate costs seconds rather than fourteen minutes, which is what makes it
affordable every few iterations.

**It uses production's exploration in full**, read from the run configuration:
``dirichlet_epsilon``, ``dirichlet_alpha``, ``temperature``,
``temperature_moves`` and ``simulations``.  The previous gate matched all of
these except ``temperature_moves``, taking the argmax of the visit counts where
production samples its first 20 moves from the distribution.  On the same
checkpoint that read 100 % where production self-play managed 64 %.

The thresholds are stricter than the blueprint's original 8-of-10, because the
A0 attempt showed a first-iteration reading is not safe to trust: completion ran
90.5 % -> 84.9 % -> 64.5 % -> 64.2 % over four iterations while training loss
*improved* from 3.06 to 2.40.  A switch gate needs margin against that.

Criteria, all of which must hold:

* aggregate completion across seeds >= 90 %
* every individual seed >= 80 %
* p90 move count <= half of ``max_moves``

Results append to a JSONL time series so the trajectory is visible, which is the
actual decision input: a gate that climbs 55 -> 65 -> 75 -> 90 % says wait, and
one that sits at 55 % for twenty iterations says something else is wrong.

Usage::

    python tools/a0_gate.py --checkpoint .../latest.pt
    python tools/a0_gate.py --checkpoint .../latest.pt --max-moves 1000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import load_checkpoint
from diamond.alphazero.config import (
    BOOTSTRAP_PRIOR_NONE,
    MCTSConfig,
    NetworkConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.native.selfplay_pool import NativeSelfPlayPool
from diamond.alphazero.network import SooModel
from diamond.alphazero.orchestration.selfplay_workers import SelfPlayJob
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.game.state import build_players

SEEDS = (12345, 67890)
"""Fixed, so successive runs are comparable. A gate whose seed set moves cannot
produce a trajectory, only a sequence of unrelated readings."""

AGGREGATE_COMPLETION = 0.90
PER_SEED_COMPLETION = 0.80
P90_FRACTION_OF_CAP = 0.5


def _report(label: str, episodes) -> dict:
    completed = [e for e in episodes if e.completed]
    moves = sorted(e.move_count for e in completed)
    aborts: dict[str, int] = {}
    for episode in episodes:
        if not episode.completed:
            reason = episode.aborted_reason or "unknown"
            aborts[reason] = aborts.get(reason, 0) + 1

    # The primary metric, not the completion rate.  Completion counts *games*
    # while training consumes *positions*, and an aborted game is long by
    # construction: at 98 % completion with 75-move games and a 500-move cap,
    # a 2 % abort rate still censors ~12 % of the moves played.  Measured from
    # the episodes themselves rather than assumed from the cap.
    kept = sum(e.move_count for e in completed)
    discarded = sum(e.move_count for e in episodes if not e.completed)
    played = kept + discarded
    return {
        "moves_kept": kept,
        "moves_discarded": discarded,
        "discarded_fraction": discarded / played if played else 0.0,
        "label": label,
        "games": len(episodes),
        "completed": len(completed),
        "completion": len(completed) / len(episodes) if episodes else 0.0,
        "median_moves": statistics.median(moves) if moves else None,
        "p90_moves": moves[min(int(0.9 * len(moves)), len(moves) - 1)] if moves else None,
        "p99_moves": moves[min(int(0.99 * len(moves)), len(moves) - 1)] if moves else None,
        "max_moves_seen": moves[-1] if moves else None,
        "samples": sum(len(e.samples) for e in completed),
        "aborts": aborts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20, help="games per seed")
    parser.add_argument("--max-moves", type=int, default=None, help="default: the config's")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help=(
            "Override the search budget. 64 is the project's reference point "
            "rather than a measured optimum, and few-simulation AlphaZero is "
            "exactly the regime where root search stops reliably improving on "
            "the network prior. Sweep this ALONE -- changing epsilon or "
            "temperature at the same time makes the result uninterpretable."
        ),
    )
    parser.add_argument("--prior", default=BOOTSTRAP_PRIOR_NONE)
    parser.add_argument(
        "--simulations-late",
        type=int,
        default=0,
        help="Search budget from --late-move-threshold onward; 0 disables adaptive search.",
    )
    parser.add_argument("--late-move-threshold", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None, help="JSONL time series to append to")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the fixed seed set. The default two exist so successive "
            "readings form a trajectory; a shadow run wants fresh seeds and a "
            "large sample instead, because 40 games put a 39/40 reading "
            "somewhere in a 87-99 % Wilson interval and cannot resolve a 2 % "
            "abort rate at all."
        ),
    )
    args = parser.parse_args()


    config = json.loads(args.config.read_text(encoding="utf-8"))
    network = NetworkConfig(**config["network"])
    version = config["model_version"]
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    trainer = AlphaZeroTrainer(
        SooModel(network, model_version=version),
        compatibility,
        TrainingConfig(**config["training"]),
    )
    load_checkpoint(
        args.checkpoint, trainer, expected=compatibility, allow_device_migration=True
    )
    device = config["training"]["device"]
    max_moves = args.max_moves if args.max_moves is not None else config["self_play"]["max_moves"]

    mcts = MCTSConfig(
        simulations=args.simulations or config["mcts"]["simulations"],
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        dirichlet_epsilon=config["mcts"]["dirichlet_epsilon"],
        seed=0,
    )
    selfplay = SelfPlayConfig(
        max_moves=max_moves,
        temperature=config["self_play"]["temperature"],
        temperature_moves=config["self_play"]["temperature_moves"],
        seed=0,
        bootstrap_prior=args.prior,
    )
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    players = build_players(2)
    initial_state = AlphaZeroGameAdapter(players).initial_state()
    model_key = ModelKey(
        model_name="Soo", model_version=version, checkpoint_sha256="0" * 64
    )

    print(
        f"[gate] step={trainer.training_step} prior={args.prior} "
        f"sims={mcts.simulations} eps={mcts.dirichlet_epsilon} "
        f"T={selfplay.temperature}/{selfplay.temperature_moves} max_moves={max_moves} "
        + (
            f"adaptive={args.simulations_late}@{args.late_move_threshold} "
            if args.simulations_late
            else ""
        )
        + 
        f"seeds={seeds} games/seed={args.games}",
        flush=True,
    )

    pool = NativeSelfPlayPool(
        trainer.model,
        device=device,
        lanes=args.lanes,
        threads=args.threads,
        max_batch=config["inference"]["max_batch_size"],
        max_wait_us=int(config["workers"].get("native_max_wait_us", 500)),
        simulations_late=args.simulations_late,
        late_move_threshold=args.late_move_threshold,
    )

    per_seed = []
    all_episodes = []
    started = time.perf_counter()
    for seed in seeds:
        jobs = tuple(
            SelfPlayJob(
                run_seed=seed,
                iteration=0,
                game_index=index,
                retry_id="gate",
                model_key=model_key,
                compatibility=compatibility,
                players=players,
                initial_state=initial_state,
                mcts_config=mcts,
                selfplay_config=selfplay,
            )
            for index in range(args.games)
        )
        episodes = pool.run(jobs)
        all_episodes.extend(episodes)
        row = _report(f"seed={seed}", episodes)
        per_seed.append(row)
        print(
            f"  seed {seed}: {row['completed']}/{row['games']} "
            f"({row['completion'] * 100:.0f}%) median={row['median_moves']} "
            f"p90={row['p90_moves']} p99={row['p99_moves']} aborts={row['aborts']}",
            flush=True,
        )

    aggregate = _report("aggregate", all_episodes)
    elapsed = time.perf_counter() - started

    p90 = aggregate["p90_moves"]
    checks = {
        "aggregate_completion": aggregate["completion"] >= AGGREGATE_COMPLETION,
        "per_seed_completion": all(r["completion"] >= PER_SEED_COMPLETION for r in per_seed),
        "p90_clear_of_cap": p90 is not None and p90 <= P90_FRACTION_OF_CAP * max_moves,
    }
    passed = all(checks.values())

    print(
        f"  aggregate: {aggregate['completed']}/{aggregate['games']} "
        f"({aggregate['completion'] * 100:.1f}%) median={aggregate['median_moves']} "
        f"p90={p90} p99={aggregate['p99_moves']} max={aggregate['max_moves_seen']} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    print(
        f"  censoring: {aggregate['moves_kept']:,} moves kept, "
        f"{aggregate['moves_discarded']:,} discarded -> "
        f"**{aggregate['discarded_fraction'] * 100:.1f}% of all moves played thrown away**",
        flush=True,
    )
    # Terminal-labelled throughput: what the run actually gets to train on per
    # second, which is the quantity a bigger search budget has to justify.
    print(
        f"  yield: {aggregate['samples']:,} terminal-labelled samples in "
        f"{elapsed:.1f}s = {aggregate['samples'] / elapsed:,.0f}/s",
        flush=True,
    )
    for name, ok in checks.items():
        print(f"    {'ok  ' if ok else 'FAIL'} {name}")
    print(f"[gate] {'PASS' if passed else 'FAIL'}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "captured_at": datetime.now(UTC).isoformat(),
                        "training_step": trainer.training_step,
                        "prior": args.prior,
                        "max_moves": max_moves,
                        "simulations": mcts.simulations,
                        "dirichlet_epsilon": mcts.dirichlet_epsilon,
                        "temperature": selfplay.temperature,
                        "temperature_moves": selfplay.temperature_moves,
                        "seeds": list(seeds),
                        "games_per_seed": args.games,
                        "per_seed": per_seed,
                        "aggregate": aggregate,
                        "checks": checks,
                        "passed": passed,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
