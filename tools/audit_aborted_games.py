"""What is actually happening in the games A0 fails to finish?

The abort tail is the whole problem — 28.6 % of every move played at 64
simulations is thrown away, and it is thrown away *because* those games never
reach a terminal state.  Before deciding whether the fix is more search, a
teacher, or a change to the training targets, it is worth knowing what the
failure looks like.  "Wandering" has been an assumption, not an observation.

Three shapes to distinguish, which want different fixes:

**Cycling.**  The game revisits a small set of positions forever.  A pure
repetition attractor; more search escapes it only if the search is deep enough
to see past the cycle, and a teacher escapes it immediately.

**Stalling.**  Positions keep changing but progress toward the target camp does
not.  Shuffling in place.  More search plausibly helps.

**Slow progress.**  Progress is real but too slow for the cap.  Then the cap is
the problem, and §5.7 already ruled that out — 500 → 750 changed nothing.

The data needed for this already exists and is already being thrown away. The
native ``Episode`` retains every ``EpisodeMove`` for aborted games; only
``NativeSelfPlayPool`` drops them, because it has no terminal outcome to label
them with. This reads the raw episodes instead.

Position identity comes from the encoded root features, which are the canonical
player-relative encoding of the position — so two moves with equal features are
the same position from the same side's perspective, which is exactly the
repetition notion that matters.

Usage::

    python tools/audit_aborted_games.py --checkpoint .../latest.pt --games 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import load_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.native import native_game, require_native
from diamond.alphazero.native.backend import policy_value_callback
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.game.state import build_players


def _position_key(move) -> str:
    """Canonical position identity, from the features the network was shown."""
    return hashlib.blake2b(move["node_features"].tobytes(), digest_size=8).hexdigest()


def _cycle_lengths(keys: list[str]) -> Counter:
    """Distance between consecutive revisits of the same position.

    A two-ply shuffle shows up as a mass of gaps at 2, a four-ply rotation at 4,
    and genuinely slow progress as no concentration at all.
    """
    last_seen: dict[str, int] = {}
    gaps: Counter = Counter()
    for index, key in enumerate(keys):
        if key in last_seen:
            gaps[index - last_seen[key]] += 1
        last_seen[key] = index
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--lanes", type=int, default=128)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
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
    load_checkpoint(args.checkpoint, trainer, expected=compatibility, allow_device_migration=True)

    module = require_native()
    players = build_players(2)
    native = native_game(players)
    state = AlphaZeroGameAdapter(players).initial_state()
    opening = module.State(
        occupancy=list(state.occupancy),
        current_player=state.current_player_id,
        turn_number=state.turn_number,
        status=0,
        finish_order=[],
    )
    simulations = args.simulations or config["mcts"]["simulations"]
    cfg = module.EpisodeConfig(
        lanes=args.lanes,
        threads=args.threads,
        max_batch=config["inference"]["max_batch_size"],
        max_wait_us=int(config["workers"].get("native_max_wait_us", 500)),
        simulations=simulations,
        max_moves=config["self_play"]["max_moves"],
        temperature=config["self_play"]["temperature"],
        temperature_moves=config["self_play"]["temperature_moves"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        dirichlet_epsilon=config["mcts"]["dirichlet_epsilon"],
    )
    jobs = [(opening, args.seed + index) for index in range(args.games)]

    print(
        f"[audit] step={trainer.training_step} sims={simulations} games={args.games} "
        f"max_moves={cfg.max_moves if hasattr(cfg, 'max_moves') else config['self_play']['max_moves']}",
        flush=True,
    )
    callback = policy_value_callback(trainer.model, device=config["training"]["device"])
    result = native.play_episodes(jobs, cfg, callback, "policy_value")

    aborted = [e for e in result["episodes"] if not e["completed"]]
    completed = [e for e in result["episodes"] if e["completed"]]
    print(
        f"  {len(completed)}/{len(result['episodes'])} completed; "
        f"auditing {len(aborted)} aborted games",
        flush=True,
    )
    if not aborted:
        print("  no aborted games to audit")
        return 0

    rows = []
    for episode in aborted:
        keys = [_position_key(m) for m in episode["moves"]]
        counts = Counter(keys)
        gaps = _cycle_lengths(keys)
        short_cycles = sum(count for gap, count in gaps.items() if gap in (2, 4, 6, 8))
        rows.append(
            {
                "moves": len(keys),
                "unique": len(counts),
                "unique_fraction": len(counts) / len(keys) if keys else 0.0,
                "max_revisits": max(counts.values()),
                "short_cycle_returns": short_cycles,
                "short_cycle_fraction": short_cycles / len(keys) if keys else 0.0,
            }
        )

    def summarise(name: str, values) -> None:
        values = list(values)
        ordered = sorted(values)
        print(
            f"  {name:22s} median={statistics.median(values):.3f} "
            f"min={ordered[0]:.3f} max={ordered[-1]:.3f}"
        )

    summarise("unique / moves", (r["unique_fraction"] for r in rows))
    summarise("max revisits", (float(r["max_revisits"]) for r in rows))
    summarise("short-cycle fraction", (r["short_cycle_fraction"] for r in rows))

    # The interpretation, stated rather than left to the reader.
    median_unique = statistics.median(r["unique_fraction"] for r in rows)
    median_cycle = statistics.median(r["short_cycle_fraction"] for r in rows)
    if median_unique < 0.25:
        verdict = "CYCLING - the tail revisits a small set of positions"
    elif median_cycle > 0.20:
        verdict = "SHORT-CYCLE SHUFFLE - distinct positions, but returning every few ply"
    elif median_unique > 0.8:
        verdict = "SLOW PROGRESS - positions keep changing; the cap is the binding limit"
    else:
        verdict = "MIXED - no single shape dominates"
    print(f"  verdict: {verdict}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "training_step": trainer.training_step,
                    "simulations": simulations,
                    "games": args.games,
                    "completed": len(completed),
                    "aborted": len(aborted),
                    "per_game": rows,
                    "verdict": verdict,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
