"""Play every checkpoint against every other, and fit one rating scale.

``arena_head_to_head.py`` answers "is this one better than that one".  A sweep
asks a different question -- where do *several* checkpoints sit relative to each
other -- and answering it by reading a pile of pairwise logs invites the error
of chaining point estimates ("A beat B by 190, B beat C by 210, so A is 400
above C") when the pairs were played at different sample sizes.

This runs the full round robin on the same ``SooArena`` the pairwise tool uses,
then fits one Bradley-Terry rating to all results at once.

Two things about the fit, stated because both are easy to over-read:

* It is **regularised**.  A participant that wins or loses every game has an
  unbounded maximum-likelihood rating; a weak L2 prior pulls it to something
  finite and reportable.  A rating whose pair was a shutout is a lower bound on
  a gap, not a measurement of one -- the pairwise table says which those are.
* Ratings are **anchored** on one participant so the scale is readable.  Only
  differences mean anything; the anchor's 0 is a choice, not a result.

The search budget is part of the result, not a detail: §6.8 of the training
document found this network's play depends strongly on it, so a table is only
comparable to another table at the same ``--simulations``.

Usage::

    python tools/arena_round_robin.py --games 40 --simulations 128 \\
        --participant a0-final=/path/latest.pt \\
        --participant transition=/path/actor.pt \\
        --participant cpu-step80=runtime/runs/soo/cpu8h-soo-20260819/latest.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.arena import SooArena
from diamond.alphazero.checkpoint import checkpoint_network_config, load_checkpoint
from diamond.alphazero.config import (
    ArenaConfig,
    MCTSConfig,
    NetworkConfig,
    TrainingConfig,
)
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.game.state import build_players


def _load(path: Path, config: dict):
    """Build a trainer at the checkpoint's own shape; see arena_head_to_head."""
    network = checkpoint_network_config(path)
    version = config["model_version"]
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    trainer = AlphaZeroTrainer(
        SooModel(network, model_version=version),
        compatibility,
        TrainingConfig(**config["training"]),
    )
    load_checkpoint(path, trainer, expected=compatibility, allow_device_migration=True)
    evaluator = TorchEvaluator(
        trainer.model, value_size=1, device=config["training"]["device"]
    )
    return trainer, evaluator, network


def _wilson(wins: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _elo(win_rate: float) -> float | None:
    if win_rate <= 0.0 or win_rate >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def _bradley_terry(
    names: list[str],
    results: dict[tuple[str, str], tuple[int, int]],
    *,
    anchor: str,
    prior: float = 1.0,
    iterations: int = 10_000,
    step: float = 8.0,
) -> dict[str, float]:
    """Ratings in Elo points, by regularised gradient ascent on the BT likelihood.

    ``prior`` is the weight of an L2 pull toward the anchor, in units of games;
    it is what keeps a shutout finite.  Deliberately weak, so it moves a
    well-determined rating very little and a shutout a lot.
    """
    scale = math.log(10.0) / 400.0
    rating = {name: 0.0 for name in names}
    for _ in range(iterations):
        gradient = {name: 0.0 for name in names}
        for (a, b), (wins, losses) in results.items():
            total = wins + losses
            if total == 0:
                continue
            expected = 1.0 / (1.0 + math.exp(-scale * (rating[a] - rating[b])))
            residual = wins - total * expected
            gradient[a] += residual
            gradient[b] -= residual
        for name in names:
            gradient[name] -= prior * scale * rating[name]
            rating[name] += step * gradient[name] * scale
        offset = rating[anchor]
        for name in names:
            rating[name] -= offset
    return rating


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json"
    )
    parser.add_argument(
        "--participant",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat once per checkpoint; NAME labels it in the table.",
    )
    parser.add_argument("--games", type=int, default=40, help="per pair; multiple of 4")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--anchor",
        default=None,
        help="Participant rated 0; default is the last one given (usually the weakest).",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON result file")
    args = parser.parse_args()

    entries: list[tuple[str, Path]] = []
    for spec in args.participant:
        if "=" not in spec:
            raise SystemExit(f"--participant must be NAME=PATH, got {spec!r}")
        name, _, raw = spec.partition("=")
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"no such checkpoint: {path}")
        entries.append((name, path))
    if len(entries) < 2:
        raise SystemExit("a round robin needs at least two participants")
    names = [name for name, _ in entries]
    if len(set(names)) != len(names):
        raise SystemExit("participant names must be unique")
    anchor = args.anchor or names[-1]
    if anchor not in names:
        raise SystemExit(f"anchor {anchor!r} is not a participant")

    import torch

    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    simulations = args.simulations or config["mcts"]["simulations"]
    max_moves = args.max_moves or config["self_play"]["max_moves"]

    loaded = {}
    for name, path in entries:
        trainer, evaluator, network = _load(path, config)
        loaded[name] = (trainer, evaluator)
        print(
            f"[load] {name:<14} step={trainer.training_step:<7} "
            f"{network.width}x{network.residual_blocks}  {path}",
            flush=True,
        )

    mcts = MCTSConfig(
        simulations=simulations,
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        dirichlet_epsilon=config["mcts"]["dirichlet_epsilon"],
        seed=args.seed,
    )
    arena_config = ArenaConfig(
        games=args.games,
        seed=args.seed,
        max_moves=max_moves,
        promotion_threshold=config["arena"]["promotion_threshold"],
    )

    def game_factory(order: tuple[int, ...]) -> DiamondSearchAdapter:
        # SooArena crosses turn order with candidate seat for a 4-fold balanced
        # cycle; ignoring ``order`` collapses it to 2-fold and biases the result.
        return DiamondSearchAdapter(
            AlphaZeroGameAdapter(build_players(len(order), order=order))
        )

    print(
        f"[round-robin] {len(names)} participants, "
        f"{len(names) * (len(names) - 1) // 2} pairs x {args.games} games, "
        f"sims={simulations} max_moves={max_moves}",
        flush=True,
    )

    results: dict[tuple[str, str], tuple[int, int]] = {}
    rows = []
    for first, second in itertools.combinations(names, 2):
        started = time.perf_counter()
        outcome = SooArena(
            candidate=loaded[first][1],
            baseline=loaded[second][1],
            mcts_config=mcts,
            arena_config=arena_config,
        ).run(game_factory)
        elapsed = time.perf_counter() - started
        completed = outcome.wins + outcome.losses
        low, high = _wilson(outcome.wins, completed)
        elo = _elo(outcome.win_rate)
        results[(first, second)] = (outcome.wins, outcome.losses)
        verdict = (
            "STRONGER" if low > 0.5
            else "WEAKER" if high < 0.5
            else "not separated"
        )
        print(
            f"  {first:>14} vs {second:<14} "
            f"{outcome.wins:>3}W-{outcome.losses:<3}L "
            f"({outcome.aborted_games} aborted)  "
            f"{outcome.win_rate * 100:>5.1f}% "
            f"[{low * 100:.0f}-{high * 100:.0f}%]  "
            f"Elo {'unbounded' if elo is None else f'{elo:+.0f}':>9}  "
            f"{verdict}  {elapsed:.0f}s",
            flush=True,
        )
        rows.append(
            {
                "candidate": first,
                "baseline": second,
                "wins": outcome.wins,
                "losses": outcome.losses,
                "aborted": outcome.aborted_games,
                "win_rate": outcome.win_rate,
                "wilson_low": low,
                "wilson_high": high,
                "pairwise_elo": elo,
                "seconds": elapsed,
            }
        )

    rating = _bradley_terry(names, results, anchor=anchor)
    print(f"\n[ratings] regularised Bradley-Terry, anchored on {anchor} = 0")
    for name in sorted(names, key=lambda n: -rating[n]):
        step = loaded[name][0].training_step
        print(f"  {name:<16} {rating[name]:+8.0f}   (step {step})")
    if any(row["pairwise_elo"] is None for row in rows):
        print(
            "  note: at least one pair was a shutout, so its rating gap is a "
            "lower bound held finite by the prior, not a measurement."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "simulations": simulations,
                    "games_per_pair": args.games,
                    "max_moves": max_moves,
                    "seed": args.seed,
                    "anchor": anchor,
                    "participants": {
                        name: {
                            "path": str(path),
                            "training_step": loaded[name][0].training_step,
                        }
                        for name, path in entries
                    },
                    "pairs": rows,
                    "ratings": rating,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
