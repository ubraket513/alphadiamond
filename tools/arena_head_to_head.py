"""Is the training actually making the network stronger?

The A0 gate answers a *health* question — can this network finish its games — and
health is what the last several days of investigation turned on.  It is not
strength.  A network could hold 98 % completion while playing no better than the
checkpoint it started from, and nothing measured so far would notice.

This plays two checkpoints against each other under the existing ``SooArena``,
which alternates colours across a balanced matchup cycle so the result is not an
artefact of who moves first.  Evaluation never inherits training root noise —
``SooArena`` zeroes ``dirichlet_epsilon`` itself — and temperature is 0, so the
games are deterministic given the seed and the two networks.

The search budget is a parameter and it matters. §6.8 of the training document
found that this network's play depends strongly on it, so a strength comparison
is only meaningful at a stated budget; running it at 64 and at 128 can give
different answers and both are true.

Usage::

    python tools/arena_head_to_head.py --candidate new.pt --baseline old.pt
    python tools/arena_head_to_head.py --candidate new.pt --baseline old.pt \
        --games 40 --simulations 128
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.arena import SooArena
from diamond.alphazero.checkpoint import load_checkpoint
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


def _load(path: Path, config: dict) -> tuple[AlphaZeroTrainer, TorchEvaluator]:
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
    load_checkpoint(path, trainer, expected=compatibility, allow_device_migration=True)
    evaluator = TorchEvaluator(
        trainer.model, value_size=1, device=config["training"]["device"]
    )
    return trainer, evaluator


def _elo(win_rate: float) -> float | None:
    """Elo difference implied by a win rate, or None when it is unbounded."""
    if win_rate <= 0.0 or win_rate >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def _wilson(wins: int, total: int) -> tuple[float, float]:
    """95 % interval, because a 40-game arena is a small sample.

    A 24/40 result is 60 % and its interval still contains 50 %: encouraging, not
    significant. Reporting the point estimate alone invites exactly the
    over-reading that a fixed 40-game gate already caused once.
    """
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40, help="must be a multiple of 4")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidate_trainer, candidate = _load(args.candidate, config)
    baseline_trainer, baseline = _load(args.baseline, config)

    simulations = args.simulations or config["mcts"]["simulations"]
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
        max_moves=args.max_moves or config["self_play"]["max_moves"],
        promotion_threshold=config["arena"]["promotion_threshold"],
    )
    def game_factory(order: tuple[int, ...]) -> DiamondSearchAdapter:
        # ``order`` must be honoured, not ignored. SooArena crosses every turn
        # order with every candidate seat to get a 4-fold balanced cycle;
        # dropping it collapses that to 2-fold and quietly biases the result.
        return DiamondSearchAdapter(
            AlphaZeroGameAdapter(build_players(len(order), order=order))
        )

    print(
        f"[arena] candidate step={candidate_trainer.training_step} "
        f"vs baseline step={baseline_trainer.training_step} "
        f"sims={simulations} games={args.games} max_moves={arena_config.max_moves}",
        flush=True,
    )
    started = time.perf_counter()
    result = SooArena(
        candidate=candidate,
        baseline=baseline,
        mcts_config=mcts,
        arena_config=arena_config,
    ).run(game_factory)
    elapsed = time.perf_counter() - started

    completed = result.wins + result.losses
    low, high = _wilson(result.wins, completed)
    elo = _elo(result.win_rate)
    print(
        f"  {result.wins}W-{result.losses}L ({result.aborted_games} aborted) "
        f"win rate {result.win_rate * 100:.1f}% "
        f"[95% CI {low * 100:.0f}-{high * 100:.0f}%] in {elapsed:.0f}s",
        flush=True,
    )
    print(f"  implied Elo: {'unbounded' if elo is None else f'{elo:+.0f}'}")
    # A 40-game arena cannot separate "slightly better" from "no different"; say
    # so rather than let the point estimate speak for itself.
    verdict = (
        "STRONGER" if low > 0.5 else "WEAKER" if high < 0.5 else "NOT SEPARATED at this sample size"
    )
    print(f"  verdict: {verdict}", flush=True)
    print(f"  promotion threshold {arena_config.promotion_threshold}: "
          f"{'met' if result.promoted else 'not met'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "candidate_step": candidate_trainer.training_step,
                        "baseline_step": baseline_trainer.training_step,
                        "simulations": simulations,
                        "games": args.games,
                        "wins": result.wins,
                        "losses": result.losses,
                        "aborted": result.aborted_games,
                        "win_rate": result.win_rate,
                        "ci95": [low, high],
                        "implied_elo": elo,
                        "verdict": verdict,
                        "promoted": result.promoted,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
