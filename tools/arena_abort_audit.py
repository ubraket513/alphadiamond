"""Where do arena aborts come from, and how close was the game to finishing?

``SooArenaResult`` reports one abort count and drops those games from the win
rate entirely -- ``win_rate = wins / (wins + losses)``.  That is fine when
aborts are rare and badly misleading when they are not: a pair reporting
"100 %" over 20 completed games out of 40 has had half its evidence discarded
by a filter that is not independent of the result.

Two things the aggregate cannot tell you, and this can:

**Which cell.** ``SooArena`` cycles four matchup cells -- two turn orders
crossed with two candidate seats -- so 40 games is 10 of each.  Aborts landing
uniformly across the four means "these two networks draw out long games".
Aborts landing entirely in one cell means something is wrong with a seat or an
orientation, and that is a different bug with a different fix.

**How close.** A two-player Soo match ends the moment *either* player fills
their target camp -- ``match_is_over`` is ``len(finish_order) >= len(players) -
1``, which is ``1 >= 1``.  So an abort is not "the weak side could not finish";
it is "**neither** side finished", the strong one included.  Reporting how many
of each player's ten target cells were occupied separates a game decided by one
blocked cell from one that never got going.

Usage::

    python tools/arena_abort_audit.py --candidate new.pt --baseline old.pt \\
        --games 40 --simulations 128
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.arena import _balanced_matchups
from diamond.alphazero.checkpoint import checkpoint_network_config, load_checkpoint
from diamond.alphazero.config import MCTSConfig, TrainingConfig
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.game.state import build_players


def _load(path: Path, config: dict):
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
    return trainer, TorchEvaluator(
        trainer.model, value_size=1, device=config["training"]["device"]
    )


def _camp_fill(game: AlphaZeroGameAdapter, state, spec) -> tuple[int, int]:
    """How many of a player's ten target cells they hold, and the camp size."""
    target = game.board.camp_positions(spec.target_camp)
    held = sum(1 for position in target if state.occupant(position) == spec.id)
    return held, len(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40, help="multiple of 4")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.games % 4:
        raise SystemExit("games must be a multiple of 4")

    candidate_trainer, candidate = _load(args.candidate, config)
    baseline_trainer, baseline = _load(args.baseline, config)
    simulations = args.simulations or config["mcts"]["simulations"]
    max_moves = args.max_moves or config["arena"]["max_moves"]
    # Evaluation never inherits training root noise; SooArena zeroes this too.
    mcts = MCTSConfig(
        simulations=simulations,
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        dirichlet_epsilon=0.0,
        seed=args.seed,
    )

    matchups = _balanced_matchups((1, 2))
    print(
        f"[audit] candidate step={candidate_trainer.training_step} "
        f"vs baseline step={baseline_trainer.training_step} "
        f"sims={simulations} games={args.games} max_moves={max_moves}",
        flush=True,
    )

    records = []
    started = time.perf_counter()
    for game_index in range(args.games):
        order, candidate_player = matchups[game_index % len(matchups)]
        players = build_players(len(order), order=order)
        inner = AlphaZeroGameAdapter(players)
        game = DiamondSearchAdapter(inner)
        state = game.initial_state()
        moves = 0
        # The seed schedule matches SooArena's exactly, so a game here is the
        # same game it plays -- otherwise this audits a different experiment.
        while not game.is_terminal(state) and moves < max_moves:
            evaluator = (
                candidate if game.current_player_id(state) == candidate_player else baseline
            )
            stepped = replace(mcts, seed=args.seed + game_index * max_moves + moves)
            action = MCTS2P(game, evaluator, stepped).run(state, temperature=0.0).selected_action
            state = game.apply_action(state, action)
            moves += 1

        terminal = game.is_terminal(state)
        fills = {
            spec.id: _camp_fill(inner, state, spec)[0]
            for spec in players
        }
        outcome = (
            "abort" if not terminal
            else "win" if game.final_order(state)[0] == candidate_player
            else "loss"
        )
        records.append(
            {
                "game": game_index,
                "order": list(order),
                "candidate_seat": candidate_player,
                "outcome": outcome,
                "moves": moves,
                "candidate_fill": fills[candidate_player],
                "baseline_fill": fills[3 - candidate_player] if len(players) == 2 else None,
                "finished_any": bool(state.finish_order),
            }
        )
        print(
            f"  game {game_index:>3}  order={order}  cand_seat={candidate_player}  "
            f"{outcome:<5} moves={moves:<5} fill cand={fills[candidate_player]}/10 "
            f"base={fills[3 - candidate_player]}/10",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    print(f"\n[cells] four balanced cells, {args.games // 4} games each   ({elapsed:.0f}s)")
    print(f"  {'order':>8}{'cand seat':>11}{'W':>4}{'L':>4}{'abort':>7}"
          f"{'median moves':>14}{'abort cand fill':>17}{'abort base fill':>17}")
    for order, candidate_player in matchups:
        cell = [
            record
            for record in records
            if tuple(record["order"]) == order and record["candidate_seat"] == candidate_player
        ]
        if not cell:
            continue
        aborts = [record for record in cell if record["outcome"] == "abort"]
        wins = sum(1 for record in cell if record["outcome"] == "win")
        losses = sum(1 for record in cell if record["outcome"] == "loss")
        median = statistics.median(record["moves"] for record in cell)
        cand_fill = (
            f"{statistics.mean(r['candidate_fill'] for r in aborts):.1f}/10" if aborts else "-"
        )
        base_fill = (
            f"{statistics.mean(r['baseline_fill'] for r in aborts):.1f}/10" if aborts else "-"
        )
        print(
            f"  {order!s:>8}{candidate_player:>11}{wins:>4}{losses:>4}{len(aborts):>7}"
            f"{median:>14.0f}{cand_fill:>17}{base_fill:>17}"
        )

    aborts = [record for record in records if record["outcome"] == "abort"]
    wins = sum(1 for record in records if record["outcome"] == "win")
    losses = sum(1 for record in records if record["outcome"] == "loss")
    print(
        f"\n  totals: {wins}W-{losses}L, {len(aborts)} aborted "
        f"({len(aborts) / len(records) * 100:.0f}% of games)"
    )
    if aborts:
        by_seat = {seat: sum(1 for r in aborts if r["candidate_seat"] == seat) for seat in (1, 2)}
        by_order = {}
        for record in aborts:
            by_order[tuple(record["order"])] = by_order.get(tuple(record["order"]), 0) + 1
        print(f"  aborts by candidate seat: {by_seat}")
        print(f"  aborts by turn order:     {by_order}")
        print(
            f"  camp fill in aborted games: candidate "
            f"{statistics.mean(r['candidate_fill'] for r in aborts):.1f}/10, "
            f"baseline {statistics.mean(r['baseline_fill'] for r in aborts):.1f}/10"
        )
        print(
            "  A concentration in one seat or one order is a different finding "
            "from a uniform spread; so is 9/10 against 3/10."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "candidate_step": candidate_trainer.training_step,
                    "baseline_step": baseline_trainer.training_step,
                    "simulations": simulations,
                    "max_moves": max_moves,
                    "seed": args.seed,
                    "games": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
