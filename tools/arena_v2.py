"""Strength measurement that is not four games wearing a rosette.

`SooArena` cycles four balanced matchup cells -- two turn orders crossed with
two candidate seats -- and reseeds per move.  Evaluation forces
`dirichlet_epsilon = 0` and `temperature = 0`, and neither code path then
touches the RNG: `select_from_visits` returns `min(visits, ...)` outright below
temperature 0, and `add_dirichlet_noise` returns its input before drawing
anything when epsilon is 0.  So the seed is inert and a cell is a pure function
of `(turn order, candidate seat)`.

**Asking for 40 games therefore plays 4 games ten times each.**  Measured, not
inferred: games 0/4/8/12/16 of a real run came back with identical move counts
and identical final camp occupancy.  Every per-game confidence interval this
project has quoted was computed over pseudo-replicates -- the `40W-0L` was
`4W-0L`, and the `30W-10L` behind the recorded `+191 Elo` was `3W-1L`, which is
why it landed on a multiple of ten.

The fix is more starting positions, not more repeats, and deliberately **not**
temperature.  Sampling moves would raise the game count but change the question
from "which model's best play is stronger" to "which model's sampled play is
stronger", and add variance exactly where the effect is small.  Fishtest varies
the opening book and keeps play deterministic; so does this.

Two structural points:

*Openings are generated per turn order.*  An action sequence is only legal
under the turn order it was generated for, so replaying one suite under both
orders would silently desynchronise.  `OpeningSuite.generate` builds from
`build_players(player_count)` and cannot express this, so the generation lives
here and reuses `BenchmarkOpening` purely for its versioned identity.

*The statistical unit is the opening pair, not the game.*  The two games from
one opening differ only in which side the candidate plays; they share a
position and are strongly correlated.  Treating them as independent
overstates precision, so the interval comes from a bootstrap over pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import checkpoint_network_config, load_checkpoint
from diamond.alphazero.config import MCTSConfig, TrainingConfig
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import (
    RULESET_FINGERPRINT,
    RULESET_VERSION,
    CheckpointCompatibilitySpec,
)
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.network import SooModel
from diamond.alphazero.rating.openings import BenchmarkOpening
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.contract.state import build_players

SUITE_VERSION = "arena-openings-v2"


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
    ), network


def _openings_for_order(
    order: tuple[int, ...], *, count: int, max_depth: int, seed: int, label: str = "dev"
) -> list[BenchmarkOpening]:
    """Distinct legal action prefixes, generated under this turn order.

    The empty opening comes first, so the standard start position is always in
    the suite and the new measurement stays comparable to the old one on at
    least that point.
    """
    players = build_players(len(order), order=order)
    adapter = AlphaZeroGameAdapter(players)
    version = f"{SUITE_VERSION}/{label}/order-{'-'.join(str(seat) for seat in order)}"
    first = BenchmarkOpening(
        suite_version=version,
        player_count=len(order),
        ruleset_version=RULESET_VERSION,
        ruleset_fingerprint=RULESET_FINGERPRINT,
        action_ids=(),
    )
    openings = [first]
    seen = {first.opening_id}
    rng = random.Random(seed)
    attempts = 0
    while len(openings) < count:
        attempts += 1
        if attempts > count * 200:
            raise SystemExit(f"could not generate {count} distinct openings for {order}")
        state = adapter.initial_state()
        action_ids: list[int] = []
        for _ in range(rng.randint(1, max_depth)):
            legal = sorted(adapter.legal_action_ids(state))
            if not legal:
                break
            action_id = rng.choice(legal)
            action_ids.append(action_id)
            state = adapter.apply_action(state, action_id)
        opening = BenchmarkOpening(
            suite_version=version,
            player_count=len(order),
            ruleset_version=RULESET_VERSION,
            ruleset_fingerprint=RULESET_FINGERPRINT,
            action_ids=tuple(action_ids),
        )
        if opening.opening_id not in seen:
            openings.append(opening)
            seen.add(opening.opening_id)
    return openings


def _camp_fill(adapter: AlphaZeroGameAdapter, state, spec) -> int:
    target = adapter.board.camp_positions(spec.target_camp)
    return sum(1 for position in target if state.occupant(position) == spec.id)


def _bootstrap(scores: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    """Percentile interval over opening pairs, the unit that is independent."""
    if not scores:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [scores[rng.randrange(len(scores))] for _ in scores]
        means.append(sum(draw) / len(draw))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return (low, high)


def _elo(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "runtime/configs/soo-rtx5090-native.json"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--openings", type=int, default=16, help="per turn order")
    parser.add_argument("--opening-depth", type=int, default=8)
    parser.add_argument("--opening-seed", type=int, default=20260823)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--suite",
        choices=("dev", "cert"),
        default="dev",
        help=(
            "Which frozen suite to draw openings from.  'dev' is the working "
            "suite used while iterating; 'cert' has a different seed and is "
            "reserved for promotion decisions and headline numbers.  Selecting "
            "candidates repeatedly against one suite lets the *chooser* overfit "
            "it even though no model trains on it, so the two are kept apart "
            "and the cert suite is used sparingly."
        ),
    )
    parser.add_argument(
        "--max-abort-fraction",
        type=float,
        default=0.1,
        help=(
            "Above this share of aborted pairs the rating is reported as "
            "censored rather than as an Elo.  An unfinished game is not a draw "
            "-- these rules have no draw -- so scoring it 0.5 would be an "
            "invention, and dropping many of them selects on the outcome."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidate_trainer, candidate, candidate_net = _load(args.candidate, config)
    baseline_trainer, baseline, baseline_net = _load(args.baseline, config)
    simulations = args.simulations or config["mcts"]["simulations"]
    max_moves = args.max_moves or config["arena"]["max_moves"]
    mcts = MCTSConfig(
        simulations=simulations,
        c_puct=config["mcts"]["c_puct"],
        dirichlet_alpha=config["mcts"]["dirichlet_alpha"],
        # Deterministic on purpose; see the module docstring.  The seed below is
        # therefore inert, and that is the property this tool is built around
        # rather than the bug it used to hide.
        dirichlet_epsilon=0.0,
        seed=0,
    )

    orders = tuple(itertools.permutations((1, 2)))
    # Distinct seeds *and* distinct suite versions, so a cert opening can never
    # collide with a dev opening by identity even if a seed is reused by mistake.
    suite_offset = 0 if args.suite == "dev" else 1_000_000
    suites = {
        order: _openings_for_order(
            order,
            count=args.openings,
            max_depth=args.opening_depth,
            seed=args.opening_seed + suite_offset + index,
            label=args.suite,
        )
        for index, order in enumerate(orders)
    }
    total = sum(len(suite) for suite in suites.values()) * 2
    print(
        f"[arena-v2] candidate step={candidate_trainer.training_step} "
        f"({candidate_net.width}x{candidate_net.residual_blocks}) vs "
        f"baseline step={baseline_trainer.training_step} "
        f"({baseline_net.width}x{baseline_net.residual_blocks})\n"
        f"  sims={simulations} max_moves={max_moves} "
        f"openings={args.openings}/order depth<={args.opening_depth} "
        f"-> {total} unique games in {total // 2} pairs",
        flush=True,
    )

    records = []
    started = time.perf_counter()
    for order in orders:
        players = build_players(len(order), order=order)
        for opening_index, opening in enumerate(suites[order]):
            for candidate_seat in order:
                adapter = AlphaZeroGameAdapter(players)
                game = DiamondSearchAdapter(adapter)
                state = adapter.initial_state()
                for action_id in opening.action_ids:
                    state = adapter.apply_action(state, action_id)
                moves = 0
                trace = hashlib.sha256()
                while not game.is_terminal(state) and moves < max_moves:
                    evaluator = (
                        candidate
                        if game.current_player_id(state) == candidate_seat
                        else baseline
                    )
                    action = (
                        MCTS2P(game, evaluator, mcts)
                        .run(state, temperature=0.0)
                        .selected_action
                    )
                    trace.update(str(action).encode())
                    state = game.apply_action(state, action)
                    moves += 1
                terminal = game.is_terminal(state)
                outcome = (
                    "abort" if not terminal
                    else "win" if game.final_order(state)[0] == candidate_seat
                    else "loss"
                )
                records.append(
                    {
                        "order": list(order),
                        "opening_index": opening_index,
                        "opening_id": opening.opening_id,
                        "opening_plies": len(opening.action_ids),
                        "candidate_seat": candidate_seat,
                        "outcome": outcome,
                        "moves": moves,
                        "trajectory": trace.hexdigest()[:16],
                        "candidate_fill": _camp_fill(
                            adapter, state, next(p for p in players if p.id == candidate_seat)
                        ),
                        "baseline_fill": _camp_fill(
                            adapter, state, next(p for p in players if p.id != candidate_seat)
                        ),
                    }
                )
                print(
                    f"  {order!s:>7} op{opening_index:<3} "
                    f"({len(opening.action_ids)} plies) seat={candidate_seat} "
                    f"{outcome:<5} moves={moves:<5} "
                    f"fill {records[-1]['candidate_fill']}/10 vs "
                    f"{records[-1]['baseline_fill']}/10",
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    # Pseudo-replication guard: the whole reason this tool exists.  Distinct
    # openings must produce distinct games; if they do not, the suite is not
    # buying independence and the interval below would be a fiction again.
    traces = [record["trajectory"] for record in records]
    distinct = len(set(traces))
    print(f"\n[distinct] {distinct} distinct trajectories out of {len(records)} games")
    if distinct < len(records):
        print(
            "  WARNING: repeated trajectories -- the effective sample is smaller "
            "than the game count, which is exactly the defect this replaces."
        )

    pairs = []
    aborted_pairs = 0
    for order in orders:
        for opening_index in range(len(suites[order])):
            cell = [
                record
                for record in records
                if tuple(record["order"]) == order
                and record["opening_index"] == opening_index
            ]
            if any(record["outcome"] == "abort" for record in cell):
                aborted_pairs += 1
                continue
            wins = sum(1 for record in cell if record["outcome"] == "win")
            pairs.append(wins / len(cell))

    aborts = [record for record in records if record["outcome"] == "abort"]
    print(f"[aborts] {len(aborts)} games, {aborted_pairs} pairs dropped")
    if aborts:
        by_order = {}
        for record in aborts:
            by_order[tuple(record["order"])] = by_order.get(tuple(record["order"]), 0) + 1
        print(f"  by turn order: {by_order}")
        print(
            f"  camp fill when aborted: candidate "
            f"{statistics.mean(r['candidate_fill'] for r in aborts):.1f}/10, "
            f"baseline {statistics.mean(r['baseline_fill'] for r in aborts):.1f}/10"
        )
        print(
            "  Dropped rather than scored: an unfinished game has no winner, and "
            "keeping it in the denominator would bias whichever side stalls."
        )

    if not pairs:
        print("\n[result] every pair contained an abort; nothing to rate.")
        return 1

    abort_fraction = aborted_pairs / (aborted_pairs + len(pairs))
    score = sum(pairs) / len(pairs)
    low, high = _bootstrap(pairs, samples=args.bootstrap, seed=args.opening_seed)
    elo, elo_low, elo_high = _elo(score), _elo(low), _elo(high)
    fmt = lambda value: "unbounded" if value is None else f"{value:+.0f}"
    print(
        f"\n[result] {len(pairs)} scored pairs, candidate score "
        f"{score * 100:.1f}%  [95% CI {low * 100:.1f}-{high * 100:.1f}%]"
    )
    print(f"  Elo {fmt(elo)}  [CI {fmt(elo_low)} to {fmt(elo_high)}]")
    print(f"  pair score spread: {sorted(set(pairs))}")
    verdict = (
        "STRONGER" if low > 0.5 else "WEAKER" if high < 0.5 else "NOT SEPARATED"
    )
    if abort_fraction > args.max_abort_fraction:
        verdict = f"CENSORED ({abort_fraction * 100:.0f}% of pairs unfinished)"
        print(f"  verdict: {verdict}   ({elapsed:.0f}s)")
        print(
            "  Treat the score above as descriptive, not as a rating.  With this "
            "many positions unresolved the more informative result is that the "
            "search cannot finish them, not the Elo of the ones it could."
        )
    else:
        print(f"  verdict: {verdict}   ({elapsed:.0f}s)")
    print(
        "  The interval is a bootstrap over opening pairs, not over games: the "
        "two games of a pair share a position and are not independent."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "candidate": str(args.candidate),
                    "baseline": str(args.baseline),
                    "candidate_step": candidate_trainer.training_step,
                    "baseline_step": baseline_trainer.training_step,
                    "simulations": simulations,
                    "max_moves": max_moves,
                    "openings_per_order": args.openings,
                    "opening_depth": args.opening_depth,
                    "opening_seed": args.opening_seed,
                    "distinct_trajectories": distinct,
                    "pair_scores": pairs,
                    "aborted_games": len(aborts),
                    "aborted_pairs": aborted_pairs,
                    "score": score,
                    "suite": args.suite,
                    "abort_pair_fraction": abort_fraction,
                    "censored": abort_fraction > args.max_abort_fraction,
                    "ci": [low, high],
                    "elo": elo,
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
