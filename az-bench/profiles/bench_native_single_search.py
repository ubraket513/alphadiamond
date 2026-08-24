"""Gate C.1: native single-search cost, measured against the Python oracle.

No threads, no batcher, no Python callback in the timed region -- this is the
cost of the lane work alone, which is the first of the three questions Gate C
has to keep separate.  A dummy evaluator stands in for inference so the number
is not contaminated by it.

The corpus is the committed Gate A fixture: full-game positions, not openings.
Section 0.3 of the design measured per-call costs varying ~3x between the two,
so an opening-only corpus would overstate the port.

Usage::

    python az-bench/profiles/native_single_search.py [--repeats N] [--simulations N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native import native_game, require_native
from diamond.alphazero.native.topology import camp_positions
from diamond.contract.camps import Camp
from diamond.contract.state import GameState, GameStatus, build_players

FIXTURE = ROOT / "tests" / "native" / "fixtures" / "positions.jsonl"


def load_positions(player_count: int = 2) -> list[dict]:
    records = [
        json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line
    ]
    return [
        r for r in records if r["player_count"] == player_count and r["status"] != "finished"
    ]


def python_state(record: dict) -> GameState:
    return GameState(
        occupancy=tuple(record["occupancy"]),
        current_player_id=record["current_player_id"],
        turn_number=record["turn_number"],
        status=GameStatus(record["status"]),
        finish_order=tuple(record["finish_order"]),
    )


class PythonStages:
    """The same four stages, timed in Python, on the same corpus.

    This is the comparison that matters. The design's section 0.1 table was
    measured on different hardware, so quoting native against it would compare
    two machines; re-measuring Python here makes the ratio meaningful.
    """

    def __init__(self, players) -> None:
        self.players = players
        self.game = AlphaZeroGameAdapter(players)
        self.search = DiamondSearchAdapter(self.game)
        self.prior = CanonicalTargetVacancyDistancePrior()
        self.pairwise = pairwise_distance_table()
        self.target = frozenset(camp_positions(Camp.Z_NEG))

    def run(self, states: list[GameState], repeats: int) -> dict[str, float]:
        totals = dict.fromkeys(("legal", "encode", "prior", "apply"), 0.0)
        count = 0
        for _ in range(repeats):
            for state in states:
                start = time.perf_counter_ns()
                legal = self.search.legal_action_ids(state)
                totals["legal"] += time.perf_counter_ns() - start
                if not legal:
                    continue

                start = time.perf_counter_ns()
                encoded, _player_ids = self.game.encode(state)
                totals["encode"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                self.prior.priors(
                    legal, self.target, self.pairwise, encoded
                )
                totals["prior"] += time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                self.search.apply_action(state, legal[0])
                totals["apply"] += time.perf_counter_ns() - start
                count += 1
        return {"count": count, **totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="cap corpus size (0 = all)")
    args = parser.parse_args()

    module = require_native()
    players = build_players(2)
    native = native_game(players)

    records = load_positions()
    if args.limit:
        records = records[: args.limit]
    py_states = [python_state(r) for r in records]
    native_states = [
        module.State(
            occupancy=r["occupancy"],
            current_player=r["current_player_id"],
            turn_number=r["turn_number"],
            status=0,
            finish_order=r["finish_order"],
        )
        for r in records
    ]

    print(f"corpus: {len(records)} full-game 2P positions, repeats={args.repeats}")
    legal_counts = [len(DiamondSearchAdapter(AlphaZeroGameAdapter(players)).legal_action_ids(s))
                    for s in py_states[:200]]
    print(f"legal actions/state: mean {statistics.mean(legal_counts):.1f} "
          f"median {statistics.median(legal_counts):.0f} max {max(legal_counts)}")

    config = module.MCTSConfig(simulations=args.simulations, dirichlet_epsilon=0.0)

    # --- stages -------------------------------------------------------------
    native_stage = native.profile(native_states, config, args.repeats, False)
    python_stage = PythonStages(players).run(py_states, args.repeats)

    n = native_stage["evaluations"]
    p = python_stage["count"]
    print(f"\n--- per-call stage cost (us), {n} native / {p} python calls ---")
    print(f"{'stage':<10} {'python':>10} {'native':>10} {'speedup':>9}")
    rows = [
        ("legal", python_stage["legal"], native_stage["legal_ns"]),
        ("prior", python_stage["prior"], native_stage["prior_ns"]),
        ("encode", python_stage["encode"], native_stage["encode_ns"]),
        ("apply", python_stage["apply"], native_stage["apply_ns"]),
    ]
    py_total = nat_total = 0.0
    for name, py_ns, nat_ns in rows:
        py_us = py_ns / p / 1000.0
        nat_us = nat_ns / n / 1000.0
        py_total += py_us
        nat_total += nat_us
        print(f"{name:<10} {py_us:>10.2f} {nat_us:>10.2f} {py_us / nat_us:>8.1f}x")
    print(f"{'TOTAL':<10} {py_total:>10.2f} {nat_total:>10.2f} {py_total / nat_total:>8.1f}x")

    # --- whole search -------------------------------------------------------
    search_records = records[:: max(1, len(records) // 60)]
    search_states = [
        module.State(
            occupancy=r["occupancy"],
            current_player=r["current_player_id"],
            turn_number=r["turn_number"],
            status=0,
            finish_order=r["finish_order"],
        )
        for r in search_records
    ]
    result = native.profile(search_states, config, 1, True)
    evals = result["evaluations"]
    print(f"\n--- whole search, {args.simulations} simulations, dummy evaluator ---")
    print(f"searches            {result['searches']}")
    print(f"evaluations         {evals}")
    print(f"evals/search        {evals / result['searches']:.1f}")
    print(f"ms/search           {result['search_ns'] / result['searches'] / 1e6:.3f}")
    print(f"us/eval             {result['search_ns'] / evals / 1000.0:.2f}")
    print(f"single-lane evals/s {evals / (result['search_ns'] / 1e9):,.0f}")


if __name__ == "__main__":
    main()
