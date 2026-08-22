"""Build the frozen Gate A position corpus.

Regenerates ``tests/native/fixtures/positions.jsonl``.  The corpus is a
committed fixture and a permanent regression test, not a one-off script, so it
is deterministic: same seeds in, same file out.

Coverage follows section 7 of ``docs/native_selfplay_phase0.md``:

* standard opening positions (2P and 3P)
* full-game trajectories driven by the production bootstrap prior -- the real
  distribution the native backend will run under, and the only way to reach
  jump-heavy midgames and partially-filled target camps
* random legal trajectories at several depths
* near-terminal and terminal positions
* positions where a destination is reachable by both a step and a jump chain

Usage::

    python tools/build_native_corpus.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.game.board import Camp, standard_board
from diamond.game.move import MoveKind
from diamond.game.rules import moves_from
from diamond.game.state import GameState, build_players

OUTPUT = ROOT / "tests" / "native" / "fixtures" / "positions.jsonl"
MAX_MOVES = 400
"""Hard cap per generated game; prior-sampled Soo games finish well inside it."""


def _record(state: GameState, player_count: int, tag: str) -> dict:
    return {
        "tag": tag,
        "player_count": player_count,
        "occupancy": list(state.occupancy),
        "current_player_id": state.current_player_id,
        "turn_number": state.turn_number,
        "status": state.status.value,
        "finish_order": list(state.finish_order),
    }


def _target_fill(state: GameState, players) -> int:
    """The fullest target camp, counted in pieces.

    Partially-filled target camps are where the vacancy prior's potential stops
    being a fixed table, so the corpus must be dense there.
    """
    board = standard_board()
    best = 0
    for spec in players:
        held = sum(
            1 for pid in board.camp_positions(spec.target_camp) if state.occupancy[pid] == spec.id
        )
        best = max(best, held)
    return best


def _has_step_jump_conflict(state: GameState) -> bool:
    """True when some piece reaches one destination by a step and another by a jump.

    The step-wins rule only bites where both kinds are generated for the same
    source, so this is the cheapest observable proxy for the interesting case.
    """
    board = standard_board()
    for source, occupant in enumerate(state.occupancy):
        if occupant != state.current_player_id:
            continue
        kinds = {move.kind for move in moves_from(board, state, source).values()}
        if MoveKind.STEP in kinds and MoveKind.JUMP in kinds:
            return True
    return False


class _PriorPolicy:
    """The production bootstrap prior, sampled -- not a search."""

    def __init__(self, player_count: int) -> None:
        self.players = build_players(player_count)
        self.game = AlphaZeroGameAdapter(self.players)
        self.search = DiamondSearchAdapter(self.game)
        self.codec = ActionCodec(ActionSpaceSpec.diamond73())
        self.prior = CanonicalTargetVacancyDistancePrior()
        self.pairwise = pairwise_distance_table(self.game.board)
        self.target = frozenset(self.game.board.camp_positions(Camp.Z_NEG))

    def choose(self, state: GameState, rng: random.Random) -> int:
        """A physical action id, sampled from the bootstrap prior."""
        request = self.search.evaluation_request(state)
        priors = self.prior.priors(
            request.legal_action_ids,
            self.codec,
            self.target,
            self.pairwise,
            request.node_features,
        )
        canonical = rng.choices(list(priors), weights=list(priors.values()), k=1)[0]
        return self.game.encoder.to_physical_action(
            canonical, self.players, state.current_player_id
        )


def _prior_games(player_count: int, seeds: range) -> list[dict]:
    policy = _PriorPolicy(player_count)
    records: list[dict] = []
    for seed in seeds:
        rng = random.Random(seed)
        state = policy.game.initial_state()
        trail: list[GameState] = []
        for move_index in range(MAX_MOVES):
            if policy.game.is_terminal(state):
                break
            state = policy.game.apply_action(state, policy.choose(state, rng))
            trail.append(state)
            # Sample the trajectory rather than storing every ply: the corpus is
            # a fixture, and adjacent plies are near-duplicates.
            if move_index % 17 == 0:
                records.append(_record(state, player_count, f"prior-s{seed}-m{move_index}"))
            elif move_index % 5 == 0 and _has_step_jump_conflict(state):
                records.append(_record(state, player_count, f"conflict-s{seed}-m{move_index}"))
            elif _target_fill(state, policy.players) >= 6:
                records.append(_record(state, player_count, f"packing-s{seed}-m{move_index}"))
        # The tail is the endgame the fixed-table prior is blind to: keep every ply.
        for offset, tail_state in enumerate(trail[-40:]):
            records.append(_record(tail_state, player_count, f"tail-s{seed}-{offset}"))
    return records


def _random_walks(player_count: int, seeds: range, depths: tuple[int, ...]) -> list[dict]:
    players = build_players(player_count)
    game = AlphaZeroGameAdapter(players)
    records: list[dict] = []
    for seed in seeds:
        for depth in depths:
            rng = random.Random(seed * 1000 + depth)
            state = game.initial_state()
            for _ in range(depth):
                if game.is_terminal(state):
                    break
                actions = game.legal_action_ids(state)
                state = game.apply_action(state, rng.choice(actions))
            records.append(_record(state, player_count, f"walk-s{seed}-d{depth}"))
    return records


def main() -> None:
    records: list[dict] = []
    for player_count in (2, 3):
        records.append(
            _record(
                AlphaZeroGameAdapter(build_players(player_count)).initial_state(),
                player_count,
                "opening",
            )
        )
        records.extend(_random_walks(player_count, range(4), (1, 3, 8, 20, 50, 120)))
    records.extend(_prior_games(2, range(3)))
    records.extend(_prior_games(3, range(2)))

    # Deduplicate on the position itself, keeping the first tag that produced it.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for record in records:
        key = (
            record["player_count"],
            tuple(record["occupancy"]),
            record["current_player_id"],
            record["status"],
            tuple(record["finish_order"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in unique:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    terminal = sum(1 for r in unique if r["status"] == "finished")
    conflict = sum(1 for r in unique if r["tag"].startswith("conflict"))
    print(f"{len(unique)} positions -> {OUTPUT}")
    print(f"  terminal: {terminal}   step/jump conflict: {conflict}")


if __name__ == "__main__":
    main()
