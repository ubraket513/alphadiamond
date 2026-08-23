"""Freeze the Python oracle's answers into a language-neutral golden file.

The C++ tests must be able to prove rules, encoding and prior behaviour on a
machine with no Python, no pybind and no pytest.  They cannot do that by
calling one C++ function and comparing it with another C++ function, so this
script records what the *Python* oracle produces for every position in the
frozen corpus, in a format a 40-line C++ reader can consume.

Outputs (all committed):

* ``tests/golden/topology/*``   the five exported topology tables
* ``tests/golden/rules-v1.txt`` one ``pos``/``exp`` line pair per position

Per-action detail is folded into FNV-1a 64 digests rather than written out:
the corpus has 1327 positions with hundreds of successors each, and a digest
catches a mutation just as reliably as the full listing.  The prior is the one
exception -- it is floating point, so it is compared as ``max`` and an
order-sensitive dot product within a tolerance instead of hashed.

Usage::

    python tools/build_golden.py
"""

from __future__ import annotations

import argparse
import json
import struct
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
from diamond.alphazero.native.topology import player_table, topology_tables
from diamond.game.board import Camp, standard_board
from diamond.game.state import GameState, GameStatus, build_players

CORPUS = ROOT / "tests" / "native" / "fixtures" / "positions.jsonl"
GOLDEN = ROOT / "tests" / "golden"
RULES = GOLDEN / "rules-v1.txt"
TOPOLOGY = GOLDEN / "topology"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a(payload: bytes) -> int:
    digest = FNV_OFFSET
    for byte in payload:
        digest = ((digest ^ byte) * FNV_PRIME) & MASK
    return digest


def _write_topology(topology: Path) -> None:
    topology.mkdir(parents=True, exist_ok=True)
    tables = topology_tables()
    for name in ("camp_positions", "pairwise_distance", "physical_to_canonical",
                 "canonical_to_physical"):
        flat = [int(value) for row in tables[name] for value in row]
        (topology / f"topology_{name}.i32").write_bytes(struct.pack(f"<{len(flat)}i", *flat))
    flat = [int(value) for row in tables["neighbour"] for value in row]
    (topology / "topology_neighbour.i8").write_bytes(struct.pack(f"<{len(flat)}b", *flat))


class Oracle:
    """One seat count's worth of Python authority."""

    def __init__(self, player_count: int) -> None:
        self.players = build_players(player_count)
        self.game = AlphaZeroGameAdapter(self.players)
        self.search = DiamondSearchAdapter(self.game)
        self.codec = ActionCodec(ActionSpaceSpec.diamond73())
        self.prior = CanonicalTargetVacancyDistancePrior()
        self.pairwise = pairwise_distance_table(self.game.board)
        self.target = frozenset(standard_board().camp_positions(Camp.Z_NEG))

    def priors(self, state: GameState) -> list[float]:
        request = self.search.evaluation_request(state)
        values = self.prior.priors(
            request.legal_action_ids,
            self.codec,
            self.target,
            self.pairwise,
            request.node_features,
        )
        return [values[action] for action in request.legal_action_ids]


def _state(record: dict) -> GameState:
    return GameState(
        occupancy=tuple(record["occupancy"]),
        current_player_id=record["current_player_id"],
        turn_number=record["turn_number"],
        status=GameStatus(record["status"]),
        finish_order=tuple(record["finish_order"]),
    )


def _state_bytes(state: GameState) -> bytes:
    payload = bytes(state.occupancy)
    payload += bytes([
        state.current_player_id,
        1 if state.status is GameStatus.FINISHED else 0,
        state.turn_number & 0xFF,
        (state.turn_number >> 8) & 0xFF,
        len(state.finish_order),
    ])
    return payload + bytes(state.finish_order)


def _actions_bytes(actions: list[int]) -> bytes:
    return struct.pack(f"<{len(actions)}i", *actions)


def _encoded_bytes(oracle: Oracle, state: GameState) -> bytes:
    encoded = oracle.game.encoder.encode(state, oracle.players)
    rows = encoded.node_features
    features = len(rows[0]) if rows else 0
    payload = struct.pack("<ii", len(rows), features)
    payload += bytes(encoded.canonical_player_ids)
    flat = [float(value) for row in rows for value in row]
    return payload + struct.pack(f"<{len(flat)}f", *flat)


def _record_line(record: dict, oracle: Oracle) -> tuple[str, str]:
    state = _state(record)
    physical = list(oracle.game.legal_action_ids(state))
    canonical = list(oracle.search.legal_action_ids(state))

    if oracle.game.is_terminal(state):
        successors = 0
        prior_count, prior_max, prior_dot = 0, 0.0, 0.0
    else:
        payload = b"".join(
            _state_bytes(oracle.game.apply_action(state, action)) for action in physical
        )
        successors = fnv1a(payload)
        priors = oracle.priors(state)
        prior_count = len(priors)
        prior_max = max(priors) if priors else 0.0
        prior_dot = sum(index * value for index, value in enumerate(priors))

    finish = ",".join(str(pid) for pid in record["finish_order"]) or "-"
    occupancy = "".join(str(cell) for cell in record["occupancy"])
    pos = (
        f"pos {record['tag']} {record['player_count']} {record['current_player_id']} "
        f"{record['turn_number']} {1 if record['status'] == 'finished' else 0} "
        f"{finish} {occupancy}"
    )
    exp = (
        f"exp {len(physical)} {fnv1a(_actions_bytes(physical)):016x} "
        f"{fnv1a(_actions_bytes(canonical)):016x} {successors:016x} "
        f"{fnv1a(_encoded_bytes(oracle, state)):016x} "
        f"{prior_count} {prior_max:.17g} {prior_dot:.17g}"
    )
    return pos, exp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN,
        help="directory to write into (default: tests/golden)",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output

    if not CORPUS.is_file():
        raise SystemExit(f"missing corpus: {CORPUS}; run tools/build_native_corpus.py")
    output.mkdir(parents=True, exist_ok=True)
    _write_topology(output / "topology")

    records = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line]
    oracles = {count: Oracle(count) for count in sorted({r["player_count"] for r in records})}

    lines = [
        "# alphadiamond rules golden v1",
        "# generated by tools/build_golden.py from tests/native/fixtures/positions.jsonl",
        "# match <player_count> <id,camp,target_camp> ...",
    ]
    for count, oracle in oracles.items():
        seats = " ".join(",".join(str(v) for v in seat) for seat in player_table(oracle.players))
        lines.append(f"match {count} {seats}")
    for record in records:
        pos, exp = _record_line(record, oracles[record["player_count"]])
        lines.append(pos)
        lines.append(exp)

    RULES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{RULES}: {len(records)} positions")
    print(f"{TOPOLOGY}: 5 tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
