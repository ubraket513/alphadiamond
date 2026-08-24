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
# The reference evaluator is oracle tooling: it defines the deterministic
# answers the golden files are generated from, and its C++ twins live in
# native/tests. It moved here when the Python parity gates that used to share
# it were retired.
sys.path.insert(0, str(ROOT / "tools"))

from reference_evaluator import (
    ReferenceEvaluator,
    _mix_u32,
    _mix_u64,
    _unit,
    request_hash,
)

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec
from diamond.alphazero.bootstrap.heuristic import (
    CanonicalTargetVacancyDistancePrior,
    pairwise_distance_table,
)
from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.encoder import CanonicalEncoder
from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import DiamondSearchAdapter
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.mcts.search_3p import MCTS3P as PythonMCTS3P
from diamond.alphazero.native.topology import player_table, topology_tables
from diamond.game.board import Camp, standard_board
from diamond.game.rules import IllegalMoveError, find_legal_move, legal_moves
from diamond.game.session import GameSession
from diamond.game.state import GameState, GameStatus, build_players, initial_state

CORPUS = ROOT / "tests" / "native" / "fixtures" / "positions.jsonl"
GOLDEN = ROOT / "tests" / "golden"
RULES = GOLDEN / "rules-v1.txt"
MCTS = GOLDEN / "mcts-v1.txt"
MCTS3P = GOLDEN / "mcts3p-v1.txt"

# Gate B's sample: a full search costs ~200x a Gate A comparison, so the corpus
# is strided rather than searched whole. Simulation counts include 1 and 2
# because that is where an off-by-one in the backup shows up.
MCTS_STRIDE = 47
MCTS_SIMULATIONS = (1, 2, 8, 33, 64)
TOPOLOGY = GOLDEN / "topology"

FNV_OFFSET = 0xCBF29CE484222325
"""The same offset basis reference_evaluator.py uses; kept local so this file
reads without chasing an import for a constant."""
FNV_PRIME = 0x100000001B3
MASK = 0xFFFFFFFFFFFFFFFF


class Fnv:
    """Incremental FNV-1a 64, for streams that are not one contiguous buffer."""

    def __init__(self) -> None:
        self.value = FNV_OFFSET

    def byte(self, value: int) -> None:
        self.value = ((self.value ^ value) * FNV_PRIME) & MASK

    def bytes(self, payload: bytes) -> None:
        for byte in payload:
            self.byte(byte)

    def i32(self, value: int) -> None:
        self.bytes(struct.pack("<i", value))

    def u64(self, value: int) -> None:
        self.bytes(struct.pack("<Q", value))


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


class PythonRulesGame:
    """The oracle's own rules, applied by the Python engine.

    ``AlphaZeroGameAdapter`` used to be this. It is not any more -- it asks the
    native core (docs/architecture/retiring_the_python_engine.md), and a golden
    corpus generated from the implementation it is meant to pin would prove
    nothing. So the generator keeps its own adapter, here, where no shipped code
    imports it. It is deliberately the plainest possible reading of the rules:
    ``legal_moves``, ``find_legal_move`` and ``GameSession.commit``.

    ``DiamondSearchAdapter`` wraps it unchanged -- it only ever needed these
    methods, the codec and the encoder.
    """

    def __init__(self, players) -> None:
        self.players = tuple(players)
        self.board = standard_board()
        self._initial = initial_state(self.players, self.board)
        self.codec = ActionCodec(
            ActionSpaceSpec(
                board_size=len(self.board),
                version=f"diamond{len(self.board)}-srcdst-v1",
            )
        )
        self.encoder = CanonicalEncoder(self.board, self.codec)

    def initial_state(self) -> GameState:
        return self._initial

    def legal_action_ids(self, state: GameState) -> tuple[int, ...]:
        return tuple(
            self.codec.encode(move.source, move.destination)
            for move in legal_moves(self.board, state)
        )

    def apply_action(self, state: GameState, action_id: int) -> GameState:
        source, destination = self.codec.decode(action_id)
        move = find_legal_move(
            self.board, state, source, destination, player_id=state.current_player_id
        )
        if move is None:
            raise IllegalMoveError(f"action {action_id} ({source} -> {destination}) is not legal")
        session = GameSession(self.players, board=self.board, initial=state)
        session.commit(move)
        return session.state

    def is_terminal(self, state: GameState) -> bool:
        return state.status is GameStatus.FINISHED

    def final_order(self, state: GameState) -> tuple[int, ...]:
        if not self.is_terminal(state) or len(state.finish_order) != len(self.players):
            raise ValueError("final order is only available for a completed match")
        return state.finish_order


class Oracle:
    """One seat count's worth of Python authority."""

    def __init__(self, player_count: int) -> None:
        self.players = build_players(player_count)
        self.game = PythonRulesGame(self.players)
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



def _doubles_bytes(values) -> bytes:
    return struct.pack(f"<{len(values)}d", *values)



class VectorEvaluator:
    """The Gate B evaluator, widened to a three-seat value vector.

    The C++ twin is native/tests/vector_evaluator.hpp; both are specified in
    integer arithmetic so the golden file compares one function with itself
    rather than two that merely look alike.

    The three components are deliberately distinct. A symmetric vector hides
    the two mistakes that matter in a 3P search -- components assigned to the
    wrong seats, and a node maximising somebody else's component -- because
    both still look right when every component is the same number.
    """

    def evaluate(self, requests):
        results = []
        for request in requests:
            digest = request_hash(request)
            weights = []
            total = 0.0
            for action in request.legal_action_ids:
                action_hash = _mix_u32(_mix_u64(FNV_OFFSET, digest), action)
                weight = _unit(action_hash) + 0.5
                weights.append(weight)
                total += weight
            priors = {
                action: weight / total
                for action, weight in zip(request.legal_action_ids, weights)
            }
            value = tuple(
                _unit(_mix_u32(_mix_u64(FNV_OFFSET, digest), 0x5EA70000 + seat)) * 2.0 - 1.0
                for seat in range(3)
            )
            results.append(EvalResult(priors=priors, value=value))
        return tuple(results)


def _mcts3p_lines(records: list[dict], oracle: Oracle) -> list[str]:
    """Min's search, frozen: root statistics and the q vector per seat.

    Sampled like the 2P section, plus every position with a seat already
    placed -- only a handful of the corpus has one, all at the tail, and they
    are the only positions that exercise the placement vector.
    """
    lines = [
        "# alphadiamond 3P mcts golden v1",
        "# mcts3p <tag> <simulations> <current> <turn> <finish_order|-> <occupancy>",
        (
            "# mexp3p <selected> <root_fnv> <visit_fnv> <policy_fnv> <q_fnv>"
            " <calls> <simulations_run>"
        ),
    ]
    searchable = [
        record for record in records
        if record["player_count"] == 3 and record["status"] != "finished"
    ]
    sampled = searchable[::MCTS_STRIDE]
    placed = [record for record in searchable if record["finish_order"]]
    seen = {id(record) for record in sampled}
    for record in sampled + [r for r in placed if id(r) not in seen]:
        state = _state(record)
        for simulations in MCTS_SIMULATIONS:
            evaluator = VectorEvaluator()
            config = MCTSConfig(
                simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
            )
            result = PythonMCTS3P(oracle.search, evaluator, config).run(state, temperature=0.0)

            actions = list(result.visit_counts)
            visits = [result.visit_counts[action] for action in actions]
            policy = [result.policy[action] for action in actions]

            # q is a vector per action: hashed by seat id in ascending order, so
            # the digest changes if a component lands on the wrong seat.
            q_digest = Fnv()
            for action in actions:
                vector = result.q_values[action]
                for seat in sorted(vector):
                    q_digest.i32(seat)
                    q_digest.bytes(struct.pack("<d", vector[seat]))

            finish = ",".join(str(pid) for pid in record["finish_order"]) or "-"
            occupancy = "".join(str(cell) for cell in record["occupancy"])
            lines.append(
                f"mcts3p {record['tag']} {simulations} {record['current_player_id']} "
                f"{record['turn_number']} {finish} {occupancy}"
            )
            lines.append(
                f"mexp3p {result.selected_action} "
                f"{fnv1a(_actions_bytes(actions)):016x} "
                f"{fnv1a(struct.pack(f'<{len(visits)}i', *visits)):016x} "
                f"{fnv1a(_doubles_bytes(policy)):016x} "
                f"{q_digest.value:016x} "
                f"{len(actions)} {simulations}"
            )
    return lines


def _mcts_lines(records: list[dict], oracle: Oracle) -> list[str]:
    """Gate B, frozen: the search's root statistics and its request sequence.

    q values are hashed as raw doubles, not compared to a tolerance. The PUCT
    key is a double comparison, so a single-ulp drift is not a rounding
    curiosity -- it can flip a selection and change the whole descent.
    """
    lines = [
        "# alphadiamond mcts golden v1",
        "# mcts <tag> <evaluator> <simulations> <current> <turn> <occupancy>",
        (
            "# mexp <selected> <root_fnv> <visit_fnv> <q_fnv> <policy_fnv> <trace_fnv>"
            " <calls> <simulations_run>"
        ),
    ]
    searchable = [
        record for record in records
        if record["player_count"] == 2 and record["status"] != "finished"
    ]
    for record in searchable[::MCTS_STRIDE]:
        state = _state(record)
        for evaluator_name in ("hash", "uniform"):
            for simulations in MCTS_SIMULATIONS:
                evaluator = ReferenceEvaluator(uniform=evaluator_name == "uniform")
                config = MCTSConfig(
                    simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0
                )
                result = MCTS2P(oracle.search, evaluator, config).run(state, temperature=0.0)

                actions = list(result.visit_counts)  # expansion order, not sorted
                visits = [result.visit_counts[action] for action in actions]
                q_values = [result.q_values[action] for action in actions]
                policy = [result.policy[action] for action in actions]

                trace = Fnv()
                for request_digest, request_actions in evaluator.trace:
                    trace.u64(request_digest)
                    for action in request_actions:
                        trace.i32(action)

                occupancy = "".join(str(cell) for cell in record["occupancy"])
                lines.append(
                    f"mcts {record['tag']} {evaluator_name} {simulations} "
                    f"{record['current_player_id']} {record['turn_number']} {occupancy}"
                )
                lines.append(
                    f"mexp {result.selected_action} "
                    f"{fnv1a(_actions_bytes(actions)):016x} "
                    f"{fnv1a(struct.pack(f'<{len(visits)}i', *visits)):016x} "
                    f"{fnv1a(_doubles_bytes(q_values)):016x} "
                    f"{fnv1a(_doubles_bytes(policy)):016x} "
                    f"{trace.value:016x} "
                    f"{evaluator.calls} {simulations}"
                )
    return lines


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

    rules = output / RULES.name
    rules.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    mcts = output / MCTS.name
    mcts.write_text(
        "\n".join(_mcts_lines(records, oracles[2])) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"{rules}: {len(records)} positions")
    mcts3p = output / MCTS3P.name
    mcts3p.write_text(
        "\n".join(_mcts3p_lines(records, oracles[3])) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"{mcts}: gate B searches")
    print(f"{mcts3p}: 3P searches")
    print(f"{output / 'topology'}: 5 tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
