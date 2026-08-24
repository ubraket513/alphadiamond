"""The Python engine's dependent set may shrink, never grow.

The C++ core is the authority for rules, encoding, search and self-play. The
Python engine in ``diamond.game`` (and the Python MCTS beside it) is kept for
two jobs only: it is the oracle that generates ``tests/golden/``, and it is the
other half of the bridge the trainer still runs through. Everything else that
imports it is migration debt.

Debt nobody counts grows, so this counts it -- in two piles, because one number
could not tell the difference between a module that *runs the rules* and one
that names a dataclass in a type hint, and those retire at completely different
times:

* **behaviour** -- ``legal_moves``, ``GameSession``, ``MCTS2P``: what the C++
  core replaced. This list must reach zero before ``diamond.game`` can be
  deleted, and it is the work queue. ``search_factory`` left it when the
  Python-search fallback was retired (decision 1).
* **definitions and types** -- ``standard_board``, ``build_players``,
  ``GameState``: the board, the seats, and the shapes that describe a position.
  Not rules; C++ receives the same tables through the topology export, and they
  survive until the trainer speaks the native ``State`` directly.

Lumping the two together made "30 dependents" a number nobody could act on.

See docs/architecture/retiring_the_python_engine.md.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENGINE_MODULES = ("diamond.game", "diamond.alphazero.mcts")

BEHAVIOUR = {
    # Rules and session: what the native core reimplements.
    "GameSession",
    "legal_moves",
    "moves_from",
    "find_legal_move",
    "find_winner",
    "next_player_id",
    "update_ranking",
    "validate_move",
    # The Python search.
    "MCTS2P",
    "MCTS3P",
    "ScalarEdge",
    "ScalarNode",
    "VectorEdge",
    "VectorNode",
    "add_dirichlet_noise",
    "exploration_bonus",
    "select_from_visits",
}
"""Importing any of these runs the Python engine, rather than describing a
position with its types.

The criterion is whether it applies rules. `initial_state` deliberately is not
here: it fills each seat's home camp and applies nothing, so it is the opening's
definition -- the same thing the topology export hands C++."""

# ---------------------------------------------------------------------------
# The work queue: modules that still run the Python engine. Every entry is debt
# with a reason, and the reason decides when it goes.
# ---------------------------------------------------------------------------
BEHAVIOUR_ALLOWED = {
    # The oracle's adapter: what tools/build_golden.py drives to produce the
    # frozen answers. Retires with the corpus generator.
    "src/diamond/alphazero/game_adapter.py",
    # The Python self-play path, still reachable through selfplay_backend.
    "src/diamond/alphazero/selfplay/runner_2p.py",
    "src/diamond/alphazero/selfplay/runner_3p.py",
    # The smokes script games with find_legal_move.
    "src/diamond/alphazero/milestone2_smoke.py",
    "src/diamond/alphazero/smoke.py",
}

# ---------------------------------------------------------------------------
# Not the work queue: the board, the seats, and the dataclasses that describe a
# position.
# ---------------------------------------------------------------------------
DEFINITIONS_ALLOWED = {
    "src/diamond/agents/alphazero_agent.py",
    "src/diamond/agents/base.py",
    "src/diamond/agents/random_agent.py",
    "src/diamond/alphazero/bootstrap/evaluator.py",
    "src/diamond/alphazero/bootstrap/heuristic.py",
    "src/diamond/alphazero/bootstrap/probe.py",
    "src/diamond/alphazero/encoder.py",
    "src/diamond/alphazero/identity.py",
    "src/diamond/alphazero/native/__init__.py",
    "src/diamond/alphazero/native/topology.py",
    "src/diamond/alphazero/network/trunk.py",
    "src/diamond/alphazero/orchestration/benchmark.py",
    "src/diamond/alphazero/orchestration/production.py",
    "src/diamond/alphazero/orchestration/selfplay_workers.py",
    "src/diamond/alphazero/rating/openings.py",
}

ENGINE_ITSELF = {
    "src/diamond/game/__init__.py",
    "src/diamond/game/board.py",
    "src/diamond/game/history.py",
    "src/diamond/game/rules.py",
    "src/diamond/game/session.py",
    "src/diamond/game/state.py",
    "src/diamond/alphazero/mcts/__init__.py",
    "src/diamond/alphazero/mcts/search_2p.py",
    "src/diamond/alphazero/mcts/search_3p.py",
}


def _tracked_modules() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "src/diamond"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:  # pragma: no cover - not a git checkout
        return []
    return [path for path in result.stdout.split() if path.endswith(".py")]


def _engine_imports(path: str) -> set[str]:
    """Names this module takes from the Python engine."""
    try:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):  # pragma: no cover - unparseable source
        return set()
    package = Path(path).parent.as_posix().removeprefix("src/")
    imported: set[str] = set()
    for node in ast.walk(tree):
        module = None
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split("/")
                parts = parts[: len(parts) - node.level + 1]
                module = ".".join(filter(None, [*parts, module]))
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            module = node.names[0].name
            names = [module]
        if module and module.startswith(ENGINE_MODULES):
            imported.update(names)
    return imported


def _classify() -> tuple[set[str], set[str]]:
    behaviour: set[str] = set()
    definitions: set[str] = set()
    for path in _tracked_modules():
        if path in ENGINE_ITSELF:
            continue
        imported = _engine_imports(path)
        if not imported:
            continue
        if imported & BEHAVIOUR:
            behaviour.add(path)
        else:
            definitions.add(path)
    return behaviour, definitions


def test_the_engine_is_still_here_to_measure() -> None:
    assert _tracked_modules(), "no shipped modules found; every check below is vacuous"
    behaviour, definitions = _classify()
    assert behaviour or definitions, "nothing imports the engine: update or delete this test"


def test_no_new_module_runs_the_python_engine() -> None:
    behaviour, _ = _classify()
    added = sorted(behaviour - BEHAVIOUR_ALLOWED)
    assert not added, (
        f"new modules run the Python engine: {added}. The C++ core is the "
        "authority -- reach for diamond.alphazero.native, or list the module in "
        "BEHAVIOUR_ALLOWED with the reason it cannot yet."
    )


def test_no_new_module_depends_on_the_engine_at_all() -> None:
    _, definitions = _classify()
    added = sorted(definitions - DEFINITIONS_ALLOWED)
    assert not added, (
        f"new modules import the engine's definitions: {added}. Not rules, so "
        "not urgent -- but the list only shrinks."
    )


def test_both_lists_shrink_rather_than_going_stale() -> None:
    """A retired dependent must be struck from its list, not left to rot."""
    behaviour, definitions = _classify()
    stale_behaviour = sorted(BEHAVIOUR_ALLOWED - behaviour)
    stale_definitions = sorted(DEFINITIONS_ALLOWED - definitions)
    assert not stale_behaviour, (
        f"these no longer run the Python engine: {stale_behaviour}. "
        "Delete their lines from BEHAVIOUR_ALLOWED -- that deletion is the progress."
    )
    assert not stale_definitions, (
        f"these no longer import the engine: {stale_definitions}. "
        "Delete their lines from DEFINITIONS_ALLOWED."
    )
