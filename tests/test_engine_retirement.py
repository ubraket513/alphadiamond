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
  core replaced. This list had to reach zero before ``diamond.game`` could be
  deleted, and it is now empty: the searches come from ``search_factory`` and
  ``game_adapter`` applies moves through the native ``Game``. It stays empty.
* **definitions and types** -- ``standard_board``, ``build_players``,
  ``GameState``: the board, the seats, and the shapes that describe a position.
  Also empty. They were never rules, and they now live in ``diamond.contract``,
  which nothing plans to delete.

Both lists being empty is the point: ``src/diamond/game`` is reachable from the
oracle (``tools/build_golden.py``) and the bridge gates in ``tests/``, and from
nothing that ships.

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
# The work queue, drained. An entry here is a module that runs the Python
# engine; there are none, and adding one is what this file exists to refuse.
# ---------------------------------------------------------------------------
BEHAVIOUR_ALLOWED: set[str] = set()
"""Empty, and it stays empty.

Nothing shipped runs the Python engine any more: the searches come from
``search_factory``, the smokes ask the core for legality, and ``game_adapter``
-- the last entry -- applies moves through the native ``Game``
(``tests/native/test_game_adapter_parity.py`` holds the two to the same
successor for every legal action of every fixture position).

Phase A is done. What ``diamond.game`` still supplies is definitions: the board,
the seats, and ``GameState``. That is Phase B, below."""

# ---------------------------------------------------------------------------
# Not the work queue: the board, the seats, and the dataclasses that describe a
# position. Empty since those moved to ``diamond.contract`` -- the same classes,
# a package that is not scheduled for deletion, and the dependency now points
# away from the engine being retired rather than into it.
# ---------------------------------------------------------------------------
DEFINITIONS_ALLOWED: set[str] = set()

ENGINE_ITSELF = {
    "src/diamond/game/__init__.py",
    "src/diamond/game/history.py",
    "src/diamond/game/rules.py",
    "src/diamond/game/session.py",
    "src/diamond/alphazero/mcts/__init__.py",
    "src/diamond/alphazero/mcts/search_2p.py",
    "src/diamond/alphazero/mcts/search_3p.py",
    "src/diamond/alphazero/mcts/tree.py",
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


def _engine_exports() -> set[str]:
    """Every name the engine's modules define, so BEHAVIOUR cannot rot silently."""
    names: set[str] = set()
    for path in sorted(ENGINE_ITSELF):
        try:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):  # pragma: no cover - unparseable source
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
    return names


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
    """The measurement must stay wired up even now that both counts are zero."""
    assert _tracked_modules(), "no shipped modules found; every check below is vacuous"
    assert (ROOT / "src" / "diamond" / "game" / "rules.py").is_file(), (
        "the Python engine is gone: delete this test with it, and the bridge "
        "gates in tests/native that compare against it"
    )
    assert BEHAVIOUR & _engine_exports(), (
        "BEHAVIOUR names nothing the engine exports; the list has gone stale "
        "and would pass whatever shipped code did"
    )


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
