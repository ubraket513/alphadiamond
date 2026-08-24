"""The Python engine's dependent set may shrink, never grow.

The C++ core is the authority for rules, encoding, search and self-play. The
Python engine in ``diamond.game`` (and the Python MCTS beside it) is kept for
two jobs only: it is the oracle that generates ``tests/golden/``, and it is the
other half of every bridge parity gate. Everything else that still imports it
is migration debt.

Debt that nobody counts grows. So this test counts it: each shipped module that
still reaches for the Python engine is listed below, and adding a new one fails.
Removing one means deleting its line -- which is the only way this list is meant
to change.

See docs/architecture/retiring_the_python_engine.md for what has to be true
before ``src/diamond/game`` can be deleted outright.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENGINE_MODULES = ("diamond.game", "diamond.alphazero.mcts")

# Sorted; every entry is debt. The count is asserted separately so that
# swapping one dependent for another cannot pass silently.
ALLOWED = {
    # The engine itself, and the oracle's own supporting modules.
    "src/diamond/game/__init__.py",
    "src/diamond/game/board.py",
    "src/diamond/game/history.py",
    "src/diamond/game/rules.py",
    "src/diamond/game/session.py",
    "src/diamond/game/state.py",
    "src/diamond/alphazero/mcts/__init__.py",
    "src/diamond/alphazero/mcts/search_2p.py",
    "src/diamond/alphazero/mcts/search_3p.py",
    # The bridge: these exist to hand the Python board to the native side.
    "src/diamond/alphazero/native/__init__.py",
    "src/diamond/alphazero/native/topology.py",
    # Training, research and orchestration, still on the Python engine.
    "src/diamond/agents/alphazero_agent.py",
    "src/diamond/agents/base.py",
    "src/diamond/agents/random_agent.py",
    "src/diamond/alphazero/arena.py",
    "src/diamond/alphazero/bootstrap/evaluator.py",
    "src/diamond/alphazero/bootstrap/heuristic.py",
    "src/diamond/alphazero/bootstrap/probe.py",
    "src/diamond/alphazero/encoder.py",
    "src/diamond/alphazero/game_adapter.py",
    "src/diamond/alphazero/identity.py",
    "src/diamond/alphazero/milestone2_smoke.py",
    "src/diamond/alphazero/network/trunk.py",
    "src/diamond/alphazero/orchestration/benchmark.py",
    "src/diamond/alphazero/orchestration/production.py",
    "src/diamond/alphazero/orchestration/selfplay_workers.py",
    "src/diamond/alphazero/rating/openings.py",
    "src/diamond/alphazero/selfplay/runner_2p.py",
    "src/diamond/alphazero/selfplay/runner_3p.py",
    "src/diamond/alphazero/smoke.py",
}


def _tracked_modules() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "src/diamond"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:  # pragma: no cover - not a git checkout
        return []
    return [path for path in result.stdout.split() if path.endswith(".py")]


def _imports_engine(path: str) -> bool:
    try:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):  # pragma: no cover - unparseable source
        return False
    package = Path(path).parent.as_posix().removeprefix("src/")
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split("/")
                parts = parts[: len(parts) - node.level + 1]
                module = ".".join(filter(None, [*parts, module]))
        elif isinstance(node, ast.Import):
            module = node.names[0].name
        if module and module.startswith(ENGINE_MODULES):
            return True
    return False


def test_no_new_dependents_on_the_python_engine() -> None:
    modules = _tracked_modules()
    assert modules, "no shipped modules found; the check would pass vacuously"
    actual = {path for path in modules if _imports_engine(path)}

    added = sorted(actual - ALLOWED)
    assert not added, (
        "new modules depend on the Python engine: "
        f"{added}. The C++ core is the authority -- reach for "
        "diamond.alphazero.native instead, or add the module to ALLOWED with a "
        "reason if it is genuinely oracle or bridge code."
    )


def test_the_list_shrinks_rather_than_going_stale() -> None:
    """A retired dependent must be struck from the list, not left to rot."""
    modules = _tracked_modules()
    assert modules, "no shipped modules found; the check would pass vacuously"
    actual = {path for path in modules if _imports_engine(path)}

    retired = sorted(ALLOWED - actual)
    assert not retired, (
        f"these no longer depend on the Python engine: {retired}. "
        "Delete their lines from ALLOWED -- that deletion is the progress."
    )
