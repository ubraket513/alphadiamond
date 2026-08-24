"""Training game execution requires the native extension, and says so.

Decision 1 in docs/architecture/decisions.md. The rule this setting has always
enforced is that a run's data is attributable to the engine that produced it;
with one engine left, enforcing it means refusing the settings that used to name
another rather than quietly substituting.

`python` and `auto` are still *recognised* on purpose. A config asking for the
Python backend was asking for a specific engine, and the useful answer is an
error that says what happened to it -- not `unknown backend`, and certainly not
a silent switch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("az_train_module", ROOT / "tools" / "az_train.py")
assert _spec and _spec.loader
az_train = importlib.util.module_from_spec(_spec)
sys.modules["az_train_module"] = az_train
_spec.loader.exec_module(az_train)

resolve = az_train.resolve_selfplay_backend
NativeExtensionRequired = az_train.NativeExtensionRequired


def test_native_is_the_only_backend() -> None:
    assert az_train.SELFPLAY_BACKENDS == ("native",)
    assert resolve("native", max_game_seconds=None) == "native"


def test_the_retired_backends_explain_themselves() -> None:
    for retired in ("python", "auto"):
        with pytest.raises(NativeExtensionRequired, match="native extension"):
            resolve(retired, max_game_seconds=None)


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(NativeExtensionRequired, match="unknown"):
        resolve("cuda-telepathy", max_game_seconds=None)


def test_a_missing_extension_is_a_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract is the compiled extension, so its absence is the error."""
    from diamond.alphazero import native

    monkeypatch.setattr(native, "is_available", lambda: False)
    monkeypatch.setattr(native, "native_error", lambda: "not built here")
    with pytest.raises(NativeExtensionRequired, match="not built here"):
        resolve("native", max_game_seconds=None)


def test_a_wall_clock_budget_is_refused_rather_than_dropped() -> None:
    """The native runner bounds a game by moves. Ignoring a configured budget
    would let a run exceed a limit its own config says it respects."""
    with pytest.raises(NativeExtensionRequired, match="max_game_seconds"):
        resolve("native", max_game_seconds=900.0)


def test_the_default_is_native() -> None:
    source = (ROOT / "tools" / "az_train.py").read_text(encoding="utf-8")
    assert 'workers.get("selfplay_backend", "native")' in source
