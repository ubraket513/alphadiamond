"""`auto` picks a self-play backend without ever guessing silently.

The rule the training loop has always held is that a run's data can be
attributed to the engine that produced it. `auto` keeps that: it resolves to a
concrete backend before the run starts, prints why, and the resolved value is
what gets recorded in the run config.
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


def test_explicit_choices_are_never_overridden():
    assert resolve("python", max_game_seconds=None) == "python"
    assert resolve("native", max_game_seconds=None) == "native"
    # Even a request the native runner cannot honour is left alone: an explicit
    # backend must fail loudly later, not be silently swapped here.
    assert resolve("native", max_game_seconds=900.0) == "native"


def test_auto_falls_back_when_the_run_needs_a_wall_clock_bound():
    """The native runner bounds a game by moves, not by seconds."""
    assert resolve("auto", max_game_seconds=900.0) == "python"


def test_auto_follows_extension_availability(monkeypatch: pytest.MonkeyPatch):
    from diamond.alphazero import native

    monkeypatch.setattr(native, "is_available", lambda: True)
    assert resolve("auto", max_game_seconds=None) == "native"

    monkeypatch.setattr(native, "is_available", lambda: False)
    monkeypatch.setattr(native, "native_error", lambda: "not built here")
    assert resolve("auto", max_game_seconds=None) == "python"


def test_auto_is_an_accepted_backend():
    assert "auto" in az_train.SELFPLAY_BACKENDS
