"""Worker-count resolution must reflect the CPUs this process may actually use.

A container or cpuset can grant a process far fewer CPUs than the machine has,
so ``os.cpu_count()`` alone would oversubscribe the self-play pool.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from diamond.alphazero.hardware import (
    RESERVED_CPUS,
    available_cpu_count,
    resolve_worker_count,
)


def test_two_cpus_are_reserved_for_the_parent_and_inference() -> None:
    assert RESERVED_CPUS == 2
    assert resolve_worker_count(available=32) == 30
    assert resolve_worker_count(available=8) == 6


def test_worker_count_never_drops_below_one() -> None:
    # Reserving two CPUs must never yield a zero or negative pool.
    assert resolve_worker_count(available=3) == 1
    assert resolve_worker_count(available=2) == 1
    assert resolve_worker_count(available=1) == 1


def test_an_explicit_worker_count_overrides_automatic_resolution() -> None:
    assert resolve_worker_count(4, available=32) == 4
    assert resolve_worker_count(64, available=8) == 64


@pytest.mark.parametrize("configured", (0, -1))
def test_an_explicit_worker_count_must_be_positive(configured: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_worker_count(configured, available=32)


def test_a_non_integer_worker_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_worker_count(True, available=32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        resolve_worker_count(4.0, available=32)  # type: ignore[arg-type]


def test_available_cpu_count_prefers_process_affinity(monkeypatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    # The machine reports 64, but this process may only use 12.
    assert available_cpu_count() == 12


def test_available_cpu_count_falls_back_to_cpu_count(monkeypatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 9)

    assert available_cpu_count() == 9


def test_available_cpu_count_falls_back_to_one(monkeypatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert available_cpu_count() == 1


def test_available_cpu_count_survives_an_unsupported_affinity_call(monkeypatch) -> None:
    def unsupported(pid):
        raise OSError("affinity is unavailable on this platform")

    monkeypatch.setattr(os, "sched_getaffinity", unsupported, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)

    assert available_cpu_count() == 7


def test_resolution_uses_the_detected_cpus_when_none_are_given(monkeypatch) -> None:
    monkeypatch.setattr(
        "diamond.alphazero.hardware.available_cpu_count", lambda: 32
    )

    assert resolve_worker_count() == 30


def test_hardware_module_does_not_import_torch() -> None:
    """Worker processes import this module; torch must stay out of that path."""
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment = os.environ | {"PYTHONPATH": source_root}
    code = """
import sys
import diamond.alphazero.hardware
assert 'torch' not in sys.modules, tuple(sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
