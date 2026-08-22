"""Shared fixtures for the native parity gates.

Every test here is skipped when the extension is not built: the Python backend
is the default and must stay green on a host with no compiler.
"""

from __future__ import annotations

import pytest

from diamond.alphazero.native import is_available, native_error


@pytest.fixture(scope="session", autouse=True)
def _require_extension() -> None:
    if not is_available():
        pytest.skip(native_error() or "native extension unavailable", allow_module_level=True)
