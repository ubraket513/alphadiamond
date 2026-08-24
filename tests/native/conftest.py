"""Shared fixtures for the pybind boundary tests.

Every test here is skipped when the extension is not built. That is not a
fallback -- there is nothing to fall back to (decision 1) -- it is so a
developer without a compiler gets a clear skip rather than a wall of import
errors. CI builds the extension, so nothing is silently skipped there.
"""

from __future__ import annotations

import pytest

from diamond.alphazero.native import is_available, native_error


@pytest.fixture(scope="session", autouse=True)
def _require_extension() -> None:
    if not is_available():
        pytest.skip(native_error() or "native extension unavailable", allow_module_level=True)
