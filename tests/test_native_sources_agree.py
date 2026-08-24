"""The two build systems must compile the same core.

CMake builds the native application and its tests; ``setup.py`` builds the
pybind extension the trainer imports. They keep separate source lists, so a new
core file added to one and not the other links in one product and fails in the
other -- which is exactly what happened when ``topology_gen.cpp`` arrived, and
the failure surfaced as an unresolved symbol at link time with the previous
``.pyd`` still sitting on disk.

The lists are not identical by nature: ``setup.py`` also compiles
``native/bindings.cpp``, which is the bridge and belongs to no CMake core
target. What must hold is that every C++ file CMake compiles into the core and
the search is also compiled into the extension.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cmake_sources() -> set[str]:
    text = (ROOT / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
    sources: set[str] = set()
    for variable in ("SOO_CORE_SOURCES", "SOO_SEARCH_SOURCES"):
        match = re.search(rf"set\({variable}\s*(.*?)\)", text, re.DOTALL)
        assert match, f"{variable} is gone from native/CMakeLists.txt"
        sources.update(f"native/{name}" for name in re.findall(r"(src/\S+\.cpp)", match.group(1)))
    assert sources, "no CMake sources parsed; this check would pass vacuously"
    return sources


def _setup_sources() -> set[str]:
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"SOURCES\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert match, "SOURCES is gone from setup.py"
    sources = set(re.findall(r'"(native/[^"]+\.cpp)"', match.group(1)))
    assert sources, "no setup.py sources parsed; this check would pass vacuously"
    return sources


def test_every_cmake_core_source_is_in_the_extension() -> None:
    missing = sorted(_cmake_sources() - _setup_sources())
    assert not missing, (
        f"CMake compiles these and setup.py does not: {missing}. The extension "
        "will fail to link, or worse, link against a stale object."
    )


def test_the_extension_compiles_nothing_the_core_does_not_have() -> None:
    extra = sorted(_setup_sources() - _cmake_sources() - {"native/bindings.cpp"})
    assert not extra, (
        f"setup.py compiles these and no CMake core target does: {extra}. Either "
        "the file belongs in a CMake source list, or the extension is carrying "
        "code the native application never builds or tests."
    )


def test_every_listed_source_exists() -> None:
    for source in sorted(_cmake_sources() | _setup_sources()):
        assert (ROOT / source).is_file(), f"a build list names a missing file: {source}"
