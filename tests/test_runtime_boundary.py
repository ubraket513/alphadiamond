from __future__ import annotations

import tomllib
from pathlib import Path

from setuptools import find_namespace_packages, find_packages


def test_python_distribution_has_no_gui_runtime() -> None:
    """The shipped Python package carries no Qt runtime.

    Checked against the packaging configuration rather than by importing: in a
    checkout -- and in the editable install CI uses -- the GUI sources are on
    the path and importable, which says nothing about what a built distribution
    contains. Running setuptools' own discovery with the configured excludes
    asks the question that actually decides it.
    """
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    project = metadata["project"]
    declared = list(project.get("dependencies", []))
    for dependencies in project.get("optional-dependencies", {}).values():
        declared.extend(dependencies)

    normalized = "\n".join(declared).lower()
    assert "pyside" not in normalized
    assert "qtawesome" not in normalized
    assert "diamond-legacy" not in project.get("scripts", {})

    find = metadata["tool"]["setuptools"]["packages"]["find"]
    excluded = find["exclude"]
    assert "diamond.app*" in excluded
    assert "diamond.assets*" in excluded
    assert "diamond.qml*" in excluded

    where = (project_root / find["where"][0]).as_posix()
    assert "diamond.alphazero" in find_packages(where=where, exclude=tuple(excluded)), (
        "discovery found no engine packages at all"
    )

    # Namespace discovery is what sees the GUI directories: they carry no
    # __init__.py, so plain find_packages would report them absent whether or
    # not the exclusion existed, and the test would pass vacuously.
    gui_prefixes = ("diamond.app", "diamond.assets", "diamond.qml")
    discovered = find_namespace_packages(where=where)
    assert any(package.startswith(gui_prefixes) for package in discovered), (
        "no GUI packages in the tree: this test can no longer prove they are excluded"
    )
    shipped = [
        package
        for package in find_namespace_packages(where=where, exclude=tuple(excluded))
        if package.startswith(gui_prefixes)
    ]
    assert not shipped, f"the distribution would ship GUI packages: {sorted(shipped)}"
