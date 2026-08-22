from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path


def test_python_distribution_has_no_gui_runtime() -> None:
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

    excluded = metadata["tool"]["setuptools"]["packages"]["find"]["exclude"]
    assert "diamond.app*" in excluded
    assert "diamond.assets*" in excluded
    assert "diamond.qml*" in excluded
    assert importlib.util.find_spec("diamond.app.controller") is None
    assert importlib.util.find_spec("diamond.main") is None
