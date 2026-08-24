"""Setuptools shim: project metadata stays in pyproject.toml.

This file exists only to declare the optional native extension.  ``pyproject``
cannot express ``ext_modules``, and the extension must stay *optional*: with
``selfplay_backend = "python"`` (the default) the package, the CLI and the whole
test suite must work on a host with no compiler and no pybind11.
"""

from __future__ import annotations

import os

from setuptools import setup

EXT_NAME = "diamond.alphazero.native._diamond_native"

SOURCES = [
    "native/bindings.cpp",
    "native/src/batcher.cpp",
    "native/src/board.cpp",
    "native/src/encoder.cpp",
    "native/src/evaluator.cpp",
    "native/src/mcts.cpp",
    "native/src/mcts3p.cpp",
    "native/src/prior.cpp",
    "native/src/profile.cpp",
    "native/src/selfplay.cpp",
    "native/src/rules.cpp",
    "native/src/topology_gen.cpp",
    "native/src/topology_io.cpp",
]

# Broadwell is the GPU training host's baseline (AVX2) and a subset of every
# development machine used so far.  Override for a different target.
ARCH = os.environ.get("DIAMOND_NATIVE_ARCH", "broadwell")


def _extensions() -> list:
    if os.environ.get("DIAMOND_NATIVE_SKIP"):
        return []
    try:
        from pybind11.setup_helpers import Pybind11Extension
    except ImportError:
        return []

    flags = ["-O3", "-fvisibility=hidden"]
    if ARCH and ARCH != "none":
        flags.append(f"-march={ARCH}")
    return [
        Pybind11Extension(
            EXT_NAME,
            sorted(SOURCES),
            include_dirs=["native/include"],
            cxx_std=20,
            extra_compile_args=flags,
            optional=True,
        )
    ]


setup(ext_modules=_extensions())
