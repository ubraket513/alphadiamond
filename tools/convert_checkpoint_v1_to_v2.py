"""Offline, trusted conversion of a Python v1 checkpoint into v2 tensor data.

This intentionally accepts only a local artifact chosen by an operator.  It
uses ``weights_only=True`` and writes a generation plus ``CURRENT`` pointer;
the native runtime never opens the Python v1 pickle container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise SystemExit("source must be a trusted Python checkpoint v1")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise SystemExit("v1 checkpoint has no tensor-only model_state_dict")
    generation = "generation-" + hashlib.sha256(args.source.read_bytes()).hexdigest()[:24]
    staged = args.destination / "generations" / ("." + generation + ".staging")
    target = args.destination / "generations" / generation
    staged.mkdir(parents=True, exist_ok=False)
    tensors: list[dict[str, object]] = []
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous().to(torch.float32)
        filename = hashlib.sha256(name.encode()).hexdigest() + ".f32"
        (staged / filename).write_bytes(tensor.numpy().tobytes())
        tensors.append({"name": name, "file": filename, "shape": list(tensor.shape), "dtype": "float32"})
    (staged / "manifest.json").write_text(json.dumps({
        "format_version": 2, "generation": generation,
        "metadata": payload.get("metadata"), "training_step": payload.get("training_step"),
        "tensors": tensors,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True), encoding="ascii")
    os.replace(staged, target)
    pointer = args.destination / ".CURRENT.tmp"
    pointer.write_text(generation + "\n", encoding="ascii")
    os.replace(pointer, args.destination / "CURRENT")


if __name__ == "__main__":
    main()
