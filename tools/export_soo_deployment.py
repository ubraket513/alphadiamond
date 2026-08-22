"""Export the authoritative Python Soo model to a portable LibTorch artifact.

This is the Q2 compatibility spike, not the final packaging format. The
artifact deliberately contains explicit metadata, a TorchScript graph, and a
small deterministic parity corpus so a native probe can reject incompatible
inputs before GUI integration begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

from diamond.alphazero.checkpoint import load_inference_checkpoint
from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network.soo import SooModel

ARTIFACT_FORMAT_VERSION = 1
MODEL_VERSION = "0.1.0"
CORPUS_SEED = 20260823
CORPUS_BATCH = 2


def _write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def export(output: Path, checkpoint: Path | None = None) -> None:
    config = NetworkConfig()
    model = SooModel(config, model_version=MODEL_VERSION)
    checkpoint_sha256: str | None = None

    if checkpoint is not None:
        expected = CheckpointCompatibilitySpec.soo(
            model_version=MODEL_VERSION,
            network_config=config,
        )
        info = load_inference_checkpoint(checkpoint, model, expected=expected)
        checkpoint_sha256 = info.checkpoint_sha256

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(CORPUS_SEED)
    inputs = torch.randn(CORPUS_BATCH, 73, 4, generator=generator)
    with torch.inference_mode():
        policy, value = model(inputs)

    output.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, inputs, strict=True)
    traced.save(str(output / "model.ts"))
    _write_f32(output / "inputs.f32", inputs)
    _write_f32(output / "expected_policy.f32", policy)
    _write_f32(output / "expected_value.f32", value)

    model_sha256 = hashlib.sha256((output / "model.ts").read_bytes()).hexdigest()
    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_name": "Soo",
        "model_version": MODEL_VERSION,
        "input_shape": [CORPUS_BATCH, 73, 4],
        "policy_shape": [CORPUS_BATCH, 5329],
        "value_shape": [CORPUS_BATCH, 1],
        "dtype": "float32",
        "width": config.width,
        "residual_blocks": config.residual_blocks,
        "board_topology_version": "diamond73-v1",
        "encoder_version": "diamond-camp-relative-v1",
        "action_space_version": "diamond73-srcdst-v1",
        "corpus_seed": CORPUS_SEED,
        "model_sha256": model_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    export(args.output, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
