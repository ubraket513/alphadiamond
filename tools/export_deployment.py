"""Export an authoritative Python model to a portable deployment artifact.

The artifact contains explicit metadata, a TorchScript graph, raw weight
tensors, the board topology and a small deterministic parity corpus, so the
native runtime can reject an incompatible bundle before loading a single
tensor.

Format 3 describes the model rather than assuming it: `model_family` and
`architecture` are declared, and the native loader checks the weights against
what was declared. That is what lets Min ship without a format redesign.

A *training checkpoint* is not a deployment artifact: it carries optimizer,
scheduler and RNG state and stays in the bucket. Only what the application
needs for inference is exported here.

Usage::

    python tools/export_deployment.py <output-dir> [--family soo|min] \\
        [--checkpoint path/to/checkpoint.pt] [--training-commit <sha>] \\
        [--training-step N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import torch

from diamond.alphazero.checkpoint import load_inference_checkpoint
from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.deployment import (
    ARTIFACT_FORMAT_VERSION,
    GAME_CONTRACT,
    validate_metadata,
)
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.native.topology import player_table, topology_tables
from diamond.alphazero.network.min import MinModel
from diamond.alphazero.network.soo import SooModel
from diamond.contract.state import build_players

MODEL_VERSIONS = {"soo": "2.0.0", "min": "0.1.0"}
MODEL_CLASSES = {"soo": SooModel, "min": MinModel}
ARCHITECTURE_TYPE = "directional_residual"
"""Both families are the same graph trunk over the same board; they differ in
input features and value-head width, which the metadata declares."""

CORPUS_SEED = 20260823
CORPUS_BATCH = 2
CORPUS_LEGAL_ACTIONS = (0, 1, 42, 5328)


def _write_f32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def _weight_filename(key: str) -> str:
    return key.replace(".", "__") + ".f32"


def _write_i32(path: Path, values: list[int] | tuple[int, ...]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def _write_i8(path: Path, rows: tuple[tuple[int, ...], ...]) -> None:
    flat = [value for row in rows for value in row]
    path.write_bytes(struct.pack(f"<{len(flat)}b", *flat))


def _export_topology(output: Path) -> None:
    """The board tables, exported rather than transcribed into C++."""
    tables = topology_tables()
    for name in (
        "neighbour",
        "camp_positions",
        "pairwise_distance",
        "physical_to_canonical",
        "canonical_to_physical",
    ):
        rows = tuple(tuple(int(value) for value in row) for row in tables[name])
        if name == "neighbour":
            _write_i8(output / "topology_neighbour.i8", rows)
        else:
            _write_i32(output / f"topology_{name}.i32", [value for row in rows for value in row])


def _export_mcts_fixture(output: Path, player_count: int) -> None:
    players = build_players(player_count)
    game = AlphaZeroGameAdapter(players)
    adapter = DiamondSearchAdapter(game)
    state = adapter.initial_state()
    _write_i8(output / "mcts_occupancy.u8", (tuple(int(value) for value in state.occupancy),))
    _write_i32(
        output / "mcts_players.i32", [value for row in player_table(players) for value in row]
    )
    (output / "mcts_current_player.u8").write_bytes(bytes([state.current_player_id]))
    _write_i32(output / "mcts_legal_actions.i32", adapter.legal_action_ids(state))


def _runtime_sha256(output: Path) -> str:
    runtime_files = [
        output / "topology_neighbour.i8",
        output / "topology_camp_positions.i32",
        output / "topology_pairwise_distance.i32",
        output / "topology_physical_to_canonical.i32",
        output / "topology_canonical_to_physical.i32",
        *sorted((output / "weights").glob("*.f32")),
    ]
    digest = hashlib.sha256()
    for path in sorted(runtime_files, key=lambda item: item.relative_to(output).as_posix()):
        relative = path.relative_to(output).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _training_commit(explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a git checkout
        return None
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def export(
    output: Path,
    *,
    family: str = "soo",
    checkpoint: Path | None = None,
    training_commit: str | None = None,
    training_step: int | None = None,
) -> dict:
    if family not in MODEL_CLASSES:
        raise SystemExit(f"unknown model family: {family}")

    config = NetworkConfig()
    model_version = MODEL_VERSIONS[family]
    model = MODEL_CLASSES[family](config, model_version=model_version)
    checkpoint_sha256: str | None = None

    if checkpoint is not None:
        builder = getattr(CheckpointCompatibilitySpec, family)
        expected = builder(model_version=model_version, network_config=config)
        info = load_inference_checkpoint(checkpoint, model, expected=expected)
        checkpoint_sha256 = info.checkpoint_sha256

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(CORPUS_SEED)
    inputs = torch.randn(CORPUS_BATCH, 73, model.input_features, generator=generator)
    with torch.inference_mode():
        policy, value = model(inputs)
        legal_actions = torch.tensor(CORPUS_LEGAL_ACTIONS, dtype=torch.long)
        legal_priors = torch.softmax(policy[:, legal_actions], dim=1)

    output.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, inputs, strict=True)
    traced.save(str(output / "model.ts"))
    _write_f32(output / "inputs.f32", inputs)
    _write_f32(output / "expected_policy.f32", policy)
    _write_f32(output / "expected_value.f32", value)
    _write_f32(output / "expected_legal_priors.f32", legal_priors)
    _write_i32(output / "legal_actions.i32", list(CORPUS_LEGAL_ACTIONS))

    weights = output / "weights"
    weights.mkdir(exist_ok=True)
    for key, tensor in model.state_dict().items():
        _write_f32(weights / _weight_filename(key), tensor)

    _export_topology(output)
    _export_mcts_fixture(output, 2 if family == "soo" else 3)

    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_family": family,
        "model_version": model_version,
        "architecture": {
            "type": ARCHITECTURE_TYPE,
            "width": config.width,
            "residual_blocks": config.residual_blocks,
        },
        "game_contract": dict(GAME_CONTRACT),
        "tensor_shapes": {
            "input": [CORPUS_BATCH, 73, model.input_features],
            "policy": [CORPUS_BATCH, 5329],
            "value": [CORPUS_BATCH, model.value_size],
        },
        "dtype": "float32",
        "source": {
            "checkpoint_sha256": checkpoint_sha256,
            "training_commit": _training_commit(training_commit),
            "training_step": training_step,
        },
        "corpus_seed": CORPUS_SEED,
        "model_sha256": hashlib.sha256((output / "model.ts").read_bytes()).hexdigest(),
        "runtime_sha256": _runtime_sha256(output),
    }
    validate_metadata(metadata)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", default="soo", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-commit", default=None)
    parser.add_argument("--training-step", type=int, default=None)
    args = parser.parse_args()
    export(
        args.output,
        family=args.family,
        checkpoint=args.checkpoint,
        training_commit=args.training_commit,
        training_step=args.training_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
