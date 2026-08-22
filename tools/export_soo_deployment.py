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
from diamond.alphazero.deployment import validate_metadata
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network.soo import SooModel
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.native.topology import player_table, topology_tables
from diamond.game.state import build_players

ARTIFACT_FORMAT_VERSION = 1
MODEL_VERSION = "0.1.0"
CORPUS_SEED = 20260823
CORPUS_BATCH = 2


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


def _export_mcts_fixture(output: Path) -> None:
    """Export Python-authoritative data needed by the native MCTS probe."""
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

    players = build_players(2)
    game = AlphaZeroGameAdapter(players)
    adapter = DiamondSearchAdapter(game)
    state = adapter.initial_state()
    _write_i8(output / "mcts_occupancy.u8", (tuple(int(value) for value in state.occupancy),))
    _write_i32(output / "mcts_players.i32", [value for row in player_table(players) for value in row])
    (output / "mcts_current_player.u8").write_bytes(bytes([state.current_player_id]))
    _write_i32(output / "mcts_legal_actions.i32", adapter.legal_action_ids(state))


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
        legal_actions = torch.tensor([0, 1, 42, 5328], dtype=torch.long)
        legal_priors = torch.softmax(policy[:, legal_actions], dim=1)

    output.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, inputs, strict=True)
    traced.save(str(output / "model.ts"))
    _write_f32(output / "inputs.f32", inputs)
    _write_f32(output / "expected_policy.f32", policy)
    _write_f32(output / "expected_value.f32", value)
    _write_f32(output / "expected_legal_priors.f32", legal_priors)
    (output / "legal_actions.i32").write_bytes(struct.pack("<4i", 0, 1, 42, 5328))
    weights = output / "weights"
    weights.mkdir(exist_ok=True)
    for key, tensor in model.state_dict().items():
        _write_f32(weights / _weight_filename(key), tensor)
    _export_mcts_fixture(output)

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
    validate_metadata(metadata)
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
