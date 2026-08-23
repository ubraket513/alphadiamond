"""Extend a trained checkpoint with extra residual blocks, as an identity.

The trunk block computes ``nodes + GELU(LayerNorm(message))``.  Zero the new
block's LayerNorm scale *and* bias and the norm emits zeros for every input;
``GELU(0) == 0``, so the block returns ``nodes`` unchanged and the deeper
network is bit-comparable to the shallower one it came from.  Training then
starts from the parent's strength rather than from scratch — the point of the
exercise is the depth, not another cold start.

Gradients still flow: ``d GELU/dx`` at 0 is 0.5, and the gradient with respect
to the LayerNorm scale is proportional to the normalised message, which is not
zero.  The block is inert at initialisation, not frozen.

Width is deliberately *not* expandable here.  Net2WiderNet's identity trick
replicates units and halves the outgoing weights, which is exact for an
elementwise nonlinearity but **not** across a LayerNorm: normalisation runs over
the width axis, so duplicated units change the mean and variance and the
function changes.  A wider network has to be trained from scratch, and that is a
different experiment with a different baseline.

The identity claim is asserted numerically, not assumed -- ``--corpus`` runs the
parent and the child over the gate corpus and reports the worst divergence.

Usage::

    python tools/deepen_checkpoint.py --source latest.pt --out deep12.pt --blocks 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import load_checkpoint, save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer


def _source_network(payload: dict) -> tuple[NetworkConfig, str]:
    """Read the shape from the checkpoint itself rather than from a config file.

    A checkpoint records its own ``network_config``; trusting a separate config
    file is how a transplant silently reads the wrong parent.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("checkpoint has no metadata mapping")
    network = metadata.get("network_config")
    if not isinstance(network, dict):
        raise SystemExit("checkpoint metadata has no network_config")
    return NetworkConfig(**network), str(metadata["model_version"])


def _trainer(network: NetworkConfig, version: str, training: TrainingConfig) -> AlphaZeroTrainer:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    model = SooModel(network, model_version=version)
    return AlphaZeroTrainer(model, compatibility, training)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True, help="target residual_blocks")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device recorded in the output checkpoint's training config.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "tests/native/fixtures/positions.jsonl",
        help="Positions to assert the identity on; '-' skips the assertion.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Worst permitted divergence from the parent's outputs.",
    )
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    source_network, version = _source_network(payload)
    if args.blocks <= source_network.residual_blocks:
        raise SystemExit(
            f"--blocks {args.blocks} does not deepen a "
            f"{source_network.residual_blocks}-block parent"
        )
    if source_network.width <= 0:
        raise SystemExit("parent width must be positive")

    training = TrainingConfig(**payload["training_config"])
    parent = _trainer(source_network, version, training)
    load_checkpoint(
        args.source, parent, expected=parent.compatibility, allow_device_migration=True
    )

    target_network = NetworkConfig(
        width=source_network.width, residual_blocks=args.blocks
    )
    # A fresh optimizer.  Adam's moments are per-parameter and the parameter set
    # has changed; carrying the parent's moments for the surviving tensors while
    # the new blocks start cold mixes two different notions of "where training
    # is", and the state is regenerated within a few hundred steps anyway.
    child_training = TrainingConfig(
        **{**payload["training_config"], "device": args.device}
    )
    child = _trainer(target_network, version, child_training)

    parent_state = parent.model.state_dict()
    child_state = child.model.state_dict()
    copied = 0
    for key, tensor in parent_state.items():
        if key not in child_state:
            raise SystemExit(f"parent parameter {key} has no home in the child")
        if child_state[key].shape != tensor.shape:
            raise SystemExit(f"shape mismatch on {key}: cannot transplant")
        child_state[key] = tensor.clone()
        copied += 1

    # Everything the parent did not have is a new trunk block; make each inert.
    zeroed = []
    for index in range(source_network.residual_blocks, args.blocks):
        for suffix in ("weight", "bias"):
            key = f"trunk.blocks.{index}.norm.{suffix}"
            if key not in child_state:
                raise SystemExit(f"expected new block parameter {key}")
            child_state[key] = torch.zeros_like(child_state[key])
            zeroed.append(key)
    child.model.load_state_dict(child_state)

    added = args.blocks - source_network.residual_blocks
    print(
        f"[transplant] {source_network.width}x{source_network.residual_blocks} "
        f"-> {target_network.width}x{target_network.residual_blocks}: "
        f"{copied} tensors copied, {added} block(s) added, "
        f"{len(zeroed)} norm tensors zeroed"
    )

    if str(args.corpus) != "-":
        worst = _assert_identity(parent.model, child.model, args.corpus)
        print(f"[identity] worst |child - parent| over the corpus: {worst:.3e}")
        if worst > args.tolerance:
            raise SystemExit(
                f"identity assertion failed: {worst:.3e} > {args.tolerance:.3e}"
            )

    # The training step carries over: the child *is* this network, further on.
    child.training_step = parent.training_step
    save_checkpoint(args.out, child)
    print(f"[written] {args.out} at training_step {child.training_step}")
    return 0


def _assert_identity(parent: SooModel, child: SooModel, corpus: Path) -> float:
    """Return the worst absolute divergence over the corpus, policy and value."""
    states = _corpus_features(corpus)
    if not states:
        raise SystemExit(f"corpus {corpus} yielded no positions")
    # Compare on the CPU whatever devices the two trainers were built for: the
    # claim is about the function, and a device mismatch here is an accident of
    # where each checkpoint happened to be recorded.
    parent = parent.to("cpu").eval()
    child = child.to("cpu").eval()
    worst = 0.0
    with torch.no_grad():
        for start in range(0, len(states), 256):
            batch = torch.stack(states[start : start + 256])
            p_policy, p_value = parent(batch)
            c_policy, c_value = child(batch)
            worst = max(
                worst,
                float((c_policy - p_policy).abs().max()),
                float((c_value - p_value).abs().max()),
            )
    return worst


def _corpus_features(corpus: Path) -> list[torch.Tensor]:
    """Encode the two-player corpus positions the way self-play would."""
    import json

    from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
    from diamond.game.state import GameState, GameStatus, build_players

    players = build_players(2)
    game = AlphaZeroGameAdapter(players)
    features = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["player_count"] != 2:
            continue
        state = GameState(
            occupancy=tuple(record["occupancy"]),
            current_player_id=record["current_player_id"],
            turn_number=record["turn_number"],
            status=GameStatus(record["status"]),
            finish_order=tuple(record["finish_order"]),
        )
        encoded = game.encoder.encode(state, players)
        features.append(torch.tensor(encoded.node_features, dtype=torch.float32))
    return features


if __name__ == "__main__":
    raise SystemExit(main())
