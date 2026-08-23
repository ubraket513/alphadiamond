"""Have the transplanted blocks started doing anything?

``deepen_checkpoint.py`` makes a new block an exact identity by zeroing its
LayerNorm scale and bias, so at initialisation the deeper network *is* its
parent.  That is the point, but it also means a gate result taken too early
cannot distinguish two very different situations:

* the new depth was used and did not help, and
* the new depth has not switched on yet.

With ``gamma = 0`` the residual branch contributes exactly zero, and the
gradient reaching the block's own projections is scaled by that same zero.
``gamma`` and ``beta`` do receive gradient, so the block opens up over the first
few hundred steps and only then does the rest of it begin to train.  Judging
depth before that has happened measures the schedule, not the architecture.

This reports, per block, whether it has opened and whether it is contributing,
and compares the whole network's function against the parent it came from.

Usage::

    python tools/inspect_deepened_blocks.py --checkpoint deep.pt --parent 44250.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import checkpoint_network_config
from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.network import SooModel
from diamond.game.state import GameState, GameStatus, build_players


def _model(path: Path) -> tuple[SooModel, NetworkConfig, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    network = checkpoint_network_config(path)
    version = payload["metadata"]["model_version"]
    model = SooModel(network, model_version=version)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval(), network, int(payload["training_step"])


def _features(corpus: Path, limit: int) -> torch.Tensor:
    players = build_players(2)
    game = AlphaZeroGameAdapter(players)
    rows = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(rows) >= limit:
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
        rows.append(
            torch.tensor(game.encoder.encode(state, players).node_features, dtype=torch.float32)
        )
    if not rows:
        raise SystemExit(f"corpus {corpus} yielded no two-player positions")
    return torch.stack(rows)


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().pow(2).mean().sqrt())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="The checkpoint this one was transplanted from.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "tests/native/fixtures/positions.jsonl"
    )
    parser.add_argument("--positions", type=int, default=512)
    args = parser.parse_args()

    child, child_network, child_step = _model(args.checkpoint)
    parent, parent_network, parent_step = _model(args.parent)
    if child_network.width != parent_network.width:
        raise SystemExit("width differs; this tool compares a depth transplant")
    added = child_network.residual_blocks - parent_network.residual_blocks
    if added <= 0:
        raise SystemExit("child is not deeper than parent")

    features = _features(args.corpus, args.positions)
    print(
        f"child {child_network.width}x{child_network.residual_blocks} step {child_step}  "
        f"<- parent {parent_network.width}x{parent_network.residual_blocks} step {parent_step}  "
        f"({child_step - parent_step} steps, {added} new blocks, "
        f"{len(features)} positions)\n"
    )

    # --- per block: has gamma opened, and is the residual branch contributing?
    contributions: dict[int, tuple[float, float]] = {}
    handles = []

    def watch(index: int):
        def hook(module, inputs, output):
            nodes = inputs[0]
            contributions[index] = (_rms(output - nodes), _rms(nodes))
        return hook

    for index, block in enumerate(child.trunk.blocks):
        handles.append(block.register_forward_hook(watch(index)))
    with torch.no_grad():
        child(features)
    for handle in handles:
        handle.remove()

    parent_state = parent.state_dict()
    print(f"  {'block':>6}{'gamma RMS':>12}{'beta RMS':>11}{'residual RMS':>14}"
          f"{'/ input':>10}{'param delta':>13}")
    for index, block in enumerate(child.trunk.blocks):
        gamma = _rms(block.norm.weight)
        beta = _rms(block.norm.bias)
        residual, incoming = contributions[index]
        share = residual / incoming if incoming else float("nan")
        delta = ""
        prefix = f"trunk.blocks.{index}."
        if index < parent_network.residual_blocks:
            moved = sum(
                float((value - parent_state[prefix + name]).pow(2).sum())
                for name, value in block.state_dict().items()
                if prefix + name in parent_state
            ) ** 0.5
            delta = f"{moved:.4f}"
        else:
            # A new block has no parent tensor; its projections started at
            # torch's default init, so distance from *that* is meaningless.
            # What matters is whether the norm has left zero, reported above.
            delta = "new"
        marker = "  <- new" if index >= parent_network.residual_blocks else ""
        print(
            f"  {index:>6}{gamma:>12.5f}{beta:>11.5f}{residual:>14.5f}"
            f"{share:>10.4f}{delta:>13}{marker}"
        )

    # --- whole-network: has the function moved away from the parent at all?
    with torch.no_grad():
        child_logits, child_value = child(features)
        parent_logits, parent_value = parent(features)
        child_log = torch.log_softmax(child_logits, dim=-1)
        parent_log = torch.log_softmax(parent_logits, dim=-1)
        kl = float((parent_log.exp() * (parent_log - child_log)).sum(-1).mean())
        value_rmse = float((child_value - parent_value).pow(2).mean().sqrt())
        value_bias = float((child_value - parent_value).mean())

    opened = [
        index
        for index in range(parent_network.residual_blocks, child_network.residual_blocks)
        if _rms(child.trunk.blocks[index].norm.weight) > 1e-4
    ]
    print(
        f"\n  new blocks with gamma off zero: {len(opened)}/{added}"
        f"   {opened if opened else ''}"
    )
    print(f"  policy KL(parent || child), full action space: {kl:.6f} nats")
    print(f"  value RMSE vs parent: {value_rmse:.6f}   (mean shift {value_bias:+.6f})")

    new_share = [contributions[i][0] / contributions[i][1]
                 for i in range(parent_network.residual_blocks, child_network.residual_blocks)
                 if contributions[i][1]]
    old_share = [contributions[i][0] / contributions[i][1]
                 for i in range(parent_network.residual_blocks)
                 if contributions[i][1]]
    if new_share and old_share:
        ratio = (sum(new_share) / len(new_share)) / (sum(old_share) / len(old_share))
        print(
            f"  mean residual share, new blocks vs inherited: {ratio:.3f}"
            "   (1.0 would be new blocks pulling their weight)"
        )
        if ratio < 0.1:
            print(
                "\n  The new blocks are still nearly inert.  A gate taken now "
                "measures the warm-up schedule, not the depth."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
