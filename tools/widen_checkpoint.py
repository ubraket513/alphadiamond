"""Double a trained checkpoint's width while preserving the function it computes.

The depth experiment ended with a clear negative: interleaved copied blocks
behind a proper residual gate opened four times wider than appended ones and
still fit the frozen teacher *worse* than the shallow control.  Width is the
next axis, and it deserves a growth operator that does not repeat the depth
arm's failure mode -- a bigger model whose new capacity never switches on.

**Why not plain Net2Wider.**  Duplicating each channel and halving the outgoing
weights preserves the function, but the two copies then receive identical
activations, identical gradients and identical optimizer state, so under
deterministic training they move together forever and a 256-wide model stays two
copies of a 128-wide one.  Net2Net says as much and prescribes breaking the
symmetry with noise; noise is a knob, and a knob that decides whether the
experiment can succeed is not a knob one wants in the experiment.

**What this does instead** is the staged-training expansion: a hidden map
``W`` becomes ``[[W, 0], [0, W]]``.  The function is preserved for the same
reason -- the second half sees the same input and computes the same output --
but the two zero blocks are real parameters that receive gradient from the first
step, so the halves are coupled immediately and can specialise without anything
random being injected.

**Where it is not just a block diagonal.**  Three places need the structure:

*The input projection* stacks rather than block-diagonalises: ``[W; W]``, so
four input features fan out to both halves.

*LayerNorm* duplicates ``gamma`` and ``beta``.  Normalisation runs over width,
and the mean and variance of ``[x, x]`` equal those of ``x`` exactly -- an
average over a multiset that has been repeated -- so ``LayerNorm(dup(x))`` is
``dup(LayerNorm(x))``.  This is why an *uneven* widening would not work.

*The policy head* scores ``source . destination / sqrt(width)``.  Duplication
doubles the dot product while the divisor grows by ``sqrt(2)``, so the logits
would come out ``sqrt(2)`` too large.  Both projections absorb ``2^(-1/4)`` --
weights and biases -- which keeps source and destination symmetric instead of
singling one out.  The value head's final ``width -> 1`` map halves instead,
``[W/2, W/2]``, with its bias untouched.

**Optimizer state is expanded, not discarded**, by the same structural map: the
inherited entries keep the parent's moments, and the new cross-half blocks start
at zero, exactly as the depth transplant treats new blocks.

Expect near-identity rather than the bit-exact result deepening gives.  The
reduction order inside LayerNorm and the matmuls changes at 256, so FP32 rounds
differently; ``--tolerance`` defaults to 1e-5, far tighter than the ~3e-8 by
which batch size already perturbs this model.

Usage::

    python tools/widen_checkpoint.py --source 44250.pt --out 256x6.pt --width 256
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import (
    checkpoint_network_config,
    load_checkpoint,
    save_checkpoint,
)
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer
from diamond.contract.state import GameState, GameStatus, build_players

POLICY_SCALE = 2.0 ** -0.25
_MODE = "block-diagonal"


def _trainer(network: NetworkConfig, version: str, training: TrainingConfig) -> AlphaZeroTrainer:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    return AlphaZeroTrainer(
        SooModel(network, model_version=version), compatibility, training
    )


def _block_diagonal(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """``[[W, 0, ...], [0, W, ...], ...]`` for a hidden-to-hidden map."""
    rows, columns = tensor.shape
    out = tensor.new_zeros(rows * factor, columns * factor)
    for index in range(factor):
        out[index * rows : (index + 1) * rows, index * columns : (index + 1) * columns] = tensor
    return out


def _duplicated(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Every block ``W / factor`` -- the plain Net2Wider expansion.

    Also function-preserving, and kept only as a symmetry control.  Each new
    channel is an exact copy of an old one and sees an identical gradient, so
    under deterministic training the halves never separate and the wider model
    stays `factor` copies of the narrower one.  If the block-diagonal morph
    learns something this cannot, the difference is the cross-half capacity and
    not the extra parameters.
    """
    return tensor.repeat(factor, factor) / factor


def _widen(key: str, tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Structural expansion for one parameter, by role rather than by shape.

    Arithmetic runs in float64 and casts back.  The policy head's `2^(-1/4)` is
    irrational, so scaling in float32 costs ~1e-7 relative per weight, and that
    error survives into every logit -- it is introduced *here*, not in the
    forward pass, so no later precision recovers it.
    """
    original = tensor.dtype
    tensor = tensor.double()
    return _widen_impl(key, tensor, factor).to(original)


def _widen_impl(key: str, tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if key == "trunk.adjacency":
        # A buffer, not a parameter: the board's [direction, node, neighbour]
        # connectivity is a property of the game and identical at every width.
        return tensor.clone()
    if key.startswith("trunk.input_projection."):
        # Fan the input features into every half.
        return tensor.repeat(factor, *([1] * (tensor.ndim - 1)))
    if key == "value_head.2.weight":
        # width -> 1: the halves are summed, so each contributes its share.
        return tensor.repeat(1, factor) / factor
    if key == "value_head.2.bias":
        return tensor.clone()
    scale = POLICY_SCALE if key.startswith(("policy_head.source.", "policy_head.destination.")) else 1.0
    if tensor.ndim == 2:
        expand = _duplicated if _MODE == "duplicate" else _block_diagonal
        return expand(tensor, factor) * scale
    if tensor.ndim == 1:
        return tensor.repeat(factor) * scale
    raise SystemExit(f"do not know how to widen {key} with shape {tuple(tensor.shape)}")


def _widen_second_moment(key: str, tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Expand Adam's second moment, filling new positions rather than zeroing.

    Same shape as `_widen` produces, but hidden-to-hidden maps tile instead of
    going block-diagonal, so no entry is zero.  Scales that `_widen` applies to
    weights are squared here, because the moment tracks the square.
    """
    original = tensor.dtype
    tensor = tensor.double()
    if key == "trunk.adjacency":
        return tensor.to(original)
    if key.startswith("trunk.input_projection."):
        return tensor.repeat(factor, *([1] * (tensor.ndim - 1))).to(original)
    if key == "value_head.2.weight":
        return (tensor.repeat(1, factor) / factor**2).to(original)
    if key == "value_head.2.bias":
        return tensor.clone().to(original)
    scale = (
        POLICY_SCALE**2
        if key.startswith(("policy_head.source.", "policy_head.destination."))
        else 1.0
    )
    if tensor.ndim == 2:
        divisor = factor**2 if _MODE == "duplicate" else 1.0
        return (tensor.repeat(factor, factor) / divisor * scale).to(original)
    if tensor.ndim == 1:
        return (tensor.repeat(factor) * scale).to(original)
    raise SystemExit(f"do not know how to widen {key} with shape {tuple(tensor.shape)}")


def _corpus_features(corpus: Path, limit: int) -> torch.Tensor:
    import json

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
        encoded, _player_ids = game.encode(state)
        rows.append(torch.tensor(encoded, dtype=torch.float32))
    if not rows:
        raise SystemExit(f"corpus {corpus} yielded no two-player positions")
    return torch.stack(rows)


def _divergence(
    parent: SooModel, child: SooModel, features: torch.Tensor
) -> tuple[float, float, float]:
    """Worst logit, prior and value divergence over the corpus.

    The gate is the **prior**, not the logit.  Logits are unnormalised and run to
    |31| on this network, so an absolute bound on them says little; what the
    search consumes is the softmax, and a bound there is directly a bound on how
    differently MCTS can behave.
    """
    parent, child = parent.to("cpu").eval(), child.to("cpu").eval()
    worst_logit = worst_prior = worst_value = 0.0
    with torch.no_grad():
        for start in range(0, len(features), 256):
            batch = features[start : start + 256]
            p_policy, p_value = parent(batch)
            c_policy, c_value = child(batch)
            worst_logit = max(worst_logit, float((c_policy - p_policy).abs().max()))
            worst_prior = max(
                worst_prior,
                float(
                    (torch.softmax(c_policy, -1) - torch.softmax(p_policy, -1)).abs().max()
                ),
            )
            worst_value = max(worst_value, float((c_value - p_value).abs().max()))
    return worst_logit, worst_prior, worst_value


def _carry_optimizer_state(
    parent: AlphaZeroTrainer, child: AlphaZeroTrainer, factor: int
) -> tuple[int, int]:
    """Expand the parent's AdamW moments through the same structural map.

    Resetting them instead restarts bias correction for every inherited weight,
    which is a change to the training trajectory rather than to the
    architecture.  The new cross-half blocks are left at zero: they have no
    history, exactly like a new block in the depth transplant.
    """
    parent_names = [name for name, _ in parent.model.named_parameters()]
    child_index = {name: i for i, (name, _) in enumerate(child.model.named_parameters())}
    parent_state = parent.optimizer.state_dict()["state"]
    child_dict = child.optimizer.state_dict()

    carried = {}
    for position, name in enumerate(parent_names):
        if position not in parent_state or name not in child_index:
            continue
        entry = copy.deepcopy(parent_state[position])
        # The two moments expand differently, and the difference matters.
        #
        # `exp_avg` is a signed direction: a weight that did not exist has no
        # history, so the new cross-half entries are zero, which the structural
        # map gives for free.
        #
        # `exp_avg_sq` is a magnitude estimate and sits under a square root in
        # the denominator.  Leaving it zero makes the first update for 1.5 M new
        # weights roughly `sqrt((1-b2))^-1 (1-b1) ~ 3x` its normal size, all at
        # once -- measured as a +0.27 held-out CE spike within ten steps, far
        # larger than anything the architecture change itself does.  Tiling the
        # parent's values instead gives each new weight the scale its
        # layer-mates already have, so it takes an ordinary step from step one.
        # The inherited entries are untouched either way.
        if isinstance(entry.get("exp_avg"), torch.Tensor):
            entry["exp_avg"] = _widen(name, entry["exp_avg"], factor)
        if isinstance(entry.get("exp_avg_sq"), torch.Tensor):
            entry["exp_avg_sq"] = _widen_second_moment(name, entry["exp_avg_sq"], factor)
        carried[child_index[name]] = entry
    child_dict["state"] = carried
    child.optimizer.load_state_dict(child_dict)
    return len(carried), len(child_index) - len(carried)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True, help="a whole multiple of the parent's")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "tests/native/fixtures/positions.jsonl"
    )
    parser.add_argument("--positions", type=int, default=1024)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Worst permitted divergence; this is a near-identity, not a bit-exact one.",
    )
    parser.add_argument(
        "--mode",
        choices=("block-diagonal", "duplicate"),
        default="block-diagonal",
        help=(
            "How a hidden-to-hidden map expands.  'block-diagonal' is the "
            "staged-training form and creates cross-half weights that start at "
            "zero and train from the first step; 'duplicate' is plain "
            "Net2Wider and is a symmetry control, not a candidate."
        ),
    )
    parser.add_argument("--reset-optimizer", action="store_true")
    args = parser.parse_args()

    global _MODE
    _MODE = args.mode

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    source_network = checkpoint_network_config(args.source)
    version = payload["metadata"]["model_version"]
    if args.width % source_network.width:
        raise SystemExit(
            f"--width {args.width} is not a whole multiple of {source_network.width}; "
            "an uneven widening does not preserve LayerNorm statistics"
        )
    factor = args.width // source_network.width
    if factor < 2:
        raise SystemExit(f"--width {args.width} does not widen a {source_network.width} parent")

    # Both sides on --device: this is a pure tensor operation and should not
    # require the GPU the parent happened to be trained on.
    training = TrainingConfig(**{**payload["training_config"], "device": args.device})
    parent = _trainer(source_network, version, training)
    load_checkpoint(args.source, parent, expected=parent.compatibility, allow_device_migration=True)

    target_network = NetworkConfig(
        width=args.width, residual_blocks=source_network.residual_blocks
    )
    child = _trainer(target_network, version, training)

    parent_state = parent.model.state_dict()
    child_state = child.model.state_dict()
    widened = 0
    for key, tensor in parent_state.items():
        if key not in child_state:
            raise SystemExit(f"parent parameter {key} has no home in the child")
        expanded = _widen(key, tensor, factor)
        if expanded.shape != child_state[key].shape:
            raise SystemExit(
                f"widened {key} to {tuple(expanded.shape)}, child wants "
                f"{tuple(child_state[key].shape)}"
            )
        child_state[key] = expanded
        widened += 1
    child.model.load_state_dict(child_state)

    print(
        f"[widen] {source_network.width}x{source_network.residual_blocks} -> "
        f"{target_network.width}x{target_network.residual_blocks} "
        f"({factor}x, {args.mode}): {widened} tensors expanded"
    )
    if args.mode == "block-diagonal":
        cross = sum(
            int(child_state[k].numel() - factor * parent_state[k].numel())
            for k in parent_state
            if child_state[k].ndim == 2 and not k.startswith("trunk.input_projection.")
            and k != "value_head.2.weight"
        )
        print(
            f"[capacity] {cross:,} cross-half weights start at zero and "
            f"receive gradient at once"
        )
    else:
        print("[capacity] none -- every new channel is an exact copy (symmetry control)")

    if str(args.corpus) != "-":
        features = _corpus_features(args.corpus, args.positions)
        logit, prior, value = _divergence(parent.model, child.model, features)
        print(
            f"[identity] worst |child - parent| over {len(features)} positions: "
            f"prior {prior:.3e}, value {value:.3e}  (logit {logit:.3e})"
        )
        if max(prior, value) > args.tolerance:
            raise SystemExit(
                f"near-identity assertion failed: {max(prior, value):.3e} > "
                f"{args.tolerance:.3e}"
            )

    if args.reset_optimizer:
        print("[optimizer] parent moments discarded (--reset-optimizer)")
    else:
        carried, fresh = _carry_optimizer_state(parent, child, factor)
        print(
            f"[optimizer] AdamW moments expanded for {carried} parameters; "
            f"{fresh} without state"
        )

    child.training_step = parent.training_step
    save_checkpoint(args.out, child)
    print(f"[written] {args.out} at training_step {child.training_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
