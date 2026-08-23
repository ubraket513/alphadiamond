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

Width is not expandable *here*, but the reason is narrower than it first looks
and the earlier claim in this file overstated it.

An **uneven** widening -- 128 -> 192, replicating only some channels -- does
break LayerNorm: normalisation runs over the width axis, and duplicating a
subset changes the mean and the variance, so the function changes.

An **integer-multiple** widening does not.  Duplicate every channel exactly k
times and both the mean and the variance are unchanged, because each is an
average over a multiset that has simply been repeated; duplicate ``gamma`` and
``beta`` the same way and ``LayerNorm(dup(x)) == dup(LayerNorm(x))`` exactly.
Net2Wider then applies as usual to the Linear layers -- replicate output rows,
halve the weights on duplicated input columns -- and 128 -> 256 is a
function-preserving morph rather than a cold start.

Soo needs one extra correction that a generic Net2Wider does not: the policy
head scores ``source . destination / sqrt(width)``.  Duplication doubles the dot
product while ``sqrt(width)`` grows by only ``sqrt(2)``, so the logits come out
``sqrt(2)`` too large; one of the two projections has to absorb a ``1/sqrt(2)``.

That is a different tool with a different verification burden -- expect
near-identity rather than the bit-exact result deepening gives, since the
reduction order changes -- so it is deliberately not folded in here.

The identity claim is asserted numerically, not assumed -- ``--corpus`` runs the
parent and the child over the gate corpus and reports the worst divergence.

Usage::

    python tools/deepen_checkpoint.py --source latest.pt --out deep12.pt --blocks 12
"""

from __future__ import annotations

import argparse
import copy
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


def _trainer(
    network: NetworkConfig,
    version: str,
    training: TrainingConfig,
    *,
    gated_blocks: list[int] | None = None,
) -> AlphaZeroTrainer:
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    model = SooModel(network, model_version=version, gated_blocks=gated_blocks)
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
        "--branch-init",
        choices=("random", "copy"),
        default="copy",
        help=(
            "What the new block's projections start from.  'copy' seeds new "
            "block i from trained block (i %% parent_blocks); 'random' leaves "
            "torch's default init, which is what the first arm used and which "
            "measured inert.  Both are exact identities -- the gate is zero "
            "either way -- so this changes only what is waiting behind it."
        ),
    )
    parser.add_argument(
        "--residual-gate",
        action="store_true",
        help=(
            "Give each new block a ReZero-style scalar `alpha` after the "
            "activation and copy the donor block whole, LayerNorm included.  "
            "Still an exact identity at alpha = 0, but the gate now sits "
            "outside the nonlinearity, so its gradient means 'how much of this "
            "branch is wanted' rather than 'what should the branch's internal "
            "scale be'.  Implies --branch-init copy and ignores --gamma."
        ),
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help=(
            "Place each new block directly after the block it was copied from "
            "(b0 n0 b1 n1 ...) instead of appending the whole set (b0..b5 "
            "n0..n5).  Appending hands copy(b0) the representation b5 produces, "
            "which is not the distribution b0 was trained on, so a copied "
            "branch can be useless there and useful in place.  Requires the "
            "target depth to be a whole multiple of the parent's."
        ),
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help=(
            "Discard the parent's AdamW moments.  The default now carries them "
            "over for every parameter the child inherits by name; new blocks "
            "still start with no state.  Resetting was the original behaviour "
            "and it confounds a depth experiment with an optimizer restart -- "
            "the parent's blocks and heads lose their trajectory too."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help=(
            "LayerNorm scale for the new blocks.  0.0 is an exact identity.  A "
            "small non-zero value opens the residual branch from the first step "
            "at the cost of perturbing the parent: measured on the step-44,250 "
            "network, 0.01 shifts policy KL by 1.5e-4 and value RMSE by 0.009, "
            "while 0.1 already costs 0.447 value RMSE and is too much."
        ),
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
    # `layout[child_index]` is the parent block the child block comes from, or
    # None for an inherited slot; `donor_of` names the copy source.
    if args.interleave:
        if args.blocks % source_network.residual_blocks:
            raise SystemExit(
                f"--interleave needs a whole multiple of {source_network.residual_blocks} "
                f"blocks, got {args.blocks}"
            )
        factor = args.blocks // source_network.residual_blocks
        inherited = {index * factor: index for index in range(source_network.residual_blocks)}
        donor_of = {
            index * factor + offset: index
            for index in range(source_network.residual_blocks)
            for offset in range(1, factor)
        }
    else:
        inherited = {index: index for index in range(source_network.residual_blocks)}
        donor_of = {
            index: index % source_network.residual_blocks
            for index in range(source_network.residual_blocks, args.blocks)
        }
    new_indices = sorted(donor_of)
    # A fresh optimizer.  Adam's moments are per-parameter and the parameter set
    # has changed; carrying the parent's moments for the surviving tensors while
    # the new blocks start cold mixes two different notions of "where training
    # is", and the state is regenerated within a few hundred steps anyway.
    child_training = TrainingConfig(
        **{**payload["training_config"], "device": args.device}
    )
    child = _trainer(
        target_network,
        version,
        child_training,
        gated_blocks=new_indices if args.residual_gate else None,
    )

    parent_state = parent.model.state_dict()
    child_state = child.model.state_dict()

    parent_to_child = {parent_index: child_index for child_index, parent_index in inherited.items()}

    def _child_key(key: str) -> str:
        """Map a parent tensor name onto the slot it occupies in the child.

        Interleaving moves the inherited blocks: parent block 3 lands at child
        index 6, not 3.  Copying by name without this remap would quietly write
        the parent's blocks into the new slots and leave the inherited ones at
        their random init -- an identity check would fail, but only after the
        transplant had already produced a plausible-looking file.
        """
        if not key.startswith("trunk.blocks."):
            return key
        _, _, index, rest = key.split(".", 3)
        return f"trunk.blocks.{parent_to_child[int(index)]}.{rest}"

    copied = 0
    for key, tensor in parent_state.items():
        target = _child_key(key)
        if target not in child_state:
            raise SystemExit(f"parent parameter {key} has no home in the child")
        if child_state[target].shape != tensor.shape:
            raise SystemExit(f"shape mismatch on {key}: cannot transplant")
        child_state[target] = tensor.clone()
        copied += 1

    # Everything the parent did not have is a new trunk block.  Two independent
    # decisions: what its projections start from, and how far its gate is open.
    #
    # The gate is what makes the transplant an identity, and it does that
    # whatever the projections hold: LayerNorm with gamma = beta = 0 emits zeros
    # for any input, GELU(0) is 0, and the block returns `nodes`.  So seeding
    # the projections from trained blocks costs nothing in fidelity.
    #
    # It is not cosmetic.  With random projections the first arm's new blocks
    # never switched on -- gamma flat at ~0.002 from 900 steps to 2,100 while
    # the inherited blocks sat at 0.8-1.6 -- because the gradient reaching gamma
    # is proportional to the normalised message, and a random message carries no
    # signal to open the gate for.  Meanwhile Adam normalises by the gradient's
    # own second moment, so those projections still took full-size steps and
    # random-walked 42 % of their init norm behind a shut gate.  Copying gives
    # gamma a structured message to have an opinion about from the first step.
    copied_branches = []
    zeroed = []
    for index in new_indices:
        if args.branch_init == "copy" or args.residual_gate:
            donor = donor_of[index]
            prefix = f"trunk.blocks.{index}."
            donor_prefix = f"trunk.blocks.{donor}."
            for key in list(child_state):
                if not key.startswith(prefix) or key.endswith(".alpha"):
                    continue
                # With an explicit gate the donor's LayerNorm comes too: the
                # branch should compute a trained function, and `alpha` alone
                # decides how much of it reaches the trunk.
                if ".norm." in key and not args.residual_gate:
                    continue
                source_key = donor_prefix + key[len(prefix):]
                if source_key not in parent_state:
                    raise SystemExit(f"donor parameter {source_key} is missing")
                child_state[key] = parent_state[source_key].clone()
            copied_branches.append((index, donor))
        if args.residual_gate:
            key = f"trunk.blocks.{index}.alpha"
            if key not in child_state:
                raise SystemExit(f"expected gate parameter {key}")
            child_state[key] = torch.zeros_like(child_state[key])
            zeroed.append(key)
        else:
            for suffix, value in (("weight", args.gamma), ("bias", 0.0)):
                key = f"trunk.blocks.{index}.norm.{suffix}"
                if key not in child_state:
                    raise SystemExit(f"expected new block parameter {key}")
                child_state[key] = torch.full_like(child_state[key], value)
                zeroed.append(key)
    child.model.load_state_dict(child_state)

    added = args.blocks - source_network.residual_blocks
    print(
        f"[transplant] {source_network.width}x{source_network.residual_blocks} "
        f"-> {target_network.width}x{target_network.residual_blocks}: "
        f"{copied} tensors copied, {added} block(s) added, "
        f"{len(zeroed)} norm tensors set (gamma={args.gamma})"
    )
    if copied_branches:
        pairs = ", ".join(f"{donor}->{index}" for index, donor in copied_branches)
        print(f"[branches] projections seeded from trained blocks: {pairs}")

    if str(args.corpus) != "-":
        worst = _assert_identity(parent.model, child.model, args.corpus)
        print(f"[identity] worst |child - parent| over the corpus: {worst:.3e}")
        if args.gamma == 0.0:
            # An exact identity is the claim, so hold it to one.
            if worst > args.tolerance:
                raise SystemExit(
                    f"identity assertion failed: {worst:.3e} > {args.tolerance:.3e}"
                )
        else:
            # A non-zero gate is a deliberate perturbation, so the number is a
            # measurement rather than a gate.  Report it and let the caller
            # judge; --tolerance would only encode a guess about how much
            # divergence this particular gamma should buy.
            print(
                f"[identity] gamma={args.gamma} is not an identity by "
                f"construction; the divergence above is the cost of opening "
                f"the gate, not a failure"
            )

    if args.reset_optimizer:
        print("[optimizer] parent moments discarded (--reset-optimizer)")
    else:
        carried, fresh = _carry_optimizer_state(parent, child)
        print(
            f"[optimizer] AdamW moments carried for {carried} inherited "
            f"parameters; {fresh} new parameters start with no state"
        )

    # The training step carries over: the child *is* this network, further on.
    child.training_step = parent.training_step
    save_checkpoint(args.out, child)
    print(f"[written] {args.out} at training_step {child.training_step}")
    return 0


def _carry_optimizer_state(parent: AlphaZeroTrainer, child: AlphaZeroTrainer) -> tuple[int, int]:
    """Move the parent's AdamW moments onto the child's inherited parameters.

    Discarding them was the original behaviour and it is not free.  A growth
    operator is supposed to preserve the *training state*, not only the
    function: Adam's first and second moments and its step count are what make
    the next update the size the schedule expects, and throwing them away
    restarts the bias correction for the inherited blocks and both heads --
    parameters the experiment is not trying to change.  A depth result measured
    against that is a depth-plus-optimizer-restart result.

    The mapping is by parameter *name*.  Optimizer state is keyed by position in
    `model.parameters()`, and inserting blocks shifts those positions, so
    copying by index would silently attach the value head's moments to a trunk
    projection.

    New blocks are deliberately left with no state: they have no history to
    continue, and inventing one from a donor would give two parameters the same
    moments while their gradients immediately diverge.
    """
    parent_names = [name for name, _ in parent.model.named_parameters()]
    child_index = {name: index for index, (name, _) in enumerate(child.model.named_parameters())}
    parent_state = parent.optimizer.state_dict()["state"]

    child_dict = child.optimizer.state_dict()
    carried_state = {}
    for parent_position, name in enumerate(parent_names):
        if parent_position not in parent_state or name not in child_index:
            continue
        carried_state[child_index[name]] = copy.deepcopy(parent_state[parent_position])
    child_dict["state"] = carried_state
    child.optimizer.load_state_dict(child_dict)
    return len(carried_state), len(child_index) - len(carried_state)


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
