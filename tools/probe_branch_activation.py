"""Does a transplanted block's gate open, and how fast?

`deepen_checkpoint.py` starts a new block as an exact identity by zeroing its
LayerNorm gate.  Whether that block ever becomes useful is a separate question,
and the first arm answered it badly: gamma sat at ~0.002 from 900 steps to
2,100 while the inherited blocks held 0.8-1.6, so four iterations of GPU time
measured a warm-up that had not happened.

This is the cheap version of that question -- a few hundred steps on real replay
batches, no self-play -- so an initialisation can be rejected in minutes instead
of hours.

**Every variant sees the same batches in the same order.**  That is the point:
gamma's gradient is proportional to the normalised message, so comparing
initialisations on different data compares the data.

Reported per variant:

``gamma RMS``
    Has the gate left zero, and by how much against the inherited blocks.
``gamma grad RMS``
    Is there a signal to open it, or only noise.  A gate that is not moving
    because its gradient is zero is a different problem from one whose gradient
    is real but cancels.
``residual / input``
    What the block actually contributes to the trunk, which is the quantity
    that matters and the one gamma is only a proxy for.
``branch drift``
    How far the projections have moved from where they started.  Large drift
    with a shut gate is a random walk, not learning: Adam normalises by the
    gradient's own second moment, so a tiny noisy gradient still buys full-size
    steps.

Usage::

    python tools/probe_branch_activation.py --parent 44250.pt \\
        --replay /path/to/run/replay --steps 300 \\
        --variant random=--branch-init,random --variant copy=--branch-init,copy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.checkpoint import checkpoint_network_config, load_checkpoint
from diamond.alphazero.config import TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import SooModel
from diamond.alphazero.orchestration.replay_store import PersistentReplayStore
from diamond.alphazero.trainer import AlphaZeroTrainer

ACTION_SIZE = 5329


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().pow(2).mean().sqrt())


def _build(path: Path, device: str) -> AlphaZeroTrainer:
    network = checkpoint_network_config(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload["metadata"]["model_version"]
    compatibility = CheckpointCompatibilitySpec.soo(
        model_version=version, network_config=network
    )
    training = TrainingConfig(**{**payload["training_config"], "device": device})
    # Gating is not recorded in the compatibility metadata, so infer it from the
    # state dict: a gated checkpoint carries `alpha` parameters and would fail a
    # strict load into an ungated model.  Loud either way, but this makes the
    # tool work on both without a flag the caller has to keep in sync.
    gated = sorted(
        int(key.split(".")[2])
        for key in payload["model_state_dict"]
        if key.startswith("trunk.blocks.") and key.endswith(".alpha")
    )
    trainer = AlphaZeroTrainer(
        SooModel(
            network,
            model_version=version,
            gate_blocks_from=gated[0] if gated else None,
        ),
        compatibility,
        training,
    )
    load_checkpoint(path, trainer, expected=compatibility, allow_device_migration=True)
    return trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True, help="a run's replay dir")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=FLAG,FLAG,...",
        help="Repeat per variant; flags are passed to deepen_checkpoint.py.",
    )
    args = parser.parse_args()

    parent_network = checkpoint_network_config(args.parent)
    old_blocks = parent_network.residual_blocks
    new_blocks = range(old_blocks, args.blocks)

    specs = []
    for raw in args.variant:
        if "=" not in raw:
            raise SystemExit(f"--variant must be NAME=FLAGS, got {raw!r}")
        name, _, flags = raw.partition("=")
        specs.append((name, [f for f in flags.split(",") if f]))

    scratch = Path(tempfile.mkdtemp(prefix="branch-probe-"))
    print(
        f"parent {parent_network.width}x{old_blocks} -> {args.blocks}, "
        f"{args.steps} steps at batch {args.batch_size}, "
        f"identical batches per variant (seed {args.seed})\n"
    )

    for name, flags in specs:
        target = scratch / f"{name}.pt"
        build = [
            sys.executable, str(ROOT / "tools/deepen_checkpoint.py"),
            "--source", str(args.parent), "--out", str(target),
            "--blocks", str(args.blocks), "--device", args.device,
            "--corpus", "-", *flags,
        ]
        result = subprocess.run(build, capture_output=True, text=True, check=False)
        if result.returncode:
            raise SystemExit(f"{name}: transplant failed\n{result.stdout}{result.stderr}")

        trainer = _build(target, args.device)
        # A fresh store per variant, seeded identically, so the sample sequence
        # is the same one every time -- see the module docstring.
        replay = PersistentReplayStore(
            args.replay,
            trainer.compatibility,
            capacity=args.capacity,
            seed=args.seed,
        )
        buffer = replay.load_buffer()
        if len(buffer) < args.batch_size:
            raise SystemExit(
                f"replay has {len(buffer)} samples, need at least {args.batch_size}"
            )

        start = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        gated = trainer.model.trunk.blocks[old_blocks].alpha is not None
        # Bind the current trainer explicitly; these close over the variant loop.
        gate_of = (
            (lambda index, model=trainer.model: model.trunk.blocks[index].alpha)
            if gated
            else (lambda index, model=trainer.model: model.trunk.blocks[index].norm.weight)
        )
        gate_gradients = {index: [] for index in new_blocks}
        gate_track = {index: [] for index in new_blocks}
        for step in range(args.steps):
            samples = replay.sample(args.batch_size)
            trainer.train_batch(buffer.collate(samples, action_size=ACTION_SIZE))
            for index in new_blocks:
                gate = gate_of(index)
                if gate.grad is not None:
                    gate_gradients[index].append(_rms(gate.grad))
                if gated and step % max(1, args.steps // 5) == 0:
                    gate_track[index].append(float(gate.detach().reshape(-1)[0]))

        share = {}
        handles = []

        def watch(index: int, sink=share):
            def hook(module, inputs, output):
                incoming = _rms(inputs[0])
                sink[index] = _rms(output - inputs[0]) / incoming if incoming else 0.0
            return hook

        for index, block in enumerate(trainer.model.trunk.blocks):
            handles.append(block.register_forward_hook(watch(index)))
        with torch.no_grad():
            samples = replay.sample(args.batch_size)
            probe = buffer.collate(samples, action_size=ACTION_SIZE)
            features = torch.tensor(
                probe.node_features, dtype=torch.float32, device=trainer.device
            )
            trainer.model(features)
        for handle in handles:
            handle.remove()

        state = trainer.model.state_dict()
        if gated:
            # A scalar gate has a sign, and the sign is the finding: a branch the
            # optimizer wants pushes alpha away from zero in one direction, while
            # one it is indifferent to leaves it wandering.  Averaging magnitudes
            # would erase exactly that.
            alphas = [float(state[f"trunk.blocks.{i}.alpha"].reshape(-1)[0]) for i in new_blocks]
        gamma_new = [
            _rms(state[f"trunk.blocks.{i}.norm.weight"]) for i in new_blocks
        ]
        gamma_old = [
            _rms(state[f"trunk.blocks.{i}.norm.weight"]) for i in range(old_blocks)
        ]
        drift = []
        for index in new_blocks:
            prefix = f"trunk.blocks.{index}."
            moved = sum(
                float((state[key] - start[key]).pow(2).sum())
                for key in state
                if key.startswith(prefix) and ".norm." not in key
            ) ** 0.5
            initial = sum(
                float(start[key].pow(2).sum())
                for key in state
                if key.startswith(prefix) and ".norm." not in key
            ) ** 0.5
            drift.append(moved / initial if initial else float("nan"))
        grad = [
            sum(values) / len(values) if values else 0.0
            for values in gate_gradients.values()
        ]
        new_share = [share[i] for i in new_blocks]
        old_share = [share[i] for i in range(old_blocks)]
        mean = lambda values: sum(values) / len(values)

        print(f"  {name}")
        if gated:
            print(f"    new-block alpha          "
                  f"{', '.join(f'{value:+.4f}' for value in alphas)}")
            print(f"    new-block alpha mean     {mean(alphas):+.5f}"
                  f"   (from 0.0; sign is the signal)")
            trail = gate_track[old_blocks]
            print(f"    block {old_blocks} alpha trajectory "
                  f"{' -> '.join(f'{value:+.4f}' for value in trail)}")
        else:
            print(f"    new-block gamma RMS      {mean(gamma_new):.5f}"
                  f"   (inherited {mean(gamma_old):.5f})")
        print(f"    new-block gate grad RMS  {mean(grad):.3e}")
        print(f"    new-block residual/input {mean(new_share):.5f}"
              f"   (inherited {mean(old_share):.5f},"
              f" ratio {mean(new_share) / mean(old_share):.4f})")
        print(f"    new-block branch drift   {mean(drift) * 100:.1f}% of init norm")
        opened = mean(new_share) / mean(old_share)
        verdict = (
            "OPENING -- worth a full warm arm" if opened > 0.05
            else "still inert -- a gate here would measure the schedule"
        )
        print(f"    -> {verdict}\n")

    print(f"(scratch checkpoints in {scratch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
