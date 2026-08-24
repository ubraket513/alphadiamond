"""Freeze deterministic CPU FP32 Soo/Min model and AdamW parity vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from diamond.alphazero.checkpoint import save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.network import MinModel, SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_clean_tree() -> None:
    result = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT)
    if result.returncode:
        raise SystemExit("refusing to freeze training vectors from a dirty worktree")


def write_bytes(root: Path, relative: str, payload: bytes, *, shape: list[int], element_type: str,
                files: dict[str, dict[str, object]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    files[relative] = {
        "shape": shape,
        "element_type": element_type,
        "byte_count": len(payload),
        "sha256": sha256(path),
    }


def write_tensor(root: Path, relative: str, tensor: torch.Tensor,
                 files: dict[str, dict[str, object]]) -> None:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.float32:
        suffix, element_type = ".f32", "float32"
    elif value.dtype == torch.int64:
        suffix, element_type = ".i64", "int64"
    else:
        raise TypeError(f"unsupported frozen tensor type: {value.dtype}")
    write_bytes(root, relative + suffix, value.numpy().tobytes(),
                shape=list(value.shape), element_type=element_type, files=files)


def freeze_family(root: Path, family: str, factory: type[SooModel] | type[MinModel],
                  files: dict[str, dict[str, object]]) -> None:
    torch.manual_seed(104729 if family == "soo" else 104759)
    network = NetworkConfig(width=8, residual_blocks=1)
    model = factory(network, model_version="1.0.0")
    compatibility = (CheckpointCompatibilitySpec.soo if family == "soo"
                     else CheckpointCompatibilitySpec.min)(
                         model_version="1.0.0", network_config=network)
    trainer = AlphaZeroTrainer(model, compatibility, TrainingConfig(
        batch_size=2, learning_rate=1e-3, weight_decay=1e-4, device="cpu", seed=271828,
    ))
    features = torch.linspace(-1.0, 1.0, 2 * 73 * model.input_features,
                              dtype=torch.float32).reshape(2, 73, model.input_features)
    policy_targets = torch.zeros((2, 73 * 73), dtype=torch.float32)
    policy_targets[0, 17] = 0.25
    policy_targets[0, 113] = 0.75
    policy_targets[1, 91] = 0.6
    policy_targets[1, 733] = 0.4
    value_targets = torch.linspace(-0.5, 0.5, 2 * model.value_size,
                                  dtype=torch.float32).reshape(2, model.value_size)

    model.eval()
    with torch.no_grad():
        logits, values = model(features)
    prefix = f"{family}/"
    write_tensor(root, prefix + "inputs", features, files)
    write_tensor(root, prefix + "policy_targets", policy_targets, files)
    write_tensor(root, prefix + "value_targets", value_targets, files)
    write_tensor(root, prefix + "initial_logits", logits, files)
    write_tensor(root, prefix + "initial_values", values, files)
    for name, tensor in model.state_dict().items():
        write_tensor(root, prefix + "initial_parameters/" + name.replace('.', '__'), tensor, files)

    metrics = trainer._train_tensors(features, policy_targets, value_targets)
    write_tensor(root, prefix + "losses", torch.tensor(
        [metrics.total_loss, metrics.policy_loss, metrics.value_loss], dtype=torch.float32), files)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        write_tensor(root, prefix + "gradients/" + name.replace('.', '__'), parameter.grad, files)
    for name, tensor in model.state_dict().items():
        write_tensor(root, prefix + "after_step_parameters/" + name.replace('.', '__'), tensor, files)
    names = [name for name, _ in model.named_parameters()]
    for index, (_, parameter) in enumerate(model.named_parameters()):
        state = trainer.optimizer.state[parameter]
        assert state and "exp_avg" in state and "exp_avg_sq" in state
        write_tensor(root, prefix + "optimizer/" + names[index].replace('.', '__') + "__exp_avg",
                     state["exp_avg"], files)
        write_tensor(root, prefix + "optimizer/" + names[index].replace('.', '__') + "__exp_avg_sq",
                     state["exp_avg_sq"], files)
        write_tensor(root, prefix + "optimizer/" + names[index].replace('.', '__') + "__step",
                     state["step"].to(dtype=torch.int64).reshape(1), files)

    checkpoint = root / prefix / "checkpoint-v1.pt"
    save_checkpoint(checkpoint, trainer)
    files[str(checkpoint.relative_to(root)).replace('\\', '/')] = {
        "shape": [], "element_type": "checkpoint-v1", "byte_count": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
    }
    resumed = trainer._train_tensors(features, policy_targets, value_targets)
    write_tensor(root, prefix + "resumed_losses", torch.tensor(
        [resumed.total_loss, resumed.policy_loss, resumed.value_loss], dtype=torch.float32), files)
    for name, tensor in model.state_dict().items():
        write_tensor(root, prefix + "resumed_parameters/" + name.replace('.', '__'), tensor, files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    require_clean_tree()
    if output.exists():
        if not args.replace:
            raise SystemExit(f"refusing to overwrite existing fixture directory: {output} (use --replace)")
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True)
    files: dict[str, dict[str, object]] = {}
    write_bytes(output, "environment/torch_version.txt", torch.__version__.encode("utf-8"),
                shape=[len(torch.__version__)], element_type="utf8", files=files)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    write_bytes(output, "environment/source_commit.txt", commit.encode("ascii"),
                shape=[len(commit)], element_type="utf8", files=files)
    freeze_family(output, "soo", SooModel, files)
    freeze_family(output, "min", MinModel, files)
    manifest = {
        "fixture_version": 1,
        "game_contract": "diamond-authoritative-rules-v1",
        "dtype": "float32",
        "device": "cpu",
        "families": ["soo", "min"],
        "files": dict(sorted(files.items())),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"frozen {len(files)} files at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
