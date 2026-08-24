"""Promotion refuses the ways it can go wrong.

The manifest has always carried a `state`; nothing moved it, so which
checkpoint was "the release" lived in someone's head. These are the three
things the transition has to refuse, because each of them produces a release
that looks fine and is not.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "promote_checkpoint", ROOT / "tools" / "promote_checkpoint.py"
)
assert _spec and _spec.loader
promote_checkpoint = importlib.util.module_from_spec(_spec)
sys.modules["promote_checkpoint"] = promote_checkpoint
_spec.loader.exec_module(promote_checkpoint)

PromotionError = promote_checkpoint.PromotionError


def _checkpoint(directory: Path, *, state: str = "archival", payload: bytes = b"weights") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoint.pt").write_bytes(payload)
    manifest = {
        "manifest_version": 1,
        "model_family": "soo",
        "run_id": "soo-scratch-20260822",
        "training_step": 44250,
        "training_commit": "c" * 40,
        "architecture": {"type": "directional_residual", "width": 128, "residual_blocks": 6},
        "game_contract": {
            "topology": "diamond73-v1",
            "encoder": "diamond-camp-relative-v1",
            "action_space": "diamond73-srcdst-v1",
        },
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "contains": {"optimizer": True, "scheduler": True, "rng": True},
        "state": state,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def test_a_checkpoint_moves_one_step(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "step-00044250")
    manifest = promote_checkpoint.promote(directory, "candidate")
    assert manifest["state"] == "candidate"
    assert json.loads((directory / "manifest.json").read_text())["state"] == "candidate"


def test_repeating_a_transition_is_not_an_error(tmp_path: Path) -> None:
    """Re-running a promotion script must not fail a pipeline."""
    directory = _checkpoint(tmp_path / "step-1", state="candidate")
    assert promote_checkpoint.promote(directory, "candidate")["state"] == "candidate"


def test_promotion_cannot_skip_the_candidate_gate(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "step-2")
    with pytest.raises(PromotionError, match="one step forward"):
        promote_checkpoint.promote(directory, "promoted", artifacts=tmp_path / "out")


def test_a_drifted_digest_stops_everything(tmp_path: Path) -> None:
    """An immutable path that was overwritten invalidates its measurements."""
    directory = _checkpoint(tmp_path / "step-3")
    (directory / "checkpoint.pt").write_bytes(b"something else entirely")
    with pytest.raises(PromotionError, match="no longer compare"):
        promote_checkpoint.promote(directory, "candidate")


def test_promotion_without_conversion_is_refused(tmp_path: Path) -> None:
    """A `promoted` state whose artifact was never exported is the bookkeeping
    this exists to remove."""
    directory = _checkpoint(tmp_path / "step-4", state="candidate")
    with pytest.raises(PromotionError, match="converted"):
        promote_checkpoint.promote(directory, "promoted")


def test_the_escape_hatch_is_explicit(tmp_path: Path) -> None:
    """Recovering a botched run must be possible, and must be asked for."""
    directory = _checkpoint(tmp_path / "step-5", state="candidate")
    manifest = promote_checkpoint.promote(directory, "promoted", export=False)
    assert manifest["state"] == "promoted"
    assert "deployment" not in manifest


def test_a_missing_or_malformed_manifest_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "step-6"
    empty.mkdir()
    with pytest.raises(PromotionError, match="no manifest"):
        promote_checkpoint.promote(empty, "candidate")

    directory = _checkpoint(tmp_path / "step-7")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["manifest_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PromotionError, match="manifest version"):
        promote_checkpoint.promote(directory, "candidate")


def test_a_missing_checkpoint_is_refused(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "step-8")
    (directory / "checkpoint.pt").unlink()
    with pytest.raises(PromotionError, match="no checkpoint"):
        promote_checkpoint.promote(directory, "candidate")
