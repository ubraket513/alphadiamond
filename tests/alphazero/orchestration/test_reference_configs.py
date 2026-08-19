"""The checked-in reference configs must stay loadable.

``ProductionConfig`` validates payloads with exact keys, so adding a field to any
config dataclass silently invalidates every config file on disk until it is
updated too. These tests turn that into a build failure instead of a runtime one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamond.alphazero.config import BOOTSTRAP_PRIOR_NONE, BOOTSTRAP_PRIORS
from diamond.alphazero.orchestration.production import ProductionConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "alphazero"
CONFIGS = sorted(CONFIG_DIR.glob("*.json"))
PRODUCTION = sorted(CONFIG_DIR.glob("*-production.json"))
BOOTSTRAP = sorted(CONFIG_DIR.glob("*-bootstrap.json"))


def load(path: Path) -> ProductionConfig:
    return ProductionConfig.from_payload(json.loads(path.read_text(encoding="utf-8")))


def test_the_config_directory_is_not_empty() -> None:
    assert CONFIGS, f"no reference configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_reference_config_round_trips(path: Path) -> None:
    config = load(path)
    assert config.self_play.bootstrap_prior in BOOTSTRAP_PRIORS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_reference_config_states_bootstrap_prior_explicitly(path: Path) -> None:
    """Never rely on the dataclass default: the file should say what it means."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "bootstrap_prior" in payload["self_play"]


@pytest.mark.parametrize("path", PRODUCTION, ids=lambda p: p.stem)
def test_production_configs_never_enable_a_bootstrap_prior(path: Path) -> None:
    assert load(path).self_play.bootstrap_prior == BOOTSTRAP_PRIOR_NONE


@pytest.mark.parametrize("path", BOOTSTRAP, ids=lambda p: p.stem)
def test_bootstrap_configs_enable_a_bootstrap_prior(path: Path) -> None:
    prior = load(path).self_play.bootstrap_prior
    assert prior != BOOTSTRAP_PRIOR_NONE
    assert prior in BOOTSTRAP_PRIORS


def test_both_models_have_a_production_and_a_bootstrap_config() -> None:
    assert {p.stem for p in PRODUCTION} == {"soo-production", "min-production"}
    assert {p.stem for p in BOOTSTRAP} == {"soo-bootstrap", "min-bootstrap"}
