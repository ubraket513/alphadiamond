"""Fixed-seed A/B probe comparing bootstrap priors on data-generation viability.

This measures whether a prior lets cold-start self-play reach real terminal
states and fill replay.  It deliberately reports raw metrics and asserts no
production thresholds; Elo and TrueSkill are the wrong instruments here and are
not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ...game.state import build_players
from ..config import MCTSConfig, NetworkConfig, SelfPlayConfig
from ..evaluator.base import Evaluator
from ..evaluator.dummy import DummyEvaluator
from ..game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from ..identity import MIN_MODEL_NAME, SOO_MODEL_NAME, CheckpointCompatibilitySpec
from ..selfplay.runner_2p import SooSelfPlayRunner
from ..selfplay.runner_3p import MinSelfPlayRunner
from .evaluator import bootstrap_evaluator
from .heuristic import BOOTSTRAP_PRIOR_NONE


@dataclass(frozen=True, slots=True)
class ProbeReport:
    model_name: str
    bootstrap_prior: str
    episodes: int
    completed: int
    move_counts: tuple[int, ...]
    samples: int
    abort_reasons: dict[str, int]

    @property
    def completion_rate(self) -> float:
        return self.completed / self.episodes if self.episodes else 0.0

    @property
    def median_moves(self) -> float | None:
        return median(self.move_counts) if self.move_counts else None

    @property
    def p90_moves(self) -> int | None:
        """Only meaningful with enough completions to rank."""
        if len(self.move_counts) < 10:
            return None
        ordered = sorted(self.move_counts)
        return ordered[min(int(0.9 * len(ordered)), len(ordered) - 1)]

    @property
    def samples_per_episode(self) -> float:
        return self.samples / self.episodes if self.episodes else 0.0


def _unless_none(**values: object) -> dict[str, object]:
    """Keep a dataclass default rather than overriding it with a stand-in.

    Passing an explicit 0.0 for "unset" would have silently turned Dirichlet
    noise off for every existing caller, because ``MCTSConfig.dirichlet_epsilon``
    defaults to 0.25 and not to zero.
    """
    return {name: value for name, value in values.items() if value is not None}


def run_probe(
    *,
    model_name: str = SOO_MODEL_NAME,
    bootstrap_prior: str = BOOTSTRAP_PRIOR_NONE,
    episodes: int = 20,
    simulations: int = 1,
    max_moves: int = 2000,
    base_seed: int = 0,
    base_evaluator: Evaluator | None = None,
    dirichlet_epsilon: float | None = None,
    dirichlet_alpha: float | None = None,
    temperature: float | None = None,
    temperature_moves: int = 0,
) -> ProbeReport:
    """Run ``episodes`` fixed-seed self-play games under one prior.

    **The exploration settings decide what this measures, and the historical
    default measures an easier question than production asks.**  ``None`` leaves
    a setting at its dataclass default, which is what every existing caller got:
    Dirichlet noise *on* at ``epsilon = 0.25``, but ``temperature_moves = 0``, so
    no temperature sampling.  Production self-play samples its first 20 moves
    from the visit distribution instead of taking the argmax.

    That difference is not academic.  Measured on the from-scratch Soo run at
    step 13,650: the probe without temperature sampling reported **100 %
    completion** and the gate passed, while production self-play on the same
    checkpoint completed only **64 %** and fell to ~64 % over four iterations.
    Sampling the opening from a distribution -- rather than perturbing it and
    then taking the best move -- is what sends games wandering past the move cap.

    A gate for "is this network ready for production self-play" therefore has to
    pass production's *whole* exploration configuration in, not just part of it.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    if model_name == SOO_MODEL_NAME:
        player_count, runner_cls = 2, SooSelfPlayRunner
        compatibility = CheckpointCompatibilitySpec.soo(
            model_version="0.1.0", network_config=NetworkConfig()
        )
        default_value: float | tuple[float, ...] = 0.0
    elif model_name == MIN_MODEL_NAME:
        player_count, runner_cls = 3, MinSelfPlayRunner
        compatibility = CheckpointCompatibilitySpec.min(
            model_version="0.7.0", network_config=NetworkConfig()
        )
        default_value = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"unsupported probe model: {model_name}")

    completed = 0
    samples = 0
    move_counts: list[int] = []
    aborts: dict[str, int] = {}

    for index in range(episodes):
        seed = base_seed + index
        base = DummyEvaluator(default_value) if base_evaluator is None else base_evaluator
        episode = runner_cls(
            DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(player_count))),
            bootstrap_evaluator(base, bootstrap_prior),
            MCTSConfig(
                simulations=simulations,
                seed=seed,
                **_unless_none(
                    dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon
                ),
            ),
            SelfPlayConfig(
                max_moves=max_moves,
                temperature_moves=temperature_moves,
                seed=seed,
                bootstrap_prior=bootstrap_prior,
                **_unless_none(temperature=temperature),
            ),
            compatibility,
        ).run()
        samples += len(episode.samples)
        if episode.completed:
            completed += 1
            move_counts.append(episode.move_count)
        else:
            reason = episode.aborted_reason or "unknown"
            aborts[reason] = aborts.get(reason, 0) + 1

    return ProbeReport(
        model_name=model_name,
        bootstrap_prior=bootstrap_prior,
        episodes=episodes,
        completed=completed,
        move_counts=tuple(move_counts),
        samples=samples,
        abort_reasons=aborts,
    )


def format_report(report: ProbeReport) -> str:
    return (
        f"{report.model_name:4s} {report.bootstrap_prior:38s} "
        f"completion {report.completion_rate:6.1%} "
        f"median {report.median_moves!s:>6s} "
        f"p90 {report.p90_moves!s:>6s} "
        f"samples/ep {report.samples_per_episode:8.1f} "
        f"aborts {report.abort_reasons or '{}'}"
    )


__all__ = ["ProbeReport", "format_report", "run_probe"]
