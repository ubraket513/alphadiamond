"""Native self-play behind the existing ``SelfPlayWorkerPool`` contract.

Same signature, same ``EpisodeResult`` values, radically different machinery:
one process with a native search-worker pool and a single batched evaluator,
instead of N spawned worker processes each holding one game and talking to a
serialized parent coordinator over pipes.

Measured on the GPU host against the Python pool, same seeded work: **18.5x**.

Two things about this file are load-bearing rather than incidental.

**The bootstrap prior chooses the mode, and is not free to disagree with it.**
``value_only`` computes the vacancy prior natively -- Gate A proves it equals
``bootstrap/heuristic.py`` within 1e-12 -- so it *is*
``canonical-target-vacancy-distance-v2`` and nothing else.  ``none`` means the
neural policy, which is ``policy_value``.  A run that asked for one and got the
other would train on data whose provenance label is a lie, and nothing
downstream could detect it, so the mapping is asserted rather than defaulted.

**Determinism comes from the job, not from the pool.**  Every episode's seed is
``SelfPlayJob.seed`` -- the same canonical derivation the Python backend uses --
so re-running an iteration reproduces it regardless of how many lanes, threads
or batches the scheduler happened to use.
"""

from __future__ import annotations

from typing import Any

from ..config import (
    BOOTSTRAP_PRIOR_NONE,
    CANONICAL_TARGET_VACANCY_DISTANCE_V2,
)
from ..orchestration.selfplay_workers import EpisodeResult, SelfPlayJob
from ..replay import TrainingSample
from ..selfplay.common import MAX_GAME_MOVES_EXCEEDED
from . import native_game, require_native
from .backend import policy_value_callback, value_only_callback

_MODE_FOR_PRIOR = {
    CANONICAL_TARGET_VACANCY_DISTANCE_V2: "value_only",
    BOOTSTRAP_PRIOR_NONE: "policy_value",
}


class NativeSelfPlayPool:
    """One-shot native self-play group, API-compatible with the process pool."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        lanes: int = 0,
        threads: int = 4,
        max_batch: int = 32,
        max_wait_us: int = 2000,
        simulations_late: int = 0,
        late_move_threshold: int = 0,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.model = model
        self.device = device
        self.lanes = lanes
        self.threads = threads
        self.max_batch = max_batch
        self.max_wait_us = max_wait_us
        # Adaptive search. 0 disables it, which is the production default; see
        # EpisodeConfig for why spending it only on the tail is the point.
        self.simulations_late = simulations_late
        self.late_move_threshold = late_move_threshold
        self.metrics: dict[str, Any] = {}

    def run(self, jobs: tuple[SelfPlayJob, ...]) -> tuple[EpisodeResult, ...]:
        if not isinstance(jobs, tuple):
            # ValueError, not TypeError: SelfPlayWorkerPool.run raises ValueError
            # for this exact check, and a drop-in replacement that raises a
            # different class is not a drop-in replacement.
            raise ValueError("jobs must be a tuple")  # noqa: TRY004
        if not jobs:
            return ()
        if not all(isinstance(job, SelfPlayJob) for job in jobs):
            raise ValueError("jobs must contain only SelfPlayJob values")
        game_ids = tuple(job.game_id for job in jobs)
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("jobs must have distinct explicit attempt identities")

        first = jobs[0]
        for job in jobs[1:]:
            if (
                job.mcts_config != first.mcts_config
                or job.selfplay_config != first.selfplay_config
                or job.compatibility != first.compatibility
                or job.players != first.players
            ):
                raise ValueError(
                    "the native pool runs one configuration per call; jobs disagree"
                )

        selfplay = first.selfplay_config
        mcts = first.mcts_config
        try:
            mode = _MODE_FOR_PRIOR[selfplay.bootstrap_prior]
        except KeyError as error:
            raise ValueError(
                f"the native backend has no evaluator mode for bootstrap_prior "
                f"{selfplay.bootstrap_prior!r}"
            ) from error

        module = require_native()
        native = native_game(first.players)

        config = module.EpisodeConfig(
            lanes=self.lanes,
            threads=self.threads,
            max_batch=self.max_batch,
            max_wait_us=self.max_wait_us,
            simulations=mcts.simulations,
            max_moves=selfplay.max_moves,
            temperature=selfplay.temperature,
            temperature_moves=selfplay.temperature_moves,
            dirichlet_alpha=mcts.dirichlet_alpha,
            dirichlet_epsilon=mcts.dirichlet_epsilon,
            simulations_late=self.simulations_late,
            late_move_threshold=self.late_move_threshold,
        )

        native_jobs = [
            (
                module.State(
                    occupancy=list(job.initial_state.occupancy),
                    current_player=job.initial_state.current_player_id,
                    turn_number=job.initial_state.turn_number,
                    status=0,
                    finish_order=list(job.initial_state.finish_order),
                ),
                job.seed & 0xFFFFFFFFFFFFFFFF,
            )
            for job in jobs
        ]

        factory = value_only_callback if mode == "value_only" else policy_value_callback
        callback = factory(self.model, device=self.device)
        result = native.play_episodes(native_jobs, config, callback, mode)
        self.metrics = {key: value for key, value in result.items() if key != "episodes"}

        return tuple(
            self._episode_result(job, record)
            for job, record in zip(jobs, result["episodes"])
        )

    def _episode_result(self, job: SelfPlayJob, record: dict) -> EpisodeResult:
        move_count = int(record["move_count"])
        if not record["completed"]:
            # Python's runner contributes zero samples from an aborted game and
            # this must match: a truncated game has no terminal outcome, so its
            # value targets would be invented.
            return EpisodeResult(
                game_id=job.game_id,
                seed=job.seed,
                retry_id=job.retry_id,
                model_key=job.model_key,
                compatibility=job.compatibility,
                samples=(),
                final_order=None,
                move_count=move_count,
                completed=False,
                aborted_reason=MAX_GAME_MOVES_EXCEEDED,
                bootstrap_prior=job.selfplay_config.bootstrap_prior,
            )

        final_order = tuple(int(seat) for seat in record["finish_order"])
        winner = final_order[0]
        samples = tuple(
            self._sample(job, move, winner) for move in record["moves"]
        )
        return EpisodeResult(
            game_id=job.game_id,
            seed=job.seed,
            retry_id=job.retry_id,
            model_key=job.model_key,
            compatibility=job.compatibility,
            samples=samples,
            final_order=final_order,
            move_count=move_count,
            completed=True,
            bootstrap_prior=job.selfplay_config.bootstrap_prior,
        )

    @staticmethod
    def _sample(job: SelfPlayJob, move: dict, winner: int) -> TrainingSample:
        visits = move["visit_counts"]
        total = sum(visits)
        if total <= 0:
            raise ValueError("a recorded move has no visits")
        # ``sparse_policy`` drops zero-probability actions and sorts by action
        # id; matching that exactly keeps replay digests comparable across
        # backends.
        policy = tuple(
            sorted(
                (int(action), count / total)
                for action, count in zip(move["root_actions"], visits)
                if count > 0
            )
        )
        players = tuple(int(seat) for seat in move["canonical_player_ids"])
        return TrainingSample(
            compatibility=job.compatibility,
            node_features=tuple(tuple(float(x) for x in row) for row in move["node_features"]),
            canonical_player_ids=players,
            sparse_policy=policy,
            value_target=(1.0 if players[0] == winner else -1.0,),
        )


__all__ = ["NativeSelfPlayPool"]
