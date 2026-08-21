"""Spawn-safe orchestration for pinned AlphaZero self-play episodes."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import traceback as traceback_module
from dataclasses import dataclass, replace
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

from ..bootstrap.evaluator import bootstrap_evaluator
from ..bootstrap.heuristic import BOOTSTRAP_PRIOR_NONE
from ..config import BOOTSTRAP_PRIORS, MCTSConfig, SelfPlayConfig
from ..evaluator.base import Evaluator
from ..game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from ..identity import (
    MIN_MODEL_NAME,
    SOO_MODEL_NAME,
    CheckpointCompatibilitySpec,
)
from ..inference.protocol import InferenceFailure, InferenceRequest, ModelKey
from ..inference.remote import RemoteEvaluator, RequestCoordinator
from ..replay import TrainingSample
from ..selfplay.runner_2p import SooSelfPlayRunner
from ..selfplay.runner_3p import MinSelfPlayRunner
from ...game.state import GameState, PlayerSpec


def _identity_digest(
    domain: str,
    *,
    run_seed: int,
    iteration: int,
    game_index: int,
    model_key: ModelKey,
    retry_id: str,
) -> bytes:
    if not isinstance(run_seed, int) or isinstance(run_seed, bool):
        raise ValueError("run_seed must be an integer")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    if not isinstance(game_index, int) or isinstance(game_index, bool) or game_index < 0:
        raise ValueError("game_index must be a non-negative integer")
    if not isinstance(model_key, ModelKey):
        raise ValueError("model_key must be a ModelKey")
    if not isinstance(retry_id, str) or not retry_id.strip():
        raise ValueError("retry_id must be a non-empty string")
    payload = json.dumps(
        {
            "domain": domain,
            "game_index": game_index,
            "iteration": iteration,
            "model_key": model_key.to_payload(),
            "retry_id": retry_id,
            "run_seed": run_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def derive_game_id(
    *,
    run_seed: int,
    iteration: int,
    game_index: int,
    model_key: ModelKey,
    retry_id: str,
) -> str:
    """Return the stable identity of one explicit self-play attempt."""
    digest = _identity_digest(
        "alphadiamond-selfplay-game-v1",
        run_seed=run_seed,
        iteration=iteration,
        game_index=game_index,
        model_key=model_key,
        retry_id=retry_id,
    )
    return f"game-{digest.hex()}"


def derive_game_seed(
    *,
    run_seed: int,
    iteration: int,
    game_index: int,
    model_key: ModelKey,
    retry_id: str,
) -> int:
    """Return a deterministic non-negative 63-bit seed for one attempt."""
    digest = _identity_digest(
        "alphadiamond-selfplay-seed-v1",
        run_seed=run_seed,
        iteration=iteration,
        game_index=game_index,
        model_key=model_key,
        retry_id=retry_id,
    )
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class SelfPlayJob:
    """Immutable inputs for one explicit, replay-safe episode attempt."""

    run_seed: int
    iteration: int
    game_index: int
    retry_id: str
    model_key: ModelKey
    compatibility: CheckpointCompatibilitySpec
    players: tuple[PlayerSpec, ...]
    initial_state: GameState
    mcts_config: MCTSConfig
    selfplay_config: SelfPlayConfig

    def __post_init__(self) -> None:
        # Validate every identity field through the canonical derivation.
        derive_game_id(**self._identity_fields())
        if not isinstance(self.compatibility, CheckpointCompatibilitySpec):
            raise ValueError("compatibility must be a CheckpointCompatibilitySpec")
        identity = self.compatibility.identity
        if (
            identity.model_name != self.model_key.model_name
            or identity.model_version != self.model_key.model_version
        ):
            raise ValueError("model_key and compatibility must identify the same model")
        if not isinstance(self.players, tuple) or len(self.players) != identity.player_count:
            raise ValueError("players must be a tuple matching the model player count")
        if not isinstance(self.initial_state, GameState):
            raise ValueError("initial_state must be an authoritative GameState")
        if self.initial_state.current_player_id not in {player.id for player in self.players}:
            raise ValueError("initial_state current player is not in players")
        if not isinstance(self.mcts_config, MCTSConfig):
            raise ValueError("mcts_config must be an MCTSConfig")
        if not isinstance(self.selfplay_config, SelfPlayConfig):
            raise ValueError("selfplay_config must be a SelfPlayConfig")

    def _identity_fields(self) -> dict[str, object]:
        return {
            "run_seed": self.run_seed,
            "iteration": self.iteration,
            "game_index": self.game_index,
            "model_key": self.model_key,
            "retry_id": self.retry_id,
        }

    @property
    def game_id(self) -> str:
        return derive_game_id(**self._identity_fields())  # type: ignore[arg-type]

    @property
    def seed(self) -> int:
        return derive_game_seed(**self._identity_fields())  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Completed or aborted episode plus its pinned identity and metrics."""

    game_id: str
    seed: int
    retry_id: str
    model_key: ModelKey
    compatibility: CheckpointCompatibilitySpec
    samples: tuple[TrainingSample, ...]
    final_order: tuple[int, ...] | None
    move_count: int
    completed: bool
    aborted_reason: str | None = None
    worker_id: int | None = None
    bootstrap_prior: str = BOOTSTRAP_PRIOR_NONE
    """Self-play provenance: which prior generated this episode."""

    def __post_init__(self) -> None:
        if self.bootstrap_prior not in BOOTSTRAP_PRIORS:
            raise ValueError(
                f"bootstrap_prior must be one of {sorted(BOOTSTRAP_PRIORS)}"
            )
        if not isinstance(self.samples, tuple):
            raise ValueError("samples must be a tuple")
        if self.move_count < 0:
            raise ValueError("move_count must be non-negative")
        if self.completed:
            if self.final_order is None or self.aborted_reason is not None:
                raise ValueError("completed episodes require an order and no abort reason")
        elif self.samples or self.final_order is not None or not self.aborted_reason:
            raise ValueError("aborted episodes require a reason and zero samples")
        if any(sample.compatibility != self.compatibility for sample in self.samples):
            raise ValueError("episode samples must preserve pinned compatibility")


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Correlated primitive/frozen details for one failed child episode."""

    worker_id: int
    game_id: str
    seed: int
    retry_id: str
    model_key: ModelKey
    error_type: str
    message: str
    traceback: str

    def __post_init__(self) -> None:
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        for value, field in (
            (self.game_id, "game_id"),
            (self.retry_id, "retry_id"),
            (self.error_type, "error_type"),
            (self.message, "message"),
            (self.traceback, "traceback"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")


class SelfPlayWorkerError(RuntimeError):
    """Raised in the parent with the exact child failure envelope."""

    def __init__(self, failure: WorkerFailure) -> None:
        self.failure = failure
        super().__init__(
            f"self-play worker {failure.worker_id} failed for {failure.game_id}: "
            f"{failure.error_type}: {failure.message}"
        )


def run_selfplay_job(
    job: SelfPlayJob,
    coordinator: RequestCoordinator,
) -> EpisodeResult:
    """Run one pinned episode through the established remote evaluator boundary."""
    if not isinstance(job, SelfPlayJob):
        raise ValueError("job must be a SelfPlayJob")
    game = DiamondSearchAdapter(
        AlphaZeroGameAdapter(job.players, initial=job.initial_state)
    )
    evaluator: Evaluator = RemoteEvaluator(
        coordinator,
        model_key=job.model_key,
        client_id=job.game_id,
    )
    mcts_config = replace(job.mcts_config, seed=job.seed)
    selfplay_config = replace(job.selfplay_config, seed=job.seed)
    evaluator = bootstrap_evaluator(evaluator, selfplay_config.bootstrap_prior)
    if job.model_key.model_name == SOO_MODEL_NAME:
        episode = SooSelfPlayRunner(
            game,
            evaluator,
            mcts_config,
            selfplay_config,
            job.compatibility,
        ).run()
    elif job.model_key.model_name == MIN_MODEL_NAME:
        episode = MinSelfPlayRunner(
            game,
            evaluator,
            mcts_config,
            selfplay_config,
            job.compatibility,
        ).run()
    else:  # ModelKey currently makes this unreachable; keep dispatch explicit.
        raise ValueError(f"unsupported self-play model: {job.model_key.model_name}")
    return EpisodeResult(
        game_id=job.game_id,
        seed=job.seed,
        retry_id=job.retry_id,
        model_key=job.model_key,
        compatibility=job.compatibility,
        samples=episode.samples if episode.completed else (),
        final_order=episode.final_order,
        move_count=episode.move_count,
        completed=episode.completed,
        bootstrap_prior=selfplay_config.bootstrap_prior,
        aborted_reason=episode.aborted_reason,
    )


@dataclass(frozen=True, slots=True)
class _RemoteConfig:
    response_timeout_s: float


class _ProcessRequestCoordinator:
    """Worker-local adapter from RemoteEvaluator to parent-owned inference."""

    def __init__(
        self,
        worker_id: int,
        request_queue: object,
        response_queue: object,
        response_timeout_s: float,
    ) -> None:
        self.worker_id = worker_id
        self.config = _RemoteConfig(response_timeout_s)
        self._requests = request_queue
        self._responses = response_queue
        self._routes: dict[tuple[str, str], Queue[object]] = {}
        self._routes_lock = Lock()
        self._closed = Event()
        self._pump = Thread(
            target=self._pump_responses,
            name=f"selfplay-response-{worker_id}",
            daemon=True,
        )
        self._pump.start()

    def submit(self, request: InferenceRequest, reply_queue: Queue[object]) -> None:
        with self._routes_lock:
            self._routes[request.correlation_id] = reply_queue
        try:
            self._requests.put(
                (self.worker_id, request),
                timeout=self.config.response_timeout_s,
            )
        except Full:
            with self._routes_lock:
                self._routes.pop(request.correlation_id, None)
            reply_queue.put(
                InferenceFailure(
                    client_id=request.client_id,
                    request_id=request.request_id,
                    model_key=request.model_key,
                    error_type="Full",
                    message="self-play inference bridge is full",
                )
            )

    def close(self) -> None:
        self._closed.set()
        self._pump.join(timeout=min(0.5, self.config.response_timeout_s))

    def _pump_responses(self) -> None:
        while not self._closed.is_set():
            try:
                response = self._responses.get(timeout=0.05)
            except Empty:
                continue
            correlation_id = getattr(response, "correlation_id", None)
            with self._routes_lock:
                reply_queue = self._routes.pop(correlation_id, None)
            if reply_queue is not None:
                reply_queue.put(response)


def selfplay_worker_entry(
    worker_id: int,
    job_queue: object,
    result_queue: object,
    inference_request_queue: object,
    inference_response_queue: object,
    response_timeout_s: float,
) -> None:
    """Top-level spawn target; all arguments are pickle-safe queues/primitives."""
    coordinator = _ProcessRequestCoordinator(
        worker_id,
        inference_request_queue,
        inference_response_queue,
        response_timeout_s,
    )
    try:
        while True:
            job = job_queue.get()
            if job is None:
                return
            try:
                result = run_selfplay_job(job, coordinator)
            except BaseException as error:
                result_queue.put(
                    WorkerFailure(
                        worker_id=worker_id,
                        game_id=job.game_id,
                        seed=job.seed,
                        retry_id=job.retry_id,
                        model_key=job.model_key,
                        error_type=type(error).__name__,
                        message=str(error) or repr(error),
                        traceback=traceback_module.format_exc(),
                    )
                )
                return
            else:
                result_queue.put(replace(result, worker_id=worker_id))
    finally:
        coordinator.close()


class _InferenceBridge:
    """Forward inference between workers and the coordinator on its own threads.

    The parent used to drain inference inline with ``results.get(timeout=0.01)``,
    so every request and every reply waited on that 10 ms episode-result tick.
    Measured on the trained Soo checkpoint the parent loop ran at 93.9 ticks/s
    while the forwarding work itself took 0.33 ms, and the worker-visible round
    trip was ~43 ms against ~5 ms of actual CPU inference.

    Both directions therefore block on queue arrival instead: no polling, no
    busy spin, and episode collection no longer paces inference.
    """

    def __init__(
        self,
        coordinator: RequestCoordinator,
        request_queue: object,
        response_queues: tuple[object, ...],
        *,
        poll_timeout_s: float = 0.05,
    ) -> None:
        self.coordinator = coordinator
        self.parent_replies: Queue[object] = Queue()
        self._requests = request_queue
        self._response_queues = response_queues
        self._routes: dict[tuple[str, str], int] = {}
        self._routes_lock = Lock()
        self._closed = Event()
        self._poll_timeout_s = poll_timeout_s
        self._failure: BaseException | None = None
        self._failed = Event()
        self._threads = (
            Thread(target=self._pump_requests, name="selfplay-bridge-requests", daemon=True),
            Thread(target=self._pump_responses, name="selfplay-bridge-responses", daemon=True),
        )

    def start(self) -> "_InferenceBridge":
        for thread in self._threads:
            thread.start()
        return self

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def wait_for_failure(self, timeout: float) -> BaseException | None:
        self._failed.wait(timeout)
        return self._failure

    def close(self) -> None:
        self._closed.set()
        for thread in self._threads:
            thread.join(timeout=max(self._poll_timeout_s * 4, 0.5))

    def _fail(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        self._failed.set()

    def _pump_requests(self) -> None:
        while not self._closed.is_set():
            try:
                item = self._requests.get(timeout=self._poll_timeout_s)
            except Empty:
                continue
            except (EOFError, OSError):
                return
            try:
                worker_id, request = item
                with self._routes_lock:
                    self._routes[request.correlation_id] = worker_id
                self.coordinator.submit(request, self.parent_replies)
            except BaseException as error:  # noqa: BLE001 - surfaced to the parent
                self._fail(error)
                return

    def _pump_responses(self) -> None:
        while not self._closed.is_set():
            try:
                response = self.parent_replies.get(timeout=self._poll_timeout_s)
            except Empty:
                continue
            try:
                correlation_id = getattr(response, "correlation_id", None)
                with self._routes_lock:
                    worker_id = self._routes.pop(correlation_id, None)
                if worker_id is None:
                    raise RuntimeError("central inference returned an uncorrelated response")
                self._response_queues[worker_id].put(
                    response,
                    timeout=self.coordinator.config.response_timeout_s,
                )
            except BaseException as error:  # noqa: BLE001 - surfaced to the parent
                self._fail(error)
                return


class SelfPlayWorkerPool:
    """One-shot spawn worker group bridged to one parent inference coordinator."""

    #: Headroom over the per-lane game budget for spawn, checkpoint load and drain.
    DEFAULT_GRACE_S = 300.0

    #: Episode-result wait.  Inference has its own pump threads, so this only
    #: bounds how often the loop re-checks liveness and the pool deadline.
    RESULT_POLL_S = 0.05

    def __init__(
        self,
        coordinator: RequestCoordinator,
        *,
        worker_count: int,
        worker_timeout_s: float | None = None,
        per_game_timeout_s: float | None = None,
        grace_s: float = DEFAULT_GRACE_S,
        join_timeout_s: float = 2.0,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if worker_timeout_s is not None and worker_timeout_s <= 0:
            raise ValueError("worker timeout must be positive")
        if per_game_timeout_s is not None and per_game_timeout_s <= 0:
            raise ValueError("per-game timeout must be positive")
        if grace_s < 0 or join_timeout_s <= 0:
            raise ValueError("grace must be non-negative and join timeout positive")
        self.coordinator = coordinator
        self.worker_count = worker_count
        self.worker_timeout_s = worker_timeout_s
        self.per_game_timeout_s = per_game_timeout_s
        self.grace_s = grace_s
        self.join_timeout_s = join_timeout_s

    @staticmethod
    def derive_pool_timeout(
        *,
        job_count: int,
        lane_count: int,
        per_game_timeout_s: float,
        grace_s: float,
    ) -> float:
        """Budget the busiest lane, not the pool as a whole.

        With more jobs than lanes some lane runs several games back to back, so
        a per-game budget does not bound the iteration.  Sizing from the deepest
        lane keeps this guard catastrophic rather than routine.
        """
        if job_count < 1 or lane_count < 1:
            raise ValueError("job_count and lane_count must be positive")
        if per_game_timeout_s <= 0:
            raise ValueError("per_game_timeout_s must be positive")
        if grace_s < 0:
            raise ValueError("grace_s must be non-negative")
        jobs_per_lane = math.ceil(job_count / lane_count)
        return jobs_per_lane * per_game_timeout_s + grace_s

    def _pool_timeout(self, *, job_count: int, lane_count: int) -> float:
        """Explicit override wins; otherwise derive, else keep the old default."""
        if self.worker_timeout_s is not None:
            return self.worker_timeout_s
        if self.per_game_timeout_s is not None:
            return self.derive_pool_timeout(
                job_count=job_count,
                lane_count=lane_count,
                per_game_timeout_s=self.per_game_timeout_s,
                grace_s=self.grace_s,
            )
        return 60.0

    def run(self, jobs: tuple[SelfPlayJob, ...]) -> tuple[EpisodeResult, ...]:
        if not isinstance(jobs, tuple):
            raise ValueError("jobs must be a tuple")
        if not jobs:
            return ()
        if not all(isinstance(job, SelfPlayJob) for job in jobs):
            raise ValueError("jobs must contain only SelfPlayJob values")
        game_ids = tuple(job.game_id for job in jobs)
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("jobs must have distinct explicit attempt identities")

        context = multiprocessing.get_context("spawn")
        lane_count = min(self.worker_count, len(jobs))
        inference_requests = context.Queue(maxsize=max(2, lane_count * 2))
        results = context.Queue(maxsize=len(jobs))
        # One shared queue, not one per lane: a lane takes the next pending job
        # when it finishes, so a slow game costs its own lane and no other. Round
        # -robin pre-assignment double-booked lanes 0 and 1 with 32 games over 30
        # lanes, and a lane's second game could not start until its first
        # finished -- worth 35% samples/hour on the RTX 3060 (Finding 3).
        job_queue = context.Queue(maxsize=len(jobs) + lane_count)
        response_queues = tuple(context.Queue(maxsize=max(2, lane_count * 2)) for _ in range(lane_count))
        processes = tuple(
            context.Process(
                target=selfplay_worker_entry,
                args=(
                    worker_id,
                    job_queue,
                    results,
                    inference_requests,
                    response_queues[worker_id],
                    self.coordinator.config.response_timeout_s,
                ),
                name=f"alphazero-selfplay-{worker_id}",
            )
            for worker_id in range(lane_count)
        )
        all_queues = (inference_requests, results, job_queue, *response_queues)
        received: dict[str, EpisodeResult] = {}
        bridge = self.start_inference_bridge(
            request_queue=inference_requests, response_queues=response_queues
        )

        try:
            for process in processes:
                process.start()
            for job in jobs:
                job_queue.put(job)
            # One sentinel per lane: whichever lane reaches it last stops last.
            for _ in range(lane_count):
                job_queue.put(None)

            deadline = monotonic() + self._pool_timeout(
                job_count=len(jobs), lane_count=lane_count
            )
            while len(received) < len(jobs):
                # Inference is pumped by its own threads, so this wait no longer
                # paces the inference path and can block properly rather than
                # spinning at a 10 ms tick.
                bridge.raise_if_failed()
                try:
                    result = results.get(timeout=self.RESULT_POLL_S)
                except Empty:
                    result = None
                if isinstance(result, EpisodeResult):
                    received[result.game_id] = result
                elif isinstance(result, WorkerFailure):
                    raise SelfPlayWorkerError(result)
                elif result is not None:
                    raise RuntimeError("self-play worker returned a malformed result")
                failed = next(
                    (
                        process
                        for process in processes
                        if process.exitcode not in (None, 0)
                    ),
                    None,
                )
                if failed is not None:
                    raise RuntimeError(
                        f"self-play worker {failed.name} exited with code {failed.exitcode}"
                    )
                if monotonic() >= deadline:
                    # Not the normal way a long game ends -- a game aborts itself
                    # on its own budget.  Reaching here means a dead child, broken
                    # IPC or an unresponsive worker.
                    pending = tuple(game_id for game_id in game_ids if game_id not in received)
                    raise TimeoutError(
                        "self-play pool exceeded its catastrophic safety deadline "
                        f"waiting for jobs: {pending}"
                    )
            return tuple(received[game_id] for game_id in game_ids)
        finally:
            bridge.close()
            self._cleanup(processes, all_queues)

    def start_inference_bridge(
        self,
        *,
        request_queue: object,
        response_queues: tuple[object, ...],
        clock: object = None,
    ) -> _InferenceBridge:
        """Start the dedicated request/response pumps for this pool's coordinator."""
        return _InferenceBridge(
            self.coordinator, request_queue, response_queues
        ).start()

    def _cleanup(self, processes: tuple[object, ...], queues: tuple[object, ...]) -> None:
        for process in processes:
            process.join(timeout=self.join_timeout_s)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=self.join_timeout_s)
            if process.is_alive():
                process.kill()
                process.join(timeout=self.join_timeout_s)
            process.close()
        for queue in queues:
            queue.cancel_join_thread()
            queue.close()


__all__ = [
    "EpisodeResult",
    "SelfPlayJob",
    "SelfPlayWorkerError",
    "SelfPlayWorkerPool",
    "WorkerFailure",
    "derive_game_id",
    "derive_game_seed",
    "run_selfplay_job",
    "selfplay_worker_entry",
]
