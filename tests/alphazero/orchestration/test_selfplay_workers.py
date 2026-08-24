from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from multiprocessing import active_children
from pathlib import Path
from queue import Queue

import pytest

from diamond.alphazero.config import MCTSConfig, NetworkConfig, SelfPlayConfig
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.coordinator import InferenceConfig
from diamond.alphazero.inference.protocol import ModelKey
from diamond.alphazero.native import native_game, require_native
from diamond.alphazero.native.topology import camp_positions, neighbour_table
from diamond.alphazero.orchestration.selfplay_workers import (
    EpisodeResult,
    SelfPlayJob,
    SelfPlayWorkerError,
    SelfPlayWorkerPool,
    WorkerFailure,
    derive_game_id,
    derive_game_seed,
    run_selfplay_job,
)
from diamond.contract.camps import PLAYABLE_HOLES
from diamond.contract.state import (
    EMPTY,
    GameState,
    PlayerSpec,
    build_players,
    initial_state,
)


def _soo_key(digest: str = "a" * 64) -> ModelKey:
    return ModelKey("Soo", "1.2.3", digest)


def _near_terminal_setup(
    players: tuple[PlayerSpec, ...], finishers: int
) -> tuple[GameState, tuple[tuple[int, int], ...]]:
    occupied = [EMPTY] * PLAYABLE_HOLES
    reserved_targets = {
        position
        for player in players[:finishers]
        for position in camp_positions(player.target_camp)
    }
    used_entries: set[int] = set()
    actions: list[tuple[int, int]] = []
    entries: dict[int, int] = {}
    destinations: dict[int, int] = {}

    for player in players[:finishers]:
        target = camp_positions(player.target_camp)
        choice: tuple[int, int] | None = None
        for destination in target:
            for neighbour in neighbour_table()[destination]:
                if (
                    neighbour >= 0
                    and neighbour not in reserved_targets
                    and neighbour not in used_entries
                ):
                    choice = (neighbour, destination)
                    break
            if choice is not None:
                break
        assert choice is not None
        entry, destination = choice
        used_entries.add(entry)
        entries[player.id] = entry
        destinations[player.id] = destination
        for position in target:
            if position != destination:
                occupied[position] = player.id
        occupied[entry] = player.id

    state = GameState(tuple(occupied), players[0].id, 40)
    game = AlphaZeroGameAdapter(players, initial=state)
    module = require_native()
    native = native_game(players)
    for player in players[:finishers]:
        physical = game.codec.encode(entries[player.id], destinations[player.id])
        canonical = game.encoder.to_canonical_action(physical, players, player.id)
        # Legality is the core's answer, asked with this seat to move.
        probe = module.State(
            occupancy=list(state.occupancy),
            current_player=player.id,
            turn_number=state.turn_number,
        )
        assert canonical in native.canonical_legal_action_ids(probe)
        actions.append((player.id, canonical))
    return state, tuple(actions)


def _job(
    player_count: int,
    *,
    game_index: int,
    retry_id: str = "attempt-0",
) -> tuple[SelfPlayJob, tuple[tuple[int, int], ...]]:
    players = build_players(player_count)
    state, actions = _near_terminal_setup(players, finishers=player_count - 1)
    network = NetworkConfig(width=16, residual_blocks=1)
    if player_count == 2:
        compatibility = CheckpointCompatibilitySpec.soo(
            model_version="1.2.3", network_config=network
        )
        key = _soo_key()
    else:
        compatibility = CheckpointCompatibilitySpec.min(
            model_version="2.3.4", network_config=network
        )
        key = ModelKey("Min", "2.3.4", "b" * 64)
    return (
        SelfPlayJob(
            run_seed=71,
            iteration=4,
            game_index=game_index,
            retry_id=retry_id,
            model_key=key,
            compatibility=compatibility,
            players=players,
            initial_state=state,
            mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
            selfplay_config=SelfPlayConfig(max_moves=2, temperature_moves=0),
        ),
        actions,
    )


class ImmediateCoordinator:
    config = InferenceConfig(
        max_batch_size=2,
        max_wait_ms=1,
        request_queue_capacity=8,
        response_timeout_s=2.0,
    )

    def __init__(self, actions: tuple[tuple[str, int, int], ...]) -> None:
        self.actions = {
            (model_name, player_id): action
            for model_name, player_id, action in actions
        }
        self.requests = []

    def submit(self, request, reply_queue: Queue[object]) -> None:
        from diamond.alphazero.evaluator.base import EvalResult
        from diamond.alphazero.inference.protocol import InferenceResponse

        self.requests.append(request)
        preferred = self.actions.get(
            (request.model_key.model_name, request.canonical_player_ids[0]),
            request.legal_action_ids[0],
        )
        priors = {
            action: 1.0 if action == preferred else 0.0
            for action in request.legal_action_ids
        }
        reply_queue.put(
            InferenceResponse.from_eval_result(
                request,
                EvalResult(
                    priors,
                    0.0
                    if request.model_key.player_count == 2
                    else (0.0, 0.0, 0.0),
                ),
            )
        )


class FailingCoordinator:
    config = ImmediateCoordinator.config

    def submit(self, request, reply_queue: Queue[object]) -> None:
        from diamond.alphazero.inference.protocol import InferenceFailure

        reply_queue.put(
            InferenceFailure(
                client_id=request.client_id,
                request_id=request.request_id,
                model_key=request.model_key,
                error_type="RuntimeError",
                message="checkpoint unavailable",
            )
        )


class SilentCoordinator:
    config = InferenceConfig(
        max_batch_size=1,
        max_wait_ms=1,
        request_queue_capacity=1,
        response_timeout_s=10.0,
    )

    def submit(self, request, reply_queue: Queue[object]) -> None:
        pass


def _identity(*, retry_id: str = "attempt-0") -> dict[str, object]:
    return {
        "run_seed": 71,
        "iteration": 4,
        "game_index": 19,
        "model_key": _soo_key(),
        "retry_id": retry_id,
    }


def test_orchestration_package_exports_task_8_interfaces() -> None:
    from diamond.alphazero import orchestration

    assert orchestration.SelfPlayJob is SelfPlayJob
    assert orchestration.EpisodeResult is EpisodeResult
    assert orchestration.SelfPlayWorkerPool is SelfPlayWorkerPool


def test_game_identity_and_seed_are_stable_and_bind_every_input() -> None:
    identity = _identity()

    assert derive_game_id(**identity) == derive_game_id(**identity)
    assert derive_game_seed(**identity) == derive_game_seed(**identity)

    variants = (
        identity | {"run_seed": 72},
        identity | {"iteration": 5},
        identity | {"game_index": 20},
        identity | {"model_key": _soo_key("b" * 64)},
        identity | {"retry_id": "attempt-1"},
    )
    assert all(derive_game_id(**variant) != derive_game_id(**identity) for variant in variants)
    assert all(
        derive_game_seed(**variant) != derive_game_seed(**identity)
        for variant in variants
    )


def test_retry_reuses_identity_only_when_its_explicit_retry_id_is_reused() -> None:
    first_attempt = _identity(retry_id="attempt-0")
    same_attempt = _identity(retry_id="attempt-0")
    next_attempt = _identity(retry_id="attempt-1")

    assert derive_game_id(**first_attempt) == derive_game_id(**same_attempt)
    assert derive_game_seed(**first_attempt) == derive_game_seed(**same_attempt)
    assert derive_game_id(**first_attempt) != derive_game_id(**next_attempt)
    assert derive_game_seed(**first_attempt) != derive_game_seed(**next_attempt)


def test_job_and_result_are_immutable_and_pin_model_compatibility() -> None:
    job, actions = _job(2, game_index=1)
    coordinator = ImmediateCoordinator(
        tuple((job.model_key.model_name, player_id, action) for player_id, action in actions)
    )

    result = run_selfplay_job(job, coordinator)

    assert isinstance(result, EpisodeResult)
    assert result.game_id == job.game_id
    assert result.seed == job.seed
    assert result.model_key == job.model_key
    assert result.compatibility == job.compatibility
    assert result.completed
    assert result.samples
    assert all(sample.compatibility == job.compatibility for sample in result.samples)
    assert {request.model_key for request in coordinator.requests} == {job.model_key}
    with pytest.raises(FrozenInstanceError):
        job.game_index = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.completed = False  # type: ignore[misc]


def test_two_spawn_workers_run_authoritative_soo_and_min_episodes() -> None:
    soo_job, soo_actions = _job(2, game_index=2)
    min_job, min_actions = _job(3, game_index=3)
    preferred = tuple(
        (job.model_key.model_name, player_id, action)
        for job, actions in ((soo_job, soo_actions), (min_job, min_actions))
        for player_id, action in actions
    )
    coordinator = ImmediateCoordinator(preferred)
    children_before = {process.pid for process in active_children()}

    results = SelfPlayWorkerPool(
        coordinator,
        worker_count=2,
        worker_timeout_s=10.0,
        join_timeout_s=1.0,
    ).run((soo_job, min_job))

    assert tuple(result.game_id for result in results) == (
        soo_job.game_id,
        min_job.game_id,
    )
    # Which lane ran which game is no longer fixed: lanes pull from one shared
    # queue, so a lane that finishes first may take the next job rather than
    # waiting for its pre-assigned turn. Only the lane ids themselves are pinned.
    assert {result.worker_id for result in results} <= {0, 1}
    assert all(result.completed for result in results)
    assert results[0].final_order == (1, 2)
    assert tuple(sample.value_target for sample in results[0].samples) == ((1.0,),)
    assert results[1].final_order == (1, 2, 3)
    assert tuple(sample.value_target for sample in results[1].samples) == (
        (1.0, 0.0, -1.0),
        (0.0, -1.0, 1.0),
    )
    action_by_model_and_player = coordinator.actions
    for result in results:
        for sample in result.samples:
            expected = action_by_model_and_player[
                (result.model_key.model_name, sample.canonical_player_ids[0])
            ]
            assert {action for action, _probability in sample.sparse_policy} == {expected}
    assert {request.model_key for request in coordinator.requests} == {
        soo_job.model_key,
        min_job.model_key,
    }
    assert not ({process.pid for process in active_children()} - children_before)


def test_authoritative_abort_keeps_metrics_but_returns_zero_samples() -> None:
    near_terminal, _actions = _job(2, game_index=4)
    job = SelfPlayJob(
        run_seed=near_terminal.run_seed,
        iteration=near_terminal.iteration,
        game_index=near_terminal.game_index,
        retry_id=near_terminal.retry_id,
        model_key=near_terminal.model_key,
        compatibility=near_terminal.compatibility,
        players=near_terminal.players,
        initial_state=initial_state(near_terminal.players),
        mcts_config=near_terminal.mcts_config,
        selfplay_config=SelfPlayConfig(max_moves=1, temperature_moves=0),
    )

    result = SelfPlayWorkerPool(
        ImmediateCoordinator(()),
        worker_count=1,
        worker_timeout_s=10.0,
        join_timeout_s=1.0,
    ).run((job,))[0]

    assert not result.completed
    assert result.aborted_reason == "max_game_moves_exceeded"
    assert result.move_count == 1
    assert result.samples == ()
    assert result.final_order is None


def test_worker_failure_is_correlated_traceable_and_cleans_up() -> None:
    job, _actions = _job(2, game_index=5)
    children_before = {process.pid for process in active_children()}

    with pytest.raises(SelfPlayWorkerError) as caught:
        SelfPlayWorkerPool(
            FailingCoordinator(),
            worker_count=1,
            worker_timeout_s=10.0,
            join_timeout_s=1.0,
        ).run((job,))

    failure = caught.value.failure
    assert isinstance(failure, WorkerFailure)
    assert failure.game_id == job.game_id
    assert failure.seed == job.seed
    assert failure.retry_id == job.retry_id
    assert failure.model_key == job.model_key
    assert failure.worker_id == 0
    assert failure.error_type == "RuntimeError"
    assert "checkpoint unavailable" in failure.message
    assert "run_selfplay_job" in failure.traceback
    with pytest.raises(FrozenInstanceError):
        failure.message = "changed"  # type: ignore[misc]
    assert not ({process.pid for process in active_children()} - children_before)


def test_timeout_terminates_worker_and_leaves_no_children() -> None:
    job, _actions = _job(2, game_index=6)
    children_before = {process.pid for process in active_children()}

    with pytest.raises(TimeoutError, match=job.game_id):
        SelfPlayWorkerPool(
            SilentCoordinator(),
            worker_count=1,
            worker_timeout_s=0.5,
            join_timeout_s=0.5,
        ).run((job,))

    assert not ({process.pid for process in active_children()} - children_before)


def test_worker_module_does_not_import_torch() -> None:
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    environment = os.environ | {"PYTHONPATH": source_root}
    code = """
import sys
import diamond.alphazero.orchestration.selfplay_workers
assert 'torch' not in sys.modules, tuple(sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pool_timeout_accounts_for_multiple_jobs_per_lane() -> None:
    """32 games over 30 lanes means some lane runs two games back to back."""
    derive = SelfPlayWorkerPool.derive_pool_timeout

    assert derive(job_count=32, lane_count=30, per_game_timeout_s=900.0, grace_s=300.0) == 2100.0
    assert derive(job_count=32, lane_count=32, per_game_timeout_s=900.0, grace_s=300.0) == 1200.0
    assert derive(job_count=64, lane_count=30, per_game_timeout_s=900.0, grace_s=300.0) == 3000.0
    # One lane draining every job is the worst case.
    assert derive(job_count=4, lane_count=1, per_game_timeout_s=10.0, grace_s=1.0) == 41.0


def test_pool_timeout_rejects_nonsensical_inputs() -> None:
    derive = SelfPlayWorkerPool.derive_pool_timeout

    for kwargs in (
        {"job_count": 0, "lane_count": 1, "per_game_timeout_s": 1.0, "grace_s": 1.0},
        {"job_count": 1, "lane_count": 0, "per_game_timeout_s": 1.0, "grace_s": 1.0},
        {"job_count": 1, "lane_count": 1, "per_game_timeout_s": 0.0, "grace_s": 1.0},
        {"job_count": 1, "lane_count": 1, "per_game_timeout_s": 1.0, "grace_s": -1.0},
    ):
        with pytest.raises(ValueError):
            derive(**kwargs)


def test_a_single_timed_out_game_does_not_discard_completed_siblings() -> None:
    """One slow game must cost one game, not the whole iteration."""
    fast_a, actions_a = _job(2, game_index=20)
    fast_b, actions_b = _job(2, game_index=21)
    slow_base, _slow_actions = _job(2, game_index=22)
    # An infinitesimal budget expires on the first clock read, so the abort is
    # immediate rather than waited for.
    slow = SelfPlayJob(
        run_seed=slow_base.run_seed,
        iteration=slow_base.iteration,
        game_index=slow_base.game_index,
        retry_id=slow_base.retry_id,
        model_key=slow_base.model_key,
        compatibility=slow_base.compatibility,
        players=slow_base.players,
        initial_state=initial_state(slow_base.players),
        mcts_config=slow_base.mcts_config,
        selfplay_config=SelfPlayConfig(
            max_moves=2000, temperature_moves=0, max_game_seconds=1e-9
        ),
    )
    preferred = tuple(
        (job.model_key.model_name, player_id, action)
        for job, actions in ((fast_a, actions_a), (fast_b, actions_b))
        for player_id, action in actions
    )
    children_before = {process.pid for process in active_children()}

    results = SelfPlayWorkerPool(
        ImmediateCoordinator(preferred),
        worker_count=3,
        per_game_timeout_s=900.0,
        join_timeout_s=1.0,
    ).run((fast_a, fast_b, slow))

    by_id = {result.game_id: result for result in results}
    assert set(by_id) == {fast_a.game_id, fast_b.game_id, slow.game_id}

    timed_out = by_id[slow.game_id]
    assert not timed_out.completed
    assert timed_out.aborted_reason == "max_game_time_exceeded"
    assert timed_out.samples == ()
    assert timed_out.final_order is None

    # The siblings survived intact, which is the entire point.
    for job in (fast_a, fast_b):
        sibling = by_id[job.game_id]
        assert sibling.completed
        assert sibling.samples
        assert sibling.final_order == (1, 2)

    assert not ({process.pid for process in active_children()} - children_before)


def test_an_explicit_worker_timeout_still_wins() -> None:
    """The existing parameter keeps working for callers that pass it."""
    job, _actions = _job(2, game_index=23)

    with pytest.raises(TimeoutError, match=job.game_id):
        SelfPlayWorkerPool(
            SilentCoordinator(),
            worker_count=1,
            worker_timeout_s=0.5,
            per_game_timeout_s=10_000.0,
            join_timeout_s=0.5,
        ).run((job,))


def test_the_catastrophic_timeout_names_itself_distinctly() -> None:
    """Pool catastrophe, game timeout and inference timeout must not read alike."""
    job, _actions = _job(2, game_index=24)

    with pytest.raises(TimeoutError) as caught:
        SelfPlayWorkerPool(
            SilentCoordinator(),
            worker_count=1,
            worker_timeout_s=0.5,
            join_timeout_s=0.5,
        ).run((job,))

    message = str(caught.value)
    assert "catastrophic" in message
    assert "max_game_time_exceeded" not in message


class _RecordingCoordinator:
    """Records submissions and answers them on the reply queue immediately."""

    config = InferenceConfig(
        max_batch_size=4,
        max_wait_ms=1,
        request_queue_capacity=64,
        response_timeout_s=5.0,
    )

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.replies: list[Queue[object]] = []

    def submit(self, request, reply_queue: Queue[object]) -> None:
        from diamond.alphazero.evaluator.base import EvalResult
        from diamond.alphazero.inference.protocol import InferenceResponse

        self.submitted.append(request)
        self.replies.append(reply_queue)
        reply_queue.put(
            InferenceResponse.from_eval_result(
                request,
                EvalResult({action: 1.0 for action in request.legal_action_ids[:1]}, 0.0),
            )
        )


def _inference_request(index: int):
    from diamond.alphazero.inference.protocol import InferenceRequest

    return InferenceRequest(
        client_id="game-x",
        request_id=f"game-x:{index}",
        model_key=_soo_key(),
        node_features=((0.0, 0.0, 0.0, 0.0),),
        legal_action_ids=(3,),
        canonical_player_ids=(1, 2),
    )


def test_inference_forwarding_does_not_wait_on_the_episode_result_tick() -> None:
    """Requests must reach the coordinator without being paced by result polling.

    The parent used to alternate ``_forward_inference()`` with a blocking
    ``results.get(timeout=0.01)``, so every request and every reply waited on
    that 10 ms tick.  Measured on a trained checkpoint this cost ~33 ms of the
    ~43 ms worker-visible round trip.  Forwarding must therefore be driven by
    queue arrival, not by the episode-result poll period.
    """
    coordinator = _RecordingCoordinator()
    pool = SelfPlayWorkerPool(coordinator, worker_count=1)

    requests: Queue[object] = Queue()
    responses: Queue[object] = Queue()
    ticks: list[float] = []

    bridge = pool.start_inference_bridge(
        request_queue=requests,
        response_queues=(responses,),
        clock=lambda: ticks.append(0.0) or 0.0,
    )
    try:
        for index in range(3):
            requests.put((0, _inference_request(index)))
        delivered = [responses.get(timeout=5.0) for _ in range(3)]
    finally:
        bridge.close()

    assert len(coordinator.submitted) == 3
    assert {response.correlation_id for response in delivered} == {
        ("game-x", f"game-x:{index}") for index in range(3)
    }


def test_inference_bridge_rejects_an_uncorrelated_response() -> None:
    """A reply the parent cannot route must surface, never be dropped silently."""
    coordinator = _RecordingCoordinator()
    pool = SelfPlayWorkerPool(coordinator, worker_count=1)
    requests: Queue[object] = Queue()
    responses: Queue[object] = Queue()

    bridge = pool.start_inference_bridge(
        request_queue=requests, response_queues=(responses,)
    )
    try:
        bridge.parent_replies.put(object())
        assert bridge.wait_for_failure(timeout=5.0) is not None
    finally:
        bridge.close()


def _slow_job(game_index: int, *, moves: int = 300) -> SelfPlayJob:
    """A long episode: a whole game from the opening, not a deeper tree.

    This used to buy its slowness with ``simulations=1500`` on a near-terminal
    position. That stopped working when the runners moved to the native search:
    the position has one move left, so the tree is exhausted long before the
    simulation budget and 60,000 simulations run no longer than 1,500 -- both in
    single-digit milliseconds. Slowness measured in simulations was slowness
    measured in one engine's units.

    Moves are engine-independent: each one is a round trip to the parent's
    coordinator, and ``moves`` of them take ~1 s against the ~1 ms the
    near-terminal jobs take. The episode hits its move cap rather than a
    terminal state, which is what makes it long.
    """
    base, _actions = _job(2, game_index=game_index)
    return SelfPlayJob(
        run_seed=base.run_seed,
        iteration=base.iteration,
        game_index=base.game_index,
        retry_id=base.retry_id,
        model_key=base.model_key,
        compatibility=base.compatibility,
        players=base.players,
        initial_state=initial_state(base.players),
        mcts_config=MCTSConfig(simulations=1, dirichlet_epsilon=0.0),
        selfplay_config=SelfPlayConfig(max_moves=moves, temperature_moves=0),
    )


def test_a_lane_that_finishes_early_takes_pending_work() -> None:
    """Round-robin pre-assignment double-books lanes; a shared queue must not.

    Ten jobs over two lanes, one of them far slower than the rest. Pre-assigning
    `index % lane_count` gives each lane exactly five jobs no matter how long
    they take, so the slow lane's remaining four wait behind it while the other
    lane sits idle after finishing its own. Pulling from one shared queue instead
    lets the idle lane take that work, so the slow lane runs strictly fewer jobs.

    The margin is structural rather than timed: whatever the slow lane manages to
    pick up, the round-robin split is always exactly 5/5, so the assertion cannot
    be satisfied by pre-assignment however the spawn timing falls.

    Measured on the RTX 3060 as 32 games over 30 lanes: lanes 0 and 1 were
    double-booked, and giving every job its own lane bought 35% samples/hour.
    """
    from collections import Counter

    slow = _slow_job(40)
    fast_jobs = tuple(_job(2, game_index=41 + offset) for offset in range(9))
    preferred = tuple(
        (job.model_key.model_name, player_id, action)
        for job, actions in fast_jobs
        for player_id, action in actions
    )
    jobs = (slow, *(job for job, _actions in fast_jobs))
    children_before = {process.pid for process in active_children()}

    results = SelfPlayWorkerPool(
        ImmediateCoordinator(preferred),
        worker_count=2,
        per_game_timeout_s=60.0,
        join_timeout_s=1.0,
    ).run(jobs)

    by_id = {result.game_id: result for result in results}
    assert set(by_id) == {job.game_id for job in jobs}
    # The slow game is long, so it ends on its move cap rather than a terminal
    # state; every other job still has to finish.
    assert all(by_id[job.game_id].completed for job, _actions in fast_jobs)
    assert by_id[slow.game_id].aborted_reason == "max_game_moves_exceeded"

    lane_loads = Counter(result.worker_id for result in results)
    slow_lane = by_id[slow.game_id].worker_id
    other_lanes = {lane: count for lane, count in lane_loads.items() if lane != slow_lane}

    assert other_lanes, "the fast jobs never reached the second lane"
    assert lane_loads[slow_lane] < max(other_lanes.values()), (
        f"lane loads {dict(lane_loads)} are pre-assigned, not stolen: the slow "
        f"lane {slow_lane} carried as much work as an idle one"
    )
    assert not ({process.pid for process in active_children()} - children_before)
