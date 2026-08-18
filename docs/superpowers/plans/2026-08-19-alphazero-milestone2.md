# AlphaDiamond Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build production training and historical strength evaluation for independently versioned Soo and Min checkpoints without changing Milestone 1 game, MCTS, or value semantics.

**Architecture:** Add three isolated subsystems: a framework-neutral rating/benchmark layer, an evaluator-compatible centralized inference layer, and a spawn-safe persistence/orchestration layer. Immutable participant, protocol, event, episode, and run-state identities make every update replayable and idempotent; GPU optimization remains behind the correct eager FP32 path.

**Tech Stack:** Python 3.11+, PyTorch 2.5+, `trueskill` 0.4.5, standard-library multiprocessing/queue/JSON/hashlib/argparse, pytest.

**Spec:** `blueprint/milestone2.md`

## Global Constraints

- The authoritative 73-hole Diamond engine remains the sole game-rules implementation.
- Soo is the independently versioned 2-player model with `current-player-scalar-winloss-v1`.
- Min is the independently versioned 3-player model with `canonical-placement-utility-1-0-minus1-v1`.
- Soo strength uses Elo; Min strength uses the official `trueskill` package with `tau=0.0` and `draw_probability=0.0`.
- Min rated events require three distinct checkpoint artifact IDs; candidate/champion/champion is promotion-only.
- Rated Soo schedules contain complete `4 * opening_count` cycles; rated Min schedules contain complete `36 * opening_count` cycles.
- Ratings never mix benchmark protocol IDs, and aborted games never change ratings.
- MCTS remains independent from PyTorch, Elo, TrueSkill, orchestration, and GUI code.
- Worker code must be Windows `spawn` safe and must not import PySide/QML/controller modules.
- Runtime artifacts live under ignored `runs/`; no external database, C++, OpenVINO, or GUI agent routing is added.
- Eager FP32 is implemented and measured before BF16; `torch.compile` is benchmark-only and retained only with evidence.
- Every behavior change follows red-green-refactor TDD.

## File Structure

- `src/diamond/alphazero/rating/protocol.py`: benchmark and rating-system configuration identities.
- `src/diamond/alphazero/rating/participants.py`: immutable checkpoint artifact identity and hashing.
- `src/diamond/alphazero/rating/events.py`: immutable Soo/Min rated match records and stable event IDs.
- `src/diamond/alphazero/rating/elo.py`: Soo Elo math and state transitions.
- `src/diamond/alphazero/rating/min_trueskill.py`: explicit official TrueSkill environment and Min transitions.
- `src/diamond/alphazero/rating/registry.py`: append-only event persistence, deterministic replay, leaderboards.
- `src/diamond/alphazero/rating/openings.py`: deterministic authoritative opening generation/reconstruction.
- `src/diamond/alphazero/rating/schedule.py`: complete Soo pair and Min triple rated schedules.
- `src/diamond/alphazero/inference/protocol.py`: serializable request/response/model-key envelopes.
- `src/diamond/alphazero/inference/model_pool.py`: compatibility-checked resident evaluators.
- `src/diamond/alphazero/inference/coordinator.py`: bounded, model-keyed batching and metrics.
- `src/diamond/alphazero/inference/remote.py`: worker-side `Evaluator` implementation.
- `src/diamond/alphazero/orchestration/selfplay_workers.py`: spawn-safe worker entry and episode envelopes.
- `src/diamond/alphazero/orchestration/replay_store.py`: idempotent episode ingestion and chunked replay snapshots.
- `src/diamond/alphazero/orchestration/run_state.py`: atomic serializable iteration state.
- `src/diamond/alphazero/orchestration/coordinator.py`: explicit resumable stage machine.
- `src/diamond/alphazero/orchestration/cli.py`: headless start/resume/benchmark/leaderboard/profile commands.
- `src/diamond/alphazero/milestone2_smoke.py`: tiny real CPU Soo/Min workflow.
- `tests/alphazero/rating/`, `tests/alphazero/inference/`, `tests/alphazero/orchestration/`: focused CPU tests.
- `tests/alphazero/test_milestone2_smoke.py`: end-to-end CPU smoke assertions.
- `docs/alphazero.md`: architecture and operator runbook.
- `pyproject.toml`, `environment.yml`, `.gitignore`: isolated dependency and runtime layout.

---

### Task 1: Dependency Surface and Immutable Rating Identities

**Files:**
- Modify: `pyproject.toml`
- Modify: `environment.yml`
- Create: `src/diamond/alphazero/rating/__init__.py`
- Create: `src/diamond/alphazero/rating/participants.py`
- Create: `src/diamond/alphazero/rating/protocol.py`
- Create: `tests/alphazero/rating/test_participants.py`
- Create: `tests/alphazero/rating/test_protocol.py`

**Interfaces:**
- Consumes: `ModelIdentity`, `CheckpointCompatibilitySpec`, `MCTSConfig`.
- Produces: `CheckpointParticipant.from_checkpoint(path)`, `EloConfig`, `TrueSkillConfig`, `BenchmarkProtocol`, `benchmark_protocol_id`.

- [ ] **Step 1: Write failing participant tests** proving content SHA-256, training step, model/version metadata, stable participant ID, same-version/different-step distinction, and metadata/hash collision rejection.
- [ ] **Step 2: Run `python -m pytest tests/alphazero/rating/test_participants.py -v`** and verify import failure for the missing rating package.
- [ ] **Step 3: Implement `CheckpointParticipant`** as a frozen dataclass with `participant_id`, `model_name`, `model_version`, `training_step`, `checkpoint_sha256`, `compatibility_metadata`, and `display_name`; load checkpoint metadata with `torch.load(..., weights_only=True)` and hash raw bytes without mutating a trainer.
- [ ] **Step 4: Run the participant tests** and verify all pass.
- [ ] **Step 5: Write failing protocol tests** for deterministic canonical JSON hashing, noise rejection, fixed compute fields, rating parameter inclusion, and changed-opening/config namespace separation.
- [ ] **Step 6: Implement protocol DTOs** with these signatures:

```python
@dataclass(frozen=True, slots=True)
class EloConfig:
    initial_rating: float = 1000.0
    k_factor: float = 32.0
    logistic_scale: float = 400.0
    rating_system_version: str = "soo-elo-v1"

@dataclass(frozen=True, slots=True)
class TrueSkillConfig:
    mu: float = 25.0
    sigma: float = 25.0 / 3.0
    beta: float = 25.0 / 6.0
    tau: float = 0.0
    draw_probability: float = 0.0
    backend: str | None = None
    rating_system_version: str = "min-trueskill-v1"

@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    compatibility: CheckpointCompatibilitySpec
    simulations: int
    c_puct: float
    dirichlet_epsilon: float
    decision_temperature: float
    max_game_moves: int
    opening_suite_version: str
    opening_suite_hash: str
    rating_system_version: str
    rating_parameters: dict[str, object]
    inference_numeric_mode: str = "fp32"

    @property
    def protocol_id(self) -> str: ...
```

- [ ] **Step 7: Run protocol tests and the Milestone 1 identity tests**; verify green.
- [ ] **Step 8: Add `trueskill>=0.4.5,<0.5` only to the `alphazero` extra and mamba training environment**, then verify `python -c "import trueskill; print(trueskill.__version__)"`.
- [ ] **Step 9: Commit** with `feat: add benchmark and checkpoint artifact identities`.

### Task 2: Immutable Rating Events

**Files:**
- Create: `src/diamond/alphazero/rating/events.py`
- Create: `tests/alphazero/rating/test_events.py`

**Interfaces:**
- Consumes: `CheckpointParticipant.participant_id`, `BenchmarkProtocol.protocol_id`.
- Produces: `SooRatingEvent`, `MinRatingEvent`, `RatingEventError`, canonical `event_id`.

- [ ] **Step 1: Write failing tests** for deterministic event IDs, completed Soo winner/loser validation, Min strict full ranking, distinct Min participants, abort-without-outcome, stable sequence index, and protocol identity retention.
- [ ] **Step 2: Run the event test file** and verify missing-module failure.
- [ ] **Step 3: Implement frozen event DTOs** whose IDs are SHA-256 hashes over canonical semantic payloads excluding non-deterministic display text; require exact participant cardinality and permutation outcomes.
- [ ] **Step 4: Run event tests** and verify all pass.
- [ ] **Step 5: Commit** with `feat: add immutable rating match events`.

### Task 3: Soo Elo and Replayable Registry

**Files:**
- Create: `src/diamond/alphazero/rating/elo.py`
- Create: `src/diamond/alphazero/rating/registry.py`
- Create: `tests/alphazero/rating/test_elo.py`
- Create: `tests/alphazero/rating/test_registry_elo.py`

**Interfaces:**
- Produces: `expected_score(rating_a, rating_b, config)`, `rate_soo_match(...)`, `RatingRegistry.add_participant`, `record_event`, `rebuild`, `soo_leaderboard`, `save`, `load`.

- [ ] **Step 1: Write failing literal Elo tests** for equal ratings (`0.5`), a hand-calculated unequal expectation, winner/loser updates from pre-match values, and no update for abort.
- [ ] **Step 2: Run Elo tests** and verify expected missing import.
- [ ] **Step 3: Implement pure Elo functions** returning both new ratings simultaneously from pre-match values.
- [ ] **Step 4: Run Elo tests** and verify green.
- [ ] **Step 5: Write failing registry tests** for default 1000, duplicate-event idempotence, protocol mismatch rejection, participant compatibility rejection, deterministic replay, atomic save/load, corrupted-cache rebuild, and descending leaderboard sorting.
- [ ] **Step 6: Implement `RatingRegistry`** with participant metadata plus ordered immutable events as source of truth; write JSON through sibling `.tmp` files and `Path.replace`; derive cached ratings on replay.
- [ ] **Step 7: Run registry and all rating tests** and verify green.
- [ ] **Step 8: Commit** with `feat: add replayable Soo Elo registry`.

### Task 4: Official Min TrueSkill Registry

**Files:**
- Create: `src/diamond/alphazero/rating/min_trueskill.py`
- Modify: `src/diamond/alphazero/rating/registry.py`
- Create: `tests/alphazero/rating/test_min_trueskill.py`
- Create: `tests/alphazero/rating/test_registry_min.py`

**Interfaces:**
- Produces: `create_environment(config) -> trueskill.TrueSkill`, `MinRating(mu, sigma, exposure, rated_games)`, native free-for-all update path.

- [ ] **Step 1: Write failing environment tests** asserting `mu=25`, `sigma=25/3`, `beta=25/6`, `tau=0`, `draw_probability=0`, and no mutation of `trueskill.global_env()`.
- [ ] **Step 2: Run environment tests** and verify missing implementation.
- [ ] **Step 3: Implement explicit environment construction** and `env.create_rating`, `env.rate(groups, ranks=ranks)`, `env.expose` usage only.
- [ ] **Step 4: Write failing semantic tests** for a three-player free-for-all, lower-rank superiority, finite sigma, winner/last movement, distinct-artifact rejection, abort no-op, and candidate/champion/champion rejection.
- [ ] **Step 5: Implement Min event application** by mapping final-order participant IDs to native ranks and updating all groups in one `env.rate` call.
- [ ] **Step 6: Write and implement registry replay tests** for deterministic mu/sigma/exposure, event ordering, duplicate handling, insufficient history, persistence, and exposure-sorted leaderboard.
- [ ] **Step 7: Run all rating tests** and verify green.
- [ ] **Step 8: Commit** with `feat: add official Min TrueSkill registry`.

### Task 5: Authoritative Opening Suites and Rated Schedules

**Files:**
- Create: `src/diamond/alphazero/rating/openings.py`
- Create: `src/diamond/alphazero/rating/schedule.py`
- Create: `tests/alphazero/rating/test_openings.py`
- Create: `tests/alphazero/rating/test_schedule.py`

**Interfaces:**
- Produces: `BenchmarkOpening`, `OpeningSuite.generate`, `OpeningSuite.reconstruct`, `SooRatedMatch`, `MinRatedMatch`, `schedule_soo_pair`, `schedule_min_triple`.

- [ ] **Step 1: Write failing opening tests** for included empty opening, deterministic fixed-seed generation, authoritative legal reconstruction, identical reconstructed state, player-count isolation, ruleset fingerprint rejection, and stable suite hash.
- [ ] **Step 2: Implement opening generation** using `AlphaZeroGameAdapter`, standard initial state, sorted legal action IDs, and a local `random.Random(seed)`; store only canonical action sequences and semantic identity.
- [ ] **Step 3: Run opening tests** and verify green.
- [ ] **Step 4: Write failing schedule tests** asserting exactly four Soo assignments per opening and 36 Min assignments per opening, deterministic ordering, distinct Min IDs, every physical-seat permutation, every turn-order permutation, and rejection of incomplete rated batches.
- [ ] **Step 5: Implement pure schedule generators and validators**; keep existing promotion arena unchanged at 4/18 games.
- [ ] **Step 6: Run rating/opening/schedule tests** and verify green.
- [ ] **Step 7: Commit** with `feat: add rated opening suites and balanced schedules`.

### Task 6: Central Inference Protocol and Model Pool

**Files:**
- Create: `src/diamond/alphazero/inference/__init__.py`
- Create: `src/diamond/alphazero/inference/protocol.py`
- Create: `src/diamond/alphazero/inference/model_pool.py`
- Modify: `src/diamond/alphazero/checkpoint.py`
- Create: `tests/alphazero/inference/test_protocol.py`
- Create: `tests/alphazero/inference/test_model_pool.py`

**Interfaces:**
- Produces: `ModelKey`, `InferenceRequest`, `InferenceResponse`, `InferenceFailure`, `InferenceModelPool.activate_checkpoint`, `evaluate`.

- [ ] **Step 1: Write failing serialization tests** for request/response correlation, malformed payload rejection, model-key identity, and EvalRequest round trips.
- [ ] **Step 2: Implement frozen transport envelopes** containing primitives/tuples only.
- [ ] **Step 3: Write failing model-pool tests** for strict compatibility before activation, two Soo keys, three Min keys, unknown-key errors, and no optimizer requirement.
- [ ] **Step 4: Add a read-only checkpoint loader** that validates the existing metadata gates and strict model state without weakening training checkpoint loading.
- [ ] **Step 5: Implement resident `TorchEvaluator` routing by immutable `ModelKey`** and verify different keys never share a forward call.
- [ ] **Step 6: Run inference protocol/model-pool and Milestone 1 checkpoint/evaluator tests**.
- [ ] **Step 7: Commit** with `feat: add compatibility-checked inference model pool`.

### Task 7: Bounded Central Batching and RemoteEvaluator

**Files:**
- Create: `src/diamond/alphazero/inference/coordinator.py`
- Create: `src/diamond/alphazero/inference/remote.py`
- Create: `tests/alphazero/inference/test_coordinator.py`
- Create: `tests/alphazero/inference/test_remote.py`

**Interfaces:**
- Produces: `InferenceConfig(max_batch_size, max_wait_ms, request_queue_capacity, response_timeout_s)`, `InferenceCoordinator.start/stop`, `RemoteEvaluator.evaluate`, `InferenceMetrics`.

- [ ] **Step 1: Write failing coordinator tests** for max-size flush, max-wait flush, batch size greater than one, no cross-key batches, bounded queue failure, malformed request response, worker error propagation, and graceful shutdown.
- [ ] **Step 2: Implement coordinator loop** using blocking queue gets with a monotonic deadline, per-key pending buckets, explicit stop sentinel, and captured batch/latency counters.
- [ ] **Step 3: Run coordinator tests** and verify green.
- [ ] **Step 4: Write failing RemoteEvaluator tests** for Evaluator protocol compatibility, concurrent clients, request correlation under reordered responses, timeout, coordinator failure, and local-vs-central FP32 parity.
- [ ] **Step 5: Implement `RemoteEvaluator`** as an EvalRequest envelope adapter with unique deterministic client/request IDs and no Torch imports.
- [ ] **Step 6: Add an architectural test** scanning MCTS imports through module loading and asserting no Torch/rating/orchestration dependency.
- [ ] **Step 7: Run all inference and MCTS tests** and verify green.
- [ ] **Step 8: Commit** with `feat: add centralized batched evaluator`.

### Task 8: Spawn-Safe Multiworker Self-Play

**Files:**
- Create: `src/diamond/alphazero/orchestration/__init__.py`
- Create: `src/diamond/alphazero/orchestration/selfplay_workers.py`
- Create: `tests/alphazero/orchestration/test_selfplay_workers.py`

**Interfaces:**
- Produces: `SelfPlayJob`, `EpisodeResult`, `derive_game_id`, `derive_game_seed`, `run_selfplay_job`, `SelfPlayWorkerPool`.

- [ ] **Step 1: Write failing identity tests** for stable game IDs and seeds derived from run/iteration/game index/checkpoint ID.
- [ ] **Step 2: Implement immutable jobs/results** that pin checkpoint ID and compatibility for an entire episode.
- [ ] **Step 3: Write a failing spawn integration test** using two workers and tiny near-terminal authoritative Soo/Min games; assert both complete, legal samples, correct Soo/Min targets, and stable IDs.
- [ ] **Step 4: Implement a top-level worker entry function** and explicit `multiprocessing.get_context("spawn")` lifecycle with bounded join/terminate cleanup.
- [ ] **Step 5: Add failing tests for abort/no-samples, surfaced worker exception, retry identity, and no hanging children**, then implement explicit error envelopes and shutdown.
- [ ] **Step 6: Run worker tests twice** to catch hidden global-state or cleanup dependence.
- [ ] **Step 7: Commit** with `feat: add spawn-safe self-play workers`.

### Task 9: Idempotent Persistent Replay

**Files:**
- Create: `src/diamond/alphazero/orchestration/replay_store.py`
- Create: `tests/alphazero/orchestration/test_replay_store.py`

**Interfaces:**
- Produces: `ReplayManifest`, `PersistentReplayStore.ingest_episode`, `load_buffer`, `sample`, `compact`.

- [ ] **Step 1: Write failing tests** for completed ingestion, duplicate game ID no-op, conflicting duplicate rejection, abort metrics without samples, Soo/Min isolation, compatibility rejection, and manifest atomicity.
- [ ] **Step 2: Implement one immutable JSON replay chunk per completed game** with sparse samples, content hash, atomic rename, and an atomic manifest containing ordered game IDs and deterministic RNG state.
- [ ] **Step 3: Write failing persistence tests** for bounded-window reload, deterministic sampling after restart, missing/corrupt chunk failure, and crash-before-manifest behavior.
- [ ] **Step 4: Implement replay reconstruction and safe orphan handling**; never load pickle from runtime data.
- [ ] **Step 5: Run replay store and Milestone 1 replay tests**.
- [ ] **Step 6: Commit** with `feat: add idempotent persistent replay`.

### Task 10: Atomic Training Run State

**Files:**
- Create: `src/diamond/alphazero/orchestration/run_state.py`
- Modify: `.gitignore`
- Create: `tests/alphazero/orchestration/test_run_state.py`

**Interfaces:**
- Produces: `RunStage` enum, `TrainingRunState`, `RunStateStore.initialize/load/save/transition`.

- [ ] **Step 1: Write failing tests** for exact stage transitions, immutable run/model identity, champion/candidate pointers, iteration/training step, replay manifest, completed game IDs, promotion/rating records, protocol IDs, and deterministic seed derivation.
- [ ] **Step 2: Implement a validated state machine** for `INITIALIZE`, `SELF_PLAY`, `REPLAY_INGEST`, `TRAIN`, `SAVE_CANDIDATE`, `PROMOTION_ARENA`, `RATING_BENCHMARK`, `PROMOTE_OR_REJECT`, `PERSIST`, and `COMPLETE`.
- [ ] **Step 3: Write failing atomicity tests** for stale write generation, invalid transition, truncated temporary file, and candidate preservation across restart.
- [ ] **Step 4: Implement versioned JSON with generation compare-and-swap and atomic replace**.
- [ ] **Step 5: Ignore `/runs/` and run state tests**.
- [ ] **Step 6: Commit** with `feat: add atomic AlphaZero training run state`.

### Task 11: Iterative Training Coordinator and Resume

**Files:**
- Create: `src/diamond/alphazero/orchestration/coordinator.py`
- Create: `tests/alphazero/orchestration/test_training_coordinator.py`
- Create: `tests/alphazero/orchestration/test_resume.py`

**Interfaces:**
- Produces: focused `WorkerConfig`, `TrainingLoopConfig`, `PersistenceConfig`, `TrainingCoordinator.run_iteration/resume`.

- [ ] **Step 1: Write failing stage-order test** using real tiny components and recording durable state after each stage.
- [ ] **Step 2: Implement one-iteration coordinator methods** rather than an implicit infinite loop; each method first checks whether its output artifact/event already exists.
- [ ] **Step 3: Write interruption tests** after self-play, replay save, training, candidate checkpoint, arena, and rating update.
- [ ] **Step 4: Implement resume guards** so completed work is reused, checkpoint hashes cannot be overwritten, game IDs are not duplicated, events are not double-rated, and champion/candidate state survives.
- [ ] **Step 5: Add compatibility/protocol mismatch tests** for Soo/Min crossover and changed benchmark namespace.
- [ ] **Step 6: Run orchestration, checkpoint, arena, replay, and rating tests**.
- [ ] **Step 7: Commit** with `feat: add resumable production training coordinator`.

### Task 12: Headless CLI and Complete CPU Smoke

**Files:**
- Create: `src/diamond/alphazero/orchestration/cli.py`
- Create: `src/diamond/alphazero/milestone2_smoke.py`
- Modify: `pyproject.toml`
- Create: `tests/alphazero/orchestration/test_cli.py`
- Create: `tests/alphazero/test_milestone2_smoke.py`

**Interfaces:**
- Produces CLI commands `train`, `resume`, `benchmark`, `leaderboard`, `profile`; module command `python -m diamond.alphazero.milestone2_smoke`.

- [ ] **Step 1: Write failing CLI tests** invoking argument parsing without PySide imports and asserting machine-readable exit codes/output for each command.
- [ ] **Step 2: Implement argparse command dispatch** through orchestration/rating services only.
- [ ] **Step 3: Write failing end-to-end smoke tests** requiring real Soo and Min multiworker self-play, replay ingest, one training update, candidate save, promotion arena, Elo/TrueSkill event where eligible, persistence, and resume.
- [ ] **Step 4: Implement tiny near-terminal authoritative fixtures** as configuration of real production stages; report insufficient Min history instead of fabricating duplicate participants.
- [ ] **Step 5: Run the smoke module and assert exit code zero plus persisted reload equivalence**.
- [ ] **Step 6: Commit** with `feat: add headless Milestone 2 training smoke`.

### Task 13: FP32 Profiling and Controlled Numeric Modes

**Files:**
- Create: `src/diamond/alphazero/inference/profile.py`
- Modify: `src/diamond/alphazero/inference/model_pool.py`
- Create: `tests/alphazero/inference/test_profile.py`
- Create: `tests/alphazero/inference/test_numeric_modes.py`

**Interfaces:**
- Produces: `ProfileReport`, eager `fp32` default, optional `bf16` and `compiled-*` benchmark modes.

- [ ] **Step 1: Write failing CPU profile tests** for evaluated states/sec, calls/sec, mean/p50/p95 batch/latency, stage timing, finite values, and JSON serialization.
- [ ] **Step 2: Implement monotonic-timer instrumentation** around queue wait, inference, self-play, replay collation, and training steps.
- [ ] **Step 3: Detect CUDA availability and hardware name**; if unavailable, emit `gpu_verified=false` and omit fabricated GPU rows.
- [ ] **Step 4: On A30/CUDA only, run and record eager FP32 baseline** before adding BF16.
- [ ] **Step 5: Write BF16 agreement tests** with the same checkpoint/request set, finite outputs, legal-policy normalization, and explicit tolerance; implement configurable autocast without changing FP32 default.
- [ ] **Step 6: Benchmark `torch.compile` only after eager modes** and retain compiled mode only if startup/steady-state/memory measurements complete successfully.
- [ ] **Step 7: Commit measured code/results** with `perf: add measured AlphaZero inference modes`.

### Task 14: Documentation, Regression, Review, and C++ Recommendation

**Files:**
- Modify: `docs/alphazero.md`
- Modify: `blueprint/milestone2.md`
- Modify: `README.md` only if a headless command needs discovery.

**Interfaces:**
- Consumes all completed Milestone 2 components and measured profile reports.
- Produces operator runbook and evidence-based final status.

- [ ] **Step 1: Document** Soo Elo, Min TrueSkill/tau rationale, distinct triple rule, 18-game promotion versus 36-game rating, benchmark namespaces/openings, artifact IDs, start/resume/leaderboard/profile commands, throughput interpretation, and environment split.
- [ ] **Step 2: Run `python -m pytest tests/alphazero -o addopts= -q`** in the training environment and record the exact count.
- [ ] **Step 3: Run `python -m pytest tests --ignore=tests/alphazero -o addopts= -q`** in the Qt-working environment and record the exact count/skips.
- [ ] **Step 4: Run both Milestone 1 and Milestone 2 smoke modules** and record exact JSON/exit codes.
- [ ] **Step 5: Run GPU integration/profile commands only when CUDA is available**; otherwise record them as not executed.
- [ ] **Step 6: Run `compileall`, `git diff --check`, and the MCTS dependency guard**.
- [ ] **Step 7: Request independent code review**, resolve every Critical/Important finding with TDD, and rerun the full verification commands.
- [ ] **Step 8: Base the C++ recommendation only on measured stage percentages**; do not implement C++.
- [ ] **Step 9: Commit** with `docs: complete AlphaZero Milestone 2 runbook`.

## Plan Self-Review

- Spec coverage: tasks map to phases 1–13, including rating, schedules/openings, central inference, workers, replay/run persistence, coordinator, CLI/smoke, profiling, and documentation.
- Deferred by design: C++ implementation, OpenVINO, Iris Xe deployment, GUI routing, and multi-node training.
- Type consistency: participant IDs feed events; protocol IDs gate events/registries; model keys pin inference and episodes; game IDs gate replay; durable stage outputs gate resume.
- No optimization phase precedes the tested eager FP32 production path.
- No runtime source of truth depends on mutable rating caches or monolithic pickle files.
