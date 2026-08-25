# Durable Iteration Loop and Replay Transactions Implementation Plan

> **For Codex:** Execute after device/model ownership is stable. Root owns every crash-recovery integration failure directly.

**Goal:** Make replay selection, checkpoints, stage results, initialization lineage, and multi-iteration progress one recoverable protocol.

**Architecture:** Immutable operation-result artifacts precede authoritative state transitions. Replay RNG selection and candidate progress commit together. Resume reuses valid results and rejects conflicts.

**Spec:** `docs/superpowers/specs/2026-08-25-native-gpu-training.md`

## Task 1: Replace full-pool sampling with bounded selection

**Files:** `native/include/diamond_pipeline/replay_store.hpp`, `native/src/replay_store.cpp`, `native/tests/replay_store_test.cpp`, `native/benchmarks/replay_benchmark.cpp`

- [ ] Add an allocation/copy observation seam and a failing test proving current sampling scales with pool size.
- [ ] Implement O(batch) without-replacement descriptor selection with deterministic IDs.
- [ ] Separate selection from authoritative RNG-state commit.
- [ ] Benchmark pool sizes while holding batch fixed; emit JSON.
- [ ] Harden Windows test cleanup with scoped store lifetimes and diagnostic `error_code` cleanup without weakening replay assertions.

**Acceptance:** exact `replay_store_test` target and `replay_benchmark` explicit run.

**Commit:** `perf(replay): bound minibatch selection memory`

## Task 2: Add replay sampling and ingest transactions

**Files:** `native/include/diamond_pipeline/replay_store.hpp`, `native/src/replay_store.cpp`, `native/tests/replay_store_test.cpp`, `native/tests/replay_schema_test.cpp`

- [ ] Define selection transaction IDs, before/after RNG state, selected IDs, and committed/aborted states.
- [ ] Recover a crash before and after selection activation without changing the next committed batch.
- [ ] Ingest an iteration in one transaction and report accepted/duplicate games and samples.
- [ ] Activate a validated manifest before deleting unreachable chunks; inject failures at each boundary.
- [ ] Never delete a chunk referenced by the active manifest.

**Acceptance:** replay schema/store tests.

**Commit:** `feat(replay): make sampling and ingest transactional`

## Task 3: Introduce checkpoint manifest v3 and initialization lineage

**Files:** `native/include/diamond_training/checkpoint.hpp`, `native/src/checkpoint.cpp`, `native/src/checkpoint_main.cpp`, checkpoint tests, compact manifest fixtures

- [ ] Add exact manifest-v3 parsing/serialization and digest verification.
- [ ] Model `scratch`, `warm_start`, `resume`, and gated `audited_legacy_import` as distinct types.
- [ ] Record canonical config/SHA, run/iteration/step, replay digest, RNG protocol, operation IDs, device/precision, environment, and source lineage.
- [ ] Prove warm start creates a new run/native step zero and `optimizer_reset=true`.
- [ ] Keep legacy step-44,250 immutable and weight-only unless optimizer decode/parity is actually demonstrated.

**Acceptance:** checkpoint roundtrip/transaction/reject tests plus CLI manifest fixture.

**Commit:** `feat(checkpoint): record durable initialization lineage`

## Task 4: Give every stage a durable idempotent operation result

**Files:** `native/include/diamond_orchestration/run_state.hpp`, `native/src/run_state.cpp`, `native/include/diamond_orchestration/coordinator.hpp`, `native/src/coordinator.cpp`, `native/tests/run_state_test.cpp`, `native/tests/coordinator_resume_test.cpp`

- [ ] Define the nine authoritative stages and remove nominal-only transitions.
- [ ] Derive stable operation IDs from run/iteration/stage/protocol/retry identity.
- [ ] Write checksummed per-stage results and validate/reuse a matching result on re-entry.
- [ ] Reject a conflicting artifact rather than overwrite or rerun it.
- [ ] Inject interruption before/after result activation and before/after state transition for every stage.

**Acceptance:** run-state and coordinator-resume tests.

**Commit:** `feat(orchestration): persist idempotent stage results`

## Task 5: Split the monolithic iteration into real stage executors

**Files:** `native/include/diamond_pipeline/pipeline.hpp`, `native/src/pipeline.cpp`, orchestration headers/sources, `native/src/train_main.cpp`, `native/tests/native_pipeline_smoke_test.cpp`, `native/tests/final_pipeline_smoke_test.cpp`

- [ ] Move self-play, ingest, training, candidate save, arena/rating, decision, and persist work behind their owning stage interfaces.
- [ ] Keep self-play and learner optimization sequential.
- [ ] Pass immutable artifact references between stages rather than live mutable aliases.
- [ ] Verify exact work accounting and no duplicate replay/checkpoint work after resume.

**Acceptance:** native and final pipeline smoke tests.

**Commit:** `refactor(orchestration): assign work to durable stages`

## Task 6: Implement bounded multi-iteration train/resume semantics

**Files:** `native/src/train_main.cpp`, orchestration sources, `native/tests/coordinator_resume_test.cpp`, `native/tests/cli_contract_test.cpp`, `native/benchmarks/end_to_end_benchmark.cpp`

- [ ] Implement exact `train`, `resume`, `evaluate`, and `report` contracts from the spec.
- [ ] Snapshot resolved canonical config/SHA during initialization and reject incompatible mutable inputs on resume.
- [ ] At PERSIST increment iteration and continue until budget/stop; add two-iteration seed/progress tests.
- [ ] Derive seeds with run/model, iteration, stage, game, retry, and opening identity.
- [ ] Add forced interruption/resume to the end-to-end benchmark with exact work accounting.

**Acceptance:** coordinator, CLI, and end-to-end smoke/benchmark evidence.

**Commit:** `feat(orchestration): run and resume bounded iterations`

## Task 7: Durability boundary verification

- [ ] Run every interruption injection once and record counts.
- [ ] Run the full CPU `native-training` preset once.
- [ ] On CUDA, repeat the two-iteration interruption/resume smoke without changing workload knobs.
