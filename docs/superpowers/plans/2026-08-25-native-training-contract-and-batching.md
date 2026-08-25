# Native Training Contract and Batching Implementation Plan

> **For Codex:** Execute with test-driven-development. Root owns environment, integration, acceptance, and commits; do not create a worktree or dispatch a review loop.

**Goal:** Land config schema v2 and make the existing CPU pipeline honor minibatch, optimizer-step, lane/thread, microsecond-wait, and game-deadline contracts without changing the scientific workload.

**Architecture:** Keep configuration Torch-free. Convert v2 into existing native `EpisodeConfig` and `IterationRequest` values at the CLI boundary. Until the shared resolver lands in PR B, CPU execution is supported and CUDA requests fail before run-directory mutation.

**Tech stack:** C++20, LibTorch CPU, MSVC/CMake/CTest, repository JSON support.

**Spec:** `docs/superpowers/specs/2026-08-25-native-gpu-training.md`

**Baseline:** `6ac0a1c7cc0edc84a418ab56e14c0d1a252400fd`

## Global constraints

- Preserve Soo 128×6 FP32, production 400 simulations, bootstrap 128 simulations, temperatures/noise/move limits, and A0 prior semantics.
- Use `tools/native_training.sh` for every CMake/CTest invocation.
- Build the exact target before its nearest test. Do not run the full suite after every task.
- Modify no Qt/QML/model-catalog file from PR #56.
- Stage only the files named by the task and make the listed atomic commit after acceptance.

## Task 1: Freeze config-v2 failure contracts

**Files:**

- Modify: `native/tests/config_test.cpp`
- Modify: `native/CMakeLists.txt` only if a separate fixture executable is necessary

**Steps:**

- [ ] Add a v1 fixture that expects the complete actionable migration error from the specification.
- [ ] Add table cases for `cpu`, `cuda`, `cuda:0`, invalid casing, empty index, negative index, trailing text, unsupported precision, unknown/missing keys, invalid budgets, and invalid opening statistics.
- [ ] Add a round-trip v2 object with lanes different from threads and `max_wait_us = 50`.
- [ ] Add exact-default assertions for all new structs.
- [ ] Run the test and record that it fails for missing schema-v2 behavior.

```bash
tools/native_training.sh cmake --build --preset native-training --target config_test --parallel
tools/native_training.sh ctest --preset native-training -R '^config_test$' --output-on-failure
```

## Task 2: Implement exact schema v2 and migrate tracked configs

**Files:**

- Modify: `native/include/diamond_orchestration/config.hpp`
- Modify: `native/src/config.cpp`
- Modify: `configs/alphazero/soo-production.json`
- Modify: `configs/alphazero/soo-bootstrap.json`
- Modify: `configs/alphazero/min-production.json`
- Modify: `configs/alphazero/min-bootstrap.json`

**Steps:**

- [ ] Add `RuntimeConfig`, `RunBudgetConfig`, `OpeningSuiteConfig`, and `PromotionStatisticsConfig` exactly as specified.
- [ ] Move device to `runtime`, accept only the exact device grammar, and accept only `fp32`.
- [ ] Add `train_steps_per_iteration`, `logical_lanes`, `search_threads`, and `max_wait_us`; remove accepted v1 names.
- [ ] Make parser key sets exact and serialization canonical/deterministic.
- [ ] Reject schema v1 before parsing nested v2-only fields so the migration error is stable.
- [ ] Migrate all four JSON files with baseline-preserving values from the specification.
- [ ] Build and run `config_test` once; inspect `git diff --check`.

```bash
tools/native_training.sh cmake --build --preset native-training --target config_test --parallel
tools/native_training.sh ctest --preset native-training -R '^config_test$' --output-on-failure
git diff --check -- native/include/diamond_orchestration/config.hpp native/src/config.cpp configs/alphazero
```

**Commit:** `feat(training): define production config v2`

## Task 3: Make pipeline batch and step counts observable and honored

**Files:**

- Modify: `native/include/diamond_pipeline/pipeline.hpp`
- Modify: `native/src/pipeline.cpp`
- Modify: `native/src/train_main.cpp`
- Modify: `native/tests/native_pipeline_smoke_test.cpp`
- Modify: `native/tests/cli_contract_test.cpp`

**Steps:**

- [ ] Add failing pipeline cases that seed enough replay, request batch `N > 1` and steps `M > 1`, and assert the actual batch/step metrics.
- [ ] Add a failing insufficient-replay case that proves the trainer and optimizer step remain unchanged.
- [ ] Extend `IterationRequest` with explicit `training_batch_size`; extend `IterationResult` with requested/completed steps, actual batch sizes, replay size, and abort counts by reason.
- [ ] Replace `replay.sample(1)` with `replay.sample(training_batch_size)` and preflight replay availability.
- [ ] Wire config batch size and train steps in `train_main.cpp`; remove the hard-coded one-step assignment.
- [ ] Add a CLI contract proving a CPU-only build rejects a CUDA request before creating the requested run directory.
- [ ] Build and run only the pipeline and CLI contract tests.

```bash
tools/native_training.sh cmake --build --preset native-training --target native_pipeline_smoke_test cli_contract_test --parallel
tools/native_training.sh ctest --preset native-training -R '^(native_pipeline_smoke_test|cli_contract_test)$' --output-on-failure
```

**Commit:** `feat(training): honor minibatch and iteration step counts`

## Task 4: Separate lanes, threads, and microsecond waits at the command boundary

**Files:**

- Modify: `native/src/train_main.cpp`
- Modify: `native/tests/cli_contract_test.cpp`
- Modify: `native/tests/selfplay_test.cpp` only for an observable scheduler contract

**Steps:**

- [ ] Add a failing CLI or extracted-wire test with lanes `7`, threads `2`, and wait `50` µs.
- [ ] Map each field independently to `soo::EpisodeConfig`; delete the millisecond multiplication.
- [ ] Ensure `games_per_iteration` controls job count rather than concurrent lane count.
- [ ] Prove queued games continue when games exceed lanes.
- [ ] Run the nearest two tests once.

```bash
tools/native_training.sh cmake --build --preset native-training --target cli_contract_test selfplay_test --parallel
tools/native_training.sh ctest --preset native-training -R '^(cli_contract_test|selfplay_test)$' --output-on-failure
```

**Commit:** `feat(selfplay): separate lanes threads and wait units`

## Task 5: Implement monotonic per-game deadlines and zero-sample aborts

**Files:**

- Modify: `native/include/soo/selfplay.hpp`
- Modify: `native/src/selfplay.cpp`
- Modify: `native/src/train_main.cpp`
- Modify: `native/src/pipeline.cpp`
- Modify: `native/tests/selfplay_test.cpp`
- Modify: `native/tests/native_pipeline_smoke_test.cpp`

**Steps:**

- [ ] Add an internal deadline contract that uses `steady_clock`; zero/absent means disabled.
- [ ] Add a deterministic failing timeout test and assert it is not classified as move-limit or interruption.
- [ ] Add a pipeline case proving a timed-out episode records reason `max_game_seconds` and contributes zero samples.
- [ ] Wire optional config seconds into the internal deadline without wall-clock timestamps.
- [ ] Preserve move-limit and cancellation behavior.
- [ ] Build and run the two nearest tests once.

```bash
tools/native_training.sh cmake --build --preset native-training --target selfplay_test native_pipeline_smoke_test --parallel
tools/native_training.sh ctest --preset native-training -R '^(selfplay_test|native_pipeline_smoke_test)$' --output-on-failure
```

**Commit:** `feat(selfplay): enforce per-game deadlines`

## Task 6: Collate dense training targets once on the host

**Files:**

- Modify: `native/src/trainer.cpp`
- Modify: `native/tests/training_step_parity_test.cpp`

**Steps:**

- [ ] Extend the frozen CPU test to compare total/policy/value loss, selected gradients, and selected post-AdamW parameters for multi-sample Soo and Min batches.
- [ ] Add validation cases for duplicate/out-of-range sparse action IDs and non-finite targets.
- [ ] Replace scalar Torch `index_put_` target construction with contiguous host feature/policy/value buffers and one tensor creation/copy per buffer.
- [ ] Do not change the objective, normalization, learning rate, weight decay, or CPU tolerances.
- [ ] Build and run the parity target once.

```bash
tools/native_training.sh cmake --build --preset native-training --target training_step_parity_test --parallel
tools/native_training.sh ctest --preset native-training -R '^training_step_parity_test$' --output-on-failure
```

**Commit:** `perf(training): collate dense targets contiguously`

## Task 7: PR-A acceptance and handoff

**Files:**

- Update: `docs/superpowers/plans/2026-08-25-native-training-contract-and-batching.md` checkboxes/results only if the project convention records execution
- Add: one focused baseline JSON only if an existing benchmark executable owns its schema

**Steps:**

- [ ] Reconfigure and build `native-training` through the wrapper.
- [ ] Run the full CPU `native-training` preset once and report exact pass/fail/skip counts, preserving any diagnosed pre-existing Windows cleanup failure.
- [ ] Run `native-ci` once at the PR boundary.
- [ ] Verify no tracked Python, Qt/QML/catalog, build output, checkpoint, or log entered the diff.
- [ ] Record the exact DLL dependency/PATH evidence for the built LibTorch executables.

```bash
tools/native_training.sh cmake --preset native-training
tools/native_training.sh cmake --build --preset native-training --parallel
tools/native_training.sh ctest --preset native-training --output-on-failure
tools/native_training.sh cmake --build --preset native-ci --parallel
tools/native_training.sh ctest --preset native-ci --output-on-failure
git diff --check
```

**Final commit if documentation/evidence changed:** `docs(training): record native contract acceptance`
