# Min Vacancy-Prior Realizability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-state-safe offline diagnostic that tests whether Min can directly fit the canonical vacancy prior with either the policy head alone or the trunk plus policy, and emits reproducible held-out metrics and disposable checkpoints for no-prior probes.

**Architecture:** Reconstruct the vacancy teacher from replay feature channel 0 and the authoritative legal actions already stored in `sparse_policy`. A dedicated distillation module owns teacher construction, legal KL evaluation, parameter freezing, and policy-only optimization; a thin CLI owns checkpoint/replay/config loading, provenance gates, JSON reporting, and optional diagnostic checkpoint output. Existing self-play remains unchanged and evaluates the disposable output through the existing Min probe.

**Tech Stack:** C++20, LibTorch, CMake/Ninja/CTest, canonical JSON, existing persistent replay and checkpoint-v3 APIs.

**Spec:** `docs/superpowers/specs/2026-09-02-min-a0-transition-diagnosis.md`

## Global Constraints

- Never mutate or resume the durable production run; checkpoint and replay inputs are read-only.
- Arm A trains only `policy_source` and `policy_destination`; Arm B freezes only `value_linear1` and `value_linear2`.
- The objective is legal-set `KL(P_vacancy || P_network)`; zero-visit legal actions remain legal.
- Reports gate clean build provenance, config digest, checkpoint model digest, seed, device, and exact experiment parameters.
- The committed operating-point config explicitly uses `canonical-target-vacancy-distance-v2`, prior weight `0.5`, and legal policy loss.
- Parallel MCTS and production curriculum changes remain out of scope.

---

### Task 1: Reproducible operating-point config

**Files:**
- Create: `configs/alphazero/min-anneal-alpha050-v1.json`
- Modify: `native/tests/config_test.cpp`

**Interfaces:**
- Consumes: `configs/alphazero/min-production-6h.json`
- Produces: a versioned config accepted by `ProductionConfig::from_json`

- [ ] **Step 1: Add a failing config fixture assertion**

Load `min-anneal-alpha050-v1.json` in `config_test.cpp` and assert model `Min`, `self_play.bootstrap_prior == "canonical-target-vacancy-distance-v2"`, `bootstrap_prior_weight == 0.5`, and `training.policy_loss_domain == "legal"`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cmake --build build/native-training --target config_test -j2 && ctest --test-dir build/native-training -R '^config_test$' --output-on-failure`

Expected: failure because the versioned config does not exist.

- [ ] **Step 3: Create the config**

Copy the six-hour production config and change only the bootstrap prior fields and policy-loss domain required by the spec.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2; expected PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/alphazero/min-anneal-alpha050-v1.json native/tests/config_test.cpp
git commit -m "config(min): pin alpha 0.50 legal-loss operating point"
```

### Task 2: Vacancy teacher reconstruction

**Files:**
- Create: `native/include/diamond_pipeline/vacancy_distillation.hpp`
- Create: `native/src/vacancy_distillation.cpp`
- Create: `native/tests/vacancy_distillation_test.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: `diamond_training::TrainingSample`
- Produces: `VacancyTarget vacancy_target(const TrainingSample&)`, containing ordered legal actions, teacher probabilities, and per-action potential reductions

- [ ] **Step 1: Write failing tests**

Construct a six-channel Min sample whose channel 0 contains ten canonical pieces and whose sparse policy contains legal actions with zero probabilities. Assert every sparse action is retained, teacher probabilities sum to one, ordering is preserved, and malformed feature widths, empty policies, duplicate/out-of-range actions, or non-binary channel-0 values throw.

- [ ] **Step 2: Register and run the test to verify RED**

Run: `cmake --build build/native-training --target vacancy_distillation_test -j2`

Expected: compile failure because `vacancy_distillation.hpp` does not exist.

- [ ] **Step 3: Implement teacher reconstruction**

Read channel 0 from the interleaved `[73,6]` feature array into `soo::PieceSet`, collect all `sparse_policy` action ids, call `soo::vacancy_prior`, and compute `Phi(before)-Phi(after)` using `decode_action` and `vacancy_potential`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `ctest --test-dir build/native-training -R '^vacancy_distillation_test$' --output-on-failure`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native/include/diamond_pipeline/vacancy_distillation.hpp native/src/vacancy_distillation.cpp native/tests/vacancy_distillation_test.cpp native/CMakeLists.txt
git commit -m "feat(training): reconstruct vacancy teacher from replay"
```

### Task 3: Freeze-aware distillation and held-out metrics

**Files:**
- Modify: `native/include/diamond_pipeline/vacancy_distillation.hpp`
- Modify: `native/src/vacancy_distillation.cpp`
- Modify: `native/tests/vacancy_distillation_test.cpp`

**Interfaces:**
- Produces: `enum class DistillationArm { policy_head, trunk_policy }`, `run_vacancy_distillation(Trainer&, train_samples, held_out_samples, config)`, and metrics for teacher KL, expected progress ratio, top-k teacher mass, top-1 agreement, and before/after policy KL

- [ ] **Step 1: Write failing freeze and learning tests**

For Arm A snapshot all named parameters, run two steps, and assert only the two policy modules change. For Arm B assert value-head parameters stay byte-identical while at least one trunk and one policy parameter changes. Assert held-out KL is finite and decreases on a repeated synthetic sample.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: link failure for the new API.

- [ ] **Step 3: Implement minimal optimizer and metrics**

Set `requires_grad` according to the arm and reuse the checkpoint-loaded `Trainer` optimizer so model and optimizer remain a consistent checkpoint-v3 pair. Collate dense legal masks and dense teacher targets, compute masked legal log-softmax and teacher cross-entropy, backpropagate only the distillation loss, step the trainer optimizer, and evaluate in batches without gradients. Reject zero steps/batches, non-Min compatibility, non-finite tensors, and empty held-out data.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused CTest twice to ensure deterministic CPU behavior; expected PASS both times.

- [ ] **Step 5: Commit**

```bash
git add native/include/diamond_pipeline/vacancy_distillation.hpp native/src/vacancy_distillation.cpp native/tests/vacancy_distillation_test.cpp
git commit -m "feat(training): add vacancy realizability distillation"
```

### Task 4: Offline diagnostic CLI and provenance gates

**Files:**
- Create: `native/benchmarks/min_vacancy_realizability.cpp`
- Create: `native/tests/min_vacancy_realizability_cli_test.cmake`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: `--checkpoint`, `--config`, `--replay`, `--arm head|full`, `--device`, `--steps`, `--batch-size`, `--eval-samples`, `--eval-batch`, `--seed`, `--expected-source-commit`, `--expected-config-sha256`, `--expected-model-sha256`, `--out`, optional `--checkpoint-out`
- Produces: schema-v1 canonical JSON and an optional disposable checkpoint-v3 tree

- [ ] **Step 1: Write failing CLI contract tests**

Assert `--help` lists every required flag, unknown arms fail, and source/config/model digest mismatches exit non-zero before training.

- [ ] **Step 2: Register and run tests to verify RED**

Expected: failure because the executable is absent.

- [ ] **Step 3: Implement CLI**

Follow `min_learning_diagnostic.cpp` for config, checkpoint, replay, device, canonical JSON, and build provenance. Sample disjoint deterministic train/eval sets, run the requested arm, serialize initial/final metrics and parameter-group drift, and save only to the explicitly provided diagnostic output path.

- [ ] **Step 4: Run focused CLI tests and verify GREEN**

Run: `ctest --test-dir build/native-training -R 'min_vacancy_realizability' --output-on-failure`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native/benchmarks/min_vacancy_realizability.cpp native/tests/min_vacancy_realizability_cli_test.cmake native/CMakeLists.txt
git commit -m "feat(training): add Min vacancy realizability CLI"
```

### Task 5: Format, regression verification, and experiment handoff

**Files:**
- Modify: only C++ files reported by the repository formatting check
- Create: `docs/superpowers/reports/2026-09-02-min-vacancy-realizability-handoff.md`

**Interfaces:**
- Produces: a clean native-format result, passing focused/regression tests, and exact commands for head/full arms plus existing no-prior 256-game probe

- [ ] **Step 1: Run clang-format-18 check and apply only reported mechanical changes**

Use the same command/target as CI run #202; inspect the diff to ensure no semantic edits.

- [ ] **Step 2: Run focused and regression verification**

Run config, vacancy distillation, CLI, training parity, replay schema/store, Min learning diagnostic, and native-format tests. Run `git diff --check` and confirm the worktree is clean except intended files.

- [ ] **Step 3: Write the handoff report**

Record the exact clean-build commands, digest derivation commands, head/full CLI invocations, disposable checkpoint paths, and existing `alphadiamond-min-probe` invocation for 256 no-prior games. State that the production run is untouched and no experimental result is claimed until artifacts are supplied and both arms run.

- [ ] **Step 4: Commit**

```bash
git add native docs/superpowers/reports/2026-09-02-min-vacancy-realizability-handoff.md
git commit -m "chore(native): verify vacancy realizability diagnostic"
```
