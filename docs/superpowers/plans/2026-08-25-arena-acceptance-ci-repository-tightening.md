# Arena, Acceptance, CI, and Repository Tightening Implementation Plan

> **For Codex:** Execute after candidate/checkpoint durability exists. Hardware acceptance is a separate evidence-producing step and never promotes/uploads its candidate.

**Goal:** Replace deterministic arena pseudo-replicates with materialized balanced opening blocks, strengthen promotion evidence, expose comparable benchmarks/metrics, and make CPU/GPU repository gates honest.

**Spec:** `docs/superpowers/specs/2026-08-25-native-gpu-training.md`

## Task 1: Materialize versioned opening suites

**Files:** arena/schedule headers and sources, config integration, `native/tests/arena_schedule_test.cpp`, compact opening fixtures

- [ ] Define suite/opening IDs, serialized state/digest, version, seed, count, and depth.
- [ ] Deterministically generate/load the same suite and reject digest/version mismatches.
- [ ] Wire configured suite fields into production evaluation.
- [ ] Keep fixture size compact and manifest every payload.

**Acceptance:** arena schedule test.

**Commit:** `feat(arena): materialize versioned opening suites`

## Task 2: Use complete Soo pairs and Min triples without pseudo-replicates

**Files:** `native/src/arena.cpp`, `native/src/schedule.cpp`, their headers, `native/src/train_main.cpp`, `native/tests/arena_schedule_test.cpp`

- [ ] Route Soo through `schedule_soo_pair()` and Min through `schedule_min_triple()`.
- [ ] Include suite/opening/cell identity in every unique match ID.
- [ ] Reject game counts that are not complete-block multiples.
- [ ] Prove increasing `arena.games` cannot cycle the same deterministic cells.
- [ ] Report completed/aborted matches and complete/incomplete blocks separately.

**Acceptance:** arena schedule and CLI contract tests.

**Commit:** `feat(arena): evaluate unique balanced opening blocks`

## Task 3: Bootstrap statistics by opening block and harden promotion

**Files:** arena/rating/promotion headers and sources, `native/tests/promotion_test.cpp`, rating tests

- [ ] Implement the exact configured opening-block bootstrap protocol and deterministic statistics seed.
- [ ] Preserve 0.55 threshold and report interval, health regressions, and decision reason.
- [ ] Exclude incomplete blocks from strength observations while reporting them as health failures.
- [ ] Require durable candidate/checkpoint/arena/rating/decision artifacts before champion activation.
- [ ] Inject incomplete/corrupt artifact failures and prove champion identity is unchanged.

**Acceptance:** promotion and rating tests.

**Commit:** `feat(promotion): require block-aware durable evidence`

## Task 4: Complete canonical metrics and native benchmarks

**Files:** report/metrics sources, existing five native benchmark executables, benchmark contract tests/docs

- [ ] Emit append-only JSONL and checksummed stage results with all required fields from the spec.
- [ ] Extend training/self-play/end-to-end benchmarks with real ModelPool/device/artifact/workload options.
- [ ] Pin before/after commits and identical workload manifests; emit median/range and completed throughput.
- [ ] Report incompatible or slower GPU results without changing the workload.

**Acceptance:** benchmark schema tests plus one CPU benchmark manifest.

**Commit:** `feat(metrics): publish comparable training evidence`

## Task 5: Add honest CPU and NVIDIA CI lanes

**Files:** `.github/workflows/ci.yml`, `CMakePresets.json`, `native/CMakeLists.txt`, branch-protection migration notes

- [ ] Add a required CPU LibTorch `native-training` configure/build/CTest job.
- [ ] Label real CUDA tests `cuda` and exclude them only from CPU presets.
- [ ] Add opt-in/self-hosted NVIDIA workflow that validates labels/device, prints environment, runs CUDA tests and bounded acceptance, and uploads JSON.
- [ ] Fail rather than mark a skipped/no-device run as GPU acceptance.
- [ ] Retain compatibility aliases until new contexts have passed and protection is explicitly migrated.

**Acceptance:** workflow syntax plus one same-commit CPU CI run; GPU remains pending until an NVIDIA runner produces artifacts.

**Commit:** `ci(training): add native CPU and NVIDIA gates`

## Task 6: Pin changed-file formatting and update current documentation

**Files:** formatting config/workflow, README, native GPU runbook, config/checkpoint/resume/benchmark/promotion references, `TrainAlphaDiamond/README.md`

- [ ] Add pinned changed-file `clang-format` while retaining `git diff --check`.
- [ ] Do not reformat untouched repository files.
- [ ] Replace stale current Python-era commands with exact native commands.
- [ ] Cross-reference historical reports instead of rewriting them.
- [ ] Document DLL/PATH/MSVC activation and no-silent-fallback behavior.
- [ ] Document exact branch-protection migration without applying it automatically.

**Acceptance:** docs command scan, format check, Python-zero gate.

**Commit:** `docs(training): publish native GPU operations`

## Task 7: CUDA-host acceptance checklist

Run on a clean checkout of the exact candidate commit with CUDA-enabled LibTorch and a real NVIDIA GPU:

- [ ] Record `git rev-parse HEAD`, `git status --short`, driver, GPU, CUDA runtime, LibTorch version, CPU/threads, and memory.
- [ ] Put MSVC developer environment plus conda `Library\bin`, `Scripts`, and env root on PATH on Windows.
- [ ] Use `tools/native_training.sh` for configure/build/CTest.
- [ ] Inspect the linked runtime dependencies and prove `c10.dll`, `torch_cpu.dll`, CUDA LibTorch DLLs, and MSVC runtimes resolve from intended paths.
- [ ] Run all `cuda` tests and record exact pass/fail/skip counts; zero skipped hardware cases are required.
- [ ] Run one real forward and AdamW step on the selected GPU and record tensor/optimizer residency.
- [ ] Run FP32 Soo 128×6 with at least 128 simulations, games > lanes > threads, batch 256, multiple steps, and two iterations.
- [ ] Checkpoint after iteration one, force interruption, resume, and prove no duplicated durable work.
- [ ] Run paired-opening candidate/champion evaluation and record complete blocks and unique matches.
- [ ] Save canonical config/lineage, throughput, completion/aborts, batch distribution, timings, memory, checkpoint digests, resume evidence, and arena statistics as JSON.
- [ ] Do not upload or promote the acceptance candidate.

**Expected evidence:** every required GPU test executed, canonical device `cuda:N`, no CPU fallback/host CUDA dereference, finite parity within named GPU tolerances, bounded memory, exact work accounting, and attached acceptance/benchmark JSON.

## Task 8: Final repository boundary verification

- [ ] Run normal CPU suites once on the final commit.
- [ ] Confirm all Linux/macOS/Windows/sanitizer/Qt/policy/Python-zero contexts remain green on that commit.
- [ ] Confirm no tracked `.py`, build output, checkpoint, unrestricted log, credential, cache, or unmanifested large fixture was added.
- [ ] State explicitly whether the NVIDIA hardware gate ran; if not, mission completion remains pending.
