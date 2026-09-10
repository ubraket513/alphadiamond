# AlphaDiamond Python-Zero Proposed PR Sequence

**Rule:** one isolated worktree and branch per PR. Every PR uses TDD, receives code review, runs its named verification once, and leaves `main` releasable.

**Exact Python deletion lists:** route IDs refer to the exhaustive file arrays in [`../inventories/2026-08-24-python-zero-migration-ledger.yaml`](../inventories/2026-08-24-python-zero-migration-ledger.yaml). The route reference is part of the PR contract; it is not a glob.

## PR 00 — Baseline, acceptance, and branch-protection integrity

- **Purpose:** land the approved inventory/spec/plan, record current measurements, and make required checks match the current workflow.
- **Files:** add/update `docs/superpowers/inventories/2026-08-24-python-zero-*`, `docs/superpowers/specs/2026-08-24-python-zero-migration-design.md`, `docs/superpowers/plans/2026-08-24-python-zero-*`, and `docs/performance-profiling/python_zero_baseline_2026-08-24.md`.
- **Python/pytest deleted:** none.
- **CTest:** none added; record current CTest/pytest outcomes and skips without changing gates.
- **Parity/frozen evidence:** ledger exactly equals current `git ls-files '*.py'`; current remote SHA and required contexts recorded.
- **Benchmark impact:** establishes, but does not compare against, the baseline.
- **Rollback boundary:** revert documentation and restore the prior protection contexts through the GitHub API.
- **Completion:** measurements have environment metadata; branch protection requires the eight current contexts and no stale context; `core`, `bridge`, and `lint` remain required.

## PR 01 — Close native contract coverage gaps

- **Purpose:** close known native gaps before retiring any Python contract test.
- **Production files:** modify `native/src/selfplay.cpp` or `native/include/soo/selfplay.hpp` only if the new test exposes a real gap; otherwise test-only.
- **Tests/build:** add `native/tests/selfplay_3p_test.cpp`; strengthen `native/tests/rules_golden_test.cpp` with the vacancy-prior stall/preference fixture; update `native/CMakeLists.txt`.
- **Python/pytest deleted:** none.
- **Parity/frozen evidence:** Min placement targets and episode determinism are frozen; vacancy-prior expected values come from the approved contract fixture.
- **Benchmark impact:** no performance claim; run native core timing once and record any material change.
- **Rollback boundary:** independent test/bug-fix PR.
- **Completion:** new tests fail under a deliberate local mutation and pass on the unmodified core; no bridge test is weakened.

## PR 02 — Retire duplicate Python action and canonical encoders

- **Purpose:** make existing Python callers consume native action/encoding results without keeping duplicate algorithms.
- **Production files:** modify `native/bindings.cpp`, `src/diamond/alphazero/game_adapter.py`, `src/diamond/alphazero/bootstrap/evaluator.py`, `src/diamond/alphazero/bootstrap/heuristic.py`, and direct callers identified by `git grep`.
- **Python/pytest deleted:** ledger route `A-native-contract-duplicates` — 4 files.
- **CTest:** existing `action_codec_test`, `topology_test`, and `rules_golden_test`; retain affected B tests because the bridge still exists.
- **Parity/frozen evidence:** exact action IDs, canonical feature tensors, and round-trip mappings for the existing frozen corpus.
- **Benchmark impact:** bridge callback/self-play median may not regress more than 5% without a measured cause.
- **Rollback boundary:** restore the four A files and caller imports; no persisted format changes.
- **Completion:** no Python implementation computes action encode/decode or canonical feature encoding; all supported Python commands still pass their current tests.

## PR 03 — Productionize the native model and freeze training vectors

- **Purpose:** turn the existing LibTorch probe model into the reusable model target and create immutable Soo/Min training fixtures.
- **Production files:** modify `native/CMakeLists.txt`, `CMakeLists.txt`, `CMakePresets.json`, `native/include/diamond_model/soo_model.hpp`, and `native/src/soo_model.cpp`; add `native/include/diamond_support/json.hpp` and `native/src/json.cpp` by extracting the existing artifact parser.
- **Fixtures/tests:** add `tests/golden/training-v1/manifest.json` plus compact tensor/state payloads; add temporary `tools/freeze_training_vectors.py`; add `native/tests/training_vector_manifest_test.cpp`.
- **Python/pytest deleted:** none; the freezer is entered in the ledger with deletion condition PR 05.
- **CTest:** model forward outputs, parameter-name mapping, fixture digests, and Soo/Min shapes.
- **Parity/frozen evidence:** logits, values, losses, named gradients, one AdamW step, resulting parameters, optimizer state, and resumed next step.
- **Benchmark impact:** record Python forward/training fixture timings; make no speed claim.
- **Rollback boundary:** model/fixture infrastructure only; current Python production command remains authoritative.
- **Completion:** required fixtures are present in `native-training` CI rather than conditionally skipped, and the existing deployment artifact v3 tests remain green.

## PR 04 — Native LibTorch training step

- **Purpose:** implement sparse sample-to-tensor construction, losses, autograd, AdamW, metrics, and deterministic seed setup.
- **Production files:** add `native/include/diamond_training/training_sample.hpp`, `native/include/diamond_training/trainer.hpp`, `native/src/training_sample.cpp`, and `native/src/trainer.cpp`; modify `native/CMakeLists.txt` and presets.
- **Python/pytest deleted:** none; Python trainer deletion condition is PR 10.
- **CTest:** add `native/tests/training_step_parity_test.cpp` for validation errors, Soo/Min shapes, losses, selected gradients, optimizer step, and finite metrics.
- **Parity/frozen evidence:** CPU FP32 `rtol=1e-5, atol=1e-6`; CUDA FP32 `rtol=1e-4, atol=1e-5` in the labelled CUDA lane.
- **Benchmark impact:** native median training-step time must be within 5% of the same fixture's Python baseline.
- **Rollback boundary:** new library and tests; no production caller or persisted format changes.
- **Completion:** all frozen one-step vectors pass and `diamond_training` exposes no Python/pybind types.

## PR 05 — Native checkpoint, legacy import, and resume

- **Purpose:** provide transactional checkpoint v2 and preserve supported Python checkpoint v1 inputs.
- **Production files:** add `native/include/diamond_training/checkpoint.hpp`, `native/include/diamond_training/rng_state.hpp`, `native/src/training_checkpoint.cpp`, `native/src/rng_state.cpp`, and `native/src/checkpoint_main.cpp`; modify CMake targets/presets.
- **Python/pytest deleted:** delete temporary `tools/freeze_training_vectors.py`; no current ledger route is deleted yet.
- **CTest:** add `native/tests/training_checkpoint_test.cpp`, `native/tests/legacy_checkpoint_import_test.cpp`, and `native/tests/training_resume_test.cpp`.
- **Parity/frozen evidence:** staged validation-before-mutation, model/AdamW/RNG state, operation ID, device-migration gate, and one further resumed step for frozen Soo/Min v1 checkpoints.
- **Benchmark impact:** save/load/resume time and peak RSS within 10% of baseline unless the manifest records a justified format cost.
- **Rollback boundary:** v2 is additive; Python v1 remains the production writer until the native pipeline cutover.
- **Completion:** direct C++ v1 import passes. If it cannot, stop this PR for an explicit compatibility decision; do not weaken the gate.

## PR 06 — Native replay schema and persistent store

- **Purpose:** implement schema-v1 records, deterministic sampling, transactional chunks/manifests, rollback, and bounded loading.
- **Production files:** add `native/include/diamond_pipeline/replay.hpp`, `native/include/diamond_pipeline/replay_store.hpp`, `native/src/replay.cpp`, and `native/src/replay_store.cpp`; extend `diamond_support` JSON only for required schema values.
- **Python/pytest deleted:** none; deletion condition is PR 07.
- **CTest:** add `native/tests/replay_schema_test.cpp` and `native/tests/replay_store_test.cpp` using frozen existing stores and failure injection around manifest replacement.
- **Parity/frozen evidence:** compatibility IDs, sparse-policy ordering, value shapes, deterministic sample IDs, rollback, and old-layout reads.
- **Benchmark impact:** measure `PersistentReplayStore.load_buffer()` first; native load/sample may not regress and physical redesign occurs only if the baseline is material.
- **Rollback boundary:** native store initially reads/writes an isolated fixture namespace; old stores remain untouched.
- **Completion:** native readers accept supported existing data and corrupt/incompatible data cannot partially mutate live state.

## PR 07 — Direct C++ self-play → replay → training pipeline

- **Purpose:** remove the callback/process transport from the training hot path and make native pipeline execution the supported training engine.
- **Production files:** add `native/include/diamond_pipeline/model_pool.hpp`, `native/include/diamond_pipeline/pipeline.hpp`, `native/src/model_pool.cpp`, `native/src/pipeline.cpp`, and `native/src/pipeline_main.cpp`; modify `soo_search` interfaces only where direct evaluator ownership requires it; adapt the still-Python outer coordinator to the typed process boundary for one transition PR.
- **Python/pytest deleted:** ledger route `C-replay-inference-selfplay-pipeline` — 38 files.
- **CTest:** add `native/tests/inference_coordinator_test.cpp` and `native/tests/native_pipeline_smoke_test.cpp`; retain bridge arena/search tests still used by Python evaluation.
- **Parity/frozen evidence:** fixed episode/job IDs, 2P/3P value targets, cancellation/deadline/error codes, replay digest, selected batch IDs, losses, and resulting checkpoint.
- **Benchmark impact:** self-play samples/s and end-to-end steps/hour within 5% of baseline; transport cost is reported, not assumed.
- **Rollback boundary:** Python outer coordinator can point back to the previous production command until the PR is reverted; persisted formats remain readable.
- **Completion:** the supported training hot path crosses no language boundary; no deleted Python module has a supported caller.

## PR 08 — Native run state, configuration, and coordinator

- **Purpose:** port typed config parsing, atomic run state, stage transitions, resume, and iteration coordination without changing rating policy yet.
- **Production files:** add `native/include/diamond_orchestration/config.hpp`, `run_state.hpp`, `coordinator.hpp`, `native/src/config.cpp`, `run_state.cpp`, and `coordinator.cpp`; add/update `native/src/train_main.cpp`.
- **Python/pytest deleted:** none; the Python coordinator remains only as the current arena/rating shell with deletion condition PR 09.
- **CTest:** add `native/tests/config_test.cpp`, `run_state_test.cpp`, and `coordinator_resume_test.cpp` from frozen JSON/run fixtures.
- **Parity/frozen evidence:** exact validation errors, stage graph, operation IDs, idempotent resume, ledger records, and exit codes.
- **Benchmark impact:** coordinator overhead and checkpoint interval are reported; no speed claim.
- **Rollback boundary:** native CLI is opt-in until PR 09 completes arena/rating.
- **Completion:** a stopped native run resumes deterministically through self-play and training without Python.

## PR 09 — Native arena, ratings, reports, and CLI cutover

- **Purpose:** complete the native control plane and make `alphadiamond-train` the sole supported run/evaluate/resume CLI.
- **Production files:** add `native/include/diamond_orchestration/arena.hpp`, `rating.hpp`, `schedule.hpp`, `native/src/arena.cpp`, `rating.cpp`, `schedule.cpp`, and report serialization; finalize `native/src/train_main.cpp`.
- **Python/pytest deleted:** ledger route `C-orchestration-evaluation-rating-cli` — 48 files.
- **CTest:** add `native/tests/arena_schedule_test.cpp`, `rating_fixture_test.cpp`, `rating_registry_test.cpp`, and `cli_contract_test.cpp`.
- **Parity/frozen evidence:** balanced openings/seats, deterministic schedules/events, Elo/TrueSkill updates, registry replay, promotion inputs, JSON reports, and CLI exits.
- **Benchmark impact:** arena games/s within 5% of baseline; rating/report CPU time recorded.
- **Rollback boundary:** revert the native CLI cutover and restore the route; checkpoint/replay formats remain compatible.
- **Completion:** configure, train, evaluate, resume, and report supported runs without Python; no deleted CLI/tool is documented as supported.

## PR 10 — Native checkpoint tooling, artifact promotion, release, and data boundary

- **Purpose:** replace supported model/checkpoint/release workflows and remove Python model/training/release code.
- **Production files:** add native subcommands under `native/src/checkpoint_main.cpp` and `native/src/release_main.cpp`; add `native/include/diamond_release/promotion.hpp`, `native/src/promotion.cpp`, and model-index/package staging support; modify root CMake/CPack and Make façade.
- **Python/pytest deleted:** ledger routes `C-libtorch-training` and `C-checkpoint-release-package-data` — 34 files.
- **CTest:** legacy/current checkpoint inspect/migrate, supported surgery fixtures, artifact-v3 contract, promotion state machine, model index, digest/provenance, and package staging.
- **Parity/frozen evidence:** supported v1/v2 checkpoints, deployment artifact v3 bytes/metadata semantics, promotion transitions, and package model selection.
- **Benchmark impact:** checkpoint surgery/export time, peak RSS, and package size compared with baseline; no raw checkpoint in package.
- **Rollback boundary:** native readers remain backward-compatible; promotion writes into a new staged directory before atomic activation.
- **Completion:** every supported checkpoint/release command is native; unsupported research surgery is explicitly retired; Make/build/train/release no longer invokes `hf sync` and accepts local paths only.

## PR 11 — Native benchmark disposition and end-to-end smokes

- **Purpose:** retain only decision-useful measurements and replace Python cross-subsystem smokes.
- **Production/files:** add focused executables under `native/benchmarks/` only for the approved Phase 0 measures; add `native/tests/final_pipeline_smoke_test.cpp`; update profiling docs with tool-by-tool disposition.
- **Python/pytest deleted:** ledger routes `C-benchmark-and-profile-tooling` and `C-final-smokes-and-package-shell` — 27 files.
- **CTest:** final native training/pipeline smokes; benchmarks remain explicit commands, not flaky pass/fail correctness tests.
- **Parity/frozen evidence:** each deleted smoke maps to a named CTest; each benchmark maps to a native command or an approved retirement reason.
- **Benchmark impact:** this PR produces the first complete Phase 0 versus native comparison table.
- **Rollback boundary:** tooling/smoke-only after production native commands exist.
- **Completion:** no historical Python benchmark is kept merely because it is tracked; no correctness contract is lost.

## PR 12 — Remove pybind/Python build and make CMake/CTest authoritative

- **Purpose:** delete the last Python/bridge/policy surface, move native application assets, and switch CI to the final native shape.
- **Production files:** delete `native/bindings.cpp`; modify root/native/Qt CMake and presets; move `src/diamond/qml` and `src/diamond/assets` to `native/qt/qml` and `native/qt/assets`; add `cmake/VerifyPythonZero.cmake`; modify `.github/workflows/ci.yml`, `Makefile`, packaging, and docs.
- **Python/pytest deleted:** ledger routes `B-contract-and-adapter-boundary`, `B-pybind-runtime-boundary`, `D-golden-and-mutation-policy`, and `D-build-policy-and-test-infrastructure` — 32 files. Delete `pyproject.toml` and `environment.yml` after their native provisioning/docs replacements are present.
- **CTest:** native golden-manifest validator, repository no-Python policy, install/package layout, Qt resource smoke, and clean supported-command scan.
- **Parity/frozen evidence:** complete pytest disposition table and no missing replacement; CMake is the sole source list.
- **Benchmark impact:** clean configure/build/test/package timings and package size recorded once.
- **Rollback boundary:** single cleanup PR after all callers are already native; revert restores bridge/build files without changing native formats.
- **Completion:** `git ls-files '*.py'` is empty; no supported workflow references Python/pip/pytest/setuptools/pybind/Ruff; final native CI jobs are green before obsolete contexts are removed from protection.

## PR 13 — Final clean-machine proof and protection lock

- **Purpose:** collect release evidence and make branch protection match the final native workflow.
- **Files:** update final architecture, operations, release, performance, and pytest-disposition documentation only unless proof exposes a defect; defects get their own fix PR.
- **Python/pytest deleted:** none; count is already zero.
- **Verification:** clean Linux/macOS/Windows configure/build/CTest, Linux sanitizers, Linux headless Qt, three-OS native training, Windows package/install/run, native resume, and final workflow scan.
- **Parity/frozen evidence:** final fixture manifest and complete test-disposition report.
- **Benchmark impact:** publish baseline versus final medians/ranges, memory, and artifact sizes without unsupported causal claims.
- **Rollback boundary:** documentation/protection update; any product defect is isolated separately.
- **Completion:** final required contexts exactly match the workflow; packaged Qt and training commands run on a machine with no Python installation; all strict end-state criteria are evidenced.
