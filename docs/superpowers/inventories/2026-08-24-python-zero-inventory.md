# AlphaDiamond Python-Zero Inventory

**Status:** Proposed first-run inventory; no production, test, build, or CI file was changed.

**Inspection date:** 2026-08-24

**Authority:** tracked files at `f0f4a8a7a3928a3770afe16d5e1e246001961759`; remote `main` and branch protection were read through the GitHub API on the inspection date.

**Machine-readable ledger:** [`2026-08-24-python-zero-migration-ledger.yaml`](2026-08-24-python-zero-migration-ledger.yaml)

## 1. Verified baseline

- Local branch: `main`, tracking `origin/main`, with both local refs at `f0f4a8a7a3928a3770afe16d5e1e246001961759`.
- Remote `main`: the same full SHA, verified read-only through `repos/ubraket513/alphadiamond/branches/main`.
- Existing user state: untracked `.claude/`; it was not read, modified, or included in these artifacts.
- PR #44 is present as merge `a6f73d6` with payload `592adcc`: Python engine, geometry, oracle, and old agent layer were deleted.
- PR #45 is present as merge `85bcd2d` with payload `91f2511`: obsolete CI lanes/guards were removed and the Python matrix was reduced.
- PR #46 is present as merge `f0f4a8a` with payload `8bc18ae`: documentation was refreshed.
- Native C++ is authoritative for topology, rules, action/canonical encoding, MCTS, batching, self-play, and native inference.
- Python remains authoritative for eager model/training, optimizer and training checkpoints, replay persistence, inference orchestration, run coordination, rating/statistics, CLI, checkpoint surgery, artifact export, and release/promotion tooling.

No build, CTest, pytest, training step, package, or benchmark was run in this planning pass. Phase 0 records one current baseline with the commands in the design; old profiling documents are evidence of past measurements, not a current baseline.

## 2. CI and branch-protection drift

The workflow contains six job IDs producing eight check contexts:

| Job ID | Check context(s) |
|---|---|
| `native-core` | `native-core (ubuntu-latest)`, `native-core (macos-latest)`, `native-core (windows-latest)` |
| `native-sanitizers` | `native-sanitizers` |
| `core` | `core (py3.12)` |
| `bridge` | `bridge (pybind boundary)` |
| `native-qt` | `native-qt` |
| `lint` | `lint (changed files)` |

Branch protection currently has ten required contexts. Seven match the workflow. The drift is exact:

- Stale required contexts: `core (py3.11)`, `core (py3.13)`, `bridge-parity (Gate A-F)`.
- Missing required context: `bridge (pybind boundary)`.
- Matching required contexts: all three `native-core` legs, `native-sanitizers`, `core (py3.12)`, `native-qt`, and `lint (changed files)`.
- Protection has `strict: false`, `enforce_admins: false`, and no required review rule.

Before the first implementation PR relies on protection, replace the three stale contexts with the current bridge context. Do not remove `core`, `bridge`, or `lint` until their Python subjects have been deleted.

## 3. Tracked Python inventory

`git ls-files '*.py'` returns **183** files.

| Area | Count | Current role |
|---|---:|---|
| `src/` | 71 | package, trainer/control plane, and bridge adapters |
| `tests/` | 67 | 63 pytest modules, two `conftest.py` files, and two package markers |
| `tools/` | 24 | train/evaluate/replay/checkpoint/release and policy tools |
| `az-bench/profiles/` | 20 | historical/current Python profiling harnesses |
| root `setup.py` | 1 | optional pybind extension build |

Ledger classification totals are **A 4, B 21, C 147, D 11**. The category is a migration disposition, not a quality label:

- A: native behavior already exists; remove the Python duplicate only with proved CTest coverage.
- B: real pybind boundary behavior; retain until its last Python caller is gone.
- C: active Python behavior/tooling that first needs a native implementation or an explicit retirement decision.
- D: repository/build/policy behavior that moves to CMake, CTest, or a native validator.

### 3.1 Product subsystems

| Subsystem | Python surface | Native destination | Owning phase |
|---|---|---|---:|
| Duplicate contracts | `action_codec.py`, `encoder.py` | `soo/action.hpp`, `encoder.cpp`, native contract CTests | 1 |
| Boundary DTO/adapters | `diamond/contract/`, `game_adapter.py`, `native/`, search result/factory | public `soo` types and direct C++ calls | 1/3/6 |
| Model/training | `network/`, `evaluator/torch.py`, `trainer.py`, `checkpoint.py` | `diamond_model`, new `diamond_training` | 2 |
| Replay/inference | `replay.py`, `inference/`, replay store, worker/pool adapters | new `diamond_pipeline` | 3 |
| Orchestration/evaluation | `orchestration/`, `arena.py`, `rating/`, CLI | new `diamond_orchestration`, native executables | 4 |
| Release/data | checkpoint surgery, export, promotion, model index, backup | native checkpoint/release commands and CPack | 5 |
| Build/test policy | setuptools, pybind source list, pytest/Ruff policy | CMake/CTest and native validators | 6 |

### 3.2 Dependency graph

```text
alphadiamond-train / tools/az_train.py
  -> orchestration.production
     -> coordinator + run-state + persistent replay + self-play workers
     -> inference coordinator/model pool/remote
        -> TorchEvaluator -> SooModel | MinModel
     -> checkpoint + trainer + replay
     -> arena + schedules + Elo/TrueSkill registry

Python bridge
  contract DTOs + game_adapter + native.topology
    -> _diamond_native (native/bindings.cpp)
  native.search + native.selfplay_pool + native.backend
    -> Python callback -> Torch tensors/model

Release
  Python checkpoint
    -> export_deployment.py -> deployment artifact v3
    -> promote_checkpoint.py / build_model_index.py
    -> CPack input under dist/models
```

External Python imports are concentrated in Torch, pytest, TrueSkill, NumPy, psutil, and pybind11. `pyproject.toml` uses setuptools, requires Python `>=3.11`, exposes `alphadiamond-train`, and defines the `dev`, `alphazero`, and `native` extras. `environment.yml` also provisions Python/PyTorch/Qt tooling. There is no lock file.

Hugging Face synchronization is not implemented by tracked Python code. The supported Make façade invokes the external `hf sync` command. The final design removes this command from the build/train/release contract rather than adding Rust/Cargo or another build graph; local filesystem inputs remain the trainer contract.

## 4. Additional repository drift

1. `src/diamond/__main__.py` imports non-tracked `diamond.main`; `python -m diamond` is currently broken.
2. Five tracked scripts still import the deleted `diamond.alphazero.mcts.search_2p` module:
   - `az-bench/profiles/bench_bridge_search.py`
   - `az-bench/profiles/bench_production_stages.py`
   - `az-bench/profiles/bench_worker_mcts.py`
   - `tools/arena_abort_audit.py`
   - `tools/arena_v2.py`
3. `native-qt` is an Ubuntu headless lane. The documented shipping target includes Windows, but no Windows package/install smoke is currently a required check.
4. LibTorch CTests are conditional on `DIAMOND_BUILD_LIBTORCH_PROBE=ON`; artifact/parity tests register only when exported fixture metadata exists. They are not part of the portable default native suite.
5. The root CMake packaging comment still tells users to run two Python exporters before `package`.

These are inventory facts, not authorization to fix them in the planning pass.

## 5. Native targets and current CTest coverage

| Native area | Current files/target | CTest coverage |
|---|---|---|
| Rules/topology/encoding | `native/src/{board,encoder,prior,rules,topology_gen,topology_io}.cpp`; `soo_core` | `action_codec_test`, `topology_test`, `rules_golden_test` |
| Search | `native/src/{mcts,mcts3p,evaluator}.cpp`; `soo_search` | `mcts_golden_test`, `mcts3p_golden_test`, `mcts_stochastic_test`, `budget_test` |
| Batching/self-play | `native/src/{batcher,selfplay,profile}.cpp`; `soo_search` | `batcher_test`, `selfplay_test` |
| LibTorch inference | `deployment_artifact.cpp`, `soo_model.cpp`, `soo_evaluator.cpp`; `diamond_model` | `model_index_test`; conditional `soo_artifact_contract`, `model_parity_test` |
| Qt | `native/qt/`; `diamond_qt_backend`, `diamond_qt` | `diamond_qt_controller_contract`, `diamond_qt_analysis_smoke` |
| Bridge | `native/bindings.cpp`; optional pybind module | pytest under `tests/native/` |

Material gaps that must be closed before Python deletion:

- Native `selfplay_test` covers two-player episodes; Min needs a native three-player episode/scheduler test.
- There is no native three-player stochastic-search gate.
- The vacancy-prior golden path lacks the explicit v1-stall/v2-preference regression currently expressed in pytest.
- Callback ABI/GIL/NumPy conversion tests remain correctly bridge-only while Python calls pybind.
- There is no native training step, optimizer checkpoint, replay store, run coordinator, rating fixture, native CLI, or package-without-Python smoke.

## 6. Pytest inventory and migration matrix

There are **63 test modules plus two conftest files**. Classification totals below exclude the two package-marker `__init__.py` files but include both conftests: **A 2, B 7, C 50, D 6**. The YAML ledger also accounts for the package markers.

### A — native behavior already authoritative (2)

| Pytest module | Existing native proof | Deletion gate |
|---|---|---|
| `tests/alphazero/test_action_codec.py` | `action_codec_test` | Delete with the Python codec after its callers use the native contract. |
| `tests/alphazero/test_encoder.py` | `topology_test`, `rules_golden_test` | Delete after direct native encoding replaces the Python wrapper shape. |

### B — pybind boundary behavior (7)

| Pytest/support module | Current subject | Final replacement/deletion gate |
|---|---|---|
| `tests/alphazero/test_game_adapter.py` | state/move/exception conversion | Native callers eliminate the adapter; rules remain covered by `rules_golden_test`. |
| `tests/native/test_topology_generation.py` | extension auto-configuration/export shape | Delete with the extension; topology remains covered by `topology_test`. |
| `tests/native/test_callback.py` | callback ABI, GIL, exceptions, NumPy views | Delete only when LibTorch is called directly. |
| `tests/native/test_search_factory.py` | selector/deadline/callback/result mapping | Delete when native orchestration owns search creation. |
| `tests/native/test_selfplay_pool.py` | Soo job/callback/result/sample conversion | Native pipeline smoke plus `selfplay_test`. |
| `tests/native/test_selfplay_pool_min.py` | Min pool and placement targets | New native `selfplay_3p_test` plus pipeline smoke. |
| `tests/native/conftest.py` | extension fixture/skip policy | Delete with the final bridge pytest. |

### C — Python production behavior not yet ported (50)

| Phase | Pytest modules | Required native evidence before deletion |
|---:|---|---|
| 2 | `tests/alphazero/test_network.py`, `test_evaluator.py`, `test_trainer.py`, `test_trainer_sample_path.py`, `test_cuda_parity.py`, `test_checkpoint.py`, `test_identity.py` | Model/training frozen vectors; selected gradients; AdamW step/state; checkpoint round-trip and resume. |
| 3 | `tests/alphazero/bootstrap/test_bootstrap_evaluator.py`, `test_heuristic.py`, `test_selfplay_integration.py`, `test_vacancy_prior.py`; all seven `tests/alphazero/inference/test_*.py`; `tests/alphazero/test_metrics.py`, `test_replay.py`, `test_selfplay.py`; `tests/alphazero/orchestration/test_replay_store.py`, `test_replay_rollback.py`, `test_selfplay_workers.py` | Native prior fixtures, replay schema/store, model pool/coordinator, cancellation/error tests, 2P/3P self-play, end-to-end replay-to-train smoke. |
| 4 | `tests/alphazero/bootstrap/test_probe.py`; `tests/alphazero/test_arena.py`, `test_deadline.py`, `test_hardware.py`; `tests/alphazero/orchestration/test_benchmark_stage.py`, `test_cli.py`, `test_production.py`, `test_reference_configs.py`, `test_resume.py`, `test_run_state.py`, `test_training_coordinator.py`; all nine `tests/alphazero/rating/test_*.py`; `tests/tools/test_cpu_b0_train.py` | Frozen run-state/config fixtures, deterministic schedules/events, Elo/TrueSkill vectors, resume transitions, CLI exits/output. |
| 5 | `tests/alphazero/test_deployment.py`, `test_run_migrate.py`, `tests/test_promote_checkpoint.py` | Native legacy import/export, artifact v3 contract, migration and promotion state tests, package smoke. |
| 7 | `tests/alphazero/test_milestone2_smoke.py`, `test_smoke.py` | Native training and whole-pipeline smokes on a clean Python-free package. |

The seven inference test modules are `test_coordinator.py`, `test_model_pool.py`, `test_numeric_modes.py`, `test_profile.py`, `test_protocol.py`, `test_remote.py`, and `test_summary.py`. The nine rating modules are `test_elo.py`, `test_events.py`, `test_min_trueskill.py`, `test_openings.py`, `test_participants.py`, `test_protocol.py`, `test_registry_elo.py`, `test_registry_min.py`, and `test_schedule.py`.

### D — repository/build policy (6)

| Pytest/support module | Native/CMake disposition |
|---|---|
| `tests/conftest.py` | Delete when pytest is removed. |
| `tests/test_golden_contract.py` | Replace with a native/CMake manifest and payload-hash validator. |
| `tests/test_native_sources_agree.py` | Becomes obsolete when CMake is the only source list. |
| `tests/test_repo_hygiene.py` | Replace with a CMake script test, including the no-tracked-Python assertion. |
| `tests/test_runtime_boundary.py` | Replace with install/package layout smoke. |
| `tests/test_selfplay_backend_choice.py` | Becomes obsolete when no backend choice or Python caller remains. |

## 7. Inventory acceptance

This inventory is complete when the machine ledger contains the same 183 unique paths as `git ls-files '*.py'`, every path resolves to one A/B/C/D route, every pytest module has a named replacement/deletion gate, and branch-protection drift is fixed before implementation merges. The one-time ledger check is part of this first-run artifact review; ongoing enforcement is introduced only in Phase 6.
