# AlphaDiamond Python-Zero Migration Design

**Status:** Proposed for approval. Phase 1 implementation is blocked until this spec and its implementation plan are approved.

**Baseline:** `main` and remote `main` at `f0f4a8a7a3928a3770afe16d5e1e246001961759` on 2026-08-24.

**Inventory:** [`../inventories/2026-08-24-python-zero-inventory.md`](../inventories/2026-08-24-python-zero-inventory.md)

**Ledger:** [`../inventories/2026-08-24-python-zero-migration-ledger.yaml`](../inventories/2026-08-24-python-zero-migration-ledger.yaml)

## 1. Decision summary

Migrate by **vertical-slice extinction**. Each PR introduces one native capability, proves parity or a frozen contract, routes its real caller to C++, and removes the Python implementation/test surface that has become obsolete. Python and pybind remain only where a still-supported Python caller crosses the boundary, with an explicit deletion PR named in the ledger.

The final runtime is:

```text
JSON config
   |
native coordinator
   +--> C++ self-play scheduler --> LibTorch model pool
   |          |
   |          +--> versioned replay records --> native replay store
   |                                              |
   +----------------------------------------------+--> LibTorch trainer
                                                        |
                                                  native checkpoint
                                                        |
                         arena/rating/promotion/package <-+
```

CMake remains the only build graph and CTest the only test runner. The final supported data boundary is the local filesystem. Hugging Face synchronization is removed from the build/train/release contract instead of introducing Rust/Cargo or retaining a Python CLI requirement.

## 2. Goals and non-goals

### Goals

1. No tracked production or test `.py` files and no supported command that invokes Python, pytest, pip, setuptools, pybind11, or Ruff.
2. Native LibTorch owns model execution, autograd, AdamW state, checkpoints, and training.
3. Self-play, replay, inference, training, evaluation, resume, promotion, packaging, and the Qt application run without a language boundary.
4. Existing game, artifact-v3, checkpoint/resume, deterministic schedule, rating, and packaging contracts are retained unless a separate versioned migration is explicitly approved.
5. Linux, macOS, and Windows C++ checks remain, with a separate sanitizer lane and clean-package proof.

### Non-goals

- Rewriting the already-authoritative C++ game core.
- Mechanical class-for-class translation of Python.
- Keeping Python as a fallback or a second authority.
- Adding another build system or a second source list.
- Claiming a speedup without Phase 0 and post-port measurements.
- Preserving broken or purely historical tools without evidence that they are a supported workflow.

## 3. Migration approaches

### 3.1 Recommended: vertical-slice extinction

Port one executable capability at a time: contract adapters, training, replay/pipeline, orchestration/rating, then release. A PR may temporarily leave old and new implementations together only when the new implementation is behind a parity gate and the PR names the later deletion condition.

**Advantages:** small rollback boundaries, continuous releasability, direct proof before deletion, and steadily shrinking Python surface.

**Risks:** temporary dual implementations can drift. Control this with frozen inputs, one authoritative output contract, and short-lived deletion conditions in the ledger.

### 3.2 Conservative alternative: bridge-first

Build every native subsystem behind pybind, keep all Python coordination until the entire C++ stack is complete, then switch callers and delete Python near the end.

**Advantages:** lowest early disruption and easy Python-vs-C++ differential tests.

**Risks:** the bridge becomes a long-lived architecture, Python and C++ remain co-authorities for months, checkpoints and process semantics are tested through the wrong boundary, and the final deletion becomes a large integration event. Use this only for short parity probes inside a vertical PR, not as the program structure.

### 3.3 Rejected: aggressive big-bang

Port all production and tests on one branch and delete Python in the same merge.

**Rejection:** it has no credible per-capability rollback, makes numerical and process failures inseparable, prevents required checks from evolving safely, and violates the requirement that `main` remain releasable. No implementation branch may adopt this approach.

## 4. Recommended native architecture

### 4.1 CMake targets

Preserve the two existing foundations:

- `alphadiamond::soo_core`: topology, state, rules, canonical encoding, prior.
- `alphadiamond::soo_search`: MCTS, batching, scheduling, self-play, profiling.

Promote and extend the existing optional LibTorch code:

- `alphadiamond::diamond_model`: the existing `DiamondModel`, artifact validation, weight loading, and inference. It becomes a production target whenever native training or Torch-backed Qt is enabled; “probe” ceases to be its architectural identity.
- `alphadiamond::diamond_training`: sparse sample-to-tensor conversion, policy/value loss, autograd, AdamW, gradient reset/validation, metrics, checkpoint save/load, and legacy checkpoint import.
- `alphadiamond::diamond_pipeline`: replay schema/store, deterministic sampling, resident model pool, direct self-play output ingestion, cancellation/deadline/error propagation, and end-to-end iteration smoke.
- `alphadiamond::diamond_orchestration`: typed configuration, run state, coordinator, deterministic arena schedule, rating registry, promotion decisions, and stable report records.

Expose focused executables rather than a single command with hidden modes:

- `alphadiamond-train`: train, resume, evaluate, and report a run.
- `alphadiamond-checkpoint`: inspect, validate, migrate, widen/deepen only supported workflows, and export deployment artifacts.
- `alphadiamond-release`: validate/promote an artifact, build the model index, and stage CPack input.

The checkpoint and release commands may share libraries; they are separate user-facing failure and rollback boundaries.

### 4.2 Typed interfaces

The core interfaces are deliberately smaller than the Python object graph:

```cpp
struct TrainingSample {
    Compatibility compatibility;
    std::vector<float> node_features;       // [73, feature_count]
    std::vector<std::pair<int32_t, float>> sparse_policy;
    std::vector<float> value_target;        // 1 for Soo, 3 for Min
};

struct TrainingMetrics {
    double total_loss;
    double policy_loss;
    double value_loss;
    uint64_t training_step;
};

class Trainer {
  public:
    TrainingMetrics train(std::span<const TrainingSample> samples);
    void save_checkpoint(const std::filesystem::path& target) const;
    void load_checkpoint(const std::filesystem::path& source,
                         const Compatibility& expected,
                         bool allow_device_migration = false);
};
```

Self-play returns native `Episode`/`EpisodeMove` records already present in `soo::selfplay`. `diamond_pipeline` converts them once into `TrainingSample`; it does not expose Python-compatible dictionaries or NumPy views.

### 4.3 Data flow and persistence

1. Parse a versioned JSON configuration into immutable native types.
2. Load or create a native checkpoint and run-state record.
3. Submit deterministic episode jobs to the existing native scheduler.
4. Evaluate batches directly through `diamond_model`; no callback/GIL/NumPy path.
5. Append samples to the replay store with transactional manifest replacement.
6. Sample a deterministic batch and call `diamond_training::Trainer`.
7. Atomically save checkpoint/run state; then schedule arena/rating/promotion work.

Do not literally port `PersistentReplayStore.load_buffer()`. Phase 0 measures it. The default design is append-only chunk metadata plus bounded streaming/reservoir loading; it is adopted only if the baseline proves the current full reload is material.

### 4.4 Stable formats

#### Game and deployment artifact

- Keep the frozen game-contract identifiers and `tests/golden/` payloads unchanged.
- Keep deployment artifact format v3 and `models/index.json` compatible.
- The release package continues to contain runtime weights and metadata, never a raw training checkpoint.

#### Replay

- Preserve sample schema version 1 fields, ordering, compatibility identifiers, sparse-policy semantics, and value-target shapes.
- Preserve accepted existing replay/run-state inputs until native readers pass frozen fixtures.
- Any storage optimization changes physical layout behind a versioned manifest and retains a read path for the previous layout.

#### Checkpoint

The current format is a Python `torch.save` mapping containing format version, compatibility metadata, training config, training step, model state, optimizer state, and optional operation ID. It is not assumed to be interchangeable with a C++ `OutputArchive`.

The native format is version 2:

```text
checkpoint/
  manifest.json        # format, compatibility, config, step, files/digests
  model.pt             # LibTorch module archive
  optimizer.pt         # AdamW archive
  rng-cpu.pt           # Torch CPU RNG state
  rng-cuda-<n>.pt       # present CUDA device states
  scheduler.bin        # native scheduler/replay RNG and counters
```

Write into a sibling temporary directory, fsync regular files and the parent where supported, validate the staged checkpoint, then rename atomically. Load into a staged model/optimizer before mutating the live trainer, matching the current Python guarantee.

Phase 2 first proves a direct C++ version-1 reader against frozen Soo and Min checkpoints with non-empty AdamW state. If LibTorch cannot robustly deserialize the current Python mapping, stop that PR and request a separate versioned migration decision. Python checkpoint compatibility must not be silently weakened.

### 4.5 Determinism and numerical parity

Current PyTorch documentation does not promise identical results across releases, platforms, or CPU versus CUDA. Therefore:

- Bit-for-bit resume is required only for the same binary, LibTorch version, device type, thread settings, and deterministic-algorithm configuration.
- Cross-language parity uses a pinned fixture environment and explicit tolerances.
- Cross-platform tests require schema, shapes, finite values, legal schedules, and tolerance-based outputs, not identical floating-point bytes.
- Checkpoints capture Torch CPU/CUDA generator states plus native scheduler, replay, and sampling RNG state.

Freeze representative Soo and Min vectors with fixed seeds:

- logits and values;
- total, policy, and value loss;
- gradients for named parameters in the trunk, policy head, and value head;
- one AdamW step and resulting parameters;
- optimizer moments/step counters;
- save/load/resume across at least one further training step.

Initial same-device tolerances are `rtol=1e-5, atol=1e-6` for CPU FP32 and `rtol=1e-4, atol=1e-5` for CUDA FP32. Tighten only from measured evidence; widen only with a recorded operation-level cause.

## 5. Test migration policy

Never translate a pytest in isolation. Apply the ledger route:

- A: identify the named CTest, add only a missing material case, then delete the Python implementation and pytest together.
- B: retain boundary tests while Python calls pybind; replace the end-to-end property only after the caller is native.
- C: implement the production capability, pass frozen/parity evidence, add CTest, then delete the Python implementation and tests in the same vertical phase.
- D: use CMake script tests, CTest fixtures, package smokes, or small native validators.

CTest labels separate fast portable gates without creating a second runner:

- `core`: existing rules/search/self-play tests.
- `training`: LibTorch model/trainer/checkpoint tests.
- `pipeline`: replay/inference/orchestration smokes.
- `package`: install/package/no-Python acceptance.
- `slow` and `cuda`: opt-in measured lanes, never hidden skips in a required CPU job.

Required fixtures must be present or configuration fails in their owning CI lane. The current conditional registration of artifact tests is acceptable for a developer checkout but not for `native-training` CI.

## 6. Phase plan

### Phase 0 — baseline and CI integrity

- Validate the 183-file ledger once.
- Capture the benchmark manifest and measurements below.
- Update branch protection to the eight current contexts before relying on it.
- Record exact current test/fixture/package sizes and failure/skip counts.
- Do not delete any Python lane.

### Phase 1 — contracts and adapters

- Close the native 3P self-play and vacancy-prior test gaps.
- Route Python callers away from duplicate action/canonical algorithms.
- Delete A files/tests when their named CTests prove the same contract.
- Keep callback/GIL/exception tests while the boundary is real.

### Phase 2 — LibTorch training

- Freeze Soo/Min training and checkpoint vectors.
- Promote `diamond_model` to a reusable production target.
- Add sparse tensor construction, losses, autograd, AdamW, metrics, deterministic state, checkpoint v2, and legacy v1 import.
- Route one real training command to native code and remove the superseded Python training surface only after parity.

### Phase 3 — replay and direct pipeline

- Add native replay records/store/sampling and model-pool coordination.
- Connect existing C++ self-play directly to replay and trainer.
- Cover 2P/3P episodes, cancellation, deadlines, error propagation, replay rollback, and deterministic resume.
- Remove the Python self-play/replay/inference caller surface and its bridge tests as their subjects disappear.

### Phase 4 — orchestration, evaluation, rating, CLI

- Port typed config, run-state transitions, coordinator, arena, schedules, Elo/TrueSkill behavior, reports, and CLI.
- Use frozen fixtures for external JSON schemas, schedules, events, rating updates, and exit behavior.
- Remove `alphadiamond-train` Python packaging and obsolete Python tools with native command parity.

### Phase 5 — checkpoint/release/data tooling

- Port only supported checkpoint surgery/migration workflows; explicitly retire unsupported research probes.
- Preserve artifact v3 and promotion state-machine behavior.
- Build model indexes and CPack input natively.
- Remove `data-push`/`data-pull` from the supported Make/build/train/release contract. Document local paths as the product boundary and any remote sync as operator-owned external work.

### Phase 6 — Python and bridge removal

- Delete pybind bindings, every remaining tracked `.py`, setuptools/pyproject Python packaging, Python environment files, pytest/Ruff commands, and Python-specific docs.
- Make CMake the sole source list; remove the pybind preset and duplicated extension source list.
- Move QML/assets from the Python package tree to the native application layout.
- Replace Python policy checks with CMake/CTest and update required checks without leaving stale contexts.

### Phase 7 — final proof

- Run clean configure/build/CTest on Linux, macOS, and Windows; sanitizers; Qt; native training/checkpoint/resume; package/install smoke.
- Prove `git ls-files '*.py'` is empty and supported workflows contain no Python/pip/pytest/setuptools/pybind references.
- Run the packaged Qt app with no Python installed.
- Compare performance, memory, and artifact sizes with Phase 0; report regressions and improvements without inference.

## 7. Benchmark plan and success metrics

Every result records commit, dirty state, OS, compiler, CMake generator, build type, LibTorch/Torch version, CPU/GPU, driver, thread counts, config/checkpoint digests, and whether the first run was warm-up. Use one warm-up and three measured repetitions; report the median and range.

| Measure | Phase 0 subject | Post-port subject | Success gate |
|---|---|---|---|
| Incremental native build | no-op `cmake --build --preset native-ci --parallel` | same | no more than 10% median regression |
| Native correctness time | `ctest --preset native-ci` | same plus labels reported separately | core lane no more than 10% regression |
| Python core/bridge time | pytest core and `tests/native` separately | disposition only | every module mapped; not a native speed claim |
| Training step | fixed Soo and Min batches, CPU and available CUDA | `training_step_benchmark` | no more than 5% median step-time regression; parity gates pass |
| Checkpoint | save, load, resume one step; bytes and peak RSS | native v2 and v1 import | correct resume; no more than 10% time/RSS regression unless format evidence justifies it |
| Replay load/sample | `PersistentReplayStore.load_buffer()` at fixed store sizes | native replay | no regression; redesign only if Phase 0 shows material cost |
| Self-play | fixed jobs/config/checkpoint | direct native pipeline | no more than 5% samples/s regression |
| End to end | fixed short training run | native short run | no more than 5% samples/s or steps/hour regression |
| Package | CPack archive and installed tree | Python-free package | no raw checkpoint; size increase no more than 5% without itemized cause |

Correctness gates override performance. A faster result that changes a contract does not pass.

## 8. CI and branch-protection end state

Final required checks:

- `native-core` on Linux, macOS, and Windows;
- `native-sanitizers` on Linux;
- `native-training` on Linux, macOS, and Windows with required fixtures;
- `native-qt` headless on Linux;
- `native-package (windows)` clean install/package smoke for the shipping target;
- `native-format` using the approved C++ formatter/static checks.

Introduce a new check and observe it green before making it required. Remove an old required context only after its production subject is gone and the replacement is required. Never rename and require a check in an order that blocks every PR.

## 9. Risk register

| Risk | Impact | Control and hard gate |
|---|---|---|
| Numerical parity | Wrong gradients or learning behavior | Frozen Soo/Min vectors, named gradients, AdamW state, CPU/CUDA tolerances; no trainer deletion before pass. |
| Checkpoint/resume compatibility | Existing runs become unusable | Frozen non-empty v1 checkpoints, staged load, direct C++ reader; stop for explicit version decision if unsupported. |
| Cross-platform LibTorch | Configure/link/runtime failures | Pinned supported LibTorch versions, three-OS training matrix, CPU required and CUDA separately labelled. |
| Process/cancellation semantics | Hangs, leaked work, corrupt replay | Stop-token/deadline tests, bounded queues, exception propagation, transactional persistence, sanitizer lane. |
| Rating reproducibility | Promotion decisions change | Frozen schedules/events/Elo/TrueSkill fixtures and deterministic replay of registries. |
| Release tooling | Invalid or oversized packages | Artifact-v3 and model-index CTests, digest validation, install-tree smoke, no raw checkpoints. |
| Branch-protection drift | Unmergeable or under-protected `main` | Fix current exact mismatch in Phase 0; staged context replacement thereafter. |
| Replay redesign | Data loss or speculative complexity | Measure `load_buffer()` first; keep current schema/read path; transactional fixtures and rollback test. |
| Stale tools | Migration scope expands without value | Mark broken/historical scripts; port only approved supported commands, otherwise delete with disposition. |
| Remote data sync | Python sneaks into final workflow | Local filesystem is the supported boundary; remove `hf sync` commands from product workflows. |

## 10. Rollback strategy

Each implementation PR is independently revertible. Formats are additive before destructive:

- Writers may emit a new version only after readers and fixtures exist.
- Python remains the active caller until its native replacement passes the owning vertical smoke.
- A temporary dual implementation has one authority selected by the production command and a named deletion PR.
- No PR deletes the only checkpoint reader, replay reader, rating fixture, bridge contract, or package smoke for its behavior.

## 11. Approval decisions

Approval of this spec authorizes planning and implementation of the PR sequence, not immediate destructive cleanup. It also confirms these design choices:

1. Vertical-slice extinction is the migration strategy; big-bang is rejected.
2. Checkpoint compatibility is preserved; failure of a direct native v1 reader triggers a new approval gate.
3. Hugging Face sync is removed from the supported build/train/release contract rather than adding another language/build system.
4. The final required CI includes three-OS native training and a Windows package smoke.
