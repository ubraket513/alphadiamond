# AlphaDiamond Python-Zero Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AlphaDiamond's Python trainer/control plane and pybind bridge with a Python-free C++20/LibTorch pipeline while preserving game, data, checkpoint/resume, deterministic, rating, and package contracts.

**Architecture:** Keep `soo_core` and `soo_search` authoritative, promote the existing LibTorch model to a production target, and add focused training, pipeline, and orchestration libraries. Migrate in vertical PRs; each native capability gains frozen/CTest evidence before its Python subject is deleted.

**Tech Stack:** C++20, CMake 3.21+, CTest, LibTorch 2.5+ (the exact fixture/CI patch version is pinned by Task 1), Qt 6.6+, GitHub Actions, CPack.

**Spec:** `docs/superpowers/specs/2026-08-24-python-zero-migration-design.md`

**PR sequence:** `docs/superpowers/plans/2026-08-24-python-zero-pr-sequence.md`

## Global Constraints

- Do not begin Task 1 until the design, this plan, the PR sequence, and the data-sync decision are approved.
- Execute every task in a fresh isolated worktree created with `superpowers:using-git-worktrees`; use a `codex/` branch name unless the user requests another prefix.
- CMake is the only authoritative build graph; CTest is the only final test runner.
- Use C++20 and keep the current CMake floor at 3.21 and Qt floor at 6.6.
- Keep `diamond-authoritative-rules-v1`, `diamond73-v1`, `diamond-camp-relative-v1`, `diamond73-srcdst-v1`, and `diamond-seat-layout-v1` unchanged.
- Keep deployment artifact format 3 and model-index semantics unchanged.
- Preserve checkpoint format 1 inputs unless a separate versioned compatibility change is explicitly approved.
- Use POSIX shell for commands; Windows shell automation runs through Git Bash when a shell script is required.
- Run each named correctness verification once per task after the focused red/green cycle; do not repeat unchanged full suites.
- Do not remove `core`, `bridge`, or Ruff CI while their Python production subjects remain.
- Do not delete the only test for any behavior; use the exact A/B/C/D disposition in the machine ledger.
- Do not add a Python fallback, a Rust/Cargo build, a second source list, or a new remote-sync implementation.

## Planned File Structure

```text
native/
  include/
    diamond_model/             existing model/artifact API
    diamond_support/json.hpp   shared strict JSON value/parser/writer
    diamond_training/          sample, trainer, checkpoint, RNG state
    diamond_pipeline/          replay, store, model pool, pipeline
    diamond_orchestration/     config, run state, coordinator, arena, rating
    diamond_release/           promotion and package staging
  src/
    json.cpp
    training_sample.cpp
    trainer.cpp
    training_checkpoint.cpp
    rng_state.cpp
    replay.cpp
    replay_store.cpp
    model_pool.cpp
    pipeline.cpp
    config.cpp
    run_state.cpp
    coordinator.cpp
    arena.cpp
    rating.cpp
    schedule.cpp
    promotion.cpp
    train_main.cpp
    checkpoint_main.cpp
    release_main.cpp
  tests/                       CTest executables, one contract per file
  benchmarks/                  explicit measured commands, not correctness gates
  qt/qml/ and qt/assets/       final native application resources
cmake/VerifyPythonZero.cmake
tests/golden/training-v1/      immutable cross-language training/checkpoint vectors
tests/golden/rating-v1/        immutable schedule/rating vectors
```

## Task 1: Baseline, Ledger, and Required-Check Integrity (PR 00)

**Files:**

- Create: `docs/performance-profiling/python_zero_baseline_2026-08-24.md`
- Modify: the approved inventory/spec/plan documents only if measured facts require a correction
- External read/write: GitHub branch protection required-status-check subresource only

**Interfaces:**

- Consumes: current `main`, the 183-file ledger, current CI workflow.
- Produces: pinned environment manifest, baseline results, and exactly eight current required contexts.

- [ ] **Step 1: Create the isolated worktree and confirm the unchanged baseline**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git ls-files '*.py' | LC_ALL=C sort > /tmp/alphadiamond-tracked-python.txt
test "$(wc -l < /tmp/alphadiamond-tracked-python.txt)" -eq 183
```

Expected: HEAD is `f0f4a8a7a3928a3770afe16d5e1e246001961759` at worktree creation, and the tracked list has 183 lines.

- [ ] **Step 2: Capture one correctness baseline**

Run each command once:

```bash
cmake --preset native-ci
cmake --build --preset native-ci --parallel
ctest --preset native-ci --output-on-failure
python -m pytest --ignore=tests/native --durations=10
python -m pytest tests/native -v --durations=10
```

Expected: record pass/fail/skip counts and durations verbatim. A failure blocks later tasks; diagnose it with `superpowers:systematic-debugging` rather than normalizing the baseline.

- [ ] **Step 3: Capture the benchmark baseline**

Use one warm-up and three measured repetitions for each approved subject. Record median/range, peak RSS where available, and artifact bytes:

```bash
/usr/bin/time -v cmake --build --preset native-ci --parallel
/usr/bin/time -v ctest --preset native-ci
python az-bench/profiles/bench_replay_pipeline.py --samples 20000 --batch 512
python az-bench/profiles/bench_native_scheduler.py --mode batch --seconds 3 --simulations 64
```

Use the fixed short-run command and checkpoint named in the approved benchmark manifest for training, checkpoint/resume, self-play, and end-to-end measures. Do not change its config between baseline and final comparison.

- [ ] **Step 4: Pin the parity environment**

Record exact values returned by:

```bash
python -c 'import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda)'
cmake --version
c++ --version
git diff --quiet
```

Expected: the exact Torch/LibTorch patch version becomes the required training-fixture version for Tasks 4–6; `git diff --quiet` exits 0 before writing results.

- [ ] **Step 5: Update only required status checks after explicit approval**

Run:

```bash
gh api --method PATCH \
  repos/ubraket513/alphadiamond/branches/main/protection/required_status_checks \
  --input - <<'JSON'
{
  "strict": false,
  "checks": [
    {"context": "native-core (ubuntu-latest)", "app_id": 15368},
    {"context": "native-core (macos-latest)", "app_id": 15368},
    {"context": "native-core (windows-latest)", "app_id": 15368},
    {"context": "native-sanitizers", "app_id": 15368},
    {"context": "core (py3.12)", "app_id": 15368},
    {"context": "bridge (pybind boundary)", "app_id": 15368},
    {"context": "native-qt", "app_id": 15368},
    {"context": "lint (changed files)", "app_id": 15368}
  ]
}
JSON
```

Read the subresource once and confirm those eight contexts, `strict=false`, and no stale context. This endpoint does not change reviews or admin enforcement.

- [ ] **Step 6: Commit and request review**

```bash
git add docs/superpowers docs/performance-profiling/python_zero_baseline_2026-08-24.md
git commit -m "docs: baseline the python-zero migration"
```

Review gate: baseline facts, ledger completeness, branch protection, and benchmark reproducibility.

## Task 2: Native 3P and Vacancy-Prior Contract Closure (PR 01)

**Files:**

- Create: `native/tests/selfplay_3p_test.cpp`
- Modify: `native/tests/rules_golden_test.cpp`, `native/CMakeLists.txt`
- Modify production only if the red test identifies a core defect: `native/src/selfplay.cpp`, `native/include/soo/selfplay.hpp`, or `native/src/prior.cpp`

**Interfaces:**

- Consumes: `soo::run_episodes`, the frozen 3P match/opening, and vacancy prior.
- Produces: native proof for Min episode targets and the explicit prior regression.

- [ ] **Step 1: Add the failing 3P episode test**

The test must load `tests/golden/rules-v1.txt`, select the three-player opening, run identical jobs twice, and assert:

```cpp
CHECK_EQ(first.size(), jobs.size());
CHECK_EQ(second.size(), jobs.size());
CHECK_EQ(first[index].moves.size(), second[index].moves.size());
CHECK_EQ(first[index].finish_order, second[index].finish_order);
CHECK_EQ(first[index].moves[move].features.feature_count, 6);
CHECK_EQ(first[index].moves[move].features.canonical_player_ids.size(), 3U);
CHECK_EQ(first[index].moves[move].selected_action,
         second[index].moves[move].selected_action);
```

- [ ] **Step 2: Add the failing vacancy-prior fixture assertion**

Add a frozen case where v1 stalls and v2 gives strict preference. Assert exact legal-action order and `kPriorTolerance = 1e-9` for maximum and order-sensitive dot product, matching the existing golden convention.

- [ ] **Step 3: Run the focused red tests**

```bash
cmake --preset native-ci
cmake --build --preset native-ci --target selfplay_3p_test rules_golden_test --parallel
ctest --preset native-ci -R 'selfplay_3p_test|rules_golden_test' --output-on-failure
```

Expected: the missing contract fails for a specific assertion; a registration/build failure is not an acceptable red state.

- [ ] **Step 4: Make the minimal production correction if required**

Preserve existing public types. Correct only the failing 3P target/order or vacancy-prior behavior; otherwise leave production unchanged.

- [ ] **Step 5: Run the focused green tests and commit**

```bash
ctest --preset native-ci -R 'selfplay_3p_test|rules_golden_test' --output-on-failure
git add native/CMakeLists.txt native/tests native/src native/include
git commit -m "test(native): close 3p selfplay and prior contracts"
```

## Task 3: Remove Duplicate Python Action/Encoding Algorithms (PR 02)

**Files:**

- Modify: `native/bindings.cpp`, `src/diamond/alphazero/game_adapter.py`, `src/diamond/alphazero/bootstrap/evaluator.py`, `src/diamond/alphazero/bootstrap/heuristic.py`, direct imports found by `git grep`
- Delete: exact files in ledger route `A-native-contract-duplicates`

**Interfaces:**

- Consumes: native `soo::encode_action`, `soo::decode_action`, and `soo::encode` behavior.
- Produces: one native algorithm with a temporary narrow bridge shape.

- [ ] **Step 1: Add boundary assertions to existing B tests**

In `tests/alphazero/test_game_adapter.py`, assert that the adapter's action ID and encoded features equal the native result for the frozen opening. Keep the assertion about the boundary; do not recreate the deleted algorithm in the test.

- [ ] **Step 2: Run the focused red test**

```bash
python -m pytest tests/alphazero/test_game_adapter.py -q
```

Expected: failure names the missing native-return surface.

- [ ] **Step 3: Expose and consume the narrow native values**

Add only these bridge operations if they are not already reachable:

```cpp
m.def("encode_action", &soo::encode_action, py::arg("source"), py::arg("destination"));
m.def("decode_action", &soo::decode_action, py::arg("action_id"));
```

Route adapter/bootstrap callers to native canonical data. Do not add a replacement Python codec/encoder class.

- [ ] **Step 4: Delete the A route and prove no import remains**

```bash
git rm src/diamond/alphazero/action_codec.py \
       src/diamond/alphazero/encoder.py \
       tests/alphazero/test_action_codec.py \
       tests/alphazero/test_encoder.py
git grep -n -E 'ActionCodec|CanonicalEncoder|alphazero\.action_codec|alphazero\.encoder' -- '*.py'
```

Expected: `git grep` prints no live import/use.

- [ ] **Step 5: Run one combined owning verification and commit**

```bash
ctest --preset native-ci -R 'action_codec_test|topology_test|rules_golden_test' --output-on-failure
python -m pytest tests/alphazero/test_game_adapter.py tests/alphazero/bootstrap -q
git add native/bindings.cpp src tests
git commit -m "refactor(contract): retire duplicate python encoding"
```

## Task 4: Production Model Target and Frozen Training Vectors (PR 03)

**Files:**

- Create: `native/include/diamond_support/json.hpp`, `native/src/json.cpp`, `native/tests/training_vector_manifest_test.cpp`, `tools/freeze_training_vectors.py`, `tests/golden/training-v1/`
- Modify: `native/include/diamond_model/soo_model.hpp`, `native/src/soo_model.cpp`, `native/src/deployment_artifact.cpp`, `native/CMakeLists.txt`, `CMakeLists.txt`, `CMakePresets.json`

**Interfaces:**

- Consumes: current Python Soo/Min models and non-empty AdamW state.
- Produces: stable raw tensor/vector fixtures and a production `diamond_model` CMake target.

- [ ] **Step 1: Write the manifest test first**

Require this exact top-level contract:

```json
{
  "fixture_version": 1,
  "game_contract": "diamond-authoritative-rules-v1",
  "dtype": "float32",
  "device": "cpu",
  "families": ["soo", "min"],
  "files": {}
}
```

The real manifest's `files` object contains each path, shape, element type, byte count, and SHA-256. The test rejects missing/extra fields, digest mismatch, or incompatible Torch fixture version.

- [ ] **Step 2: Run the manifest test red**

```bash
cmake --preset native-training
cmake --build --preset native-training --target training_vector_manifest_test --parallel
ctest --preset native-training -R training_vector_manifest_test --output-on-failure
```

Expected: failure reports the missing fixture manifest.

- [ ] **Step 3: Add the one-purpose freezer**

`tools/freeze_training_vectors.py` uses fixed seeds and writes raw little-endian `.f32`/`.i64` files plus two Python checkpoint-v1 fixtures. It records `torch.__version__`, the source commit, tensor shapes, and SHA-256. It refuses a dirty tree and refuses to overwrite an existing fixture without `--replace`.

- [ ] **Step 4: Extract the existing strict JSON parser**

Move the private value/parser logic from `deployment_artifact.cpp` into `diamond_support/json.hpp` and `json.cpp`. Add finite floating-point support required by replay/config while preserving duplicate-key, type, and trailing-data rejection. Keep deployment artifact behavior byte-for-byte compatible.

- [ ] **Step 5: Promote the model target**

Add `DIAMOND_BUILD_LIBTORCH` and a `native-training` preset. During migration, accept the old `DIAMOND_BUILD_LIBTORCH_PROBE` option as a deprecated alias to the same target so existing commands do not break in this PR.

- [ ] **Step 6: Generate once, verify once, and commit**

```bash
python tools/freeze_training_vectors.py --output tests/golden/training-v1
cmake --preset native-training
cmake --build --preset native-training --parallel
ctest --preset native-training -R 'training_vector_manifest_test|model_parity_test|soo_artifact_contract' --output-on-failure
git add CMakeLists.txt CMakePresets.json native tools/freeze_training_vectors.py tests/golden/training-v1
git commit -m "feat(training): freeze native model parity vectors"
```

## Task 5: Native Sparse AdamW Training Step (PR 04)

**Files:**

- Create: `native/include/diamond_training/training_sample.hpp`, `native/include/diamond_training/trainer.hpp`, `native/src/training_sample.cpp`, `native/src/trainer.cpp`, `native/tests/training_step_parity_test.cpp`
- Modify: `native/CMakeLists.txt`, `CMakePresets.json`

**Interfaces:**

- Produces:

```cpp
namespace diamond_training {
struct TrainingSample {
    Compatibility compatibility;
    std::vector<float> node_features;
    std::vector<std::pair<int32_t, float>> sparse_policy;
    std::vector<float> value_target;
};
struct TrainingMetrics {
    double total_loss;
    double policy_loss;
    double value_loss;
    uint64_t training_step;
};
class Trainer {
  public:
    Trainer(diamond_model::DiamondModel model, Compatibility compatibility,
            TrainingConfig config);
    TrainingMetrics train(std::span<const TrainingSample> samples);
};
}
```

- [ ] **Step 1: Write validation and parity tests**

Cover empty batches, compatibility mismatch, non-finite features/targets, negative policy values, policy sums outside `1e-5`, action IDs outside 5329, Soo/Min value widths, logits/values/losses, named gradients, AdamW state, and one resulting parameter update.

- [ ] **Step 2: Run the focused tests red**

```bash
cmake --build --preset native-training --target training_step_parity_test --parallel
ctest --preset native-training -R training_step_parity_test --output-on-failure
```

Expected: link/compile fails because `diamond_training::Trainer` does not exist.

- [ ] **Step 3: Implement sparse tensor construction and the training step**

The body follows the current contract:

```cpp
optimizer_.zero_grad();
auto [policy_logits, predicted_values] = model_->forward(features);
auto policy_loss = -(policy_targets * torch::log_softmax(policy_logits, 1)).sum(1).mean();
auto value_loss = torch::mse_loss(predicted_values, value_targets);
auto total_loss = policy_loss + value_loss;
if (!torch::isfinite(total_loss).item<bool>()) {
    throw std::invalid_argument("training produced a non-finite loss");
}
total_loss.backward();
optimizer_.step();
++training_step_;
```

Scatter sparse policy entries directly into a zero tensor; never materialize dense C++ vectors per sample.

- [ ] **Step 4: Run the training label once and commit**

```bash
ctest --preset native-training -L training --output-on-failure
git add native CMakePresets.json
git commit -m "feat(training): add native libtorch adamw step"
```

## Task 6: Transactional Checkpoint v2 and Legacy-v1 Resume (PR 05)

**Files:**

- Create: `native/include/diamond_training/checkpoint.hpp`, `native/include/diamond_training/rng_state.hpp`, `native/src/training_checkpoint.cpp`, `native/src/rng_state.cpp`, `native/src/checkpoint_main.cpp`, `native/tests/training_checkpoint_test.cpp`, `native/tests/legacy_checkpoint_import_test.cpp`, `native/tests/training_resume_test.cpp`
- Delete: `tools/freeze_training_vectors.py`
- Modify: `native/CMakeLists.txt`, `CMakePresets.json`, migration ledger

**Interfaces:**

```cpp
void save_checkpoint(const std::filesystem::path& target, const Trainer& trainer,
                     const RunRngState& rng, std::optional<std::string> operation_id);
CheckpointInfo load_checkpoint(const std::filesystem::path& source, Trainer& trainer,
                               const Compatibility& expected,
                               bool allow_device_migration = false);
CheckpointInfo import_legacy_v1(const std::filesystem::path& source,
                                const std::filesystem::path& target_v2,
                                const Compatibility& expected);
```

- [ ] **Step 1: Write failing save/load atomicity tests**

Assert exact manifest keys, SHA-256, non-empty optimizer state, staged validation before live mutation, optional operation ID, rejected incompatible metadata/device, temporary-directory cleanup, and uninterrupted old checkpoint after injected write failure.

- [ ] **Step 2: Write failing legacy import/resume tests**

Load both frozen Python-v1 checkpoints. Compare model parameters, AdamW moments/counters, step, config, and the next fixed training step against the fixture.

- [ ] **Step 3: Run red**

```bash
cmake --build --preset native-training --target training_checkpoint_test legacy_checkpoint_import_test training_resume_test --parallel
ctest --preset native-training -R 'training_checkpoint_test|legacy_checkpoint_import_test|training_resume_test' --output-on-failure
```

- [ ] **Step 4: Implement v2 archives and staged activation**

Use `torch::serialize::OutputArchive`/`InputArchive` for the native model and AdamW state. Serialize CPU/CUDA generator states and native RNG state as named payloads. Write into `<target>.tmp-<operation-id>`, validate every digest, then rename to the target.

- [ ] **Step 5: Implement the direct v1 importer or stop**

Read the current Python mapping through supported LibTorch serialization primitives, map exact Python state-dict names to registered native parameters, and validate all metadata before applying state. If the frozen fixture cannot be read without Python, mark the PR blocked and request approval for a versioned pre-conversion policy; do not delete or reinterpret v1.

- [ ] **Step 6: Run once, delete the freezer, and commit**

```bash
ctest --preset native-training -L training --output-on-failure
git rm tools/freeze_training_vectors.py
git add native CMakePresets.json docs/superpowers/inventories/2026-08-24-python-zero-migration-ledger.yaml
git commit -m "feat(checkpoint): add native transactional resume"
```

## Task 7: Native Replay Schema and Store (PR 06)

**Files:**

- Create: `native/include/diamond_pipeline/replay.hpp`, `native/include/diamond_pipeline/replay_store.hpp`, `native/src/replay.cpp`, `native/src/replay_store.cpp`, `native/tests/replay_schema_test.cpp`, `native/tests/replay_store_test.cpp`
- Modify: `native/include/diamond_support/json.hpp`, `native/src/json.cpp`, `native/CMakeLists.txt`

**Interfaces:**

```cpp
class ReplayStore {
  public:
    ReplayStore(std::filesystem::path root, Compatibility compatibility,
                std::size_t capacity, uint64_t seed);
    std::size_t ingest(std::span<const Episode> episodes);
    std::vector<TrainingSample> sample(std::size_t count);
    void prune();
    void restore_manifest(const std::filesystem::path& snapshot);
};
```

- [ ] **Step 1: Freeze legacy store fixtures and write failing readers**

Copy minimal completed, aborted, duplicate-id, corrupt-digest, rollback, capacity-prune, and deterministic-sampling stores from existing pytest fixtures into `tests/golden/replay-v1/`. Assert canonical JSON and exact next sample IDs.

- [ ] **Step 2: Run red**

```bash
cmake --build --preset native-training --target replay_schema_test replay_store_test --parallel
ctest --preset native-training -R 'replay_schema_test|replay_store_test' --output-on-failure
```

- [ ] **Step 3: Implement schema-v1 parsing and transaction order**

Preserve `schema_version`, compatibility metadata, `game_ids`, ordered chunks, aborted records, sample fields, SHA-256, and canonical JSON sorting. Write immutable chunks first and atomically replace the authoritative manifest last.

- [ ] **Step 4: Preserve legacy deterministic sampling**

For existing manifests, decode the CPython 3.12 MT19937 state and reproduce the frozen next sample IDs. New v2 manifests name their RNG algorithm explicitly and serialize its complete state; no platform library distribution is used as an implicit contract.

- [ ] **Step 5: Apply the measured loading decision**

If Task 1 shows full reload is material, keep a bounded in-memory index and stream newest chunks until capacity. If not, use the simpler full rebuild. Record the measured decision in `docs/performance-profiling/python_zero_baseline_2026-08-24.md`.

- [ ] **Step 6: Run once and commit**

```bash
ctest --preset native-training -R 'replay_schema_test|replay_store_test' --output-on-failure
git add native tests/golden/replay-v1 docs/performance-profiling/python_zero_baseline_2026-08-24.md
git commit -m "feat(replay): add native transactional store"
```

## Task 8: Direct Native Self-Play-to-Training Pipeline (PR 07)

**Files:**

- Create: `native/include/diamond_pipeline/model_pool.hpp`, `native/include/diamond_pipeline/pipeline.hpp`, `native/src/model_pool.cpp`, `native/src/pipeline.cpp`, `native/src/pipeline_main.cpp`, `native/tests/inference_coordinator_test.cpp`, `native/tests/native_pipeline_smoke_test.cpp`
- Modify: `native/CMakeLists.txt` and the remaining Python outer coordinator to invoke the typed native command during the transition
- Delete: every file in ledger route `C-replay-inference-selfplay-pipeline` (38 exact files)

**Interfaces:**

```cpp
struct IterationRequest {
    std::string operation_id;
    std::vector<soo::EpisodeJob> jobs;
    soo::EpisodeConfig selfplay;
    std::size_t training_steps;
};
struct IterationResult {
    std::string operation_id;
    std::size_t completed_games;
    std::size_t aborted_games;
    std::size_t new_samples;
    uint64_t training_step;
};
IterationResult run_iteration(const IterationRequest& request,
                              ModelPool& models, ReplayStore& replay,
                              diamond_training::Trainer& trainer,
                              std::stop_token stop);
```

- [ ] **Step 1: Write failing model-pool and pipeline tests**

Cover per-key model residency, incompatible checkpoint rejection, bounded batches, exception propagation, stop token, deadline, 2P/3P episodes, aborted episode record, replay digest, selected training samples, losses, and resulting checkpoint.

- [ ] **Step 2: Run red**

```bash
cmake --build --preset native-training --target inference_coordinator_test native_pipeline_smoke_test --parallel
ctest --preset native-training -R 'inference_coordinator_test|native_pipeline_smoke_test' --output-on-failure
```

- [ ] **Step 3: Implement direct evaluator ownership**

Connect the existing `soo::BatchEvaluator` contract to `diamond_model` without callback or NumPy conversion. Use bounded queues and `std::stop_token`; surface one typed failure record to the coordinator and drain/stop the batcher according to the existing `batcher_test` contract.

- [ ] **Step 4: Cut over the supported training engine**

The transitional Python coordinator invokes `pipeline_main` with versioned JSON request/result files and treats a non-zero exit as a failed stage. It does not import the deleted replay/inference/self-play modules and has no Python execution inside the native command.

- [ ] **Step 5: Delete the route and verify references**

```bash
sed -n '/id: "C-replay-inference-selfplay-pipeline"/,/^  - id:/ {
  s/^      - "\(.*\.py\)"$/\1/p
}' docs/superpowers/inventories/2026-08-24-python-zero-migration-ledger.yaml \
  > /tmp/alphadiamond-route-files.txt
test "$(wc -l < /tmp/alphadiamond-route-files.txt)" -eq 38
git rm --pathspec-from-file=/tmp/alphadiamond-route-files.txt
git grep -n -E 'alphazero\.(replay|inference|selfplay)|orchestration\.(replay_store|selfplay_workers)' -- '*.py'
```

Expected: the extracted route has exactly 38 paths and no supported import remains.

- [ ] **Step 6: Run owning verification and commit**

```bash
ctest --preset native-training -L pipeline --output-on-failure
python -m pytest tests/alphazero/orchestration -q
git add native src tests tools docs/superpowers/inventories/2026-08-24-python-zero-migration-ledger.yaml
git commit -m "feat(pipeline): train directly from native selfplay"
```

## Task 9: Native Config, Run State, and Coordinator (PR 08)

**Files:**

- Create: `native/include/diamond_orchestration/config.hpp`, `run_state.hpp`, `coordinator.hpp`, `native/src/config.cpp`, `run_state.cpp`, `coordinator.cpp`, `native/src/train_main.cpp`, `native/tests/config_test.cpp`, `run_state_test.cpp`, `coordinator_resume_test.cpp`
- Modify: `native/CMakeLists.txt`, `CMakePresets.json`

**Interfaces:**

```cpp
enum class RunStage { selfplay, train, benchmark, rate, checkpoint, complete };
struct TrainingRunState {
    uint32_t schema_version;
    std::string run_id;
    RunStage stage;
    uint64_t iteration;
    uint64_t training_step;
    std::optional<std::string> operation_id;
    std::optional<std::string> replay_manifest;
};
TrainingRunState run_or_resume(const RunConfig& config,
                               const std::filesystem::path& runtime_dir,
                               std::string_view run_id);
```

- [ ] **Step 1: Freeze config/run-state fixtures and write failing tests**

Cover exact key sets, unknown/missing keys, numeric ranges, run ID validation, legal stage transitions, atomic save, operation idempotency, interrupted-stage resume, checkpoint/replay pointers, and reference configs.

- [ ] **Step 2: Run red**

```bash
cmake --build --preset native-training --target config_test run_state_test coordinator_resume_test --parallel
ctest --preset native-training -R 'config_test|run_state_test|coordinator_resume_test' --output-on-failure
```

- [ ] **Step 3: Implement typed parsing and atomic state**

Use `diamond_support::Json`; reject unknown keys and non-finite numbers. Stage state in a sibling file, validate its round trip, then atomically replace. Reusing an operation ID returns the prior committed result instead of duplicating work.

- [ ] **Step 4: Implement coordinator transitions**

Call the native pipeline for self-play/train stages and persist state after each committed boundary. On process stop, leave the last committed stage and operation ID intact for deterministic resume.

- [ ] **Step 5: Run once and commit**

```bash
ctest --preset native-training -R 'config_test|run_state_test|coordinator_resume_test|native_pipeline_smoke_test' --output-on-failure
git add native CMakePresets.json tests/golden
git commit -m "feat(orchestration): add native run coordinator"
```

## Task 10: Native Arena, Ratings, Reports, and CLI Cutover (PR 09)

**Files:**

- Create: `native/include/diamond_orchestration/arena.hpp`, `rating.hpp`, `schedule.hpp`, `native/src/arena.cpp`, `rating.cpp`, `schedule.cpp`, `native/tests/arena_schedule_test.cpp`, `rating_fixture_test.cpp`, `rating_registry_test.cpp`, `cli_contract_test.cpp`, `tests/golden/rating-v1/`
- Modify: `native/src/train_main.cpp`, `native/CMakeLists.txt`, docs/config command examples
- Delete: every file in ledger route `C-orchestration-evaluation-rating-cli` (48 exact files)

**Interfaces:**

```cpp
using RatingEvent = std::variant<SooRatingEvent, MinRatingEvent>;
class RatingRegistry {
  public:
    bool apply(const RatingEvent& event);
    RatingSnapshot snapshot() const;
};
std::vector<Matchup> balanced_schedule(const BenchmarkProtocol& protocol,
                                       uint64_t seed);
```

- [ ] **Step 1: Freeze and test schedules/rating vectors**

Commit exact 2P balanced matchups, 3P seat permutations, event IDs, Elo updates, TrueSkill mean/sigma updates, duplicate-event idempotency, registry replay, and JSON report records. Use `1e-12` for pure-double rating fixtures and exact equality for schedule/event identity.

- [ ] **Step 2: Run red**

```bash
cmake --build --preset native-training --target arena_schedule_test rating_fixture_test rating_registry_test cli_contract_test --parallel
ctest --preset native-training -R 'arena_schedule_test|rating_fixture_test|rating_registry_test|cli_contract_test' --output-on-failure
```

- [ ] **Step 3: Implement the current approved algorithms**

Port the current Elo and TrueSkill equations used by supported Soo/Min events, not the full third-party Python package. Preserve schedule ordering, event canonicalization, default parameters, registry replay, and promotion inputs shown by the fixtures.

- [ ] **Step 4: Finalize the native CLI**

Support `train`, `resume`, `evaluate`, and `report` with stable JSON output and documented exit codes: `0` success, `2` invalid input/config, `3` incompatible artifact/checkpoint, `4` interrupted/incomplete stage, `5` internal execution failure.

- [ ] **Step 5: Delete the route and run the owning verification**

Use the reviewed static path list from the ledger route, then run:

```bash
ctest --preset native-training -L pipeline --output-on-failure
cmake --build --preset native-training --target alphadiamond-train --parallel
build/native-training/native/alphadiamond-train --help
git grep -n 'diamond\.alphazero\.orchestration\|diamond\.alphazero\.rating' -- '*.py'
```

Expected: no live import and native help exits 0.

- [ ] **Step 6: Commit**

```bash
git add native tests/golden/rating-v1 src tests tools docs
git commit -m "feat(orchestration): cut over native training cli"
```

## Task 11: Native Checkpoint/Release Commands and Data Boundary (PR 10)

**Files:**

- Create: `native/include/diamond_release/promotion.hpp`, `native/src/promotion.cpp`, `native/src/release_main.cpp`, `native/tests/checkpoint_surgery_test.cpp`, `promotion_state_test.cpp`, `package_smoke_test.cpp`
- Modify: `native/src/checkpoint_main.cpp`, `native/src/deployment_artifact.cpp`, root/native CMake, CPack settings, `Makefile`, release docs
- Delete: ledger routes `C-libtorch-training` and `C-checkpoint-release-package-data` (34 exact files)

**Interfaces:**

```cpp
enum class PromotionState { archival, candidate, promoted };
PromotionRecord promote(const std::filesystem::path& artifact,
                        PromotionState target,
                        const std::filesystem::path& destination);
void stage_release(const std::filesystem::path& model_index,
                   const std::filesystem::path& output_root);
```

- [ ] **Step 1: Write failing supported-workflow tests**

Cover checkpoint inspect/migrate, every surgery operation explicitly present in the approved supported-workflow list, deployment-v3 export, promotion transitions, digest/provenance rejection, model index, no raw checkpoint, and staged package layout.

- [ ] **Step 2: Run red**

```bash
cmake --build --preset native-training --target checkpoint_surgery_test promotion_state_test package_smoke_test --parallel
ctest --preset native-training -R 'checkpoint_surgery_test|promotion_state_test|package_smoke_test' --output-on-failure
```

- [ ] **Step 3: Implement only supported transformations**

Use named parameter maps and exact frozen output fixtures for widen/deepen/migrate operations retained by approval. A historical probe without a supported command receives a retirement entry instead of a native port.

- [ ] **Step 4: Make package staging native**

`alphadiamond-release` validates artifacts, writes `models/index.json`, copies runtime-only weights/metadata/topology, validates the staged tree, and only then exposes it to CPack. Remove Python exporter commands from the root CMake comment.

- [ ] **Step 5: Formalize the local data boundary**

Remove `data-push*` and `data-pull*` Make targets and all supported `hf sync` instructions. Document that supported commands read/write local paths; remote synchronization is an external operator responsibility and is not required to build, train, package, or release.

- [ ] **Step 6: Delete the routes, verify, and commit**

```bash
ctest --preset native-training -R 'training_|legacy_checkpoint|checkpoint_surgery|promotion_state|package_smoke|artifact_contract|model_index' --output-on-failure
git grep -n -E 'tools/(export_deployment|promote_checkpoint|build_model_index)|diamond\.alphazero\.(trainer|checkpoint|network)' -- ':!docs/superpowers/**'
git add native CMakeLists.txt Makefile src tests tools docs
git commit -m "feat(release): replace python checkpoint and packaging tools"
```

Expected: the grep finds no live source/build command.

## Task 12: Native Benchmark Disposition and Final Smokes (PR 11)

**Files:**

- Create: focused files under `native/benchmarks/`, `native/tests/final_pipeline_smoke_test.cpp`
- Modify: `native/CMakeLists.txt`, performance docs, migration ledger
- Delete: ledger routes `C-benchmark-and-profile-tooling` and `C-final-smokes-and-package-shell` (27 exact files)

**Interfaces:**

- Consumes: fixed Task 1 benchmark manifest.
- Produces: native commands for training step, checkpoint, replay, self-play, and end-to-end measures; explicit retirement records for the rest.

- [ ] **Step 1: Approve the tool disposition table before coding**

For each of the 20 benchmark scripts and seven final-smoke/package-shell files, mark exactly one disposition: native benchmark command, named CTest, or retire with the reason that no supported decision consumes it.

- [ ] **Step 2: Write the final smoke red**

The test creates a temporary run, generates fixed 2P and 3P episodes, ingests replay, trains one step per family, saves/reloads, resumes one step, evaluates a small arena, applies rating events, and stages a release artifact.

- [ ] **Step 3: Add only the five approved benchmark executables**

Each accepts `--manifest`, emits one canonical JSON result, records environment metadata, performs one warm-up and a caller-specified repetition count, and never controls correctness pass/fail by elapsed time.

- [ ] **Step 4: Run once, delete routes, and commit**

```bash
cmake --build --preset native-training --parallel
ctest --preset native-training -R 'final_pipeline_smoke_test|native_pipeline_smoke_test' --output-on-failure
git add native docs src tests tools az-bench
git commit -m "test(native): replace python smokes and benchmark harnesses"
```

## Task 13: Remove Bridge/Python Build and Enforce Python-Zero (PR 12)

**Files:**

- Delete: `native/bindings.cpp`, exact files in both B routes and both D routes, `pyproject.toml`, `environment.yml`
- Move: `src/diamond/qml` to `native/qt/qml`; `src/diamond/assets` to `native/qt/assets`
- Create: `cmake/VerifyPythonZero.cmake`, native golden-manifest validator/test
- Modify: root/native/Qt CMake, `CMakePresets.json`, `.github/workflows/ci.yml`, `Makefile`, docs

**Interfaces:**

- Produces: zero tracked Python, one CMake source graph, CTest policy checks, final native CI contexts.

- [ ] **Step 1: Add the failing Python-zero CMake test**

`cmake/VerifyPythonZero.cmake` must contain the equivalent of:

```cmake
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" ls-files "*.py"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE tracked_python
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT tracked_python STREQUAL "")
  message(FATAL_ERROR "tracked Python remains:\n${tracked_python}")
endif()
```

Register it as `python_zero_policy` and verify it fails before deletion.

- [ ] **Step 2: Replace remaining D policy tests**

Add native/CMake checks for golden manifest fields and SHA-256, source-list authority, repository hygiene, installed runtime layout, and no supported-command Python references.

- [ ] **Step 3: Remove every remaining caller and bridge/build file**

Delete both B routes and both D routes, then delete `native/bindings.cpp`, `pyproject.toml`, and `environment.yml`. Remove `DIAMOND_BUILD_PYBIND`, the pybind preset, setuptools source lists, pytest/Ruff commands, and Python setup steps.

- [ ] **Step 4: Move Qt resources and keep install behavior**

Update `native/qt/CMakeLists.txt` resource roots to `native/qt/qml` and `native/qt/assets`; preserve QML resource prefixes, sound deployment, font files, and package paths. Make the final Qt/package preset enable both the native Qt application and its LibTorch-backed Soo runtime.

- [ ] **Step 5: Change CI in a non-blocking order**

Add and observe final `native-training` three-OS, `native-package (windows)`, and `native-format` contexts before requiring them. Keep old required contexts until their last green merge; after this PR merges, patch protection to the final contexts and remove `core`, `bridge`, and Ruff jobs.

- [ ] **Step 6: Run the single final local verification and commit**

```bash
test -z "$(git ls-files '*.py')"
cmake --preset native-training
cmake --build --preset native-training --parallel
ctest --preset native-training --output-on-failure
cmake --preset native-qt
cmake --build --preset native-qt --parallel
ctest --preset native-qt --output-on-failure
git add -A
git commit -m "build: remove python and the pybind bridge"
```

## Task 14: Final Clean-Machine Proof (PR 13)

**Files:**

- Modify: final architecture, operations, release, performance, and pytest-disposition documents
- Production fixes: none in this PR; a discovered defect gets its own focused fix PR and review

**Interfaces:**

- Consumes: final native CI/package artifacts.
- Produces: auditable evidence for every strict end-state criterion and final required checks.

- [ ] **Step 1: Run the supported workflow scan**

```bash
test -z "$(git ls-files '*.py')"
rg -n -i 'python|pytest|pip|setuptools|pybind|ruff' \
  CMakeLists.txt CMakePresets.json Makefile .github native cmake README.md docs
```

Expected: only historical migration evidence or explicit “no Python” assertions remain; no supported command invokes a banned tool.

- [ ] **Step 2: Collect CI proof once per required lane**

Require green Linux/macOS/Windows native core and training, Linux sanitizers, Linux headless Qt, Windows package/install/run, and native format/static checks. Record workflow run URLs and artifact digests.

- [ ] **Step 3: Prove clean package behavior**

On a clean machine/image with no Python installation, configure/build/test/package, run `alphadiamond-train --help`, execute the native training/checkpoint/resume smoke, install the package, and launch the Qt analysis smoke.

- [ ] **Step 4: Compare baseline and final measurements**

Use the Task 1 manifest and the same repetition policy. Publish medians/ranges, peak RSS, package/artifact bytes, and any regression with its measured cause. Do not infer a speedup from architectural intent.

- [ ] **Step 5: Update final branch protection**

Read workflow check names, patch required checks to the exact final contexts, then read the subresource once. Confirm no stale `core`, `bridge`, or Ruff context remains.

- [ ] **Step 6: Commit documentation and request final review**

```bash
git add README.md docs
git commit -m "docs: record the python-zero release proof"
```

Review gate: complete pytest disposition, all strict end-state evidence, performance comparison, package digest, and exact required contexts.

## Execution Handoff

After approval, execute one PR task at a time with `superpowers:subagent-driven-development` (recommended). Each task gets a fresh implementer, spec compliance review, code quality review, and the single named verification before moving to the next PR. Stop at any compatibility gate that requires a product decision.

## Approval Checklist

- [ ] Accept the verified baseline, 183-file ledger, and pytest dispositions.
- [ ] Approve vertical-slice extinction and reject the big-bang approach.
- [ ] Require direct native checkpoint-v1 compatibility; a failed importer returns for a separate decision.
- [ ] Remove Hugging Face sync from the supported build/train/release contract and use local paths as the product boundary.
- [ ] Approve the final three-OS native-training and Windows package CI shape.
- [ ] Authorize Task 1 to replace the stale branch-protection contexts with the eight current contexts.
- [ ] Approve PR 00 only; Phase 1 production work begins after PR 00 evidence is reviewed.
