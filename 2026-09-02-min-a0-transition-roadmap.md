# Min A0 Transition and Parallel MCTS Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Min's B0 plateau is caused by weak search targets, ineffective policy learning, or heuristic dependence; then move toward A0 with the smallest validated change, implementing parallel MCTS only when stronger serial search proves valuable.

**Architecture:** Work in three gated increments. First add observability without changing training behavior. Second run a paired serial-search sweep at 128/256/400 simulations, including repetition-triggered adaptive search, and classify the bottleneck. Third implement exactly one evidence-backed intervention: parallel MCTS when deeper search helps but is too expensive, or vacancy-prior annealing when deeper search does not independently solve A0 stability.

**Tech Stack:** C++20 targets built with GCC 14+, LibTorch, CUDA, CMake/Ninja/CTest, canonical JSON, Git/GitHub CLI, shell and Python only for experiment aggregation—not for runtime or model execution.

**Spec:** `docs/model-training/min_bootstrap.md`, `docs/model-training/min_arena_throughput.md`, and `docs/performance-profiling/min_autopilot_log_2026-08-31.md` on the execution branch's base commit.

## Global Constraints

- `ubraket513/alphadiamond` is the source of truth. Begin from freshly fetched `origin/main` and record its SHA in every experiment report.
- At plan authoring, the latest observed `main` commit is `589d6b452242f2a7d96a3277af76fec54582326d`. If `origin/main` has moved, re-read the three spec files above and preserve their newer contracts.
- The released baseline is `min-v1.0.1`, iteration 25, training step 27,776. Use its released checkpoint, replay, and run-state assets after verifying `SHA256SUMS`.
- The actual B0 operating point is taken from the released run's `resolved-config.json`, not from `configs/alphazero/min-production.json`. The release ran FP32, 1,024 games, 512 lanes, 16 search threads, batch 256, wait 100 microseconds, 128 simulations, and 1,408 learner steps.
- Do not mutate or resume the live production run while implementing or measuring Tasks 1–9. All training diagnostics operate on copied artifacts and never save a checkpoint.
- Keep runtime precision FP32. FP16 is not an experimental arm because it already produced non-finite Min inference; BF16 is not an arm because it was slower than FP32.
- Do not reintroduce Soo as a policy teacher. Soo-policy transfer is closed. This plan does not mix Soo logits into Min.
- Preserve Min's global-seat `ValueVector` semantics. Parallel-search bookkeeping must never add a scalar virtual loss to a value component.
- Preserve all existing frozen golden fixtures. The serial path and `parallel_leaves=1` path must remain bit-exact against existing 3P golden tests.
- PR 1 is observability-only. It must not alter search choices, training targets, optimizer updates, checkpoint/replay schemas, production defaults, or resolved-config compatibility.
- PR 2 may enable already-configured adaptive simulations for 3P but keeps defaults unchanged (`simulations_late=0`, `repeat_window=0`).
- Parallel MCTS is forbidden until the serial sweep meets the gate in Task 9.
- Vacancy-prior annealing is forbidden until the diagnostics and serial sweep meet the gate in Task 9.
- Every new JSON output includes `schema_version`, base commit, model digest, resolved-config digest or path, exact command-line parameters, seed, device, precision, and wall time.
- Build on Linux with GCC 14 or newer. Set `TORCH_CUDA_ARCH_LIST` to the host GPU compute capability. Run LibTorch-linked binaries with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` on the documented wheel-based environment.

---

## Phase 0: Establish a Reproducible, Read-Only Baseline

### Task 1: Create the worktree and acquire verified Min v1.0.1 artifacts

**Files:**
- Create in repository: `docs/superpowers/reports/2026-09-01-min-a0-baseline.md`
- Create outside repository: `/workspace/alphadiamond-experiments/min-v1.0.1/`

**Interfaces:**
- Consumes: `origin/main`, GitHub release `min-v1.0.1`
- Produces: `BASE_SHA`, `CHECKPOINT_ROOT`, `REPLAY_ROOT`, `RESOLVED_CONFIG`, and a verified artifact inventory used by every later task

- [ ] **Step 1: Create an isolated worktree from current `origin/main`.**

```bash
git -C /workspace/alphadiamond fetch origin --prune
BASE_SHA=$(git -C /workspace/alphadiamond rev-parse origin/main)
git -C /workspace/alphadiamond worktree add \
  /workspace/alphadiamond-min-a0 \
  -b codex/min-a0-diagnostics "$BASE_SHA"
cd /workspace/alphadiamond-min-a0
printf '%s\n' "$BASE_SHA" > /tmp/min-a0-base-sha
```

Expected: the new worktree is clean and `git status --short` prints nothing.

- [ ] **Step 2: Read the authoritative handoff documents before editing.**

```bash
sed -n '1,260p' docs/model-training/min_bootstrap.md
sed -n '1,260p' docs/model-training/min_arena_throughput.md
sed -n '1,280p' docs/performance-profiling/min_autopilot_log_2026-08-31.md
```

Expected: the documents identify the iteration-25 champion as the durable baseline, record the failed no-prior gate, and describe the policy/value-loss plateau. If newer documents supersede these facts, record the newer facts and use them in all commands below.

- [ ] **Step 3: Download the release into a directory that is never used as a run directory.**

```bash
EXP=/workspace/alphadiamond-experiments/min-v1.0.1
rm -rf "$EXP"
mkdir -p "$EXP/downloads" \
         "$EXP/extracted/checkpoint" \
         "$EXP/extracted/replay" \
         "$EXP/extracted/run-state" \
         "$EXP/results"
gh release download min-v1.0.1 \
  --repo ubraket513/alphadiamond \
  --dir "$EXP/downloads" \
  --pattern 'latest.tar.gz' \
  --pattern 'replay.tar.gz' \
  --pattern 'run-state.tar.gz' \
  --pattern 'SHA256SUMS'
(
  cd "$EXP/downloads"
  sha256sum -c SHA256SUMS
)
```

Expected: every entry in `SHA256SUMS` prints `OK`.

- [ ] **Step 4: Extract each asset into a separate tree and discover roots by content, not guessed archive layout.**

```bash
tar -xzf "$EXP/downloads/latest.tar.gz" -C "$EXP/extracted/checkpoint"
tar -xzf "$EXP/downloads/replay.tar.gz" -C "$EXP/extracted/replay"
tar -xzf "$EXP/downloads/run-state.tar.gz" -C "$EXP/extracted/run-state"

CHECKPOINT_CURRENT=$(find "$EXP/extracted/checkpoint" -type f -name CURRENT -print -quit)
REPLAY_MANIFEST=$(find "$EXP/extracted/replay" -type f -name manifest.json -print -quit)
RESOLVED_CONFIG=$(find "$EXP/extracted/run-state" -type f -name resolved-config.json -print -quit)

test -n "$CHECKPOINT_CURRENT"
test -n "$REPLAY_MANIFEST"
test -n "$RESOLVED_CONFIG"
CHECKPOINT_ROOT=$(dirname "$CHECKPOINT_CURRENT")
REPLAY_ROOT=$(dirname "$REPLAY_MANIFEST")

cat > "$EXP/env.sh" <<ENV
export MIN_A0_BASE_SHA='$BASE_SHA'
export MIN_A0_EXP='$EXP'
export MIN_A0_CHECKPOINT='$CHECKPOINT_ROOT'
export MIN_A0_REPLAY='$REPLAY_ROOT'
export MIN_A0_CONFIG='$RESOLVED_CONFIG'
ENV
cat "$EXP/env.sh"
```

Expected: the three paths point inside `$EXP/extracted`; none point into a live run directory.

- [ ] **Step 5: Validate checkpoint identity and record the baseline.**

```bash
source "$EXP/env.sh"
mkdir -p docs/superpowers/reports
{
  echo '# Min A0 Baseline — 2026-09-01'
  echo
  echo "- Base commit: \`$MIN_A0_BASE_SHA\`"
  echo "- Release: \`min-v1.0.1\`"
  echo "- Checkpoint root: \`$MIN_A0_CHECKPOINT\`"
  echo "- Replay root: \`$MIN_A0_REPLAY\`"
  echo "- Resolved config: \`$MIN_A0_CONFIG\`"
  echo
  echo '## Resolved config'
  echo '```json'
  cat "$MIN_A0_CONFIG"
  echo '```'
} > docs/superpowers/reports/2026-09-01-min-a0-baseline.md
```

After the native build exists, append the output of:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  build/native-training/native/alphadiamond-checkpoint validate \
  "$MIN_A0_CHECKPOINT"
```

Expected: checkpoint format v3 is valid, optimizer state is readable, and training step is 27,776 unless a newer authoritative release supersedes it.

- [ ] **Step 6: Commit only the baseline record.**

```bash
git add docs/superpowers/reports/2026-09-01-min-a0-baseline.md
git commit -m "docs(min): pin the A0 diagnostic baseline"
```

---

## Phase 1: Add Observability Without Changing Behavior

### Task 2: Implement pure MCTS visit-target statistics

**Files:**
- Create: `native/include/soo/search_target_metrics.hpp`
- Create: `native/src/search_target_metrics.cpp`
- Create: `native/tests/search_target_metrics_test.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: authoritative root `visit_counts` from `soo::EpisodeMove`
- Produces:

```cpp
namespace soo {

struct VisitTargetObservation {
    uint32_t legal_actions = 0;
    uint64_t visits = 0;
    double entropy = 0.0;
    double normalized_entropy = 0.0;
    double max_probability = 0.0;
    double top3_mass = 0.0;
    double effective_actions = 0.0;
    double zero_visit_fraction = 0.0;
};

struct VisitTargetSummary {
    uint64_t rows = 0;
    double legal_actions_mean = 0.0;
    double entropy_mean = 0.0;
    double entropy_p50 = 0.0;
    double entropy_p90 = 0.0;
    double normalized_entropy_mean = 0.0;
    double max_probability_mean = 0.0;
    double top3_mass_mean = 0.0;
    double effective_actions_mean = 0.0;
    double zero_visit_fraction_mean = 0.0;
};

VisitTargetObservation inspect_visit_target(std::span<const uint32_t> visit_counts);
VisitTargetSummary summarize_visit_targets(std::span<const VisitTargetObservation> rows);

}  // namespace soo
```

- [ ] **Step 1: Write failing unit tests.**

Test all of the following in `search_target_metrics_test.cpp`. The repository's `check.hpp` exposes only `CHECK`, `CHECK_EQ`, and `REQUIRE`, so define local `close(double,double,double)` and `throws_invalid_argument(callable)` helpers rather than inventing new assertion macros:

```cpp
CHECK(throws_invalid_argument([] { (void)inspect_visit_target({}); }));
CHECK(throws_invalid_argument([] {
    (void)inspect_visit_target(std::array<uint32_t, 2>{0, 0});
}));

const auto deterministic = inspect_visit_target(std::array<uint32_t, 3>{8, 0, 0});
CHECK(close(deterministic.entropy, 0.0, 1e-12));
CHECK(close(deterministic.normalized_entropy, 0.0, 1e-12));
CHECK(close(deterministic.max_probability, 1.0, 1e-12));
CHECK(close(deterministic.top3_mass, 1.0, 1e-12));
CHECK(close(deterministic.effective_actions, 1.0, 1e-12));
CHECK(close(deterministic.zero_visit_fraction, 2.0 / 3.0, 1e-12));

const auto uniform = inspect_visit_target(std::array<uint32_t, 4>{1, 1, 1, 1});
CHECK(close(uniform.entropy, std::log(4.0), 1e-12));
CHECK(close(uniform.normalized_entropy, 1.0, 1e-12));
CHECK(close(uniform.max_probability, 0.25, 1e-12));
CHECK(close(uniform.top3_mass, 0.75, 1e-12));
CHECK(close(uniform.effective_actions, 4.0, 1e-12));
```

Also assert exact p50/p90 selection on a known five-row input and reject non-finite derived values.

- [ ] **Step 2: Register the source and test, then run it to verify failure.**

Add `src/search_target_metrics.cpp` to `SOO_SEARCH_SOURCES`, create the test target with `diamond_add_native_test(search_target_metrics_test)`, then run:

```bash
cmake --build build/native-training --target search_target_metrics_test -j2
ctest --test-dir build/native-training -R '^search_target_metrics_test$' --output-on-failure
```

Expected before implementation: compile/link failure because the declared functions are undefined.

- [ ] **Step 3: Implement the metric functions.**

Requirements:

```cpp
const double probability = static_cast<double>(count) / static_cast<double>(visits);
entropy -= probability * std::log(probability);  // skip zero counts
normalized_entropy = legal_actions > 1 ? entropy / std::log(legal_actions) : 0.0;
effective_actions = std::exp(entropy);
```

Sort only a short local probability vector for top-3 mass. Quantiles use the same lower-index convention already used by repository benchmarks: `index = q * (size - 1)`.

- [ ] **Step 4: Run the focused tests.**

```bash
cmake --build build/native-training --target search_target_metrics_test -j2
ctest --test-dir build/native-training -R '^search_target_metrics_test$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add native/include/soo/search_target_metrics.hpp \
        native/src/search_target_metrics.cpp \
        native/tests/search_target_metrics_test.cpp \
        native/CMakeLists.txt
git commit -m "feat(search): measure MCTS visit-target entropy"
```

### Task 3: Surface target statistics in self-play results and the durable sidecar

**Files:**
- Modify: `native/include/diamond_pipeline/pipeline.hpp`
- Modify: `native/src/pipeline.cpp`
- Modify: `native/src/train_main.cpp`
- Modify: `native/tests/native_pipeline_smoke_test.cpp`
- Modify: `native/tests/cli_contract_test.cpp`

**Interfaces:**
- Consumes: Task 2 `inspect_visit_target()` and `summarize_visit_targets()`
- Produces:

```cpp
struct SelfPlayMetrics {
    // existing fields remain unchanged
    soo::VisitTargetSummary all_targets;
    soo::VisitTargetSummary completed_targets;
    soo::VisitTargetSummary aborted_targets;
};
```

The durable `selfplay.metrics.json` becomes schema version 2 and adds:

```json
"search_targets": {
  "all": { "rows": 0, "entropy_mean": 0.0, "entropy_p50": 0.0, "entropy_p90": 0.0,
           "normalized_entropy_mean": 0.0, "max_probability_mean": 0.0,
           "top3_mass_mean": 0.0, "effective_actions_mean": 0.0,
           "legal_actions_mean": 0.0, "zero_visit_fraction_mean": 0.0 },
  "completed": { "...": "same fields" },
  "aborted": { "...": "same fields" }
}
```

- [ ] **Step 1: Write failing pipeline assertions.**

Extend `native_pipeline_smoke_test.cpp` with a completed synthetic episode or a deterministic small self-play fixture and assert:

```cpp
CHECK(result.metrics.all_targets.rows > 0);
CHECK(result.metrics.all_targets.entropy_mean >= 0.0);
CHECK(result.metrics.all_targets.normalized_entropy_mean >= 0.0);
CHECK(result.metrics.all_targets.normalized_entropy_mean <= 1.0 + 1e-12);
CHECK_EQ(result.metrics.all_targets.rows,
         result.metrics.completed_targets.rows + result.metrics.aborted_targets.rows);
```

Extend `cli_contract_test.cpp` to read the self-play sidecar generated by its existing training smoke and require schema 2 plus `search_targets.all.rows`.

- [ ] **Step 2: Run tests and verify they fail on absent fields.**

```bash
cmake --build build/native-training --target native_pipeline_smoke_test cli_contract_test -j2
ctest --test-dir build/native-training \
  -R '^(native_pipeline_smoke_test|cli_contract_test)$' --output-on-failure
```

Expected: compile failure for missing C++ members or JSON-contract failure for missing sidecar fields.

- [ ] **Step 3: Aggregate raw root targets before pipeline conversion discards zero-visit legal actions.**

In `run_self_play()` collect observations from every raw `soo::EpisodeMove`:

```cpp
std::vector<soo::VisitTargetObservation> all;
std::vector<soo::VisitTargetObservation> completed;
std::vector<soo::VisitTargetObservation> aborted;

for (const auto& episode : episodes) {
    auto& bucket = episode.completed ? completed : aborted;
    for (const auto& move : episode.moves) {
        auto row = soo::inspect_visit_target(move.visit_counts);
        all.push_back(row);
        bucket.push_back(row);
    }
}
result.metrics.all_targets = soo::summarize_visit_targets(all);
result.metrics.completed_targets = soo::summarize_visit_targets(completed);
result.metrics.aborted_targets = soo::summarize_visit_targets(aborted);
```

Do not add these fields to `TrainingSample`, replay chunks, or checkpoint manifests.

- [ ] **Step 4: Add one JSON serializer helper in `train_main.cpp`.**

Use a function with this exact signature to avoid duplicating field spelling:

```cpp
Object visit_target_json(const soo::VisitTargetSummary& summary);
```

Bump only `selfplay.metrics.json` to schema 2. Keep existing field names and values byte-for-byte equivalent except for the schema number and new object.

- [ ] **Step 5: Run focused tests.**

```bash
cmake --build build/native-training --target native_pipeline_smoke_test cli_contract_test -j2
ctest --test-dir build/native-training \
  -R '^(native_pipeline_smoke_test|cli_contract_test)$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add native/include/diamond_pipeline/pipeline.hpp \
        native/src/pipeline.cpp native/src/train_main.cpp \
        native/tests/native_pipeline_smoke_test.cpp \
        native/tests/cli_contract_test.cpp
git commit -m "feat(training): persist Min search-target diagnostics"
```

### Task 4: Add legal-set policy-fit diagnostics to `selfplay_benchmark`

**Files:**
- Create: `native/include/diamond_pipeline/policy_diagnostics.hpp`
- Create: `native/src/policy_diagnostics.cpp`
- Create: `native/tests/policy_diagnostics_test.cpp`
- Modify: `native/benchmarks/selfplay_benchmark.cpp`
- Modify: `native/tests/benchmark_schema_test.cmake`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: full 5,329-action model logits, authoritative root actions, MCTS visit counts
- Produces:

```cpp
namespace diamond_pipeline {

struct PolicyRowDiagnostics {
    double target_entropy = 0.0;
    double full_cross_entropy = 0.0;
    double legal_cross_entropy = 0.0;
    double full_kl = 0.0;
    double legal_kl = 0.0;
    double legal_probability_mass = 0.0;
    bool top1_agrees = false;
};

PolicyRowDiagnostics diagnose_policy_row(
    std::span<const float> full_logits,
    std::span<const int32_t> legal_actions,
    std::span<const uint32_t> visit_counts);

}  // namespace diamond_pipeline
```

The benchmark adds CLI options:

```text
--bootstrap-prior config|vacancy|none
--simulations-late N
--repeat-window N
--max-game-seconds F
--diagnostic-roots N
--diagnostic-batch N
```

And JSON fields:

```json
"domain": {
  "search_targets": { "...": "Task 2 summary" },
  "policy_fit": {
    "sampled_roots": 0,
    "target_entropy_mean": 0.0,
    "full_cross_entropy_mean": 0.0,
    "legal_cross_entropy_mean": 0.0,
    "full_kl_mean": 0.0,
    "legal_kl_mean": 0.0,
    "legal_probability_mass_mean": 0.0,
    "top1_agreement": 0.0
  },
  "repetition": {
    "revisit_fraction_mean": 0.0,
    "repeat_within_8_fraction_mean": 0.0,
    "max_revisits_mean": 0.0,
    "cycling_games": 0
  }
}
```

- [ ] **Step 1: Write failing mathematical tests.**

Use small 5,329-element vectors with only the first four entries varied. Test:

Use the same local `close()` and exception helpers as Task 2:

```cpp
// Uniform logits over the entire action space and target [1,0] over two legal actions.
CHECK(close(row.legal_cross_entropy, std::log(2.0), 1e-6));
CHECK(close(row.legal_kl, std::log(2.0), 1e-6));
CHECK(close(row.legal_probability_mass, 2.0 / 5329.0, 1e-7));

// Identity that must hold when legal probabilities are renormalized.
CHECK(close(row.full_cross_entropy - row.legal_cross_entropy,
            -std::log(row.legal_probability_mass), 1e-6));
CHECK(close(row.full_kl - row.legal_kl,
            -std::log(row.legal_probability_mass), 1e-6));
```

Also reject: wrong logit count, mismatched action/visit widths, duplicate legal actions, out-of-range actions, all-zero visits, and non-finite logits.

- [ ] **Step 2: Run the unit test and verify failure.**

```bash
cmake --build build/native-training --target policy_diagnostics_test -j2
ctest --test-dir build/native-training -R '^policy_diagnostics_test$' --output-on-failure
```

Expected before implementation: compile/link failure.

- [ ] **Step 3: Implement stable full and legal log-softmax calculations.**

Compute `logsumexp` by subtracting the maximum logit. Tie-break both target and network top-1 by the smallest canonical action ID, matching search finalization. Clamp only roundoff-level negative KL to zero:

```cpp
if (kl < 0.0 && kl > -1e-10) kl = 0.0;
```

Throw for larger negative KL because it indicates an implementation error.

- [ ] **Step 4: Extend benchmark parsing without changing old defaults.**

Use `std::optional` for override fields. Rules:

```text
--bootstrap-prior config   inherit checkpoint config; artifact mode means none
--bootstrap-prior vacancy force vacancy prior
--bootstrap-prior none    force network prior
--simulations-late 0      disable adaptive budget
--repeat-window 0         disable repetition trigger
--max-game-seconds 0      disable per-game deadline
--diagnostic-roots 0      skip extra model-forward diagnostics
```

Reject `--repeat-window > 0` when `--simulations-late == 0` and reject `--diagnostic-roots > 0` in artifact mode only if model family is not Min or Soo. Do not silently change temperature or Dirichlet settings.

- [ ] **Step 5: Aggregate repetition diagnostics from every returned raw episode.**

Use the existing `EpisodeDiagnostics` denominator `observations`, and duplicate the already-tested `dominant_cycle_period()` logic into a shared helper if necessary. Do not infer whole-game repetition from the 64-position tail.

- [ ] **Step 6: Evaluate a deterministic subset of roots after self-play.**

Select at most `--diagnostic-roots` moves by deterministic reservoir sampling keyed by `--seed`. Batch root features through the loaded model in FP32. Keep all root actions and visit counts in memory until diagnostics finish; do not write a new replay format.

- [ ] **Step 7: Extend the benchmark contract test.**

Require the six new help options. Run a two-game, one-simulation CPU benchmark with `--diagnostic-roots 1` and require finite/non-negative fields, `legal_probability_mass_mean` in `(0,1]`, and both KL values no smaller than `-1e-10`.

- [ ] **Step 8: Run focused tests.**

```bash
cmake --build build/native-training \
  --target policy_diagnostics_test selfplay_benchmark -j2
ctest --test-dir build/native-training \
  -R '^(policy_diagnostics_test|benchmark_schema_test)$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add native/include/diamond_pipeline/policy_diagnostics.hpp \
        native/src/policy_diagnostics.cpp \
        native/tests/policy_diagnostics_test.cpp \
        native/benchmarks/selfplay_benchmark.cpp \
        native/tests/benchmark_schema_test.cmake \
        native/CMakeLists.txt
git commit -m "feat(benchmark): diagnose Min search targets and legal policy fit"
```

### Task 5: Add an in-memory learner diagnostic executable

**Files:**
- Create: `native/include/diamond_training/parameter_diagnostics.hpp`
- Create: `native/src/parameter_diagnostics.cpp`
- Create: `native/tests/parameter_diagnostics_test.cpp`
- Create: `native/include/diamond_pipeline/learning_diagnostic.hpp`
- Create: `native/src/learning_diagnostic.cpp`
- Create: `native/tests/min_learning_diagnostic_test.cpp`
- Create: `native/benchmarks/min_learning_diagnostic.cpp`
- Modify: `native/tests/benchmark_schema_test.cmake`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: an in-memory `Trainer`, a const `ReplayStore`, and deterministic sampling seeds
- Produces no checkpoint and performs no replay ingest, prune, or manifest write. The CLI loads copied checkpoint/replay artifacts, runs the in-memory core, and emits target entropy, full policy KL, gradient norms, relative update norms, and start-to-end policy drift.

```cpp
namespace diamond_training {

enum class ParameterGroup {
    input_projection,
    residual_trunk,
    last_residual_block,
    output_norm,
    policy_source,
    policy_destination,
    value_hidden,
    value_output,
};

ParameterGroup classify_parameter(std::string_view name, int64_t residual_blocks);
const char* parameter_group_name(ParameterGroup group);

struct GroupNorms {
    double parameter_l2 = 0.0;
    double gradient_l2 = 0.0;
    double update_l2 = 0.0;
    double relative_update = 0.0;
};

}  // namespace diamond_training
```

```cpp
namespace diamond_pipeline {

struct LearningDiagnosticConfig {
    uint64_t iteration = 0;
    std::size_t steps = 0;
    std::size_t batch_size = 0;
    std::size_t evaluation_samples = 0;
    std::size_t evaluation_batch = 0;
    std::size_t log_every = 0;
    uint64_t seed = 0;
};

struct LearningDiagnosticResult {
    // Define typed nested structs for held-out policy/value metrics and
    // per-step grouped norms. Do not expose JSON as the library API.
};

LearningDiagnosticResult run_min_learning_diagnostic(
    diamond_training::Trainer& trainer,
    const ReplayStore& replay,
    const LearningDiagnosticConfig& config);

}  // namespace diamond_pipeline
```

CLI contract:

```text
min_learning_diagnostic
  --checkpoint DIR --config FILE --replay DIR
  --device cpu|cuda|cuda:N
  --iteration N --steps N --batch-size N
  --eval-samples N --eval-batch N --log-every N
  --seed N --out FILE
```

The learner diagnostic deliberately reports **full-action** policy cross-entropy/KL because replay samples retain non-zero MCTS target support but not the complete legal-action set. Legal-set CE/KL and legal probability mass come from Task 4's raw self-play benchmark.

- [ ] **Step 1: Write failing parameter-group tests against actual model names.**

Build a width-8, two-block Min model and assert its real `named_parameters()` classify as follows:

```text
input_projection.*                    -> input_projection
block_0.*                             -> residual_trunk
block_1.*                             -> last_residual_block
output_norm.*                         -> output_norm
policy_source.*                       -> policy_source
policy_destination.*                  -> policy_destination
value_linear1.*                       -> value_hidden
value_linear2.*                       -> value_output
```

Reject unknown parameter names instead of silently putting them in `other`. Use local exception helpers because `native/tests/check.hpp` does not provide `CHECK_THROWS`.

- [ ] **Step 2: Run the classifier test and verify failure.**

```bash
cmake --build build/native-training --target parameter_diagnostics_test -j2
ctest --test-dir build/native-training -R '^parameter_diagnostics_test$' --output-on-failure
```

Expected before implementation: compile/link failure.

- [ ] **Step 3: Implement the classifier and grouped norm calculation.**

Before each diagnostic training step, clone named parameters into a map keyed by parameter name. After `trainer.train(samples)`:

```cpp
gradient_sq += parameter.grad().detach().to(torch::kFloat64).pow(2).sum().item<double>();
update_sq += (parameter.detach() - before).to(torch::kFloat64).pow(2).sum().item<double>();
parameter_sq += parameter.detach().to(torch::kFloat64).pow(2).sum().item<double>();
```

Finalize with square roots and:

```cpp
relative_update = parameter_l2 > 0.0 ? update_l2 / parameter_l2 : 0.0;
```

Treat undefined gradients, non-finite norms, or a missing named group as errors.

- [ ] **Step 4: Write a failing integration test for the in-memory diagnostic.**

`min_learning_diagnostic_test.cpp` must:

1. Create a small width-8, one-block Min model and `Compatibility::min(...)`.
2. Create a scratch replay store and ingest at least eight valid Min samples derived from encoded non-terminal Min states, with two or more policy actions and valid three-component value targets.
3. Save a checkpoint-v3 fixture with explicit lineage/provenance.
4. Record the checkpoint tree digest and replay manifest digest.
5. Construct a second trainer, restore the checkpoint with `exact_resume`, and call `run_min_learning_diagnostic()` for two CPU steps.
6. Assert finite losses and group norms, and non-zero gradients for `policy_source` and `policy_destination` by step 2.
7. Assert the trainer's in-memory training step advanced by two.
8. Assert the checkpoint tree and replay manifest digests are byte-identical after the call.

Run and verify it fails because the API is absent:

```bash
cmake --build build/native-training --target min_learning_diagnostic_test -j2
ctest --test-dir build/native-training -R '^min_learning_diagnostic_test$' --output-on-failure
```

- [ ] **Step 5: Implement `run_min_learning_diagnostic()`.**

Requirements:

1. Require Min compatibility and non-zero steps/batch/evaluation sizes.
2. Draw one fixed held-out set using a seed domain distinct from training minibatches.
3. Evaluate held-out metrics before training:
   - target entropy from `TrainingSample::sparse_policy`;
   - full 5,329-action cross-entropy;
   - full KL = cross-entropy - target entropy;
   - target/network top-1 agreement using smallest-action tie-break;
   - value MSE.
4. For each local training step, draw with `replay_sampling_seed(replay.replay_seed(), iteration, local_step)`, clone parameters, call `Trainer::train()`, and collect grouped gradient/update norms.
5. Retain detailed rows for local steps 1, 2, every `log_every`, and the final step.
6. Re-evaluate the unchanged held-out set and emit:
   - final held-out metrics;
   - start-to-end policy KL `KL(p_start || p_end)` over all 5,329 actions;
   - top-1 agreement change;
   - logit RMS delta;
   - value RMS delta;
   - policy/value loss deltas.
7. Never call `ReplayStore::ingest*`, `ReplayStore::prune`, or any checkpoint save function.

Use batched tensor operations. Do not copy all 5,329 logits to CPU one row at a time.

- [ ] **Step 6: Implement the CLI wrapper.**

The executable must:

1. Parse the resolved config and require model family Min.
2. Construct model dimensions from config, not hard-coded width/block counts.
3. Construct `Trainer` and call `load_checkpoint_v3(..., exact_resume)` so AdamW state is restored in memory.
4. Open `ReplayStore` with `ReplayContents::full` and pass it as const to the core.
5. Include checkpoint training step, model digest, replay size/manifest digest, config path, source commit, device, and exact options in the JSON preamble.
6. Serialize the typed result to canonical JSON.
7. Exit without writing any file except the explicit `--out` report.

- [ ] **Step 7: Register and test the executable contract.**

Add `learning_diagnostic.cpp` to `diamond_native_pipeline`; add `parameter_diagnostics.cpp` to `diamond_training`; register both tests and the benchmark executable. Extend `benchmark_schema_test.cmake` only to require that `min_learning_diagnostic --help` succeeds and lists every CLI option above. The no-mutation and numeric contract lives in `min_learning_diagnostic_test.cpp`, not in CMake scripting.

- [ ] **Step 8: Run focused tests.**

```bash
cmake --build build/native-training \
  --target parameter_diagnostics_test min_learning_diagnostic_test \
           min_learning_diagnostic -j2
ctest --test-dir build/native-training \
  -R '^(parameter_diagnostics_test|min_learning_diagnostic_test|benchmark_schema_test)$' \
  --output-on-failure
```

Expected: PASS and no source artifact changes.

- [ ] **Step 9: Commit.**

```bash
git add native/include/diamond_training/parameter_diagnostics.hpp \
        native/src/parameter_diagnostics.cpp \
        native/tests/parameter_diagnostics_test.cpp \
        native/include/diamond_pipeline/learning_diagnostic.hpp \
        native/src/learning_diagnostic.cpp \
        native/tests/min_learning_diagnostic_test.cpp \
        native/benchmarks/min_learning_diagnostic.cpp \
        native/tests/benchmark_schema_test.cmake \
        native/CMakeLists.txt
git commit -m "feat(training): add a read-only Min learning diagnostic"
```

### Task 6: Verify PR 1 and document exactly what it does not change

**Files:**
- Modify: `docs/superpowers/reports/2026-09-01-min-a0-baseline.md`

**Interfaces:**
- Consumes: Tasks 2–5
- Produces: an observability-only PR ready for review

- [ ] **Step 1: Configure the native training build.**

```bash
cd /workspace/alphadiamond-min-a0
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
cmake --preset native-training -G Ninja \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DDIAMOND_BUILD_BENCHMARKS=ON
```

If `CMAKE_PREFIX_PATH` is required on the host, pass the installed LibTorch CMake directory already used by the repository's native build.

- [ ] **Step 2: Build and run the complete suite.**

```bash
cmake --build build/native-training -j2
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

Expected: every test registered on this host passes. CUDA-labelled tests run only when the host/build enables them.

- [ ] **Step 3: Prove serial search outputs are unchanged.**

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training \
  -R '^(mcts3p_golden_test|mcts_golden_test|selfplay_test)$' \
  --output-on-failure
```

Expected: all frozen golden digests and trajectories remain unchanged.

- [ ] **Step 4: Append verification and scope to the baseline report.**

Record:

```text
- Full test count and result
- Base SHA and head SHA
- The new JSON schema fields
- Explicit statement: no config defaults, search selection, replay bytes,
  checkpoint bytes, optimizer behavior, or production run state changed
```

- [ ] **Step 5: Commit and push PR 1.**

```bash
git add docs/superpowers/reports/2026-09-01-min-a0-baseline.md
git commit -m "docs(min): record learning-diagnostic verification"
git push -u origin codex/min-a0-diagnostics
gh pr create \
  --repo ubraket513/alphadiamond \
  --base main \
  --head codex/min-a0-diagnostics \
  --title "Add observability for the Min A0 transition" \
  --body-file docs/superpowers/reports/2026-09-01-min-a0-baseline.md
```

Expected: PR 1 contains observability only. Do not add adaptive search or parallel MCTS to this PR.

---

## Phase 2: Enable and Measure Stronger Serial Search

### Task 7: Make adaptive simulation budgets work for Min's 3P search

**Files:**
- Modify: `native/include/soo/mcts3p.hpp`
- Modify: `native/include/soo/episode_search.hpp`
- Modify: `native/tests/mcts3p_golden_test.cpp`
- Modify: `native/tests/selfplay_test.cpp`
- Modify: `native/benchmarks/selfplay_benchmark.cpp`
- Modify: `native/tests/benchmark_schema_test.cmake`

**Interfaces:**
- Consumes: existing `EpisodeConfig.simulations_late`, `repeat_window`, and `selfplay.cpp::simulations_for()`
- Produces:

```cpp
void SearchSession3P::set_simulations(int simulations);
```

`EpisodeSearch::set_simulations()` forwards to both 2P and 3P sessions and updates its cached value.

- [ ] **Step 1: Create a new branch after PR 1 is merged or rebase onto its accepted head.**

```bash
git fetch origin --prune
git switch -c codex/min-3p-adaptive-search origin/main
```

If PR 1 is not merged, branch from its reviewed head and state that dependency in PR 2.

- [ ] **Step 2: Add failing direct-budget tests.**

In `mcts3p_golden_test.cpp`, retain every frozen case unchanged and add a non-golden contract case:

```cpp
soo::SearchSession3P session(match, config_with_4_sims);
session.set_simulations(9);
run_to_completion(session, state, evaluator);
CHECK_EQ(session.result().simulations_run, uint32_t{9});
CHECK(throws_invalid_argument([&] { session.set_simulations(0); }));
CHECK(throws_invalid_argument([&] { session.set_simulations(-1); }));
```

Run:

```bash
cmake --build build/native-training --target mcts3p_golden_test -j2
ctest --test-dir build/native-training -R '^mcts3p_golden_test$' --output-on-failure
```

Expected: compile failure because 3P has no setter.

- [ ] **Step 3: Implement the setter.**

```cpp
void set_simulations(int simulations) {
    if (simulations <= 0)
        throw std::invalid_argument("simulations must be positive");
    config_.simulations = simulations;
}
```

In `EpisodeSearch::set_simulations()`:

```cpp
if (two_)
    two_->set_simulations(simulations);
else
    three_->set_simulations(simulations);
simulations_ = simulations;
```

Do not recreate a tree mid-search. The setter is called only before `begin()` for the next move.

- [ ] **Step 4: Add a behavioral self-play test for the 3P repetition trigger.**

Use a deterministic Min evaluator and a short job known to revisit a state. Compare identical jobs/seeds with:

```cpp
baseline.simulations = 4;
baseline.simulations_late = 0;
baseline.repeat_window = 0;

adaptive = baseline;
adaptive.simulations_late = 9;
adaptive.repeat_window = 8;
```

Assert:

```cpp
CHECK(adaptive_metrics.boosted_moves > 0);
CHECK(adaptive_metrics.evaluations > baseline_metrics.evaluations);
CHECK_EQ(adaptive_episodes.size(), baseline_episodes.size());
```

Then deliberately restore the old throwing branch locally, confirm the test fails, and restore the implementation.

- [ ] **Step 5: Wire benchmark CLI fields added in Task 4 to `EpisodeConfig`.**

Require the JSON workload to report `simulations_late`, `repeat_window`, `boosted_moves`, and `boosted_fraction`.

- [ ] **Step 6: Run all search and scheduler tests.**

```bash
cmake --build build/native-training \
  --target mcts3p_golden_test selfplay_test selfplay_benchmark -j2
ctest --test-dir build/native-training \
  -R '^(mcts3p_golden_test|mcts_golden_test|selfplay_test|budget_test|benchmark_schema_test)$' \
  --output-on-failure
```

Expected: existing serial golden output remains exact; adaptive 3P test passes.

- [ ] **Step 7: Commit.**

```bash
git add native/include/soo/mcts3p.hpp \
        native/include/soo/episode_search.hpp \
        native/tests/mcts3p_golden_test.cpp \
        native/tests/selfplay_test.cpp \
        native/benchmarks/selfplay_benchmark.cpp \
        native/tests/benchmark_schema_test.cmake
git commit -m "feat(search): enable adaptive simulation budgets for Min"
```

### Task 8: Run the read-only learner diagnostic on Min v1.0.1

**Files:**
- Create: `docs/superpowers/reports/2026-09-01-min-learning-diagnostic.md`
- Create outside repository: `$MIN_A0_EXP/results/learning.json`

**Interfaces:**
- Consumes: `min_learning_diagnostic`, verified release checkpoint/replay/config
- Produces: classification evidence for target-limited, optimizer-limited, disconnected-head, or illegal-mass-dominated learning

- [ ] **Step 1: Run a 2-step smoke.**

```bash
source /workspace/alphadiamond-experiments/min-v1.0.1/env.sh
BIN=/workspace/alphadiamond-min-a0/build/native-training/native/min_learning_diagnostic
PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

LD_PRELOAD="$PRELOAD" "$BIN" \
  --checkpoint "$MIN_A0_CHECKPOINT" \
  --config "$MIN_A0_CONFIG" \
  --replay "$MIN_A0_REPLAY" \
  --device cuda \
  --iteration 26 \
  --steps 2 \
  --batch-size 256 \
  --eval-samples 512 \
  --eval-batch 256 \
  --log-every 1 \
  --seed 20260901 \
  --out "$MIN_A0_EXP/results/learning-smoke.json"
```

Expected: no mutation error, finite losses, and non-zero policy-head gradients by step 2.

- [ ] **Step 2: Verify the source artifacts are unchanged.**

```bash
(
  cd "$MIN_A0_EXP/downloads"
  sha256sum -c SHA256SUMS
)
```

Expected: all release archives still pass. Also compare checkpoint/replay file hashes recorded by the diagnostic's no-mutation preamble.

- [ ] **Step 3: Run the 256-step diagnostic.**

```bash
LD_PRELOAD="$PRELOAD" "$BIN" \
  --checkpoint "$MIN_A0_CHECKPOINT" \
  --config "$MIN_A0_CONFIG" \
  --replay "$MIN_A0_REPLAY" \
  --device cuda \
  --iteration 26 \
  --steps 256 \
  --batch-size 256 \
  --eval-samples 4096 \
  --eval-batch 256 \
  --log-every 32 \
  --seed 20260901 \
  --out "$MIN_A0_EXP/results/learning.json"
```

- [ ] **Step 4: Classify the result using these rules.**

Apply rules in order:

```text
1. Any non-finite loss, gradient, update, KL, or output:
   classify as LEARNER_CORRECTNESS_FAILURE and stop all search experiments.

2. policy_source or policy_destination gradient_l2 == 0 for two consecutive
   reported steps after step 1:
   classify as DISCONNECTED_POLICY_HEAD and stop all search experiments.

3. Mean relative update for both policy heads < 1e-8 and held-out start-to-end
   policy KL < 1e-5 over 256 steps:
   classify as EFFECTIVE_STEP_TOO_SMALL; inspect optimizer/LR before search work.

4. Full policy KL = full_cross_entropy - target_entropy <= 0.10 nat while
   target normalized entropy is high:
   classify as TARGET_LIMITED; the model is already close to the supplied target.

5. Policy-head gradients and updates are healthy, full KL remains materially
   positive, and held-out policy KL moves:
   classify as LEARNING_BUT_NOT_BEHAVIORALLY_READY; continue to serial search sweep.
```

The numeric thresholds are engineering gates, not scientific constants. Report raw distributions alongside the classification.

- [ ] **Step 5: Write and commit the diagnostic report.**

Include the exact command, input hashes, target entropy, policy/value losses, every parameter-group gradient/update summary, and start-to-end drift.

```bash
git add docs/superpowers/reports/2026-09-01-min-learning-diagnostic.md
git commit -m "docs(min): classify the iteration-25 learning plateau"
```

### Task 9: Run the paired serial-search matrix

**Files:**
- Create: `tools/run_min_serial_search_sweep.sh`
- Create: `tools/summarize_min_serial_search.py`
- Create: `docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md`
- Modify: the repository's existing script-policy/hygiene test only when that test explicitly requires registration of new files; otherwise do not touch policy files

**Interfaces:**
- Consumes: verified Min v1.0.1 checkpoint/config, Task 4 benchmark, Task 7 adaptive 3P search
- Produces: paired 128/256/400 B0 and A0 evidence plus adaptive-search arms

The eight arms are fixed:

```text
b0-128          vacancy prior, constant 128
b0-256          vacancy prior, constant 256
b0-400          vacancy prior, constant 400
a0-128          network prior, constant 128
a0-256          network prior, constant 256
a0-400          network prior, constant 400
a0-adaptive-256 network prior, base 128, repeat-triggered 256, window 8
a0-adaptive-400 network prior, base 128, repeat-triggered 400, window 8
```

- [ ] **Step 1: Write the sweep script with fixed common arguments.**

The script starts with:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "${MIN_A0_ENV:-/workspace/alphadiamond-experiments/min-v1.0.1/env.sh}"
BIN=${MIN_A0_SELFPLAY_BIN:-/workspace/alphadiamond-min-a0/build/native-training/native/selfplay_benchmark}
OUT=${MIN_A0_SWEEP_OUT:-$MIN_A0_EXP/results/serial-search}
PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
mkdir -p "$OUT"

common=(
  --checkpoint "$MIN_A0_CHECKPOINT"
  --config "$MIN_A0_CONFIG"
  --device cuda
  --precision fp32
  --threads 16
  --max-batch 256
  --max-wait-us 100
  --max-moves 800
  --max-game-seconds 180
  --temperature 1.0
  --temperature-moves 20
  --dirichlet-epsilon 0.25
  --diagnostic-roots 4096
  --diagnostic-batch 256
  --warmups 0
  --repetitions 1
  --seed 20260901
)
```

Provide a `run_arm name games lanes prior simulations late repeat` function that writes stdout atomically to `$OUT/$scale/$name.json.tmp` and renames only on exit code 0.

- [ ] **Step 2: Add script contract tests or a `--dry-run` mode.**

`--dry-run smoke` must print exactly eight commands and show identical seed/common arguments. Assert the adaptive arms alone have non-zero `simulations_late` and `repeat_window`.

- [ ] **Step 3: Run the 32-game smoke matrix.**

```bash
MIN_A0_SWEEP_SCALE=smoke \
MIN_A0_GAMES=32 \
MIN_A0_LANES=32 \
  tools/run_min_serial_search_sweep.sh
```

Every arm uses the same 32 game seeds. Do not run arms concurrently on one GPU.

- [ ] **Step 4: Apply the smoke kill gate.**

Always retain all three B0 arms. Scale an A0 arm to 256 games only when at least one condition holds:

```text
- completion >= 50%, or
- completion improves by >= 10 percentage points over a0-128, or
- mean repeat-within-8 fraction is <= 50% of a0-128, or
- cycling games are <= 50% of a0-128.
```

If no A0 arm qualifies, stop the sweep after smoke and proceed directly to Task 10's “no deeper-search benefit” branch.

- [ ] **Step 5: Run the 256-game matrix for qualifying arms.**

```bash
MIN_A0_SWEEP_SCALE=full \
MIN_A0_GAMES=256 \
MIN_A0_LANES=128 \
MIN_A0_ARM_FILTER='comma-separated qualifying arms plus b0-128,b0-256,b0-400' \
  tools/run_min_serial_search_sweep.sh
```

- [ ] **Step 6: Implement the deterministic summary script.**

`tools/summarize_min_serial_search.py INPUT_DIR OUTPUT_MD` must:

1. Reject mixed base commits, model digests, seeds, precision, or game counts within one scale.
2. Report completion, abort causes, p50/p90/p99/max moves, revisit fraction, repeat-within-8, max revisits, cycling games, samples/hour, evaluations/second, batch mean/p50/p90, target entropy, normalized entropy, full/legal KL, legal probability mass, top-1 agreement, boosted fraction, and wall time.
3. Compare every arm against `b0-128` and every A0 arm against `a0-128` when present.
4. Print percentage changes with explicit denominator.
5. Emit the decision classification from Task 10 but never alter code or config.

- [ ] **Step 7: Generate the report.**

```bash
python3 tools/summarize_min_serial_search.py \
  "$MIN_A0_EXP/results/serial-search" \
  docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md
```

- [ ] **Step 8: Commit scripts and the report.**

```bash
git add tools/run_min_serial_search_sweep.sh \
        tools/summarize_min_serial_search.py \
        docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md
git commit -m "bench(min): compare serial MCTS budgets at the A0 gate"
```

### Task 10: Make the evidence-backed branch decision

**Files:**
- Modify: `docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md`
- Create: `docs/superpowers/specs/2026-09-01-min-next-intervention.md`

**Interfaces:**
- Consumes: learner diagnostic and serial-search sweep
- Produces: exactly one approved implementation branch

- [ ] **Step 1: Check whether the existing policy-loss plateau is mainly a reporting artifact.**

The trainer currently computes full-action cross-entropy over all 5,329 actions, while inference softmax is restricted to the authoritative legal set. Use the benchmark relation:

```text
full_cross_entropy = legal_cross_entropy - log(legal_probability_mass)
full_KL            = legal_KL            - log(legal_probability_mass)
```

Classify as `ILLEGAL_MASS_DOMINATED` when all are true:

```text
- legal_KL <= 0.10 nat,
- full_KL >= legal_KL + 0.50 nat,
- legal_probability_mass <= exp(-0.50) ~= 0.607.
```

In that case, do not claim policy learning is stalled solely from the reported full policy loss. Keep search experiments, but schedule a separate legal-mask training-loss design before changing optimizer settings.

- [ ] **Step 2: Determine whether stronger serial search improves A0.**

Classify `DEEPER_SEARCH_HELPS` when a 256- or 400-simulation A0 arm meets all applicable conditions against `a0-128`:

```text
- completion improves by at least 10 percentage points, or reaches >= 90%;
- repeat-within-8 drops by at least 30%;
- cycling games do not increase;
- legal policy KL does not worsen by more than 10%;
- no new non-finite inference or deadline-abort failure appears.
```

- [ ] **Step 3: Determine whether adaptive search captures the benefit cheaply.**

Classify `ADAPTIVE_SEARCH_WINS` when `a0-adaptive-400` or `a0-adaptive-256`:

```text
- reaches within 5 percentage points of the matching constant-budget completion;
- has p90 moves no worse than 10% above the constant-budget arm;
- has repeat-within-8 no worse than 10% above the constant-budget arm;
- uses boosted_fraction <= 0.15;
- improves samples/hour by at least 25% over the matching constant-budget arm.
```

- [ ] **Step 4: Select exactly one next branch.**

Use this ordered decision table:

```text
A. Learner correctness failure or disconnected policy head
   -> Stop. Fix learner correctness. Do not implement parallel MCTS or annealing.

B. Adaptive search wins
   -> Adopt repetition-triggered serial search first. Do not implement parallel MCTS yet.

C. Deeper search helps, adaptive search does not capture it, and constant 256/400 costs
   at least 1.5x a0-128 wall time
   -> Authorize Phase 3A: parallel MCTS.

D. Deeper search does not help materially, but learner path is healthy
   -> Authorize Phase 3B: vacancy-prior annealing.

E. A0 reaches >=97% completion over 256 games with cycling <=1% and no seat/turn-order
   anomaly under an existing serial arm
   -> Do not implement either intervention. Run the two-checkpoint A0 acceptance gate.
```

- [ ] **Step 5: Write the decision spec.**

`docs/superpowers/specs/2026-09-01-min-next-intervention.md` must include:

```text
- Selected branch and rejected branches
- Raw gate values, not just pass/fail
- Base/model/config/replay digests
- Expected behavioral improvement
- Rollback condition
- Exact production fields that may change
- Explicit statement that Min remains B0 until the final acceptance gate passes
```

- [ ] **Step 6: Commit and open PR 2.**

```bash
git add docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md \
        docs/superpowers/specs/2026-09-01-min-next-intervention.md
git commit -m "docs(min): select the next A0 intervention from measured gates"
git push -u origin codex/min-3p-adaptive-search
gh pr create \
  --repo ubraket513/alphadiamond \
  --base main \
  --head codex/min-3p-adaptive-search \
  --title "Measure stronger serial search for the Min A0 transition" \
  --body-file docs/superpowers/reports/2026-09-01-min-serial-search-sweep.md
```

---

## Phase 3A: Conditional Parallel MCTS Implementation

> Execute this phase only when Task 10 selects branch C. Otherwise skip every task in this phase.

### Task 11A: Specify batched 3P leaf selection with virtual visits

**Files:**
- Create: `docs/superpowers/specs/2026-09-01-min-parallel-mcts-design.md`

**Interfaces:**
- Consumes: Task 10 evidence that deeper serial search improves A0
- Produces: a reviewable state-machine contract before code changes

- [ ] **Step 1: Write the state-machine contract with these exact invariants.**

```text
- parallel_leaves is a positive integer and defaults to 1.
- parallel_leaves=1 follows the existing request sequence and produces bit-exact
  root actions, visits, policy, Q vectors, selected action, evaluator-call count,
  and simulation count.
- Q uses completed visits and completed value sums only.
- PUCT exploration counts completed + in-flight visits for parent and edge N.
- No scalar or seat-specific virtual value is added.
- A leaf node may have at most one expansion evaluation in flight.
- Every reservation has a unique token and an owned path.
- Every supplied or cancelled token releases exactly its virtual visits.
- Exactly config.simulations completed simulations contribute to final visits.
- Root expansion remains a single request before simulation waves begin.
- Dirichlet noise is sampled once at root and is independent of parallel width.
- Results are finalized only when there are no outstanding tokens.
```

- [ ] **Step 2: Define the public API in the spec.**

```cpp
struct PendingEvaluation3P {
    uint32_t token = 0;
    const State* state = nullptr;
    const Encoded* encoded = nullptr;
    const std::vector<int32_t>* actions = nullptr;
};

class SearchSession3P {
  public:
    enum class BatchStatus : uint8_t { NeedsEvaluation, Ready };
    BatchStatus advance_batch(int max_pending);
    std::span<const PendingEvaluation3P> pending_evaluations() const;
    void supply(uint32_t token, const EvalOutcome3P& outcome);
};
```

The old `advance()/pending_*/supply(outcome)` API remains as a width-1 adapter until every existing caller and golden test stays green.

- [ ] **Step 3: Define scheduler ticketing.**

```cpp
struct EvaluationTicket {
    int lane = -1;
    uint16_t slot = 0;
};
```

The batcher queues tickets, not lane IDs. `slot` indexes stable lane-owned pending request storage.

- [ ] **Step 4: Commit the design only.**

```bash
git add docs/superpowers/specs/2026-09-01-min-parallel-mcts-design.md
git commit -m "docs(search): specify parallel 3P MCTS semantics"
```

### Task 12A: Implement the parallel 3P search core under width-1 compatibility

**Files:**
- Modify: `native/include/soo/mcts3p.hpp`
- Modify: `native/src/mcts3p.cpp`
- Create: `native/tests/mcts3p_parallel_test.cpp`
- Modify: `native/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 11A API and invariants
- Produces: multiple outstanding leaf evaluations in one `SearchSession3P`

Internal structures:

```cpp
struct PendingSimulation {
    uint32_t token = 0;
    uint32_t node = 0;
    std::vector<std::pair<uint32_t, uint32_t>> path;
    State state;
    Encoded encoded;
    std::vector<int32_t> actions;
    std::optional<EvalOutcome3P> outcome;
};

struct VectorEdge {
    // existing fields
    uint16_t in_flight = 0;
};

struct VectorNode {
    // existing fields
    uint16_t in_flight = 0;
    bool evaluation_pending = false;
};
```

- [ ] **Step 1: Write failing width-1 parity tests.**

For every frozen 3P golden case, drive both the legacy width-1 API and `advance_batch(1)` with the same evaluator and assert every result field and request sequence is identical.

- [ ] **Step 2: Write failing parallel invariants tests.**

Add deterministic cases for:

```text
- width 4 returns four distinct tokens when the tree has alternatives;
- one unexpanded leaf is never returned twice in the same wave;
- virtual edge/node visits are fully released after supply;
- out-of-order token supply is accepted and deterministic after all outcomes arrive;
- unknown, duplicate, and already-supplied tokens are rejected;
- terminal simulations consume budget without evaluator requests;
- final root visit sum equals configured simulations;
- vector backup updates all three completed value sums but virtual visits update none;
- begin() rejects or clears no outstanding work only after prior search is Ready.
```

- [ ] **Step 3: Run the tests and verify failure.**

```bash
cmake --build build/native-training --target mcts3p_parallel_test -j2
ctest --test-dir build/native-training -R '^mcts3p_parallel_test$' --output-on-failure
```

- [ ] **Step 4: Implement reservation and batch advancement.**

Selection score uses:

```cpp
const uint32_t effective_parent = node.total_visits + node.in_flight;
const uint32_t effective_edge = edge.visits + edge.in_flight;
const double score = edge.q(node.seat) +
    exploration_bonus(edge.prior, effective_parent, effective_edge, config_.c_puct);
```

When reserving a path, increment node/edge `in_flight` once. On supply, release virtual counts before completed backup. If expansion produced a terminal child, release and back up synchronously without creating a pending token.

- [ ] **Step 5: Preserve the width-1 adapter.**

Implement old `advance()` by calling `advance_batch(1)`, exposing the single pending row through existing accessors, and mapping old `supply(outcome)` to its token. Do not special-case a different selection algorithm.

- [ ] **Step 6: Run core tests.**

```bash
cmake --build build/native-training \
  --target mcts3p_parallel_test mcts3p_golden_test budget_test -j2
ctest --test-dir build/native-training \
  -R '^(mcts3p_parallel_test|mcts3p_golden_test|budget_test)$' \
  --output-on-failure
```

Expected: new invariants pass and all old golden digests remain exact.

- [ ] **Step 7: Commit.**

```bash
git add native/include/soo/mcts3p.hpp native/src/mcts3p.cpp \
        native/tests/mcts3p_parallel_test.cpp native/CMakeLists.txt
git commit -m "feat(search): add batched leaf selection to 3P MCTS"
```

### Task 13A: Generalize the scheduler from lane IDs to evaluation tickets

**Files:**
- Modify: `native/include/soo/batcher.hpp`
- Modify: `native/src/batcher.cpp`
- Modify: `native/include/soo/selfplay.hpp`
- Modify: `native/src/selfplay.cpp`
- Modify: `native/include/soo/episode_search.hpp`
- Modify: `native/tests/batcher_test.cpp`
- Modify: `native/tests/selfplay_test.cpp`

**Interfaces:**
- Consumes: Task 12A batched `SearchSession3P`
- Produces: multiple outstanding evaluator rows per Min lane while preserving one outstanding row for Soo

- [ ] **Step 1: Write failing batcher ticket tests.**

Assert FIFO collection of `(lane, slot)` pairs, repeated lane IDs with distinct slots, bounded collection, stop-and-drain behavior, and no loss/duplication under concurrent submitters.

- [ ] **Step 2: Write failing scheduler tests.**

Cover:

```text
- Min parallel_leaves=1 reproduces grouped and solitary trajectories exactly;
- Min parallel_leaves=4 completes without deadlock when lanes < max_batch;
- multiple slots from one lane enter one evaluator batch;
- evaluator replies delivered out of ticket order still route to the correct slot;
- a lane is requeued only after every ticket in its current wave has returned;
- deadline/cancellation releases all pending reservations;
- 2P rejects parallel_leaves > 1 with a clear error or forces exactly 1 by contract;
- BatchItem.job remains the episode job index, not lane or slot.
```

- [ ] **Step 3: Change `Batcher` storage and API.**

```cpp
void submit(EvaluationTicket ticket);
bool collect(std::vector<EvaluationTicket>& batch);
```

Update depth/wakeup accounting without changing wait semantics.

- [ ] **Step 4: Add `parallel_leaves` to `EpisodeConfig`, defaulting to 1.**

Do not add it to `ProductionConfig` in this task. It is benchmark-only until accepted.

- [ ] **Step 5: Give each Min lane stable pending storage.**

Reserve `parallel_leaves` slots when seating a lane so pointers in `BatchItem` remain valid. Each slot owns its outcome and token. Submit one ticket per pending evaluation. After the evaluator fills outcomes, call `supply(token, outcome)` for all returned slots, then return the lane to the ready queue once.

- [ ] **Step 6: Run scheduler tests under sanitizers if available.**

```bash
cmake --build build/native-training --target batcher_test selfplay_test -j2
ctest --test-dir build/native-training \
  -R '^(batcher_test|selfplay_test|mcts3p_parallel_test|mcts3p_golden_test)$' \
  --output-on-failure
```

- [ ] **Step 7: Commit.**

```bash
git add native/include/soo/batcher.hpp native/src/batcher.cpp \
        native/include/soo/selfplay.hpp native/src/selfplay.cpp \
        native/include/soo/episode_search.hpp \
        native/tests/batcher_test.cpp native/tests/selfplay_test.cpp
git commit -m "feat(selfplay): batch multiple Min leaves per lane"
```

### Task 14A: Benchmark parallel width and accept only a quality-preserving operating point

**Files:**
- Modify: `native/benchmarks/selfplay_benchmark.cpp`
- Modify: `native/tests/benchmark_schema_test.cmake`
- Create: `tools/run_min_parallel_search_sweep.sh`
- Create: `docs/superpowers/reports/2026-09-01-min-parallel-mcts-sweep.md`

**Interfaces:**
- Consumes: parallel search and scheduler
- Produces: accepted width or explicit rejection

- [ ] **Step 1: Add benchmark-only `--parallel-leaves N`.**

Report it in workload JSON along with:

```text
pending_rows_per_wave_mean
pending_rows_per_wave_p50
pending_rows_per_wave_p90
single_lane_batch_fraction
```

- [ ] **Step 2: Run width-1 schema and golden regression tests.**

`--parallel-leaves 1` must remain the default and produce the same deterministic trace as before on a fixed CPU fixture.

- [ ] **Step 3: Run the fixed grid on the same Min release and seeds.**

```text
serial references: width 1 at 128, 256, 400 simulations
parallel candidates: width 2, 4, 8 at 256 and 400 simulations
prior modes: vacancy and none
smoke: 32 games, 32 lanes
full: qualifying arms, 256 games, 128 lanes
```

Keep all common arguments identical to Task 9.

- [ ] **Step 4: Apply the adoption gate.**

Accept a parallel arm only when all hold:

```text
- A0 completion is no worse than 5 percentage points below serial at the same budget;
- p90 moves is no worse than 10% above serial at the same budget;
- repeat-within-8 and cycling games are no worse than 10% above serial;
- legal policy KL is no worse than 10% above serial;
- samples/hour improves by at least 25%;
- no deadlock, deadline regression, non-finite inference, or simulation-accounting error;
- width-1 remains bit-exact.
```

The preferred win is `parallel 400` reaching serial-400 behavior near serial-128 wall time. Merely making parallel-128 faster is insufficient if search quality degrades.

- [ ] **Step 5: If accepted, add the production config field in a separate commit.**

Modify config parser/serializer/tests with:

```json
"mcts": {
  "parallel_leaves": 1
}
```

Allow only positive integers, default 1 for old configs, include it in resolved config and checkpoint provenance, and add it to the explicit config-transition allow-list only after the benchmark report names the accepted value.

- [ ] **Step 6: If rejected, remove benchmark-only production wiring but retain no unused core code without an explicit research flag.**

A rejected implementation may remain only when `parallel_leaves=1` is default, tests prove it, and the report documents why it is retained. Otherwise revert the feature commits and keep the design/report as the negative result.

- [ ] **Step 7: Run the complete suite and open the parallel-MCTS PR.**

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

Commit the report and create a PR whose body includes the gate table and exact serial references.

---

## Phase 3B: Conditional Vacancy-Prior Annealing

> Execute this phase only when Task 10 selects branch D. Otherwise skip every task in this phase.

### Task 11B: Replace the boolean bootstrap substitution with a gated legal-set blend

**Files:**
- Modify: `native/include/soo/selfplay.hpp`
- Modify: `native/src/selfplay.cpp`
- Modify: `native/include/diamond_orchestration/config.hpp`
- Modify: `native/src/config.cpp`
- Modify: `native/src/training_wiring.cpp`
- Modify: `native/tests/selfplay_test.cpp`
- Modify: `native/tests/config_test.cpp`
- Modify: `native/tests/cli_contract_test.cpp`
- Modify: `configs/alphazero/min-bootstrap.json` only after tests pass

**Interfaces:**
- Consumes: normalized network prior and normalized vacancy prior over the same authoritative legal-action vector
- Produces:

```cpp
struct EpisodeConfig {
    bool bootstrap_prior = false;
    double bootstrap_prior_weight = 1.0;
};
```

Resolved config field:

```json
"self_play": {
  "bootstrap_prior": "canonical-target-vacancy-distance-v2",
  "bootstrap_prior_weight": 1.0
}
```

- [ ] **Step 1: Write failing exact-endpoint tests.**

For identical evaluator/jobs/seeds with temperature and Dirichlet zero:

```text
weight 1.0 -> move-for-move identical to current vacancy bootstrap
weight 0.0 -> move-for-move identical to bootstrap_prior none/network prior
weight 0.5 -> each supplied prior equals 0.5*vacancy + 0.5*network
```

Test invalid negative, greater-than-one, NaN, and infinity values.

- [ ] **Step 2: Implement legal-set normalization and blending.**

Both input distributions must already align with `pending_actions`. Validate finite, non-negative, positive sum, normalize each independently, then:

```cpp
mixed[i] = alpha * vacancy[i] + (1.0 - alpha) * network[i];
```

Renormalize once after blending to absorb floating-point roundoff. Do not blend logits. Do not change the value vector.

- [ ] **Step 3: Preserve compatibility.**

Old configs without the new field resolve to weight 1.0 when bootstrap prior is vacancy and 0.0 when bootstrap prior is none. Serialization always emits the field so future run provenance is explicit.

- [ ] **Step 4: Add a config-transition gate.**

Allow decreasing the weight at durable iteration boundaries without rewriting previous resolved config bytes. Reject increases during an A0 transition unless an explicit rollback record names the failed gate.

- [ ] **Step 5: Run focused and full tests.**

```bash
cmake --build build/native-training --target selfplay_test config_test cli_contract_test -j2
ctest --test-dir build/native-training \
  -R '^(selfplay_test|config_test|cli_contract_test|mcts3p_golden_test)$' \
  --output-on-failure
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

- [ ] **Step 6: Commit.**

```bash
git add native/include/soo/selfplay.hpp native/src/selfplay.cpp \
        native/include/diamond_orchestration/config.hpp native/src/config.cpp \
        native/src/training_wiring.cpp \
        native/tests/selfplay_test.cpp native/tests/config_test.cpp \
        native/tests/cli_contract_test.cpp \
        configs/alphazero/min-bootstrap.json
git commit -m "feat(training): anneal Min's vacancy bootstrap prior"
```

### Task 12B: Run an annealing curriculum as short, reversible segments

**Files:**
- Create: `docs/superpowers/reports/2026-09-01-min-prior-annealing.md`
- Create outside repository: copied experiment run directories for each alpha

**Interfaces:**
- Consumes: accepted serial search operating point from Task 10, Min v1.0.1 checkpoint/replay
- Produces: an A0-ready checkpoint or a documented rollback

- [ ] **Step 1: Create a new experiment run from copied release state.**

Do not point the trainer at the released archive tree directly. Copy checkpoint/replay/run state into a new run directory and record checksums before first resume.

- [ ] **Step 2: Use this fixed stage order.**

```text
alpha 1.00: one confirmation iteration
alpha 0.75: two iterations
alpha 0.50: two iterations
alpha 0.25: two iterations
alpha 0.00: bounded probe only; no long run until it passes
```

Use the serial simulation mode selected by Task 10. Keep replay capacity, learner steps, FP32, seeds, and all non-prior fields unchanged.

- [ ] **Step 3: Gate every transition.**

Before lowering alpha require, over the newest 256-game probe:

```text
- completion >= 97%;
- repeat-within-8 <= previous stage * 1.25;
- p90 moves <= previous stage * 1.20;
- cycling games <= 1%;
- no seat first-finisher count differs from 1/3 by more than 10 percentage points;
- no non-finite inference or deadline abort;
- policy-head gradients remain non-zero and held-out policy KL moves.
```

Failure means restore the previous checkpoint/config transition and stop. Do not compensate by increasing max moves alone.

- [ ] **Step 4: Run the alpha-zero acceptance gate.**

Require two consecutive checkpoints, each with:

```text
- 768 no-prior games at intended search settings;
- completion >= 97%;
- cycling <= 1%;
- stable p90/p99 tails;
- no seat/turn-order block below 95% completion;
- a complete paired arena block against the previous checkpoint.
```

Only then classify the run as A0 and update production defaults in a separate release PR.

- [ ] **Step 5: Record every transition and rollback in the report.**

Include config digests, checkpoint digests, replay digest, exact commands, per-stage gates, and the final accepted or rolled-back alpha.

---

## Phase 4: Final A0 Acceptance and Production Resume

### Task 15: Resume production only after a two-checkpoint A0 gate

**Files:**
- Modify: `docs/model-training/min_bootstrap.md`
- Modify: `docs/performance-profiling/min_autopilot_log_2026-08-31.md` or create a new dated continuation log
- Modify: production config only after acceptance

**Interfaces:**
- Consumes: accepted branch from Phase 3A or 3B
- Produces: a provenance-preserving A0 production transition or an explicit continued-B0 decision

- [ ] **Step 1: Preserve the iteration-25 B0 release and latest pre-A0 checkpoint.**

Verify local and remote hashes before deleting no files. Record rollback commands.

- [ ] **Step 2: Run the final A0 gate twice on consecutive checkpoints.**

Use intended production simulations/parallel width, 768 games, fixed paired seeds, max 800, and no vacancy prior. Require all Task 12B alpha-zero criteria even when the chosen intervention was parallel MCTS.

- [ ] **Step 3: Run a complete paired arena.**

No incomplete opening block may be extrapolated. A promotion result with any aborted block is invalid.

- [ ] **Step 4: Apply a durable, allow-listed config transition.**

Change only fields approved by the measured report. Never rewrite the original resolved config or checkpoint provenance.

- [ ] **Step 5: Resume in short ledgered segments.**

Run one iteration per command initially, verify sidecars/checkpoint/replay after each, and back up before cleanup. Do not launch an unbounded background process until three A0 iterations complete with zero infrastructure errors.

- [ ] **Step 6: Publish only after the full suite and artifact validation pass.**

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

Validate checkpoint v3, optimizer restoration, run-state digest, replay manifest digest, deployment artifact parity, and release SHA-256 inventory.

---

## Codex Execution Order

Codex must execute in this order and stop at every decision gate:

```text
1. Task 1: pin and verify baseline
2. Tasks 2–6: observability-only PR
3. Task 7: adaptive 3P simulation support
4. Task 8: learner diagnostic
5. Task 9: paired serial-search sweep
6. Task 10: select exactly one branch
7. Phase 3A OR Phase 3B, never both concurrently
8. Task 15: two-checkpoint A0 acceptance
```

The first implementation session should stop after Task 6 with PR 1 open. The second should stop after Task 10 with the measured decision spec committed. This prevents a large parallel-MCTS implementation from being built before the repository proves that deeper search is useful.

## Expected Decision Value

The plan answers the following questions in order:

```text
- Is policy loss ~6.06 mostly MCTS target entropy?
- Is it mostly probability mass wasted on illegal actions because training uses a
  full 5,329-action softmax while inference normalizes only legal actions?
- Are policy-head gradients and optimizer updates actually non-zero and large
  enough to move behavior?
- Does 256/400-simulation serial MCTS improve A0 stability?
- Can repetition-triggered adaptive search obtain that benefit cheaply?
- Only if not: can parallel leaf evaluation make the stronger search affordable?
- If deeper search does not help: can gradual vacancy-prior annealing transfer
  responsibility to the learned policy without an abrupt B0-to-A0 cliff?
```

## Self-Review

- **Spec coverage:** The plan preserves the released iteration-25 baseline, measures the documented policy plateau, tests 128/256/400 simulations, enables existing but nonfunctional 3P adaptive budgets, gates parallel MCTS, provides a non-parallel fallback, and defines the final A0 transition criteria.
- **Behavior isolation:** PR 1 is observability-only; PR 2 changes only a disabled-by-default adaptive capability; each later intervention is conditional and independently reviewable.
- **Replay/checkpoint safety:** Diagnostics load copied artifacts, never ingest, prune, save, or resume the production run; no replay or checkpoint schema change is required for visit-target metrics.
- **3P correctness:** Parallel search uses virtual visits only, keeps completed Q vectors separate, and requires width-1 bit-exact golden parity.
- **Metric consistency:** Target entropy uses raw visit counts; legal policy fit uses all authoritative root actions; full policy fit explains the trainer's current 5,329-action loss; the report never compares these quantities without naming their normalization domain.
- **Placeholder scan:** No task contains an unspecified implementation step, unbounded “add tests” instruction, or deferred acceptance criterion.
- **Type consistency:** Task 3 consumes Task 2 summaries; Task 4 consumes the raw root actions retained by `EpisodeMove`; Task 5 uses existing `Trainer`, `ReplayStore`, and checkpoint APIs; Tasks 12A–14A consume the exact batched-search/ticket interfaces defined by Task 11A.

