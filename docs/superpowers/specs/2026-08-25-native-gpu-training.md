# AlphaDiamond Native GPU Training Design

**Status:** Approved implementation specification derived from the user brief and the 2026-08-25 operating rulings. CPU-verifiable work may proceed; NVIDIA acceptance remains a hardware gate.

**Audited baseline:** `main` at `6ac0a1c7cc0edc84a418ab56e14c0d1a252400fd`.

**Working branch:** `codex/native-gpu-training` in `C:\Users\dzk55\alphadiamond`.

**Related evidence:** the immutable scientific/performance reports named in the mission brief and `../reports/2026-08-25-python-zero-final-acceptance.md`.

## 1. Decision summary

Deliver the program as four dependency-ordered, reviewable subsystems:

1. **Native training contract:** exact config v2, independent lanes/threads, microsecond batching, real minibatch and optimizer-step semantics, deadlines, and baseline metrics.
2. **CUDA runtime:** one device resolver, CUDA-safe batched inference/training, immutable actor and mutable learner separation, candidate identity, and device-portable checkpoints.
3. **Durable sustained loop:** transactional replay sampling/ingest, explicit stage ownership, operation-result reuse, bounded multi-iteration execution, resume, and initialization lineage.
4. **Arena and acceptance:** materialized opening suites, block-aware statistics and promotion, comparable benchmarks, CPU/GPU CI, formatting, and current operational documentation.

No subsystem may claim CUDA correctness without execution on a real NVIDIA GPU. This CPU-only host will implement and verify portable behavior and produce an exact CUDA-host checklist.

The original brief requested an isolated worktree. The later user ruling supersedes that operational detail: work continues on the current repository branch, existing user-owned untracked files are preserved, and subagents must not create worktrees. The branch remains based on the audited SHA and does not touch PR #56's Qt/QML/model-catalog files.

## 2. Snapshot and baseline evidence

| Item | Audited state |
|---|---|
| Base | `6ac0a1c7cc0edc84a418ab56e14c0d1a252400fd` |
| PR #56 | Open on `codex/models-manager`; outside this change set |
| Submodules / Git LFS | No submodules; no tracked LFS objects |
| Toolchain | CMake 4.4.2, MSVC 14.51.36231/19.51, Windows SDK 10.0.26100 |
| LibTorch | 2.13.0 CPU/MKL from `C:\ProgramData\miniforge3\envs\alphadiamond` |
| CUDA | No CUDA-enabled LibTorch and no NVIDIA device on this host |
| `native-ci` | Configure/build succeeded; 14/15 tests passed |
| Pre-existing failure | `replay_store_test` reaches final cleanup and then fails while deleting its scratch directory on Windows; it is not evidence of replay corruption |
| `native-training` | Configure succeeded; the baseline build was interrupted by the user before completion, so no suite result is claimed |

Windows native execution requires both the MSVC developer environment and these PATH prefixes:

```text
C:\ProgramData\miniforge3\envs\alphadiamond\Library\bin
C:\ProgramData\miniforge3\envs\alphadiamond\Scripts
C:\ProgramData\miniforge3\envs\alphadiamond
```

Missing `c10.dll`, `torch_cpu.dll`, Qt DLLs, or their transitive dependencies must be diagnosed from the executable's dependency table and the effective process PATH. The known `replay_store_test` failure is a separate cleanup/handle problem: the test does not link LibTorch and completed its replay operations before cleanup failed.

## 3. Goals and non-goals

### Goals

- Resolve and enforce one explicit `torch::Device` for a run with no silent fallback.
- Preserve Soo and Min game/model/training contracts while making GPU inference and AdamW training correct.
- Make the actor immutable during a self-play generation and give every serialized model a weight-derived identity.
- Honor configured batch size, optimizer steps, lanes, threads, waits, deadlines, run budget, checkpoint cadence, and seeds.
- Make replay, checkpoints, stage results, and run progress crash-consistent and resumable.
- Evaluate candidates with unique balanced opening blocks and statistics whose sampling unit is the opening block.
- Produce comparable JSON evidence and explicit CPU/GPU test lanes while keeping the repository Python-zero.

### Non-goals

- No width/depth experiment, search-workload weakening, AMP, FP16, BF16, quantization, CUDA graphs, or custom CUDA kernel in the initial program.
- No concurrent actor self-play and learner optimization on one GPU until ownership and memory behavior are measured.
- No new replay binary format without measured need and a versioned compatibility design.
- No strength claim from checkpoint number, loss, or a deterministic pseudo-replicate.
- No model upload, bucket mutation, promotion to the shipped model, history rewrite, or historical-report deletion.
- No Qt/QML/model-catalog change owned by PR #56.

## 4. Scientific invariants

1. Soo remains directional-residual width 128, six residual blocks, FP32.
2. Heuristic-free A0 uses `bootstrap_prior = "none"`; heuristic bootstrap configs remain visibly non-A0.
3. Search never drops below 128 simulations. The production value of 400 remains unchanged until a controlled frozen-actor GPU experiment supports a change.
4. Before/after comparisons hold simulations, games, move limit, temperature, noise, replay ratio, batch size, optimizer steps, openings, and completion semantics constant.
5. Every aborted game contributes zero replay samples and reports one explicit reason.
6. Exact topology, encoder, action space, seat layout, value semantics, and sparse visit-policy ordering remain compatible.
7. Min continues to compile and pass CPU tests.
8. Step 44,250 remains the promoted reference until a valid paired-opening evaluation proves otherwise.
9. Statistical independence is defined at a complete balanced opening block: a Soo candidate/champion pair or a Min three-player balance block.
10. Historical evidence remains immutable. Current-state corrections live in new runbooks or explicit cross-references.

## 5. System architecture and ownership

```text
canonical config v2 + SHA-256
              |
       device resolver
              |
   immutable champion/actor ----> self-play ----> transactional replay
              |                                      |
              |                               batch-sized sample txn
              |                                      |
              +--> distinct mutable learner <---- AdamW trainer
                                      |
                              immutable candidate
                                      |
                     paired-opening arena + rating
                                      |
                       atomic promote or reject
                                      |
                    checkpoint/run-state/ledger
```

### 5.1 Model roles

- **Champion:** immutable promoted model identity.
- **Actor:** an immutable snapshot of the champion used for one self-play generation. It is in eval mode.
- **Learner:** a distinct deep copy owned by `Trainer`, in train mode, with its own optimizer.
- **Candidate:** an immutable eval-mode snapshot serialized after training. Its key is derived from canonical serialized weights, never inherited from the actor.

Copying a LibTorch `ModuleHolder` is shared ownership, not cloning. Deep snapshots must copy parameter and buffer values into a separately constructed compatible module. Promotion alone changes the champion identity.

### 5.2 Device ownership

A Torch-free parser accepts only `cpu`, `cuda`, or `cuda:N`. A shared LibTorch resolver returns:

```cpp
struct ResolvedDevice {
    torch::Device torch_device;
    std::string canonical_name;  // cpu or cuda:N
    std::optional<int> cuda_index;
};
```

Resolution occurs once before creation or mutation of a run directory. `cuda` resolves to `cuda:0`; `cuda:N` validates availability and index range. An unavailable request fails with an actionable error. The exact canonical name is persisted and reported.

## 6. Production configuration v2

The parser is exact: unknown or missing keys fail. Schema v1 is rejected with:

```text
production config schema_version 1 is not supported; migrate max_wait_ms to max_wait_us, worker_count to logical_lanes and search_threads, and add runtime, run_budget, train_steps_per_iteration, opening_suite, and promotion_statistics
```

No implicit unit conversion occurs at runtime. Tracked configs are explicitly migrated.

### 6.1 Canonical shape

```json
{
  "schema_version": 2,
  "model_name": "Soo",
  "model_version": "2.0.0",
  "network": {"width": 128, "residual_blocks": 6},
  "runtime": {"device": "cuda", "precision": "fp32"},
  "mcts": {
    "simulations": 400,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "seed": 7
  },
  "self_play": {
    "max_moves": 2000,
    "max_game_seconds": null,
    "temperature": 1.0,
    "temperature_moves": 20,
    "bootstrap_prior": "none",
    "seed": 7
  },
  "workers": {
    "logical_lanes": 8,
    "search_threads": 8,
    "games_per_iteration": 64,
    "retry_id": "attempt-0"
  },
  "inference": {
    "max_batch_size": 64,
    "max_wait_us": 5000,
    "request_queue_capacity": 1024,
    "response_timeout_s": 600.0
  },
  "replay": {"capacity": 200000, "seed": 7},
  "training": {
    "batch_size": 256,
    "train_steps_per_iteration": 1,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "seed": 7
  },
  "run_budget": {
    "max_iterations": 1,
    "max_wall_clock_seconds": null,
    "checkpoint_every_iterations": 1
  },
  "arena": {
    "games": 40,
    "max_moves": 2000,
    "promotion_threshold": 0.55,
    "seed": 7
  },
  "opening_suite": {
    "id": "production-openings-v1",
    "version": 1,
    "seed": 7,
    "count": 4,
    "max_depth": 6
  },
  "promotion_statistics": {
    "method": "opening-block-bootstrap-v1",
    "resampling_unit": "opening_block",
    "confidence_level": 0.95,
    "bootstrap_replicates": 10000,
    "seed": 7
  },
  "run_seed": 7
}
```

`runtime.device` is the requested device in a source config and the canonical resolved device in the immutable run snapshot. A run snapshot never contains bare `cuda` after resolution.

### 6.2 Baseline-preserving migration values

- Soo/Min production retain 400 simulations, batch size 256, 64 games, replay capacity 200,000, CUDA request, and A0 prior `none`.
- Soo/Min bootstrap retain 128 simulations, batch size 256, 32 games, replay capacity 50,000, CPU, and the named heuristic prior.
- Existing `worker_count` maps to both explicit fields only in the committed config migration: production `8/8`, bootstrap `4/4`. The parser does not perform that migration.
- Existing 5 ms maps explicitly to 5000 µs in committed JSON. Tests prove that 50 µs remains 50 µs.
- The previous hard-coded one optimizer step and one-pass completion become explicit `1` values. Multi-step and two-iteration behavior is proved with test/acceptance configs before production tuning.

### 6.3 Validation

- Only FP32 is accepted.
- All counts are positive except `opening_suite.max_depth`, which may be zero only when count is one.
- `batch_size <= replay.capacity` and `logical_lanes <= games_per_iteration`.
- `search_threads`, `logical_lanes`, and `max_batch_size` are independently validated.
- `max_game_seconds`, `max_wall_clock_seconds`, and response timeout must be finite and positive when present.
- At least one run-budget limit is present; checkpoint cadence is positive and no larger than `max_iterations` when that limit exists.
- Soo keeps width 128 and blocks 6 in production configs; compatibility validation continues to own general model-shape checks.
- Opening statistics use the exact supported method/unit and confidence lies in `(0, 1)`.

## 7. Self-play, batching, and abort semantics

`workers.logical_lanes` maps only to `soo::EpisodeConfig::lanes`; `workers.search_threads` maps only to `threads`. Inference fields map without unit conversion.

`soo::EpisodeConfig` gains an optional monotonic deadline duration. Each game computes its own `steady_clock` deadline. Deadline termination sets `max_game_seconds_exceeded`; move-limit termination remains distinct. The pipeline serializes `aborted_reason` as one of:

- `max_moves`
- `max_game_seconds`
- `interrupted`
- an explicit evaluator/search failure category

Every non-completed episode serializes zero `TrainingSample` records.

Each optimizer step requests exactly `training.batch_size` distinct sample descriptors. If fewer samples exist, training fails before an optimizer mutation with:

```text
insufficient replay samples: requested <N>, available <M>
```

The pipeline performs exactly `training.train_steps_per_iteration` successful optimizer steps or reports the interruption/failure; requested and completed counts are separate metrics.

## 8. CUDA-safe inference and training

### 8.1 Inference

For each evaluator batch:

1. Validate compatibility, feature sizes, legal-action ranges/order, and non-empty legal sets.
2. Collate features, padded legal-action indices, and masks into contiguous host buffers.
3. Transfer each buffer once to the actor device.
4. Run one forward pass, gather legal logits, mask, and softmax on-device.
5. Check finite logits/priors/values.
6. Copy compact priors and values to host once and scatter without per-row device synchronization.

No CUDA tensor is host-dereferenced. No per-row `.cpu()`, `.item()`, or host `data_ptr<float>()` occurs.

### 8.2 Training

- Construct/move the learner before creating or restoring AdamW.
- Collate features, dense policy targets, and value targets in contiguous host buffers and transfer each once.
- Preserve normalized sparse-visit policy cross-entropy + value MSE + AdamW exactly.
- Seed LibTorch CPU and CUDA RNGs from recorded run seeds.
- Deterministic test mode enables supported deterministic behavior and fixed thread settings; throughput mode records but does not overpromise device-level bitwise reproducibility.
- Metrics expose actual tensor device, minibatch size, step timings, finite losses, and GPU memory where available.

CPU FP32 fixtures retain their existing tolerances. Separately named GPU parity tolerances begin at `rtol=1e-4, atol=1e-5` and may change only with operation-level evidence.

## 9. Replay transaction design

Replay descriptors remain authoritative and existing schema-v1 samples remain readable.

- Sampling uses an O(batch) partial selection/shuffle of descriptor indices without replacement; it does not copy the sample pool.
- A sampling transaction contains RNG state before/after, selected IDs, iteration, candidate operation ID, and completion state.
- RNG advancement becomes authoritative only when the corresponding candidate/run progress is committed. Recovery reuses a valid selection or rolls back an incomplete one.
- An iteration is ingested as one batch with accepted/duplicate game and sample counts.
- A new manifest is written and validated before activation. Only then may chunks unreachable from the active manifest be deleted.
- Cleanup failures are reported and retried; they do not roll back an already active valid manifest or delete referenced chunks.

Required metrics include replay size, accepted/duplicate games, new/pruned samples and chunks, selection/ingest I/O time, and transaction IDs.

## 10. Checkpoint and initialization manifest v3

Three modes are mutually exclusive:

1. `scratch`: new run and optimizer, no source artifact.
2. `warm_start`: validated deployment weights, new run/step lineage, fresh optimizer, `optimizer_reset=true`, source digest and source training step recorded.
3. `resume`: validated native checkpoint/run identity with model, optimizer, RNG/progress, replay identity, and canonical config.

`audited_legacy_import` is a separate offline checkpoint command and is enabled only after model and optimizer decode plus parity evidence. Until then, step 44,250 is weight-only warm start and is never described as a resumed native step.

Manifest v3 contains format version, run ID, iteration, native step, initialization mode/lineage, source digests/step, source Git commit, canonical config and SHA-256, compatibility/architecture, resolved device/precision, model and optimizer digests, replay manifest digest, RNG protocol/state presence, operation IDs, creation time, and environment metadata.

Save and load are staged. Loading validates all bytes and compatibility before mutating live objects. Model and optimizer tensors migrate explicitly to the target device. Required portability matrix: CPU→CPU, CPU→CUDA, CUDA→CUDA, CUDA→CPU, including AdamW state.

## 11. Durable orchestration

Authoritative stages are:

```text
INITIALIZE -> SELF_PLAY -> REPLAY_INGEST -> TRAIN -> SAVE_CANDIDATE
 -> PROMOTION_ARENA -> RATING_BENCHMARK -> PROMOTE_OR_REJECT -> PERSIST
```

Each stage has a stable operation ID derived from run ID, iteration, stage, protocol IDs, and retry identity. It writes a checksummed result artifact before transactional state advancement. Re-entry reuses a matching valid result and rejects conflicts.

At `PERSIST`, iteration increments and execution continues at `SELF_PLAY` until the first configured budget/stop condition. Graceful interruption leaves the current stage resumable and returns the documented interrupted exit code. Seeds derive from run/model identity, iteration, stage, game index, retry, and opening ID; `run_seed + game` is forbidden.

Exact CLI contract:

```text
alphadiamond-train train --config <v2.json> --run-dir <dir> --scratch
alphadiamond-train train --config <v2.json> --run-dir <dir> --warm-start <artifact>
alphadiamond-train train --config <v2.json> --run-dir <dir> --checkpoint <native-checkpoint>
alphadiamond-train resume --run-dir <dir>
alphadiamond-train evaluate --candidate <artifact> --champion <artifact> --opening-suite <suite>
alphadiamond-train report --run-dir <dir>
```

`train` accepts exactly one initialization selector. `resume` and `report` read authoritative paths/config from the run directory and do not require an unrelated checkpoint argument.

## 12. Opening suites, arena, and promotion

An opening suite is materialized or deterministically derived from `(id, version, seed, count, max_depth)`. Every opening has a stable ID and serialized state/digest.

- Soo uses `schedule_soo_pair()` to cover candidate/champion seat assignment and turn order for each opening.
- Min uses `schedule_min_triple()` and retains the complete three-player balance block.
- Match IDs include suite/opening/cell identity and are unique.
- Requested game counts must equal a whole number of complete blocks; cycling a fixed cell set to manufacture more games is rejected.
- Confidence intervals bootstrap complete opening blocks. Aborted/incomplete blocks are reported separately and are not partial strength observations.
- The 0.55 promotion threshold remains. Promotion activates the candidate only after candidate, arena, rating, decision, checkpoint, and ledger artifacts are durable.

## 13. Metrics and benchmarks

Each stage writes a result JSON and one append-only canonical JSONL event. Required fields are those enumerated in the mission brief, including environment/provenance, game completion and move percentiles, evaluator/batch/worker utilization, replay I/O, detailed training timings/losses/memory, checkpoint digests, arena block counts, intervals, and decision reason.

Benchmarks extend existing native executables. They accept explicit device/artifact/workload/thread/warmup/repetition arguments and emit canonical JSON. Before/after comparisons pin commits, host, artifact, seeds, openings, and all workload knobs. Median and range, completed samples/hour, aborts, and incompatibilities are reported. No speedup is claimed from historical cross-host numbers.

## 14. Testing and CI

CPU tests own parser/schema errors, workload wiring, Soo/Min parity, actor immutability, candidate identity, abort-zero-sample behavior, replay bounds/transactions, stage idempotency/resume, warm-start labeling, checkpoint corruption, schedule uniqueness, and promotion transaction safety.

GPU tests carry the `cuda` label and require a real CUDA-enabled LibTorch plus NVIDIA device. A skipped or unavailable GPU lane is pending, never passing acceptance. The GPU matrix covers selected device, full tensor residency, forward/legal/prior/value parity, one-step gradient/update parity, batches 1/17/64/256, checkpoint portability, two iterations, interruption/resume, no fallback/host dereference, finite metrics, and bounded memory.

Normal CI gains one CPU `native-training` configure/build/CTest job. An opt-in/self-hosted NVIDIA job prints environment data, runs CUDA tests and the bounded acceptance workload, and uploads JSON evidence. Existing compatibility aliases stay until branch protection is deliberately migrated. Formatting adds pinned changed-file `clang-format` plus `git diff --check`, never a repository-wide reformat.

## 15. Delivery boundaries and defect coverage

| PR | Primary defects | Excluded until dependency exists |
|---|---|---|
| A — contract | 1, 4–10, 28 baseline | CUDA tensor movement, model cloning, durable format changes |
| B — CUDA runtime | 1–3, 11–12, checkpoint device portability | sustained orchestration and promotion statistics |
| C — durability | 13–19, 23 | arena/CI/docs tightening |
| D — arena/acceptance | 20–22, 24–28 | no model promotion/upload during acceptance |

Every known defect is assigned. Later PRs may add interfaces in earlier libraries, but they may not silently change an earlier scientific or persistence contract.

## 16. Operating and verification policy

- Root directly handles branch state, environment, DLL/PATH, build/link, integration, and small acceptance fixes.
- Subagents are used only for independent file sets; at most three run concurrently and receive repository/branch/base SHA, owned/prohibited files, Git Bash/POSIX-script policy, conda/MSVC paths, exact build target, one acceptance command, no-worktree rule, and stale-work exclusions.
- Do not repeat independent review agents or bounce plan→implementation→review across agents. Root self-reviews plans and diffs.
- Every supported CMake/CTest command goes through `tools/native_training.sh`. On Windows, invoke the repository POSIX script with Git Bash when available; PowerShell may only prepare/pass the MSVC and conda environment.
- Build the exact changed target and run the nearest contract/smoke test once for normal tasks. Full suites and hardware acceptance run at subsystem/final boundaries.
- Stage explicit files; never `git add .`. Commit atomically after the named acceptance passes. Push/PR/merge remain separate visible actions.

## 17. Completion and hardware gate

CPU completion means all CPU-verifiable contracts and suites pass, the CUDA tests/checklist exist, and no GPU claim is made. Full mission completion additionally requires the production-shaped workload from the brief on a real NVIDIA host: FP32 Soo 128×6, at least 128 simulations, games greater than lanes, lanes greater than threads, batch 256, multiple optimizer steps, two iterations, first-iteration checkpoint, forced interruption/resume, and paired-opening evaluation.

The resulting candidate is neither uploaded nor promoted. The acceptance report records the exact command, SHA/dirty state, GPU/driver/CUDA/LibTorch, resolved config/lineage, throughput/completion/abort/batch/training/memory metrics, checkpoint digests, resume evidence, and unique arena block/match counts.
