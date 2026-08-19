# RTX 3060 AlphaZero Training — Implementation Planning Prompt

You are working in the `alphadiamond` repository.

Your task is to produce a detailed, executable implementation plan for converting the existing Soo AlphaZero training workflow into a GPU-accelerated training workflow optimized for the hardware below.

Do **not** implement the changes yet.

First inspect the current repository thoroughly, verify every assumption in this prompt against the actual code, and then write the implementation plan.

The plan should be detailed enough that another coding agent with no prior context can implement it task-by-task using TDD.

## Hardware target

Assume this machine:

```text
GPU:
  NVIDIA RTX 3060
  VRAM: 12 GB
  memory bandwidth: ~315.2 GB/s

CPU:
  32 CPUs available to the training job

PCIe:
  PCIe 4.0 x16
  practical bandwidth approximately 25 GB/s
```

CPU allocation policy:

```text
available CPU count - 2 = self-play worker count
```

For this machine, the expected result is:

```text
32 available
 2 reserved
--------------
30 self-play workers
```

Do not blindly use `os.cpu_count()` if the process may be restricted by cpuset/container/job affinity.

Prefer detecting CPUs actually available to the current process, for example using `os.sched_getaffinity(0)` when supported, with an appropriate fallback.

The resolved worker count must never be less than 1.

Keep an explicit override mechanism if that can be done without complicating the design, but automatic `available - 2` behavior should be the normal GPU-training path.

## Existing architecture that must be inspected

Start by reading at least:

```text
tools/cpu_b0_train.py

runtime/configs/soo-cpu8h.json

src/diamond/alphazero/config.py
src/diamond/alphazero/trainer.py

src/diamond/alphazero/orchestration/selfplay_workers.py

src/diamond/alphazero/inference/coordinator.py
src/diamond/alphazero/inference/model_pool.py
src/diamond/alphazero/inference/remote.py

src/diamond/alphazero/evaluator/torch.py

src/diamond/alphazero/mcts/search_2p.py
src/diamond/alphazero/selfplay/runner_2p.py

src/diamond/alphazero/network/*
```

Also inspect the relevant tests, particularly:

```text
tests/alphazero/orchestration/test_selfplay_workers.py
tests/alphazero/inference/*
tests/alphazero/test_mcts_2p.py
tests/alphazero/test_evaluator.py
tests/alphazero/test_checkpoint.py
tests/alphazero/orchestration/test_resume.py
tests/alphazero/orchestration/test_reference_configs.py
```

Search for any additional tests/configuration code that would be affected.

Before planning changes, run:

```bash
git status
git log --oneline -10
```

and understand the current branch/repository state.

Do not assume the repository still exactly matches the description below if the code says otherwise.

## Important existing behavior

The current implementation appears to already have much of the GPU architecture we need.

Verify these facts:

* self-play workers use separate multiprocessing `spawn` processes;
* worker processes run games/MCTS but send NN evaluation requests to the parent;
* the parent owns a centralized `InferenceCoordinator`;
* `InferenceCoordinator` already batches requests using `max_batch_size` and `max_wait_ms`;
* `InferenceModelPool` already accepts a device;
* `TorchEvaluator` already performs real batched PyTorch inference and supports CUDA;
* `TorchEvaluator` already has FP32/BF16 inference support;
* `AlphaZeroTrainer` already moves its model and training tensors according to `TrainingConfig.device`;
* Soo MCTS currently evaluates one leaf synchronously at a time;
* therefore each self-play worker normally contributes approximately one outstanding inference request at a time;
* the current worker pool computes:

```python
lane_count = min(worker_count, len(jobs))
```

* therefore increasing worker count above `games_per_iteration` does not increase concurrency.

Do not build a new GPU inference server if the existing coordinator/model-pool architecture can cleanly support the requirement.

Prefer extending and instrumenting the existing design.

## Primary implementation goals

The final system should support RTX 3060 GPU training while preserving the existing CPU behavior as much as practical.

The intended initial GPU configuration is:

```text
device: cuda:0
precision: FP32

self-play workers:
  automatic = available CPUs - 2
  expected on this machine = 30

games per iteration: 32
train steps per iteration: 8

MCTS threads per worker: 1

inference max batch size: initially 32
inference max wait: benchmark around 1-2 ms

self-play simulations:
  benchmark both 32 and 64
```

The old CPU workflow must continue to work.

Do not create a completely duplicated `gpu_b0_train.py` implementation if a shared/common execution path is practical.

If the existing `cpu_b0_train.py` name has become misleading, evaluate whether a hardware-neutral trainer entry point plus backward-compatible wrapper is worth doing.

Do not break the current CPU command unnecessarily.

## Preserve the self-play/training ratio

The existing reference configuration effectively uses:

```text
16 games / iteration
4 training updates / iteration
```

The GPU configuration should use:

```text
32 games / iteration
8 training updates / iteration
```

so that increasing self-play concurrency does not accidentally halve the number of optimizer updates per generated game.

Treat this ratio preservation as an intentional learning-semantics requirement, not merely a performance setting.

## GPU precision policy

The first production-capable GPU path must use FP32.

Do not combine the initial CUDA migration with BF16/other mixed-precision changes.

The desired progression is:

```text
CPU baseline
→ CUDA FP32 correctness
→ CUDA FP32 batching/throughput
→ optional BF16 optimization later
```

Existing BF16 inference support may remain available, but FP32 should be the baseline used to validate CPU-vs-GPU semantics.

Do not introduce mixed-precision training in the first implementation unless the repository already does it safely and there is a compelling reason.

## Separate GPU run / checkpoint fork

Do not continue writing into the existing CPU run.

The current CPU source run is:

```text
runtime/runs/soo/cpu8h-soo-20260819
```

The source checkpoint is conceptually:

```text
runtime/runs/soo/cpu8h-soo-20260819/latest.pt
```

At the point this design was discussed it was around training step 72, but do not hard-code `72`; inspect and record the actual checkpoint metadata/hash.

We want the GPU experiment to fork from an explicit source checkpoint into a **new run**.

For example:

```text
runtime/runs/soo/rtx3060-soo-20260820/
```

The GPU run should have its own:

```text
loop state
ledger
replay store
checkpoint archive
latest.pt
configuration snapshot
```

The existing CPU run must remain untouched.

Design a clean way to initialize a fresh run from an existing compatible checkpoint.

A CLI option such as:

```text
--init-checkpoint PATH
```

is one possible design, but inspect the existing checkpoint/resume semantics and choose the cleanest compatible interface.

Required semantics:

```text
new run directory does not exist / is uninitialized
        +
explicit source checkpoint
        ↓
load compatible model + trainer state
        ↓
create new run's durable latest checkpoint
        ↓
new run has independent iteration/replay/ledger state
```

Normal resume semantics must remain distinct:

```text
existing run/latest.pt
        ↓
resume that same run
```

Do not silently overwrite an initialized run with `--init-checkpoint`.

Record the source checkpoint path/hash in the new run's ledger/config provenance.

## Per-game 15-minute wall-clock abort

This is an important robustness change.

Current behavior has produced failures like:

```text
TimeoutError: timed out waiting for self-play jobs: ('game-...',)
```

when running heuristic-off A0 with 64 MCTS simulations.

The current training configuration has:

```text
self_play.max_moves = 2000
inference.response_timeout_s = 600
```

and `cpu_b0_train.py` currently appears to create a worker-pool timeout based on roughly:

```python
max(600.0, inference_response_timeout * 4)
```

With a 600-second response timeout, that becomes approximately 2400 seconds / 40 minutes for the entire worker-pool call.

The current pool timeout is a global deadline and can therefore fail an entire iteration because one pathological game remains unfinished.

Change this behavior.

Each individual self-play game should have:

```text
max wall-clock runtime = 15 minutes = 900 seconds
```

When one game exceeds its wall-clock budget:

```text
completed = False
samples = ()
final_order = None
aborted_reason = "max_game_time_exceeded"
```

That game contributes zero training samples, consistent with existing aborted-game behavior.

The rest of the self-play jobs must continue.

A single timed-out game must **not** crash the iteration or discard completed sibling games.

The result should look conceptually like:

```text
31/32 usable
1 aborted: max_game_time_exceeded
→ ingest completed episodes
→ record abort
→ continue training
```

Keep `max_game_moves_exceeded` as a separate abort reason.

Do not conflate:

```text
game wall-clock timeout
inference response timeout
worker-process failure
pool-level catastrophic timeout
```

They represent different failure modes.

## Deadline propagation

The 900-second timeout should not only be checked after an entire move finishes if doing so can substantially exceed the budget.

Inspect the MCTS structure and design deadline/cancellation propagation so the game can stop at sensible boundaries inside MCTS simulation.

A likely shape is:

```text
SelfPlayRunner owns game deadline
        ↓
MCTS receives deadline/cancellation predicate
        ↓
checks between simulations / before expensive expansion
        ↓
deadline exceeded
        ↓
controlled game-timeout result
```

Do not use `sleep()`-based watchdog hacks.

Prefer monotonic time.

Tests for timeout behavior should use deterministic/fake/monkeypatched time where possible rather than actually waiting.

The existing inference timeout should remain the protection against a genuinely stuck inference operation.

## Pool-level timeout

Do not simply delete all parent-side worker timeout protection.

Keep a pool-level catastrophic safety guard for:

```text
dead child
broken IPC
unresponsive worker
unexpected logic bug
```

but it should no longer be the normal mechanism for terminating a long game.

When deriving the pool guard, account for the fact that:

```text
games_per_iteration = 32
workers ≈ 30
```

means some worker lanes can process more than one game sequentially.

A per-game 900-second budget does not imply the entire pool can always finish in 900 seconds.

Design a sensible pool timeout/grace policy based on the maximum jobs assigned to a lane rather than using an arbitrary fixed deadline.

## GPU inference batching

Reuse the existing centralized batching architecture.

With approximately 30 self-play processes and synchronous leaf evaluation, a practical batch will likely be much smaller than the old configured maximum of 64 and naturally capped by concurrent workers.

Initial target:

```text
max_batch_size = 32
max_wait_ms = 1 or 2 initially
```

Do not assume batch size 32 will actually be achieved.

Instrument actual batch behavior.

Important metrics:

```text
requests completed
batches completed
mean batch size
median batch size
p90 batch size
max observed batch size

queue-to-dispatch latency
inference latency
response latency

NN evaluations / second
batches / second
```

The existing `InferenceMetrics` already appears to capture much of this.

Prefer summarizing the existing metrics rather than introducing a second metrics system.

Avoid storing or logging giant raw latency arrays in the durable ledger if summaries are enough.

## Training/self-play throughput metrics

Extend iteration/run reporting so GPU-vs-CPU comparisons can answer:

```text
games/hour
completed games/hour
samples/hour
training steps/hour

MCTS / NN evaluations per second where measurable

self-play wall-clock seconds
training wall-clock seconds

completed count
aborted count
abort reasons

median game moves
p90 game moves

inference batch p50/p90/mean
inference latency summaries
```

Also record useful GPU environment metadata without introducing unnecessary dependencies:

```text
torch version
torch CUDA runtime version
CUDA available
selected CUDA device
GPU device name
VRAM total if available
```

If GPU utilization requires adding NVML/pynvml solely for this feature, do not add that dependency in the first implementation.

It is acceptable to benchmark utilization externally with `nvidia-smi`.

## CUDA correctness tests

Add a CUDA smoke/numerical comparison path.

When CUDA is available:

1. load the exact same checkpoint on CPU and CUDA;
2. evaluate identical states;
3. verify legal-action identity is identical;
4. compare policy probabilities/logits as appropriate;
5. compare value outputs with explicit, reasonable FP32 tolerances;
6. ensure checkpoint load/save remains device-portable.

CUDA-specific tests must skip cleanly on machines without CUDA.

The normal repository test suite must remain runnable on CPU-only CI.

Do not make ordinary unit tests require an NVIDIA GPU.

## Existing batching semantic constraint

Soo MCTS currently appears to do roughly:

```python
result = evaluator.evaluate((request,))[0]
```

for one leaf at a time.

Do **not** implement intra-tree multithreaded MCTS in this first project.

The intended parallelism is:

```text
many independent games
×
single-threaded MCTS per worker
×
central batched GPU inference
```

Only mention MCTS tree-parallelism as a possible follow-up if profiling later shows that the GPU cannot be fed adequately.

Do not add virtual loss/tree locking/etc. in this implementation.

## RTX 3060 benchmark design

The implementation plan must include a reproducible benchmark procedure.

Use the same immutable source checkpoint for comparisons.

At minimum compare:

```text
CPU baseline / 32 sims
RTX 3060 FP32 / 32 sims
RTX 3060 FP32 / 64 sims
```

Where useful, include a smaller worker/batch sweep after the first baseline, but do not design a combinatorial benchmark explosion.

Potential useful values:

```text
workers:
  auto-30
  perhaps one lower comparison if profiling justifies it

max_batch:
  16
  32

max_wait_ms:
  1
  2
  perhaps current 5 as baseline

simulations:
  32
  64
```

The final performance criterion is not just raw `sec/game`.

Measure both:

```text
throughput
learning quality per wall-clock time
```

## Learning-quality benchmark

The relevant historical heuristic-off behavior around the source model was approximately:

```text
32 sims:
  29/30 complete
  median moves ≈ 144
  p90 ≈ 1667
  one max_game_moves_exceeded

64 sims:
  30/30 complete
  median moves ≈ 105
  p90 ≈ 147
  no aborts
```

Treat these numbers as historical context, not hard-coded tests.

The key observation is that 64 simulations produced much more stable heuristic-off trajectories despite doing more search.

Therefore the GPU benchmark should explicitly determine whether:

```text
30 workers × 64 sims × batched GPU inference
```

produces better useful training throughput than:

```text
30 workers × 32 sims × batched GPU inference
```

Define "useful throughput" using metrics such as:

```text
completed games/hour
non-aborted samples/hour
move-count tail
off-probe completion
off-probe p90
```

After throughput benchmarking, the plan should include a fair wall-clock learning experiment:

```text
same initial checkpoint
same intended learning semantics

CPU train for fixed wall-clock interval
GPU train for same wall-clock interval

then run the same heuristic-off evaluation
```

Compare:

```text
samples generated
training steps
loss metrics as secondary information

off-probe completion
median moves
p90 moves
abort rate
```

Do not use training loss alone as the success criterion.

## GPU config

Plan a dedicated RTX 3060 configuration rather than silently modifying the existing CPU reference config.

A reasonable location is:

```text
runtime/configs/soo-rtx3060.json
```

The existing CPU config should remain a useful baseline.

The GPU config should reflect approximately:

```text
training.device = "cuda:0"

workers:
  automatic CPU-minus-two behavior

games_per_iteration = 32

training batch size:
  keep the existing semantic value initially unless measurement demonstrates a reason to change it

inference:
  max_batch_size = 32
  max_wait_ms = 1 or 2
  response timeout remains a separate inference safety limit

self_play:
  max_moves = 2000
  max_game_seconds = 900

MCTS:
  benchmark simulations 32 and 64
```

Decide the exact configuration schema only after inspecting existing validation/backward compatibility patterns.

Old configs without `max_game_seconds` must continue to behave sensibly.

## Avoid premature optimizations

Do not include the following in the initial implementation unless profiling proves they are required:

```text
intra-tree multithreaded MCTS
custom CUDA kernels
torch.compile
CUDA graphs
pinned-memory pipelines
async CUDA streams
distributed training
multiple GPUs
NVML runtime dependency
mixed-precision training
major network architecture changes
```

One potential later optimization worth profiling is `TorchEvaluator` CPU/GPU transfer behavior.

If it currently performs repeated per-row GPU→CPU synchronizations, mention batching those transfers as a follow-up optimization, but do not automatically fold it into the first correctness patch unless profiling shows it materially matters.

## Backward compatibility requirements

Preserve:

```text
existing CPU training
existing checkpoint compatibility
existing resume safety
existing replay semantics
existing deterministic game IDs/seeds
existing bootstrap-prior behavior
existing max_game_moves_exceeded behavior
existing worker cleanup guarantees
CPU-only pytest capability
```

Do not change AlphaZero learning semantics merely for throughput.

Aborted games must continue to contribute zero samples.

Do not alter policy/value targets.

## Testing strategy

Use TDD.

Every behavior change in the plan should first add or modify a failing test.

The implementation plan should explicitly include tests for at least:

```text
automatic available-CPU-minus-two resolution
minimum worker count
affinity-aware CPU detection
fallback CPU detection

32 games can utilize ~30 worker lanes

per-game 900-second deadline
one timed-out game becomes EpisodeResult abort
timeout abort has zero samples
timeout abort reason is max_game_time_exceeded

one game timeout does not discard sibling completed games
worker remains usable / pool completes normally

max_game_moves_exceeded remains distinct

pool catastrophic timeout still cleans up all child processes
pool timeout accounts for multiple queued jobs per lane

fresh run initialized from external checkpoint
source run remains untouched
new run has independent loop state/replay/ledger
initialized run cannot accidentally be reinitialized
normal resume still works

CUDA FP32 evaluator correctness
CUDA tests skip on CPU-only hosts

inference metric summarization
iteration ledger contains throughput/batch metrics

old CPU/reference config remains valid
new RTX 3060 config is valid
```

Do not write tests that actually wait 15 minutes.

Use dependency injection, a clock callable, a deadline abstraction, monkeypatching, or another deterministic mechanism consistent with the repository style.

## Plan quality requirements

Produce a real implementation plan, not a high-level checklist.

Before writing the plan:

1. inspect the exact current files;
2. identify the exact classes/functions involved;
3. identify existing tests that should be extended;
4. identify any new files that are genuinely needed;
5. avoid unnecessary parallel implementations;
6. follow existing repository conventions.

For every task in the plan specify:

```text
exact files to create/modify
exact functions/classes/interfaces affected

failing test first
exact pytest command
expected failure

minimal implementation
exact pytest command
expected pass

broader regression test command

commit boundary and suggested commit message
```

Keep tasks small enough to review independently.

Do not use placeholders such as:

```text
TODO
TBD
add tests
handle errors
implement logic
etc.
```

If code snippets are needed to make an interface unambiguous, include concrete snippets.

## Recommended task decomposition

Do not blindly copy this decomposition if repository inspection shows a better boundary, but the final plan will probably cover these concerns:

```text
1. hardware/worker-count resolution
2. per-game wall-clock deadline abstraction
3. MCTS deadline propagation
4. worker-pool behavior for individual game timeout
5. fresh-run checkpoint fork semantics
6. RTX 3060 config
7. CUDA FP32 correctness path
8. inference/throughput metrics
9. benchmark tooling or documented benchmark commands
10. end-to-end CPU regression + GPU smoke verification
```

If two of these naturally belong in one atomic change, combine them.

If one is too large, split it.

## Questions / ambiguity policy

Do not ask me questions whose answers can be discovered from the repository.

Inspect the code first.

If there is a genuine product/learning-semantics decision that cannot be inferred from this prompt or the repository, call it out explicitly before finalizing the plan.

Otherwise make the simplest backward-compatible engineering choice and document it.

## Deliverable

Write the implementation plan to:

```text
docs/superpowers/plans/2026-08-20-rtx3060-alphazero-training.md
```

The plan should begin with:

```markdown
# RTX 3060 AlphaZero Training Implementation Plan

**Goal:** Run Soo AlphaZero self-play and training efficiently on an RTX 3060 while using available CPU parallelism, preserving learning semantics and CPU compatibility, and making pathological games abort individually instead of terminating an iteration.

**Architecture:** Keep independent multiprocessing self-play workers and the existing centralized inference coordinator. Resolve worker count from available CPUs minus two, run policy/value inference and training on CUDA, batch inference across games, propagate a 900-second per-game monotonic deadline through MCTS, and keep the parent pool timeout only as a catastrophic safety guard.

**Tech Stack:** Python 3.11+, PyTorch, multiprocessing spawn, pytest, NVIDIA CUDA / RTX 3060.
```

Then include:

```text
Global Constraints
File/Component Map
Implementation Tasks
Test Matrix
Benchmark Procedure
Rollout Procedure
Rollback Criteria
Expected CLI examples
```

The expected CLI examples should show how to:

```text
fork a new RTX 3060 run from the existing CPU checkpoint
run a short 32-sim GPU smoke test
run a short 64-sim GPU smoke test
run a fixed-duration GPU training session
run the existing heuristic-off probe against the resulting checkpoint
```

Do not execute the implementation after writing the plan.

At the end, perform a self-review of the plan for:

```text
requirements coverage
missing tests
backward compatibility
incorrect file/function names
placeholder language
CPU-only CI compatibility
timeout edge cases
resume/fork safety
benchmark fairness
```

Fix any issues you find in the plan itself.

Then stop and report:

```text
1. plan file path
2. 5-10 sentence summary of the proposed architecture
3. files expected to change
4. major risks/tradeoffs
5. any genuinely unresolved decision
```

Do not modify production code until I explicitly approve the implementation plan.
