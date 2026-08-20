You are working in the `alphadiamond` repository.

Your task is to continue the measured CPU-side performance investigation of Soo's AlphaZero self-play pipeline and implement the next optimization only after decomposing the remaining latency.

Do NOT use the RTX 3060 machine yet.

The GPU machine costs money and is accessed remotely, so all hardware-independent profiling and optimization must be completed locally on CPU first.

Do not touch MCTS search semantics.

Do not implement tree parallelism, virtual loss, speculative leaves, async MCTS, or batched MCTS.

We already measured that MCTS itself is not the current bottleneck.

---

# Current verified state

Read these first:

docs/cpu_profile_findings.md
docs/gpu_benchmark_findings.md

src/diamond/alphazero/orchestration/selfplay_workers.py
src/diamond/alphazero/inference/coordinator.py
src/diamond/alphazero/inference/remote.py
src/diamond/alphazero/evaluator/torch.py

and all relevant tests.

Run:

git status
git log --oneline -15

The repository is expected to be clean and current.

Do not assume this prompt is more authoritative than the code.

Verify the current implementation.

---

# What has already been proven

The trained checkpoint used for profiling is:

sha256:
4b2a32ff15179e890d4266346bca178d9a255eebe16af3a6e3d0482f0ceb1320

training step:
72

The first CPU profiling pass found:

In-process inference round trip:
~9.6 ms p50

Old multiprocessing worker-pool round trip:
~43.0 ms

Pure CPU NN forward:
~4.8 ms

The original parent loop was the dominant bottleneck.

`SelfPlayWorkerPool.run()` used to alternate:

_forward_inference()

with:

results.get(timeout=0.01)

The parent effectively ran at a ~10.2 ms period, and every inference request and response crossed that tick.

That was fixed by adding `_InferenceBridge` with dedicated request and response pump threads.

The fix preserved:

- request correlation IDs
- worker-specific routing
- failure propagation
- worker cleanup
- per-game 900-second timeout
- catastrophic pool timeout
- sequential MCTS semantics

The measured before/after result on identical work was:

wall clock:
236.6 s → 112.3 s

samples/hour:
8,429 → 17,765

evals/second:
93.0 → 196.1

worker-visible inference round trip:
43.0 ms → 20.4 ms

same:
- checkpoint
- requests
- samples
- seeds/config
- completed games
- move distribution
- abort result

This was a genuine 2.11x speedup.

All tests remained green:

451 passed
7 skipped

---

# Important conclusion from the first profile

MCTS's own Python work is not the problem.

Measured per simulation:

selection/traversal:
~0.204 ms

state/request preparation:
~0.320 ms

backup:
~0.306 ms

total non-NN MCTS Python work:
~0.83 ms

CPU NN forward:
~4.8 ms

Therefore do NOT optimize:

- tree selection
- game-state handling
- MCTS backup
- MCTS parallelism

unless new measurements contradict the existing profile.

---

# The next question

The worker-visible round trip after the first fix is still approximately:

20.4 ms

while CPU NN work is only approximately:

5 ms

So roughly 15 ms remains outside the neural network.

However:

DO NOT assume that the entire remaining 15 ms is multiprocessing/pickle overhead.

The current measurements also show:

coordinator queue → dispatch p50:
~10.34 ms

mean batch size:
~1.28

Therefore a significant fraction of the remaining latency may be deliberate batching wait rather than process transport.

We need to measure the remaining latency precisely before making another architecture change.

---

# Goal

Decompose the remaining ~20.4 ms round trip into these broad components:

A. worker → parent multiprocessing transport

B. parent arrival → coordinator dispatch
   including batching wait

C. actual NN evaluation

D. completed NN response → worker receives response

Conceptually:

worker submit
    ↓
[A]
    ↓
parent bridge receives
    ↓
[B]
    ↓
coordinator dispatch
    ↓
[C]
    ↓
NN finishes
    ↓
[D]
    ↓
worker receives result

The exact timestamp boundaries should follow the actual code.

Use high-resolution monotonic timing.

Prefer:

time.monotonic_ns()

or the existing timing abstraction if suitable.

---

# Phase 1 — inspect existing instrumentation

Before adding anything, identify what is already measured.

`InferenceCoordinator` already tracks metrics such as:

- admission latency
- queue-to-dispatch latency
- inference latency
- response latency
- batch size

Reuse those metrics.

Do not duplicate them.

Identify only the missing process-boundary timings.

We particularly need enough evidence to distinguish:

worker → parent transport

batch waiting

NN compute

parent → worker transport

---

# Phase 2 — add minimal missing instrumentation

Add only the instrumentation required to decompose the remaining round trip.

The pipeline may issue tens of thousands of requests.

Do NOT retain one unbounded timestamp record per request.

Use the repository's bounded streaming/reservoir metric style.

Metrics should expose summaries such as:

worker_to_parent_ms:
  mean
  p50
  p90

parent_to_dispatch_ms:
  mean
  p50
  p90

evaluator_ms:
  mean
  p50
  p90

response_to_worker_ms:
  mean
  p50
  p90

worker_round_trip_ms:
  mean
  p50
  p90

Use better field names if they match repository conventions.

Keep the durable ledger bounded and JSON-friendly.

---

# Phase 3 — measure the pure multiprocessing transport floor

Create a controlled benchmark using a fake or immediate evaluator/coordinator path.

The goal is to measure the minimum cost of:

worker process
    ↓
multiprocessing queue
    ↓
parent
    ↓
immediate response
    ↓
multiprocessing queue
    ↓
worker

No neural network.

No artificial batching wait.

No real game required if the same transport boundary can be exercised safely in isolation.

This benchmark should answer:

"What is the process/serialization round-trip floor?"

For example, if it measures ~4–6 ms, then the remaining 20 ms cannot reasonably be described as 15 ms of IPC.

If it measures ~12–15 ms, then multiprocessing transport really is the dominant remaining bottleneck.

Do not guess.

Measure it.

---

# Fake transport benchmark requirements

Use the real transport objects and routing machinery where practical.

Do not replace multiprocessing.Queue with queue.Queue, because that would remove exactly the boundary we are trying to measure.

Preserve:

- InferenceRequest shape
- correlation IDs
- response routing
- multiprocessing spawn semantics

The fake evaluator may immediately echo a valid InferenceResponse.

The benchmark should run enough requests to stabilize p50/p90 without taking long.

Do not make it part of normal slow CI if inappropriate.

It can be a profiling tool or optional benchmark.

---

# Phase 4 — CPU batching-wait experiment

Run a very small controlled sweep on CPU.

Use the same:

- trained checkpoint
- worker count
- simulations
- max_moves
- bootstrap prior
- seeds where possible
- games
- torch thread count

Change ONLY:

max_wait_ms

Test:

1 ms
2 ms
5 ms

The purpose is not to tune the final GPU value.

The purpose is to discover how much of the current ~20.4 ms round trip is caused by waiting for a batch that rarely fills on the 4-worker CPU machine.

For each value report:

worker-visible round trip

queue→dispatch p50/p90

CPU evaluator p50/mean

mean batch size

p50/p90 batch size if available

evals/sec

games/hour if meaningful

samples/hour if meaningful

Do not interpret lower CPU max_wait as automatically correct for the GPU.

The RTX 3060 machine will have approximately 30 workers, so its batching tradeoff is different.

---

# Important CPU concurrency result already known

On the local 8-core machine:

4 workers:
~14.0 sec/game

8 workers:
~16.2 sec/game

8 workers were slower.

Mean batch size also fell:

1.28 → 1.04

This was attributed to CPU contention / oversubscription:

8 self-play workers
+
Torch CPU threads

Do not run a large worker-count sweep.

4 workers is the baseline for the remaining local investigation.

Do not interpret the 8-core CPU ceiling as evidence about the 30-vCPU GPU host.

---

# Phase 5 — classify the remaining bottleneck

After the new measurements, classify the residual ~15 ms.

Possible outcomes:

Case A — batching wait dominates

Example:

worker→parent transport: 2 ms
queue/batch wait: 9 ms
NN: 5 ms
response transport: 3 ms

Then do NOT redesign multiprocessing IPC first.

The next question becomes batching policy.

Case B — multiprocessing transport dominates

Example:

worker→parent: 6 ms
batch wait: 2 ms
NN: 5 ms
response→worker: 7 ms

Then IPC/serialization is the correct next architecture target.

Case C — both are significant

Then choose the single largest avoidable component.

Do not optimize both at once.

We need attribution.

---

# Phase 6 — one root-cause hypothesis

State one hypothesis using measured numbers.

Example:

"The dominant remaining avoidable latency is parent↔worker multiprocessing transport, which accounts for X ms of the 20.4 ms worker-visible round trip, while batching contributes only Y ms."

Or:

"The dominant remaining latency is max_wait batching behavior, not IPC: queue→dispatch accounts for X ms while pure multiprocessing transport is only Y ms."

Do not proceed to a production fix until this statement is supported by measurement.

---

# Phase 7 — implement exactly one next optimization

Use TDD.

Choose the change based on the measurements.

Do NOT combine unrelated optimizations.

---

# If batching wait is the dominant issue

Do not remove batching globally.

Remember:

CPU:
4 workers
batch ~1

GPU:
~30 workers
potentially much larger concurrent request set

The CPU and GPU optimal `max_wait_ms` may differ.

Possible implementation directions, only if supported by evidence:

- allow separate CPU/GPU config values
- reduce CPU reference wait
- make batching wait configurable more explicitly
- improve batching behavior without changing search semantics

Do not create clever adaptive batching unless there is a strong measured reason.

Prefer simple configuration first.

---

# If multiprocessing IPC is the dominant issue

Then inspect the current path carefully.

Current rough architecture:

worker process
    ↓ multiprocessing.Queue
_InferenceBridge
    ↓ queue.Queue / thread boundary
InferenceCoordinator
    ↓
evaluator
    ↓
InferenceCoordinator
    ↓ queue.Queue / thread boundary
_InferenceBridge
    ↓ multiprocessing.Queue
worker

Important distinction:

The expensive process boundaries are primarily:

worker → parent

parent → worker

The coordinator runs inside the parent process on a thread and uses `queue.Queue`.

Do not call every queue hop "pickle/IPC".

If transport dominates, investigate the smallest way to reduce process-boundary overhead.

Potential avenues to evaluate, not blindly implement:

- reduce serialized payload size
- avoid reconstructing/copying immutable request objects unnecessarily
- use shared memory for large fixed-size tensors/features if justified
- use a more direct worker↔inference process architecture
- reduce intermediate transport envelopes
- batch multiple requests per IPC message if possible without changing MCTS semantics

But choose only the smallest measured high-value change first.

Do not start with a full shared-memory redesign unless profiling proves serialization payload size is the cause.

---

# Serialization investigation if IPC dominates

Measure before redesigning.

Determine:

- serialized request size
- serialized response size
- pickle encode/decode cost
- multiprocessing queue enqueue/dequeue latency
- whether payload size or process scheduling dominates

Use focused microbenchmarks.

For example:

pickle.dumps(request)

pickle.loads(...)

Queue.put/get round trip

But remember:

pickle microbenchmark time != multiprocessing transport latency

Both should be measured separately.

---

# Do not touch MCTS semantics

Still prohibited:

- virtual loss
- shared-tree parallelism
- async MCTS
- speculative leaf selection
- multiple outstanding leaf requests from one tree

Current MCTS remains sequential.

The profiling evidence says its own work is only ~0.83 ms per simulation.

Changing search semantics to optimize a sub-millisecond component would be unjustified.

---

# Do not optimize CUDA yet

Do not rent/start the RTX 3060 during this task.

Do not implement:

- BF16
- mixed precision
- CUDA graphs
- pinned memory
- CUDA streams
- torch.compile
- GPU-specific evaluator refactoring

There is a known possible later issue in TorchEvaluator:

per-row GPU→CPU synchronization.

Leave it alone for now.

First eliminate hardware-independent latency.

---

# TDD requirements

Before any production behavior change:

1. write failing test
2. prove it fails
3. implement minimal change
4. run focused tests
5. run broader regression tests
6. benchmark identical workload
7. compare before/after

Use deterministic concurrency tests.

Prefer:

threading.Event
Barrier
fake coordinator
controlled multiprocessing queues
injected clock

Avoid flaky sleep-based tests.

Preserve tests for:

- correlation IDs
- correct worker response routing
- bridge failure propagation
- worker cleanup
- per-game 900 s timeout
- catastrophic pool timeout
- CPU-only operation
- deterministic identities/seeds

---

# Benchmark fairness

Every before/after production optimization comparison must use identical:

checkpoint
workers
simulations
max_moves
max_wait_ms
torch thread count
bootstrap prior
games
seeds where practical

If max_wait_ms itself is the experiment, clearly mark it as the only changed variable.

Report:

completed games
aborted games
abort reasons

Do not interpret sec/game alone.

Prefer:

samples/hour
completed games/hour
evals/sec
round-trip latency

An optimization that becomes faster by aborting/discarding games is not a speedup.

---

# Required deliverables

Work in this order.

## 1. Repository verification

Report current relevant architecture and confirm the first `_InferenceBridge` fix is present.

## 2. Residual-latency instrumentation plan

Identify exact existing metrics and exact missing boundaries.

## 3. Tests for instrumentation

Add focused tests before implementation where applicable.

## 4. Instrumentation implementation

Keep bounded memory.

## 5. Pure transport benchmark

Measure worker↔parent multiprocessing round-trip floor with fake/immediate inference.

## 6. CPU max_wait experiment

Run:

1 ms
2 ms
5 ms

at the same 4-worker trained-checkpoint configuration.

## 7. Residual latency report

Produce a table:

Component                   mean   p50   p90
worker → parent             X      X     X
batch/coordinator wait      X      X     X
CPU NN                      X      X     X
response → worker           X      X     X
total                       X      X     X

Also show the pure transport floor.

## 8. Root-cause decision

State the largest avoidable component.

## 9. One optimization

Use TDD.

Implement exactly one fix.

## 10. Identical before/after benchmark

Report:

worker round trip
evals/sec
samples/hour
batch size
queue→dispatch
completed/aborted games

## 11. Regression verification

Run the relevant full CPU test suite.

Do not claim success without the test output.

## 12. GPU validation recommendation

Do NOT run GPU.

Based on the new measurements, write the exact short RTX 3060 validation recipe we should use later.

---

# Final report format

## What remained after the first 2.11x fix

Plain-language explanation.

## Residual latency breakdown

Table.

## Pure multiprocessing transport floor

Measured value.

## Batching wait experiment

1 / 2 / 5 ms comparison table.

## Root cause

One sentence, backed by measurements.

## Second optimization

What changed and why.

## Before vs after

Identical-work comparison.

## Correctness

Tests run and result.

## What not to optimize yet

Explicitly mention MCTS and GPU-specific work if still unjustified.

## Next RTX 3060 experiment

Give exact short validation procedure but do not execute it.

---

# Success criterion

The goal is NOT:

"reduce GPU idle time"

because there is no GPU in this task.

The goal is:

"identify and reduce the largest remaining hardware-independent component of the worker-visible inference round trip while preserving identical AlphaZero search semantics."

Do not optimize based on intuition.

Measure → isolate → hypothesize → test → fix one thing → benchmark again.