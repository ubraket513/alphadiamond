You are working in the current `alphadiamond` repository.

Your task is to fix the highest-priority measured GPU throughput defect in Soo's AlphaZero inference coordinator:

**the batching deadline is anchored to request arrival time instead of the moment a pending batch actually opens.**

This defect has already been measured on the RTX 3060 and is reproducible as two stable operating regimes under identical checkpoint/config/seeds:

- healthy batching: mean batch ~8.35, self-play wall ~721 s, ~15,309 samples/hour
- collapsed batching: mean batch ~1.05, self-play wall ~1001 s, ~11,020 samples/hour

That is roughly a **39% throughput swing** with identical work.

Do not implement unrelated optimizations in this task.

Do not change MCTS semantics.

Do not vectorize `TorchEvaluator` yet.

Do not implement work stealing yet.

Do not change the default simulation count yet.

Do not add adaptive batching wait yet.

The purpose of this task is:

1. verify the current coordinator behavior;
2. reproduce the defect in a deterministic test;
3. fix the batch-window semantics with the smallest correct change;
4. run CPU tests;
5. prepare exact GPU verification commands;
6. if the GPU host is available in the current environment, run only the minimal verification needed to prove the bistability is gone.

---

# Current repository state

Inspect the repository first.

At minimum read:

- `docs/gpu_profile_findings.md`
- `docs/cpu_profile_findings.md`
- `src/diamond/alphazero/inference/coordinator.py`
- `src/diamond/alphazero/orchestration/selfplay_workers.py`
- relevant inference coordinator tests
- relevant worker/inference integration tests
- `runtime/configs/soo-rtx3060.json`
- benchmark configs under `az-bench/configs/`

Run:

```bash
git status
git log --oneline -15
```

Expected recent main commit includes the RTX 3060 profiling findings.

Do not trust this prompt over the code. Verify everything.

---

# Measured defect

The current coordinator logic is expected to look conceptually like:

```python
submitted_at = monotonic()
...
batch.append(item)
...
if batch[0].submitted_at + max_wait <= now:
    flush(batch)
```

The problem is that `submitted_at` is captured when the request enters the parent/coordinator submission path, before it waits in the coordinator's request queue.

Under backlog:

```text
request arrives at coordinator API
t = 0
submitted_at = 0

request waits in internal queue
...

batching thread finally dequeues it
t = 80 ms

max_wait_ms = 20
deadline = submitted_at + 20 ms = 20 ms

current time = 80 ms
deadline already expired
```

The newly opened pending batch immediately flushes with one request.

If arrivals exceed the resulting batch-1 dispatch capacity, the queue backlog deepens, causing subsequent requests to also arrive at the batching thread already expired.

This creates a self-sustaining collapsed regime.

Measured on the RTX 3060:

Healthy regime:

```text
mean batch             8.35
queue→dispatch p50    16.75 ms
inference p50          5.74 ms
evals/s               272
self-play wall        721 s
```

Collapsed regime:

```text
mean batch             ~1.05
queue→dispatch p50     ~87 ms
inference p50          ~3 ms
evals/s                ~195–196
self-play wall         ~1001 s
```

Two collapsed repeats agreed closely with each other despite identical config, checkpoint, seeds, and game outcomes to the healthy run.

---

# Correct intended semantics

`max_wait_ms` should mean:

> maximum time elapsed since the current pending batch for a model key actually opened.

It should NOT mean:

> maximum age of the first request since it was originally submitted to the coordinator.

If a request has already waited 80 ms in the request queue, and it becomes the first item in a new pending batch at time 80 ms, that pending batch should still receive its full configured 20 ms batching opportunity.

Conceptually:

```text
request arrives          t=0
request waits in queue   0→80 ms
batch opens              t=80
batch deadline           t=100
```

not:

```text
request arrives          t=0
batch deadline           t=20
request dequeued         t=80
immediate flush
```

---

# Design requirement

Introduce an explicit batch-open timestamp.

Do not try to reuse `submitted_at`.

Do not try to reuse `admitted_at`.

Both are stamped before the coordinator batching thread actually opens the pending batch.

A clean design may be one of:

```python
pending: dict[ModelKey, PendingBatch]
```

where `PendingBatch` stores:

- requests
- opened_at

or a parallel structure:

```python
pending: dict[ModelKey, list[_QueuedRequest]]
batch_opened_at: dict[ModelKey, float]
```

Prefer whichever is clearer and harder to misuse.

Do not overengineer it.

---

# Important correctness properties

The fix must preserve all current behavior unrelated to the timing bug:

- batching by `ModelKey`
- max batch size
- bounded admission
- queue capacity
- malformed request handling
- correlated `InferenceFailure`
- response routing
- shutdown
- outstanding request cleanup
- metrics
- thread safety
- CPU operation
- deterministic behavior
- response timeout semantics

Do not weaken any validation.

---

# TDD is mandatory

Before changing production behavior, write a deterministic failing test that reproduces the backlog bug.

Do not rely on hoping that scheduler timing creates the defect.

Create a controlled test.

The test should demonstrate:

1. requests are submitted;
2. they become older than `max_wait_ms` before the batching worker is allowed to process them;
3. when processing resumes, the first newly opened batch still gets a fresh batching window;
4. multiple queued requests are therefore grouped instead of each immediately flushing alone.

The old implementation should fail this test.

The fixed implementation should pass it.

---

# Preferred test strategy

Avoid flaky sleeps as the primary synchronization mechanism.

Use controlled synchronization such as:

- `threading.Event`
- `Barrier`
- a fake evaluator that blocks on an Event
- a test clock if introducing one is justified
- controlled queue backlog

One possible structure:

```text
1. Start coordinator with max_batch_size > 1 and max_wait_ms = small value.
2. Block the batching/evaluator path so a backlog accumulates.
3. Submit a burst larger than one request.
4. Ensure those requests have technically existed longer than max_wait_ms.
5. Release the batching worker.
6. Assert that the next dispatch contains multiple requests.
```

However, make sure the test actually distinguishes:

request submission age

from

batch-open age.

Do not write a test that would pass the old implementation.

---

# Also test normal wait semantics

Add or preserve tests verifying:

- a single request still flushes after approximately `max_wait_ms` from batch-open time;
- reaching `max_batch_size` flushes immediately;
- different `ModelKey`s maintain independent batch timers;
- backlog does not cause every request to dispatch individually;
- shutdown still flushes/fails outstanding requests correctly.

Timing assertions should have reasonable tolerance.

Prefer injected/controlled clocks if the current design permits that cleanly.

Do not refactor the entire coordinator solely to inject a clock unless necessary.

---

# Metrics semantics

Be careful with metrics.

Current metrics include:

- admission latency
- queue-to-dispatch latency
- inference latency
- response latency
- batch size

The fix should NOT falsify queue-to-dispatch metrics.

A request that waited 80 ms in the coordinator queue should still report approximately 80 ms queue-to-dispatch, even though its newly opened batch gets a fresh wait window.

In other words:

```text
queue_to_dispatch
```

continues to measure the user's real request latency.

The new batch-open timestamp is for scheduling semantics only.

Do not redefine historical metrics just to make the numbers look better.

If you add a new metric such as batch-open-to-dispatch latency, only do so if it materially helps validation and remains bounded-memory.

It is not required for the minimal fix.

---

# Production implementation goal

The batching loop should behave conceptually like:

```python
pending = {}

while running:
    now = monotonic()

    flush batches whose:
        opened_at + max_wait <= now

    determine next deadline from:
        batch.opened_at

    dequeue request

    if this ModelKey has no pending batch:
        create batch
        opened_at = monotonic()

    append request

    if size == max_batch_size:
        flush immediately
```

Be precise about when `opened_at` is stamped.

It should be associated with the first validated request being appended to an empty pending batch.

If a full batch is flushed and a later request opens a new batch for the same key, that new batch receives a new timestamp.

---

# Avoid hidden regressions

Pay attention to these edge cases:

### Multiple model keys

Each key needs its own batching window.

Do not use one global `opened_at`.

### Invalid request

An invalid request should not open or distort a valid model batch timer unless that is already current intended behavior.

### Full batch

If max batch size is reached, flush immediately regardless of timer.

### Stop signal

Shutdown behavior must remain deterministic.

### Evaluator exception

Failures must still correlate to every request in that batch.

### Empty pending map

No busy-spin.

### Queue backlog

A deep backlog must not repeatedly produce batch size 1 solely because request age exceeds the batch wait.

---

# Do not implement adaptive batching in this task

There is a separate measured issue where the run spends a long tail at exactly one active lane.

At one active lane:

```text
one outstanding request
→ no batch can form
→ fixed 20 ms wait is pure waste
```

That makes adaptive batching potentially valuable later.

But do NOT combine it with this fix.

Why:

- the current batch-window bug directly affects all load regimes;
- adaptive waiting changes timer policy;
- combining both would make attribution impossible.

First stabilize batching semantics.

Then re-measure the tail.

---

# Do not change `max_wait_ms` defaults in this task

The canonical RTX 3060 config has been restored to:

```text
max_wait_ms = 2
```

Benchmark configs exist separately under `az-bench/configs/`.

Do not change the canonical config as part of this bug fix.

GPU validation should use an explicit benchmark config, likely the 20 ms variant used in the measured s64 runs.

---

# Do not change simulations in production config yet

The GPU profiling shows 64 simulations clearly beat 32:

```text
32 sims:
26/32 completed
p90 moves 282
samples/hour 10,431

64 sims:
32/32 completed
p90 moves 114
samples/hour 15,309
```

That is an important config change, but do not bundle it into this coordinator bug fix.

For validation, use the known 64-simulation benchmark because it removes the 32-sim abort-tail confound.

Production config adoption can be a separate commit afterward.

---

# GPU validation workload

After the code fix and CPU tests pass, validate with the known clean workload:

```text
trained checkpoint:
training_step = 80

simulations:
64

workers:
30

games_per_iteration:
32

max_moves:
2000

max_game_seconds:
900

precision:
FP32

max_wait_ms:
20
```

Use the existing immutable trained checkpoint and benchmark config documented in `docs/gpu_profile_findings.md`.

Every verification run must log:

```text
[resume] loaded ... at training_step=80
```

If it logs:

```text
[init] wrote initial checkpoint
```

the run is invalid.

Do not compare a fresh random network.

---

# Critical GPU verification requirement

The defect is bistability.

Therefore one successful run is not enough.

Run the exact same benchmark at least 3 times if GPU time permits.

Same:

- checkpoint
- config
- simulations
- workers
- games
- seeds
- max_wait
- precision

The goal is to prove the collapsed regime is gone.

Before fix, identical runs produced:

```text
healthy:
mean batch 8.35
q→dispatch p50 16.75 ms

collapsed:
mean batch ~1.05
q→dispatch p50 ~87 ms
```

After fix, we want repeated runs to remain in one stable batching regime.

Do not require identical throughput to the millisecond.

Do require that no repeat shows the pathological combination:

```text
mean batch ≈ 1
AND
queue→dispatch ≫ max_wait_ms
AND
batches ≈ requests
```

---

# GPU success criteria

Primary correctness criterion:

```text
queue backlog no longer causes already-expired requests
to open and immediately flush one-item batches.
```

Operational criteria across repeated identical s64 runs:

- mean batch consistently materially > 1
- max batch reaches useful sizes
- queue→dispatch p50 no longer jumps to ~87 ms
- batches dispatched is far below requests completed
- evals/s no longer has the ~195 collapsed floor
- samples/hour no longer swings by ~39% solely due to batching regime

Record:

- self-play wall
- samples/hour
- completed games
- aborts
- median/p90 moves
- mean/max batch
- queue→dispatch p50/p90
- inference p50
- response p50
- evals/s
- batches dispatched
- requests completed

---

# Important comparison caveat

Do not compare the new fixed run directly against old 32-sim rows as proof of the coordinator fix.

Use:

```text
old s64-w30
old s64-prof
old s64-prof2
```

as the relevant historical comparison.

All three used the same step-80 trained checkpoint and 64-simulation game distribution.

---

# CPU regression tests

Run at minimum:

```bash
pytest tests/alphazero tests/agents tests/tools
```

Expected pre-change baseline from the latest profiling commit:

```text
474 passed
0 failed
```

CUDA-specific tests may skip on CPU-only environments.

Also run focused inference coordinator tests separately during TDD.

---

# Commit discipline

Prefer two commits if appropriate:

1. test reproducing the backlog batch-window bug
2. production fix

Or one clean commit if the repository convention prefers that.

Do not mix documentation-only benchmark changes or unrelated refactors into the production fix.

---

# Documentation

After validation, update `docs/gpu_profile_findings.md` with a new section such as:

```text
## Coordinator batch-window fix validation
```

Record:

- the failing mechanism
- the code-level fix
- focused test
- CPU test suite result
- repeated GPU before/after results
- whether bistability disappeared

Do not delete the old findings.

They are valuable evidence explaining why the fix exists.

---

# Deliverables

Work in this order.

## 1. Repository verification

Report:

- current commit
- current coordinator timing semantics
- relevant existing tests

## 2. Reproduction test design

Explain how the test deterministically makes request age exceed max_wait before batch-open.

## 3. Red test

Implement it and show that it fails on current main.

## 4. Minimal production fix

Implement explicit batch-open timing.

## 5. Focused tests

Run relevant coordinator/inference tests.

## 6. Full CPU regression

Run:

```bash
pytest tests/alphazero tests/agents tests/tools
```

## 7. Code review of the diff

Check specifically:

- timer semantics
- model-key independence
- shutdown
- metrics
- no busy-spin
- no unrelated changes

## 8. GPU validation commands

If GPU is not currently accessible, give exact commands only.

If GPU is accessible and I explicitly authorize execution, run at least 3 identical s64/w30/wait20 iterations from the same step-80 checkpoint.

## 9. Final report

Use this structure:

### Root cause

One concise paragraph.

### Failing test

What it reproduces.

### Fix

Exact scheduling semantic change.

### CPU correctness

Tests/results.

### GPU before

Healthy and collapsed historical rows.

### GPU after

Repeated identical rows.

### Bistability status

Explicitly one of:

- eliminated
- reduced but still present
- not proven

### Throughput impact

Samples/hour and evals/sec.

### Remaining bottlenecks

Rank:

1. 64-sim production config adoption
2. TorchEvaluator vectorization
3. lane scheduling/work stealing
4. adaptive batching wait

Adjust only if new evidence changes the order.

---

# Scope constraints

Do NOT:

- change AlphaZero search semantics
- introduce async MCTS
- introduce virtual loss
- implement speculative leaves
- change multiprocessing transport
- introduce shared memory
- add BF16
- add CUDA graphs
- add torch.compile
- redesign the entire coordinator
- optimize TorchEvaluator in this same change
- implement work stealing in this same change
- implement adaptive wait in this same change

This task should result in one narrow engineering claim:

> A request's age before joining a pending batch no longer consumes that batch's batching window.

Prove that claim with a failing-before/passing-after test and repeated GPU measurements.