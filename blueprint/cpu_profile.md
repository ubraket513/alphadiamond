You are working in the `alphadiamond` repository.

Your task is to perform a CPU-first performance investigation of Soo's AlphaZero self-play pipeline.

Do NOT use the external RTX 3060 machine yet.

The GPU machine is accessed remotely over SSH and incurs hourly cost, so all profiling and architectural investigation that can be done locally on CPU should be completed first.

The goal is not to make the CPU implementation itself maximally fast.

The goal is to understand exactly where Soo spends time during one MCTS simulation and one neural-network inference round trip, identify hardware-independent bottlenecks, fix those bottlenecks locally, and only then prepare a very short GPU validation experiment.

Do not optimize blindly.

Measure first.

---

# Context

Soo currently uses AlphaZero-style self-play with:

- multiprocessing self-play workers
- MCTS running inside each worker
- neural-network evaluation through a centralized inference path
- `InferenceCoordinator`
- `InferenceModelPool`
- `TorchEvaluator`
- synchronous leaf evaluation from MCTS
- worker processes communicating with the parent through multiprocessing queues

The current GPU experiments suggested:

- the RTX 3060 was often mostly idle
- worker-visible inference round-trip latency was much larger than actual GPU compute time
- a representative observed round trip was around 97 ms
- actual GPU evaluator time was around 8 ms
- therefore the GPU itself may not be the primary bottleneck

There is another important observation:

The old CPU training runs were not dramatically slower than some of the GPU runs.

That suggests the bottleneck may exist in the hardware-independent part of the pipeline:

- MCTS traversal
- game-state manipulation
- request construction
- Python object overhead
- multiprocessing IPC
- serialization
- parent forwarding
- queue scheduling
- synchronous waiting
- tree backup

We therefore want to profile the entire MCTS simulation, not only neural-network inference.

---

# Repository inspection first

Before changing anything, inspect the actual repository.

At minimum inspect:

tools/az_train.py

runtime/configs/soo-cpu8h.json
runtime/configs/soo-rtx3060.json

src/diamond/alphazero/orchestration/selfplay_workers.py

src/diamond/alphazero/inference/coordinator.py
src/diamond/alphazero/inference/model_pool.py
src/diamond/alphazero/inference/remote.py
src/diamond/alphazero/inference/protocol.py

src/diamond/alphazero/evaluator/torch.py

src/diamond/alphazero/mcts/search_2p.py
src/diamond/alphazero/selfplay/runner_2p.py

src/diamond/alphazero/game_adapter.py

tests/alphazero/orchestration/test_selfplay_workers.py
tests/alphazero/inference/*
tests covering MCTS
tests covering self-play
tests covering training metrics/reporting

docs/gpu_benchmark_findings.md
docs/gpu_training.md

Run:

git status
git log --oneline -15

The local repository is expected to match remote `main`, but verify that.

Do not modify production code until you understand the actual call path.

---

# Primary question

Answer this:

"When one Soo MCTS simulation runs, where does the wall-clock time actually go?"

We want to decompose one simulation conceptually into:

simulation begins

→ tree selection / traversal

→ leaf state reached

→ game / state / feature preparation

→ inference request creation

→ worker-to-parent IPC

→ parent-to-inference coordinator forwarding

→ coordinator queue / batching wait

→ CPU neural-network evaluation

→ inference response forwarding

→ parent-to-worker IPC

→ MCTS backup / tree update

→ simulation ends

The exact boundaries should follow the real implementation.

Do not force this exact decomposition if the code suggests cleaner boundaries.

---

# CPU-first profiling requirement

All initial profiling must run on CPU.

Do not require CUDA.

Do not require the RTX 3060 machine.

The normal CPU-only pytest suite must continue working.

Use the existing trained Soo checkpoint if available locally.

Do not accidentally profile a freshly initialized/random network when a trained checkpoint was intended.

Record the checkpoint identity / training step used in every performance run.

---

# Phase 1 — MCTS simulation latency instrumentation

Instrument one MCTS simulation using a high-resolution monotonic clock.

Prefer:

time.monotonic_ns()

or the existing repository timing abstraction if one already exists.

We need aggregated timings for stages such as:

MCTS selection/traversal

leaf/state construction

inference wait

backup

total simulation time

Conceptually:

S0 simulation begins

S1 selection/traversal complete

S2 inference request begins

S3 inference response arrives

S4 backup complete / simulation ends

From this derive:

selection_ms

pre_inference_state_ms

inference_round_trip_ms

backup_ms

simulation_total_ms

If the current MCTS code has additional meaningful stages, measure them separately.

---

# Phase 2 — Inference transport latency decomposition

Inside the inference round trip, measure the hardware-independent transport path.

Conceptually:

T0 worker submits request

T1 parent receives request from multiprocessing queue

T2 parent submits request to InferenceCoordinator

T3 coordinator admits request

T4 batch is dispatched to evaluator

T5 CPU evaluator finishes

T6 parent receives coordinator reply

T7 worker receives final reply

We want to separate at least:

worker → parent IPC

parent forwarding delay

coordinator queue / batching wait

actual CPU NN inference

coordinator/parent response delay

parent → worker IPC

worker-visible total inference round trip

Reuse existing `InferenceMetrics` where they already measure stages.

Do not build duplicate metrics for data already available.

Focus new instrumentation on missing boundaries.

---

# Important: separate absolute timing from percentages

CPU inference may be slower than GPU inference.

Therefore do NOT conclude:

"IPC is only 10% of CPU runtime, so it does not matter."

We need absolute milliseconds for every stage.

For example:

CPU:
NN = 70 ms
IPC = 15 ms

GPU later:
NN = 8 ms
IPC = 15 ms

The same 15 ms IPC becomes much more important after moving NN compute to GPU.

Therefore every timing report must include absolute latency, not only percentages.

---

# Phase 3 — worker state accounting

If practical without invasive changes, measure how much time self-play workers spend in broad states:

CPU/MCTS computation

waiting for inference

waiting on IPC/queue

other

The goal is to answer whether workers are mostly:

1. computing MCTS/game logic
2. blocked waiting for NN results
3. blocked because of orchestration/IPC

Do not implement OS-level profiling infrastructure unless necessary.

Application-level instrumentation is preferred first.

---

# Metrics design constraints

This pipeline may generate hundreds of thousands of NN requests.

Do NOT keep an unbounded record for every request or simulation.

Use bounded-memory aggregation.

Prefer the style already used by the repository's current inference metrics.

Acceptable approaches:

counters

sum / count

min / max

bounded reservoir samples

p50 / p90 / p99 summaries

sampling 1/N events if needed

Durable run ledgers should contain summary metrics only.

Do not dump hundreds of thousands of timestamps to JSON.

---

# Desired CPU profiling output

At the end of one controlled CPU benchmark, produce a table like:

MCTS stage                         mean      p50      p90
--------------------------------------------------------
selection/traversal                X ms      X        X
state/request preparation          X ms      X        X
inference round trip               X ms      X        X
backup                             X ms      X        X
total simulation                   X ms      X        X

And:

Inference stage                    mean      p50      p90
--------------------------------------------------------
worker → parent IPC                X ms      X        X
parent forwarding                  X ms      X        X
coordinator batching wait          X ms      X        X
CPU NN evaluator                   X ms      X        X
response forwarding                X ms      X        X
parent → worker IPC                X ms      X        X
worker-visible total               X ms      X        X

Also include:

requests/sec

batches/sec

mean batch size

p50 batch size

p90 batch size

max batch size

completed games/hour

usable samples/hour if benchmark length is sufficient

median game moves

p90 game moves

abort counts and reasons

---

# Controlled benchmark

Use one fixed benchmark configuration.

Do not change several knobs while profiling.

Suggested starting point:

trained Soo checkpoint

CPU inference

32 MCTS simulations

4 self-play workers initially, because this matches the historical CPU baseline

same max_moves as the baseline

same bootstrap-prior semantics

same self-play seeds where possible

same games-per-iteration semantics

Run a short benchmark sufficient to gather thousands of inference events.

We are profiling pipeline behavior, not training strength.

Do not spend hours training.

---

# Then run one CPU concurrency comparison

After establishing the baseline, run one additional CPU profiling point with more workers if the machine permits.

For example:

4 workers

vs

8 workers

The purpose is not tuning.

The purpose is to see whether:

latency increases with worker count

batch size improves

parent forwarding saturates

IPC becomes dominant

CPU NN contention increases

This can help distinguish:

single-request overhead

from

central coordination saturation

Do not perform a huge worker-count sweep.

---

# Important current hypothesis to test

There is a specific current implementation detail that may matter.

Inspect `SelfPlayWorkerPool.run()`.

It may still have a loop conceptually similar to:

forward inference

then

results.get(timeout=0.01)

then

forward inference again

If so, one hypothesis is:

"Episode-result polling delays inference request/response forwarding."

But this is only a hypothesis.

Do not change it before measuring.

Also inspect worker-side response pumping.

A timeout value such as 50 ms on `queue.get(timeout=...)` does NOT automatically mean every response incurs 50 ms latency, because a blocking queue wakes when data arrives.

Measure actual delay.

---

# Profiling methodology

Do application-level timing first.

Do NOT start by running a generic profiler and staring at a giant call graph.

The system crosses multiple processes and threads, so function CPU-time alone may hide blocking/queue latency.

First add explicit boundary timing.

After the application-level latency breakdown exists, use tools such as:

cProfile

py-spy

scalene

or similar

only if they are already available or clearly useful for a specific CPU hot path.

Do not add a dependency solely for convenience without justification.

If a generic profiler is used, profile:

MCTS selection hot loops

game-state operations

feature encoding

Torch CPU forward

serialization/pickling

but treat it as complementary evidence.

---

# Root-cause decision tree

After profiling, classify the primary bottleneck.

Case A:

Inference round-trip dominates simulation time.

Then inspect its internal breakdown.

If parent forwarding / IPC dominates:

optimize transport before touching MCTS semantics.

Case B:

CPU neural-network evaluation dominates.

Then GPU acceleration is likely worthwhile, but still record the fixed transport overhead that will remain after migration.

Case C:

MCTS selection / game-state logic dominates.

Then profile and optimize those CPU hot paths before doing further GPU work.

Case D:

No single compute stage dominates, but workers spend most of their time blocked/scheduled.

Then multiprocessing/orchestration architecture is the likely problem.

Do not assume which case will win.

---

# Stage 2 — choose one fix only after measurement

After the first CPU profile, identify the largest avoidable hardware-independent component.

Form one explicit hypothesis:

"I believe X is the primary avoidable bottleneck because Y ms out of Z ms is spent there."

Then implement only one performance change.

Examples depending on evidence:

- decouple inference forwarding from episode-result polling
- remove unnecessary queue hops
- eliminate avoidable serialization
- reduce an identified MCTS Python hot loop
- reduce game-state copying
- reduce repeated feature construction
- reduce unnecessary CPU tensor creation

Do NOT combine multiple optimizations in one benchmark.

We need attribution.

---

# If parent forwarding is the culprit

If measurements show inference requests/replies wait substantially in the parent bridge, then consider separating inference forwarding from episode result collection.

Potential architecture:

main/result collector

independent inference request pump thread

independent inference response pump thread

using blocking queue operations rather than application-level polling

Conceptually:

workers
   ↓
multiprocessing request queue
   ↓
dedicated request pump
   ↓
InferenceCoordinator
   ↓
CPU evaluator
   ↓
dedicated response pump
   ↓
worker response queues

Requirements:

preserve correlation IDs

preserve worker response routing

preserve shutdown semantics

preserve error propagation

preserve per-game timeout

preserve catastrophic pool timeout

no busy spinning

no unbounded queues

no response loss during shutdown

Do not implement this unless measurements support it.

---

# Do not parallelize MCTS yet

Do NOT implement:

tree parallelism

virtual loss

async MCTS

speculative leaves

batched leaf selection

multiple outstanding leaf evaluations per one MCTS tree

These change search semantics.

Our first goal is to remove implementation overhead while keeping current sequential MCTS behavior unchanged.

Only reconsider MCTS parallelism after CPU/system overhead has been measured and optimized.

---

# Do not optimize GPU code

No CUDA optimization in this task.

Do not implement:

mixed precision

BF16

CUDA graphs

torch.compile

custom kernels

pinned memory

CUDA streams

GPU evaluator restructuring

The GPU machine is not needed for the current investigation.

---

# TDD requirements

All production behavior changes must use TDD.

Instrumentation should also have tests where practical.

Tests should cover:

metric aggregation is bounded

metric summaries are correct

timing instrumentation does not alter MCTS outputs

sequential MCTS semantics remain unchanged

worker request/response correlation remains correct

existing worker timeout behavior remains correct

individual 900-second game abort behavior remains correct

catastrophic pool timeout remains correct

CPU-only tests pass

For concurrency changes, use:

threading.Event

fake coordinators

barriers

controlled queues

fake/injected clocks

instead of flaky wall-clock sleeps.

Do not write tests that rely on waiting real milliseconds unless unavoidable.

---

# Before/after benchmark discipline

Any optimization benchmark must use exactly the same:

checkpoint

worker count

simulations

max_moves

bootstrap prior

inference batching configuration

seeds where practical

before and after.

Report both runs side by side.

Success metrics:

MCTS simulation latency

inference round-trip latency

requests/sec

games/hour

usable samples/hour

Do not call an optimization successful solely because one internal microbenchmark improved.

---

# GPU validation comes last

After CPU profiling and at least one justified local optimization, prepare a minimal GPU validation plan.

Do not execute it unless explicitly asked.

The later RTX 3060 validation should be short.

The purpose will be to answer:

"Did the hardware-independent latency we removed on CPU also reduce worker-visible GPU inference latency?"

A 10–20 minute benchmark should be enough.

The GPU validation should reuse the exact same instrumentation added during the CPU phase.

That avoids paying GPU rental time for diagnosis.

---

# Deliverables

Work in this order.

## Deliverable 1 — repository verification

Report:

actual current inference call path

actual current MCTS call path

current relevant metrics already present

current polling/queue architecture

## Deliverable 2 — CPU profiling plan

Specify:

exact files to modify

exact timing boundaries

metric structure

bounded-memory approach

tests

exact benchmark commands

## Deliverable 3 — instrumentation implementation

Implement only profiling/metrics needed for the diagnosis.

Run focused tests.

Run full relevant CPU regression tests.

## Deliverable 4 — CPU benchmark result

Run the controlled CPU benchmark.

Produce the latency decomposition tables.

State where time actually goes.

## Deliverable 5 — one root-cause hypothesis

Choose one measured bottleneck.

State:

"I believe X is the first thing to optimize because..."

with evidence.

## Deliverable 6 — one minimal fix

Use TDD.

Implement only that fix.

Run identical benchmark again.

## Deliverable 7 — before/after report

Include:

simulation latency

inference transport breakdown

CPU NN latency

requests/sec

batch behavior

games/hour where meaningful

samples/hour where meaningful

tests passed

remaining bottlenecks

## Deliverable 8 — GPU validation recipe

Give exact commands for a later short RTX 3060 validation run.

Do not start the GPU machine.

---

# Final report format

Use:

## CPU root cause

Explain the dominant bottleneck in plain language.

## Simulation breakdown

table

## Inference breakdown

table

## First optimization

what changed and why

## Before vs after

table

## Correctness

tests run

## Remaining bottlenecks

ranked

## Is GPU still expected to help?

Explain based on measured CPU NN time versus hardware-independent overhead.

## Short RTX 3060 validation plan

exact configuration and commands

---

# Engineering constraints

Python 3.11+

CPU-only environment

no required CUDA

no new heavy dependency unless justified

bounded profiling memory

monotonic clocks

preserve existing self-play semantics

preserve deterministic identities/seeds

preserve replay/checkpoint behavior

preserve 15-minute per-game abort

preserve catastrophic pool timeout

preserve CPU reference configurations

small commits

one hypothesis / one optimization at a time

Most importantly:

Do not treat "GPU utilization" as the problem yet.

The immediate question is:

"Where does one Soo MCTS simulation spend its wall-clock time on CPU?"

Measure that first.

Everything else follows from the answer.