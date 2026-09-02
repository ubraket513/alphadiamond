# Min A0 transition status — 2026-09-02

## Executive summary

Min has **not** transitioned from B0 to A0. The durable production state is the
iteration-100 champion at training step 133,376, while the active configuration
still blends the canonical vacancy prior at weight `0.50`. Iteration 101 was
started but deliberately interrupted during `SELF_PLAY`; it produced no replay
ingest or learner update, so iteration 100 remains the authoritative recovery
point.

The latest transition check is decisive. A read-only production-shape probe of
the iteration-100 checkpoint at prior weight `0.25` completed only 12 of 256
games (4.69%); 244 games hit the 500-move cap. The required completion gate is
97%. Since `0.25` fails, the ordered curriculum does not authorize an A0
(`0.00`) probe or production switch.

The experiment did produce useful engineering and scientific results:

- the learner, search-target, policy-fit, and transition paths are now
  observable and ledgered;
- deeper serial search and the tested adaptive-search trigger do not solve the
  A0 failure;
- legal-set policy loss fixed the objective mismatch but did not make the
  learned policy behaviorally independent of the vacancy prior;
- prior annealing is reversible and records named rollback gates;
- B0/weight-0.50 production remains stable, but further training at that point
  did not make weight 0.25 viable;
- the long run recovered cleanly from a full `/dev/shm` replay-transaction
  failure without losing checkpoints, durable replay, or audit JSON.

## Terminology and phase status

The run directory retains the historical name `min-b0-6h-20260831`, and the
experiment root contains `min-a0` because this work investigates an A0
transition. Neither name proves a phase transition.

For this experiment:

| prior weight | interpretation | observed status |
|---:|---|---|
| `1.00` | full vacancy-prior B0 endpoint | passed |
| `0.75` | annealing stage | passed bounded segment |
| `0.50` | annealing stage / current operating point | stable production loop |
| `0.25` | annealing stage required before A0 | failed twice |
| `0.00` | A0, network prior only | not authorized and not adopted |

The controlling specification says Min remains B0 until the final
two-checkpoint A0 acceptance gate passes. That gate has not been attempted,
because its prerequisite stages have not passed.

## Starting point

The released baseline was Min v1.0.1 iteration 25, training step 27,776, with a
valid format-v3 checkpoint, restored optimizer state, a one-million-sample
binary replay, FP32 inference, 1,024 games per production iteration, 128 MCTS
simulations, and 1,408 learner steps per iteration.

The initial learning diagnostic classified this checkpoint as
`LEARNING_BUT_NOT_BEHAVIORALLY_READY`:

- gradients and parameter updates were finite and non-zero;
- mean relative policy-head updates were `3.18e-6` and `2.88e-6`;
- held-out full KL was `3.07686` nat;
- start-to-end policy KL across 256 diagnostic learner steps was only
  `2.83e-11`;
- policy loss remained approximately `6.063785`.

This ruled out a disconnected or numerically dead learner, but showed that the
updates barely changed the policy distribution.

## Experiment sequence

### 1. Observability and legal-policy learning

The branch added durable search-target metrics, legal/full policy diagnostics,
parameter-group gradient and update diagnostics, and a read-only Min learning
diagnostic. It also changed the production policy objective to optimize over the
authoritative legal action set. Existing checkpoint/replay compatibility and
legacy provenance were preserved.

The legal loss addresses a real mismatch: the full 5,329-action softmax spends
most of its mass outside the legal set. It does not, by itself, guarantee that
the network learns a directional policy strong enough to replace the heuristic.

### 2. Serial-search sweep

The paired 256-game sweep compared B0 and A0 at 128, 256, and 400 simulations,
plus adaptive 256/400 arms. B0 completed 256/256 games in every arm. Every A0 arm
completed 0/256.

Increasing search depth reduced legal KL but did not improve completion. The
256- and 400-simulation A0 arms merely changed the terminal failure from the
move cap to the 180-second deadline. The adaptive trigger fired on only about
`9.77e-6` of moves and was behaviorally equivalent to A0-128. The measured
classification was `NO_DEEPER_SEARCH_BENEFIT`, so parallel MCTS was correctly
rejected and reversible vacancy-prior annealing was selected.

### 3. Prior annealing

The durable transition ledger records:

| UTC time | iteration | transition | result |
|---|---:|---:|---|
| 2026-09-01 21:26:47 | 31 | `0.50 → 0.25` | attempted |
| 2026-09-01 21:44:17 | 32 | `0.25 → 0.50` | rollback: `max_moves_completion_below_97_percent` |

At iteration 31, weight 0.25 completed 107/1,024 games (10.45%) and aborted 917
at the 500-move cap. It generated only 40,670 completed-game samples. The
rollback restored weight 0.50, after which iteration 32 returned immediately to
1,024/1,024 completion.

Weight 0.50 then remained operationally stable through iteration 100. Isolated
single max-move aborts occurred at iterations 51, 63, and 73; each iteration
still passed the 97% gate, and the failures were not consecutive. Iterations 74
through 100 were otherwise observed at 1,024/1,024 completion, including
iteration 100 with 197,494 moves.

### 4. Autopilot run and storage failure

The long-running resume began at 2026-09-01 21:44 UTC. At iteration 84,
`REPLAY_INGEST` failed with `cannot write replay transaction`. The failure was
not a model or replay-schema defect: `/dev/shm` was 100% full. Raw
`selfplay.episodes` files occupied about 24 GB even though their contents had
already been committed to the durable replay.

Recovery was conservative:

1. Each candidate raw episode file was matched to its iteration's successful
   `replay-ingest.json`.
2. Only already-ingested raw `selfplay.episodes` files were removed.
3. JSON audit records, checkpoints, the one-million-sample durable replay, and
   the current un-ingested iteration-84 episode were preserved.
4. The same resume command restarted from iteration 84 `REPLAY_INGEST`, accepted
   all 1,024 games, and completed training step 110,848.
5. Subsequent raw episodes were removed only after both replay ingest and train
   completion were present.

This reduced the run from roughly 27 GB to 3.7 GB. At this report's snapshot,
`/dev/shm` uses 8.2 GB of 31 GB and has 23 GB free. The local raw episode files
that were removed are not recoverable locally; their durable replay content and
audit records remain.

The requested ten-hour target was not reached. The first process ran for about
5 hours 6 minutes before the storage failure, and the resumed process ran about
1 hour 36 minutes before it was intentionally stopped for the transition probe:
approximately 6 hours 42 minutes of actual process runtime in total. Continuing
the weight-0.50 loop solely to fill the remaining wall time would not constitute
proper A0 training and was stopped in favor of a direct gate measurement.

### 5. Latest transition probe

The latest read-only probe used the iteration-100 champion and the production
search shape:

- checkpoint step: 133,376;
- model SHA-256:
  `bd0cf4370f14448f7bbb5515c006cdd3215749811ac7af123a0699de968e71b5`;
- prior weight: `0.25`;
- 256 games, 128 lanes, 16 search threads;
- 128 simulations, FP32, batch 256, wait 100 microseconds;
- temperature 1.0 for 20 moves, Dirichlet epsilon 0.25;
- 500-move cap and seed 20260901.

Results:

| metric | result | gate |
|---|---:|---:|
| completion | 12/256 (4.69%) | ≥97% |
| max-move aborts | 244 | 0–7 allowed by completion gate |
| moves p50/p90/p99 | 500/500/500 | not materially worse than B0 |
| legal KL | 0.117481 | diagnostic only |
| legal probability mass | 0.124309 | diagnostic only |
| target normalized entropy | 0.971235 | diagnostic only |
| top-1 agreement | 0.0244141 | diagnostic only |
| cycling games | 0 | pass, but insufficient |
| wall time | 214.284 s | informational |

The result is worse than the iteration-31 production attempt in completion
percentage (4.69% versus 10.45%). More weight-0.50 training therefore did not
move this checkpoint toward the next annealing gate.

The canonical raw result is
`docs/superpowers/reports/2026-09-02-min-alpha025-transition-probe.json`.

## Failure mechanism

The evidence supports heuristic dependence combined with a censored-data
feedback loop.

1. **The network policy is nearly diffuse.** At weight 0.25, normalized target
   entropy is 0.971 and top-1 agreement is 2.44%. Only about 12.4% of full
   softmax mass lies on legal moves. Search can fit the relative legal target
   better without learning a strong global directional preference.
2. **The vacancy prior supplies the missing direction.** At weight 0.50 the
   same production loop completes essentially every game. Reducing its weight
   to 0.25 makes most games wander until the hard cap. Full A0 removes this
   stabilizer entirely and performed 0/256 in the serial sweep.
3. **More serial simulations do not change the attractor.** Deeper search lowers
   legal KL but does not produce terminal trajectories; it spends more compute
   exploring a weak, high-entropy policy.
4. **Abort censoring prevents recovery.** An aborted game contributes no
   training samples. The replay contains successful prefixes and completed
   games, but not the states and recovery behavior from the wandering tail.
   Training on this censored distribution cannot directly teach the policy how
   to escape the failure states; weakening the heuristic then exposes the same
   tail again.
5. **Additional B0 training is not an independent cure.** Weight 0.50 keeps
   generation healthy, but the latest 0.25 probe degraded rather than improved.
   Stable B0 completion is therefore evidence of heuristic support, not evidence
   of A0 readiness.

This mechanism is consistent with all measured facts and does not require a
cycling bug: revisit metrics and cycling counts are low while games still fail
to terminate.

## Gate audit

| requirement | evidence | status |
|---|---|---|
| production search configuration | 128 simulations, FP32, production exploration | pass |
| ≥256 games before decreasing alpha | latest probe has 256 | pass |
| completion ≥97% | 12/256 | **fail** |
| stable move distribution | p50/p90/p99 all at cap | **fail** |
| cycling safety | zero cycling games | pass |
| seat balance | only 12 completions; 7/3/2 first finishers | insufficient |
| two consecutive checkpoint passes | no passing checkpoint | **missing** |
| final 768-game A0 gate | prerequisite 0.25 stage failed | not authorized |
| rollback armed | named rollback supported and previously exercised | pass |

Conclusion: the transition gate is closed. Min remains B0/annealing-stage, not
A0.

## Current durable state and artifacts

- Run root: `/dev/shm/alphadiamond-min-a0-legal-alpha-1/run`
- Durable champion: iteration 100 candidate checkpoint
- Training step: 133,376
- Champion model digest:
  `bd0cf4370f14448f7bbb5515c006cdd3215749811ac7af123a0699de968e71b5`
- Active prior: `canonical-target-vacancy-distance-v2`, weight `0.50`
- Replay size: 1,000,000 samples
- State cursor: iteration 101 `SELF_PLAY`; only `initialize.json` exists and no
  iteration-101 data was ingested
- Training process: stopped intentionally for the gate probe
- Run size: 3.7 GB
- Free shared memory: 23 GB

The state is resumable from iteration 100. Resuming iteration 101 will rerun its
self-play without losing the champion or replay.

## Implemented changes

The integrated branch contains the complete diagnostics, sweep, and annealing
stack:

- MCTS visit-target entropy and durable self-play metrics;
- checkpoint policy-fit and parameter-update diagnostics;
- read-only learning diagnostics and replay-path normalization;
- serial 128/256/400 and adaptive-search sweep tooling with deterministic gates;
- legal-set policy loss configuration and training path;
- weighted vacancy/network prior blending;
- bounded resume segments and durable config transitions;
- semantic predecessor compatibility for authorized transitions;
- named rollback-gate recording and reporting;
- tests for default-preserving behavior, transition legality, rollback
  requirements, schemas, and frozen serial trajectories.

## Integration verification

The complete native build and 43-test CTest suite were run from the integration
branch. The first attempt exposed an environmental prerequisite rather than a
source failure: historical core dumps had filled the 16 GB overlay, preventing
the linker and transaction tests from creating temporary files. Nineteen core
dumps (9.4 GB) were moved, not deleted, to
`/dev/shm/alphadiamond-core-dumps-20260902`, restoring 9.4 GB of overlay space.

After the complete build, LibTorch-linked tests also confirmed the roadmap's
documented wheel-environment requirement. Without the system `libstdc++`
preload, seven LibTorch tests segfaulted; a focused rerun passed with the
preload. The authoritative full verification command was:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  ctest --test-dir build/native-training --output-on-failure
```

Result: 43/43 tests passed, zero failures.

## Recommended next experiment

Do not resume a long weight-0.50 run and do not jump to A0. The next intervention
must address censored failure trajectories while keeping the experiment
reversible. The leading candidate is to retain bounded aborted trajectories as
explicit training data with a defined target/value semantics, or to add a
separately measured recovery curriculum derived from those states. Either change
alters the training-data contract and requires its own design, schema, tests,
read-only comparison, and rollback gate.

Any next attempt should first prove improvement at weight 0.25 on at least 256
games, then repeat on a second consecutive checkpoint. Only after the complete
annealing schedule passes should the 768-game A0 acceptance gate run.

## Final disposition

The software work is suitable for integration because it makes the failure
observable, reproducible, reversible, and auditable. The scientific outcome is
negative: the current Min checkpoint is not ready for A0, and the ten-hour A0
training objective was not achieved. No report or directory name should describe
this run as a successful B0→A0 transition.
