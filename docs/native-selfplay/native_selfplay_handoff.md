# Native Soo Self-Play — Handoff

**Start here** if you are picking this work up in a new session, especially on
the GPU training host.

Everything below is current as of the A0 (heuristic-free) investigation. **All
six gates A–F are closed and the native backend is in production use.** The
project is no longer a C++ port; it is an AlphaZero training system, and the open
questions are about training dynamics rather than about the backend.

If you are picking this up cold, read this file, then §6 of
[soo_scratch_training.md](soo_scratch_training.md) — that section contains the
result that matters most and is the least obvious.

---

## 0. Where things stand right now

**Heuristic-free (A0) training works, it is producing a stronger network, and
architecture scaling has been tried in both directions and closed.**

| | |
|---|---|
| phase | **A0** — `bootstrap_prior = none`, 128 simulations |
| architecture | **`128x6`, frozen.** Both `128x12` and `256x6` were built, measured and rejected — §3.0.2 |
| run | `soo-scratch-20260822`, outside the repo at `/workspace/alphadiamond-training`, at step **51,450** |
| shipped | **step 44,250**, not the newest — 51,450 measured *weaker*, see below |
| health | 96.5 % of games completing (94.4–98.3), median 65 moves, over 23 iterations |
| strength | **+394 Elo** against the checkpoint A0 started from, 90.6 % over 32 opening pairs [95 % CI 82.8–96.9] |
| throughput | **495 terminal samples/s** rolling (median 498, IQR 5.8 %) |

The single non-obvious thing to know before changing anything: **A0 stability is
decided by the fraction of policy targets produced by a search strong enough to
improve on the network's own prior.** 128 simulations is enough, 64 is not, and
neither censoring nor actor-refresh cadence is the operative variable — both were
tested directly and neither explains it. §3.0.1 below, and §6 of
[soo_scratch_training.md](../model-training/soo_scratch_training.md).

### Two numbers that were being conflated, and are now separate

The old headline of "~247 terminal samples/s (A0 at 128)" was a **frozen-actor
gate** figure from §6.6 of the training document, not a production one.  The same
run's ledger, over its A0 iterations at 128 simulations and *before* any
optimisation, already read **423 samples/s**.  Comparing a candidate against 247
therefore compared it against the wrong ruler, and that is what `128x12` was
asked to beat.

Keep these apart and label every future measurement as one or the other:

| measurement | what it is for |
|---|---|
| **frozen-actor throughput** | the inference/self-play engine itself; the A/B ruler for optimisation work |
| **rolling production throughput** | samples per hour while actually training; the input to Elo per wall-clock |

Anything that changes the actor changes the second and not the first.

### The measurement that was wrong for longer

`SooArena` reseeds per move, but evaluation forces `dirichlet_epsilon = 0` and
`temperature = 0` and neither path then touches the RNG.  The seed is inert, so a
matchup cell is a pure function of `(turn order, candidate seat)` and **asking
for 40 games played 4 games ten times**.  Every per-game confidence interval this
project quoted was over pseudo-replicates: the `40W-0L` was `4W-0L` and the
`30W-10L` behind the retracted `+191 Elo` was `3W-1L`, which is why it landed on
a multiple of ten.  Use `tools/arena_v2.py`, which varies the opening instead of
repeating the game, keeps play deterministic, and takes its interval from a
bootstrap over opening pairs.  §3.0.3.

### Training past step 44,250 has not bought anything

The run continued to step 51,450 — 24 further iterations, 7,200 training steps —
and the result is worth recording because it is the opposite of the assumption
that a later checkpoint is a better one.  On the cert opening suite at 128
simulations, **51,450 scored 26W–38L against 44,250 over 64 games**: 40.6 % over
32 pairs, CI 29.7–51.6 %, Elo **−66 [−150, +11]**.  Not separated, so this is not
"51,450 is worse" with confidence — but it is certainly not better, and the point
estimate is negative.  Completion drifted the same way, 97.4 % down to 96.5 %.

So `runtime/runs/soo/soo-scratch-20260822/latest.pt` stays at **44,250**, and the
Qt artifact is exported from it.  Check a new checkpoint on the cert suite before
shipping it; a later step number is not evidence.

**The next experiment** is not a bigger network.  It is the search budget: 64
fails as a teacher and 128 works, and the threshold between them has never been
located.  Start at 96 — generation quality on 768 fresh games first, then a
frozen-actor learner arm over about one replay turnover, judged on whether the
*learner* holds its actor's completion rather than on actor health.  If 96 holds,
try 80; if it breaks, try 112.  A budget of 112 would already be a 12.5 % saving
with no architecture change, and the callback fix in #27 raised the share of
wall-clock that simulations account for, so the payoff is larger than it was.

---

## 1. What this project is

Moving the Soo self-play/search hot path out of Python multiprocessing into a
native multithreaded subsystem, while training, replay, checkpoints and the
PyTorch model stay in Python.

Read in this order:

| document | what it is |
|---|---|
| [native_selfplay_phase0.md](native_selfplay_phase0.md) | **The contract.** Measurements, proposed shapes, the four gates, the risk list. A design record — *not* rewritten as work lands. |
| [native_selfplay_phase1_progress.md](native_selfplay_phase1_progress.md) | **What actually happened.** Per-gate results, every number measured, every bug found and why it was possible. |
| this file | Where to resume, how to run things, what must not break. |
| [rtx5060_bottleneck_findings.md](rtx5060_bottleneck_findings.md) | The GPU-host measurements Phase 0 rests on. |

---

## 2. Gate status

| gate | question | status |
|---|---|---|
| **A** | rules / encoding / prior parity | **pass** — 1,327-position corpus, exact |
| **B** | deterministic MCTS parity | **pass** — bit-identical q, identical request sequence |
| **C** | native + batcher throughput, no Python | **pass** — 8.2x–18x the Python ceiling at measured value-only latencies |
| **D** | end-to-end with PyTorch | **pass** — 18.5x the Python backend at cap 32, same host, same seeded work |
| **E** | stochastic MCTS: Dirichlet + temperature | **pass** — distribution parity with Python, deterministic per seed, 16 lanes → 16 games |
| **F** | native backend behind the pool contract | **pass** — episode parity with `SooSelfPlayRunner`, 18.5x end to end, trains from scratch |

Beyond the gates, the backend is now in production use: the from-scratch Soo run
(`docs/soo_scratch_training.md`) has trained tens of thousands of steps on it,
peaking at **1,444 samples/s — 25.9x the Python backend** in the real training
loop, which is past Gate D's 18.5x microbenchmark headline.

Merged: #2 (A), #3 (B), #4 (gui CI fix), #5 (C), #6 (D-on-CPU).

---

## 3. What is left

### 3.0 Done on the GPU host (progress doc §10)

1. **The A/B throughput table.** Done. **18.5x** at the production cap of 32,
   against a Python baseline re-tuned on the same host (3,573 evals/s at 32
   workers, *not* the 641 in `rtx5060_bottleneck_findings.md` — that is a
   different, much weaker machine and none of its absolutes transfer).
2. **Risk 5.** Answered. The evaluator thread *is* the ceiling — 98.8–99.6 %
   busy in all 20 configurations measured — but 86 % of that is the forward
   itself and only ~1 % is the boundary. The ceiling is the GPU, which is where
   the design wanted it.

### 3.0.1 The A0 investigation — read this before changing anything

Three attempts to switch off the bootstrap heuristic failed and were rolled
back. The full account is §5.4–§6.9 of the training document; the conclusion is
short and it is not what any of the intermediate hypotheses predicted.

**What decides whether A0 training is stable is the fraction of policy targets
produced by a search strong enough to improve on the network's own prior.**

At 64 simulations the root search does not reliably beat the prior it started
from, so its visit distribution is not a policy-improvement target and training
on it degrades the network. At 128 it does, and the loop becomes stable. Three
frozen-actor arms, each four iterations from the same checkpoint and replay:

| generator | targets from 128-sim search | actor | learner | |
|---|---|---|---|---|
| flat 64 | 0 % | 93.2 % | 86.3 % | degrades |
| flat 128 | **100 %** | 97.8 % | **97.9 %** | **holds** |
| repetition trigger | 5 % | 98.0 % | 91.5 % | degrades |

Two earlier explanations were wrong and are recorded as wrong, because both are
plausible enough to be re-invented:

- **It is not the actor-refresh feedback loop.** The learner degrades with the
  actor completely frozen.
- **It is not the censored dataset.** The repetition trigger censors the *least*
  of the three (9.6 % of moves, against flat-128's 11.6 %) and its learner
  degrades anyway.

It also explains why B0 was always healthy at 64 simulations: B0 does not use the
neural prior at all — the vacancy heuristic replaces it — so the search is
anchored externally and the self-referential loop that fails in A0 does not
exist.

**Production A0 is therefore flat 128 simulations**, at ~247 terminal samples/s
against B0's ~1,444. That is the price of heuristic-free training at this network
size, and it is the number any proposed improvement has to beat.

### 3.0.2 Architecture scaling — tried both axes, both rejected

Both arms were function-preserving transplants of the step-44,250 network, both
carried the parent's AdamW moments (mapped by parameter *name*: optimizer state
is keyed by position in `model.parameters()`, and growing the model shifts those
positions), and both were judged on **held-out fit against a frozen replay of
128-simulation searches** — not on self-play loss, which pitfall 7.14 rules out,
and not on a gate taken before the new parameters had switched on.

**Depth, `128x12`.**  Four initialisations were tried and each fixed a real
defect in the previous one:

| initialisation | gate after 300 steps | new-block contribution | held-out CE vs shallow |
|---|---|---|---|
| random branch, LayerNorm gamma = 0 | 0.0019 | 0.0009 | — |
| copied branch, gamma = 0 | 0.0032 | 0.0010 | — |
| copied branch, appended, ReZero alpha | +0.0021 | 0.0007 | −0.0001 |
| copied branch, **interleaved**, alpha | **+0.0095** | **0.0060** | **+0.0026** |

Three things worth keeping.  Zeroing a LayerNorm *scale* is not a residual gate:
that parameter sits inside the branch and before a nonlinearity, so it rescales
the representation rather than the amplitude, and `GELU(gamma z) != gamma GELU(z)`.
A copied branch behind a shut gate still random-walks — Adam normalises by the
gradient's own second moment, so a tiny noisy gradient still buys full-size steps
(27 % of init norm in 300 steps, cut to 8 % by copying).  And appending puts
`copy(b0)` after `b5`, which is not the distribution it was trained on;
interleaving raised the gate fourfold and made all six blocks open in the same
direction.  None of it helped: the best arm fit the teacher *worse*.

**Width, `256x6`.**  A staged-training expansion, `W -> [[W,0],[0,W]]`, so
1.47 M cross-half weights start at zero and receive gradient immediately —
deliberately not plain Net2Wider, whose duplicated channels see identical
gradients and never separate (confirmed: that control's half-asymmetry stayed at
**exactly** 0.000e+00).  Here the capacity demonstrably switched on — asymmetry
grew from 5e-7 to 3.8e-3 and cross-half weights to 8 % of in-half — and held-out
CE still degraded monotonically against the shallow control:

    step    150     300     600     900    1200    1500
    CE   +0.0024 +0.0051 +0.0102 +0.0128 +0.0124 +0.0170

So the two failures are different, and the width one is the stronger evidence:
depth never opened, width opened and cost more anyway.  The accurate conclusion
is narrow, and worth stating exactly:

> **At the current 128-simulation teacher, replay size and AdamW regime, there is
> no evidence that `128x6`'s representational capacity is the binding
> constraint.**

Not "`256x6` is a bad architecture" — a 4x model may well want its own learning
rate, regularisation and data scale.  But the objective here is Elo per
wall-clock, and a candidate that already costs 4x while fitting the same teacher
worse does not justify starting that tuning project.

One measurement saved the width arm from a wrong answer, and it is the reason to
validate a growth operator empirically rather than trusting the algebra: Adam's
second moment cannot expand block-diagonally.  Zeroed `exp_avg_sq` on 1.5 M new
weights makes their first update ~3x normal size, and held-out CE spiked **+0.266**
within ten steps — a hundred times the architecture effect.  `exp_avg` stays
structural (a weight with no history has no direction); `exp_avg_sq` tiles, so a
new weight inherits its layer-mates' scale.  The same measurement then read
+0.0006.

### 3.0.3 Arena — what the old numbers were actually measuring

`tools/arena_head_to_head.py` had two defects, both fixed, and the second
invalidates every strength interval recorded before it.

It read `self_play.max_moves` (500) rather than `arena.max_moves` (2000), so
every arena was a 500-move probe.  And because evaluation zeroes both
`dirichlet_epsilon` and `temperature`, neither `select_from_visits` nor
`add_dirichlet_noise` consumes randomness, so the per-move seed is inert and the
four balanced cells are pure functions of `(turn order, candidate seat)`.  Forty
games were four games, ten times each — confirmed by identical move counts and
identical final camp occupancy across repeats.

`tools/arena_v2.py` replaces it: openings vary, play stays deterministic
(temperature would change the question from "whose best play is stronger" to
"whose sampled play is stronger"), openings are generated **per turn order**
because an action sequence is only legal under the order it was made for, and
the interval is a bootstrap over **opening pairs** rather than games, since the
two games of a pair share a position.  Two disjoint suites, `--suite dev` for
iterating and `--suite cert` for promotion decisions, because selecting
candidates against one suite lets the chooser overfit it.

Aborted pairs are dropped and reported, and above `--max-abort-fraction` the
result is labelled CENSORED rather than given an Elo — these rules have no draw,
so scoring an unfinished game 0.5 would be an invention.  That case is real: at
64 simulations, 6 of 32 pairs did not finish, and "the search cannot resolve
19 % of these positions" is the more useful finding than the rating of the rest.

### 3.0.4 Rolling a replay store back

Undoing one bad iteration by hand broke the next run twice.  An orphaned chunk
file conflicts with the regenerated game even after its manifest entry is gone,
and `manifest.aborted` — which holds no samples and which `load_buffer` never
reads — is checked by `ingest_episodes`, so a rolled-back abort keeps its
`game_id` permanently unable to complete.  Use `tools/replay_transaction.py`:
snapshot before an iteration, and `restore` puts `chunks`, `game_ids`, `aborted`
and `rng_state` back as one unit and disposes of unreferenced chunk files.
`verify` reports all of this up front.  Both failures are pinned by
`tests/alphazero/orchestration/test_replay_rollback.py`.

### 3.1 The actual remaining work, in order

1. ~~**Stochastic MCTS in the native backend.**~~ **Done — Gate E**, §11 of the
   progress doc. Dirichlet noise at the root and temperature sampling of moves,
   gated on distribution rather than on stream, per §9's RNG policy. 16 lanes
   now play 16 distinct games where they played 1.
2. ~~**`selfplay_backend = "native"`.**~~ **Done — Gate F**, §12. `az_train.py
   --selfplay-backend native`; `"python"` remains the default and the oracle.
3. ~~**A training loop consuming native self-play samples**~~ **Done** — §12.5.

### 3.1.0 Configuring the native backend

Two rules, both paid for:

1. **`games_per_iteration` must exceed `native_lanes`**, and by a good margin.
   Lanes are a fixed pool and surplus jobs are queued; with one lane per job a
   handful of long games hold the whole run at batch 1 for the second half of
   its dispatches (§12.2 — it cost 3x before it was found).
2. **A measurement harness needs `games > lanes` too.** The same rule, and it
   was violated in a benchmark and produced yields wrong by 2.5x (§6.7). Rates
   are unaffected; anything timing-based is not.
3. **Set `native_max_wait_us` from the forward, not from
   `inference.max_wait_ms`.** The batch never fills, so the wait is spent in
   full every cycle; 2 ms against a 0.9 ms forward costs **2.6x**. See pitfall
   7.13.

The measured operating point on the RTX 5090 is `768 jobs / 256 lanes / cap 128
/ 12 threads / 50 µs`: **1,444 samples/s in the live training loop, 25.9x the
Python backend**, evaluator 93 % busy. Lane count and thread count are not
sensitive; the wait is.

### 3.1.1 Worth doing when PolicyValue matters

`policy_value_callback` reaches only **53 %** of its roofline where ValueOnly
reaches 89 %. The gap is the per-row Python `for` loop doing the segmented
softmax. B0 does not use this path, so it is not urgent — but it is measured
now, so it no longer falls under invariant 7.

### 3.2 Known optimisation target, deliberately untouched

The **vacancy prior is 79 % of native lane cost** (7.5 µs/eval vs 1.44 for legal
move generation). It is *not* optimised, on purpose: Gate C showed the lane side
is nowhere near binding, so tuning it would be tuning against an unmeasured
constraint. Revisit only if a GPU-host measurement says it binds.

---

## 4. Environment setup on a new host

```bash
# deps: pybind11 is build-time only and deliberately NOT in build-system.requires
python -m pip install "pybind11>=2.12" pytest
python -m pip install -e .          # works with no compiler; extension is optional

python tools/build_native.py        # -> src/diamond/alphazero/native/_diamond_native*.so
pytest tests/native -q              # gates A-D, ~40s
```

`DIAMOND_NATIVE_ARCH` overrides the `-march` flag (default `broadwell`, the
training host's baseline; `none` disables it).

The extension is **optional by construction**. If it is absent, `tests/native`
skips and everything else stays green — verify with `mv` on the `.so` if you
change anything in that area.

### Fixtures the tests need

- `tests/native/fixtures/positions.jsonl` — committed, regenerate with
  `python tools/build_native_corpus.py` (deterministic, byte-identical).
- `runtime/runs/soo/cpu8h-soo-20260819/latest.pt` — the **immutable step-80
  checkpoint**, `sha256:1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af`.
  Tracked in git and hash-asserted by a test. Every measurement in the design
  rests on it; do not replace it.
- `runtime/runs/soo/soo-scratch-20260822/latest.pt` — the **A0 checkpoint at
  step 44,250**, `sha256:ad0b80321136a1919aac56ec077bf4d786a1454399dc19a78e55c312d7e0a8eb`.
  The strongest network the project has produced, and what the Qt GUI is
  exported from for human play. It is *not* a gate fixture and nothing is
  defined against it — the two files exist for opposite reasons, and the
  step-80 one is the immutable one. Beats the checkpoint A0 started from
  (step 34,650) 40W–0L at 128 simulations, no aborted games.

---

## 5. How to run the benchmarks

```bash
# C.1 native single-search cost vs Python, same machine, full-game corpus
python az-bench/profiles/bench_native_single_search.py --repeats 3

# C.2 scheduler: lane starvation / batch cap / latency sweeps
python az-bench/profiles/bench_native_scheduler.py --mode starve
python az-bench/profiles/bench_native_scheduler.py --mode batch
python az-bench/profiles/bench_native_scheduler.py --mode latency

# D  callback boundary cost + real model through the callback
python az-bench/profiles/bench_native_callback.py --seconds 3
```

```bash
# The Gate D A/B table (GPU host).  The Python side is a full training run:
# drive it with run_point.sh first and pass its self-play seconds in.
python az-bench/profiles/bench_native_gpu_ab.py --device cuda:0 \
    --python-seconds 79 --python-samples 4410
python az-bench/profiles/bench_native_callback.py --seconds 5 --device cuda:0
```

**The training run and its instruments:**

```bash
tools/train_soo_scratch.sh 6                     # B0, heuristics on
tools/train_soo_scratch.sh 6 none A0             # A0 -- but see 3.0.1, use --simulations 128

# A0 readiness, on the production engine and production exploration.
# 2 fixed seeds x 20 games is a regression probe; use fresh seeds and 768 games
# for anything you intend to act on (pitfall 7.15).
python tools/a0_gate.py --checkpoint <ckpt>
python tools/a0_gate.py --checkpoint <ckpt> --games 192 \
    --seeds 20260823 20260824 20260825 20260826 --lanes 256 --threads 12

# What the unfinished games are doing.
python tools/audit_aborted_games.py --checkpoint <ckpt> --games 192

# Durable copy: this host has workspace_is_volume=false.
python tools/backup_training_run.py --publish
```

**GPU-host baseline** (RTX 5090 / Ryzen 9 9950X3D, 16 physical cores), ValueOnly:

| measurement | value |
|---|---|
| Python backend, tuned (32 workers) | 3,573 evals/s |
| native, cap 32, diverse lanes at measured GPU latency | 66,041 evals/s (**18.5x**) |
| native, cap 256 | 147,326 evals/s (41.2x) |
| value-only forward, batch 32 | 0.401 ms |
| callback boundary | ~5 µs/crossing, ~1 % of the forward |
| evaluator thread occupancy | 98.8–99.6 %, every configuration |

**Local baseline for comparison** (8-logical-core laptop, i7-1165G7 — neither
host's absolutes transfer to the other; that is the point of re-running):

| measurement | local value |
|---|---|
| native lane stages, total | 9.43 µs/eval (Python 416.28 on the same machine) |
| whole search, dummy evaluator | 2.6 µs/eval, ~380k evals/s single lane |
| scheduler ceiling, 0 latency | 770k evals/s at 4 threads |
| scheduler at B32 / 3.784 ms | 7,775 evals/s, 92 % of roofline |
| callback boundary | 30–55 µs/batch, 0.4–1.0 µs/eval |

---

## 6. Invariants that must not break

1. **The Python implementation is the oracle and stays.** `game/rules.py`,
   `alphazero/encoder.py`, `bootstrap/heuristic.py`, `mcts/search_2p.py` and
   `game_adapter.py` are not dead code duplicated by C++ — every gate is defined
   as equality against them, and CI re-runs that comparison on every change.
   They are the only defence against risk 1 (two authoritative rule
   implementations drifting). Do not delete them.
2. **Topology is generated, never transcribed.** `native/topology.py` derives
   every board fact from `diamond.game.board` and injects it at import.
   No neighbour, camp or rotation is hard-coded in C++.
3. **Legal action ordering is exact parity, in every mode.** It decides which
   action each Dirichlet noise component lands on. Compare *sequences*, never
   sets — a set-based gate passes a backend that misassigns every noise
   component.
4. **Stochastic MCTS is unimplemented and must refuse, not approximate.**
   `dirichlet_epsilon > 0` or `temperature > 0` raises. The RNG policy is §9 of
   the design: cross-backend bit-exact RNG parity is explicitly *not* required,
   but each backend must be deterministic per seed.
5. **An evaluator must use `SearchSession::pending_state()`**, the node awaiting
   evaluation — never the lane's root. See §7.2.
6. **The batcher thread must not be starved.** See §7.5.
7. **Do not micro-optimise before measuring.** No SIMD, no bitset tricks, no
   custom allocators until a measurement names the constraint.

---

## 7. Pitfalls already paid for — do not rediscover these

### 7.1 The salt must avalanche (Gate C)
The dummy evaluator's per-lane salt was mixed as
`hash ^= salt + K + (hash << 6) + (hash >> 2)`, which moves mostly low bits —
and the value is read from `hash >> 11`, which discards them. Lane values
differed at the 11th decimal place, **32 lanes played 1 distinct game**, and
batching looked flawless *because* every lane marched in lockstep. Throughput
looked healthy the entire time. Pinned by `test_lanes_play_different_games`.

### 7.2 Root state vs node state (Gate D)
`PythonBatchEvaluator` computed the vacancy prior from the lane's **root**
instead of the node being expanded; they coincide only on the first expansion of
each move. Gate C structurally could not catch it — the dummy evaluator derives
priors from a request hash and never reads a state. It took an end-to-end
comparison against the Python oracle to surface.

### 7.3 Batch size changes the model's output
Batch size perturbs the Soo model's values by **~3e-8** — far larger than the
≤1e-12 by which the native and Python vacancy priors differ. Any parity
comparison must hold batch size fixed on both sides, or it tests the model's
numerics rather than the backend. `trunk_only` *is* bit-identical to the full
forward, so ValueOnly is a pure saving.

### 7.4 Exceptions must not escape a `std::thread`
A raising Python callback used to terminate the process. Worker and evaluator
threads capture the first failure, tear the run down so nothing waits on a dead
thread, and rethrow on the calling thread. Keep that when adding threads.

### 7.5 Oversubscription collapses throughput
8 workers + the batcher on 4 physical cores dropped the scheduler from 770k to
260k evals/s: the batcher stopped getting CPU and everything queued behind it.
Leave a core for the batcher when choosing thread counts on the 36-core host.

### 7.6 `games ≈ 2 × max_batch`, not more
512 lanes at cap 32 gave the same throughput as 64 lanes while inflating
per-eval latency 3.1 ms → 49 ms. `max_batch` is the only knob that moves
throughput. This contradicted the pre-measurement expectation.

### 7.7 Tie-breaks need tied priors to test
With distinct priors, no two PUCT keys are ever exactly equal, so the
`(-score, action)` tie-break is never consulted and a reversed tie-break passes
every test. Production ties constantly — the vacancy prior gives equal
probability to every action with the same integer progress score. Gate B runs a
**uniform-prior** evaluator specifically for this.

### 7.8 Step-beats-jump can never fire
Jump chains land on `source + 2v`, whose cube distance is always even; a step
destination is at distance 1. The sets are disjoint. Python's
`if landing not in moves` guard and its C++ mirror are both unreachable. Kept
deliberately — the mirror is exact, and the corpus cannot police a divergence
there.

### 7.9 A real evaluator has no salt (Gate D, GPU host)
§7.1 is about the dummy evaluator's salt. The mirror-image trap is that a *real*
model callback has no salt at all — it is a pure function of the position — so
in the only mode native MCTS accepts (`epsilon = 0`, `temperature = 0`) all
lanes from one opening play **one** game. Measured: 32 lanes, 1 trajectory.
Throughput does not notice, because a dense forward is indifferent to equal rows
and the evaluator is 99 % of wall. `test_lanes_play_different_games` cannot
catch it — it tests the component that has the salt. This is *why* Gate E
existed; it is now pinned from both sides, by
`test_a_real_evaluator_locks_lanes_together_without_exploration` (the cause) and
`test_exploration_makes_lanes_diverge_under_a_real_evaluator` (the fix).

### 7.10 A permissive RNG policy makes stream tests worthless (Gate E)
§9 allows the native draw sequence to differ from Python's. The consequence is
easy to miss: **no comparison of sequences can then catch a wrong sampler.** An
inverted boost exponent, a double-normalised weight vector, noise applied after
the first selection — each produces a perfectly plausible stream. Only the
distribution shows them. Gate the samplers on moments, CDFs and frequencies, and
expose whatever the search would otherwise hide (`root_priors` exists solely so
the Dirichlet mixture is observable). Related: at α = 0.3 a quantile-for-quantile
comparison is statistically hopeless — see §11.5.

---

### 7.11 A training sample is the root, not the pending node (Gate F)
§7.2 says an *evaluator* must use `pending_state()`. The mirror image bites
whoever records training data: after a completed search, `pending_features()` is
whichever leaf was expanded last, so recording it mislabels every sample's
player-to-act — and since the value target is `+1` when
`canonical_player_ids[0]` is the winner, **half the labels come out inverted**.
The games are legal, the policies are right, and the loss still goes down. Only
an end-to-end comparison against the oracle finds it. Use
`SearchSession::root_features()`.

### 7.12 A deterministic parity test may be comparing nothing (Gate F)
Cross-backend episode parity is only definable where both backends are
deterministic — and **deterministic self-play does not terminate**. Greedy play
from the opening burns the full 2000-move cap at 4, 8 and 16 simulations, so
both sides produce zero samples and every field compares `0 == 0`. The first
version of the Gate F parity test passed that way. Start such comparisons from a
near-terminal corpus position, pin it by tag (depth does not predict
termination: turns 300 and 375 never finish while 340 finishes in six moves), and
assert the comparison is non-empty.

### 7.13 The native batcher's wait is not the coordinator's wait
`inference.max_wait_ms` belongs to the **Python** coordinator, where a batch
takes milliseconds to assemble and milliseconds are the right unit. Deriving the
native batcher's wait from it costs **2.6x**. With 256 lanes running
64-simulation searches only ~50 are ready to submit at any instant, so the batch
never reaches the cap and the batcher spends the *whole* wait, every cycle — 2 ms
of it against a 0.9 ms forward. Measured on the training checkpoint: 2000 µs
gives 355 samples/s at 39 % evaluator occupancy, 50 µs gives 1,077 at 88 %.

Two things make this hard to spot. Lane count and thread count change *nothing*,
so the obvious knobs all read as dead ends; and worker threads sit at 4 %
utilisation while the evaluator reads 46 %, so neither number points at the gap
between them. Lowering the *cap* to meet the lane supply is also worse, not
better — it shrinks batches without removing the wait. Keep the cap high and the
wait short. `native_max_wait_us` is a separate knob for this reason.

### 7.14 Loss is not a health metric for a self-play loop
Both failed A0 switches showed training loss *improving* — 3.06 → 2.40 and
2.94 → 2.18 — while the thing that mattered collapsed. Loss measures how well the
policy head reproduces the targets the current search produced; it says nothing
about whether that search-policy loop generates good games, and since only
completed games reach replay, the dataset is progressively censored while the
number improves. Health metrics in priority order: **completion rate**,
move-count distribution, abort rate and reasons, throughput, external strength,
and loss last as a training sanity check.

### 7.15 A small fixed gate cannot estimate a population rate
The 40-game gate exists to compare checkpoints, and it is good at that. It is not
an estimator: 39/40 carries a Wilson interval of roughly 87–99.6 %, so it cannot
distinguish 92 % from 98 %. This was over-read twice, in both directions —
once to conclude B0 had a structural ceiling, once to conclude it had cleared
one. Use the fixed 40-game set as a **regression probe** and a fresh large
sample (768 games, SE ~0.7 pt) as a **promotion estimate**, and never mix them
in one trajectory.

### 7.16 The pinned snapshot goes stale against `tools/`
`train_soo_scratch.sh` imports `diamond` from a pinned copy of `src/` so repo
work cannot reach a run in flight — but `tools/az_train.py` is read from the
working tree. Change a signature in `src/` and the two halves disagree, with a
`TypeError` at the first self-play call rather than at startup. Re-pin the
snapshot whenever `src/` changes.

## 8. Repo conventions

- **Branch + PR for everything**; `main` is the only long-lived branch and is
  kept clean of stale branches.
- **CI jobs**: `core` (3.11/3.12/3.13, no compiler, no pybind11 — proves the
  extension stays optional), `gui` (Qt, offscreen), `lint` (changed files only),
  `native (Gate A-F)` (builds the extension, installs CPU torch, asserts the
  checkpoint digest, runs all gates).
- **CI must not be allowed to skip silently.** Both the `gui` job and the
  `native` job assert their fixtures are present in a dedicated step, so a
  missing dependency is a red build rather than quiet skips. Follow that pattern.
- `.serena/` is **deliberately deleted and untracked**. Do not re-add it; stage
  explicit paths rather than `git add -A`.
- Benchmarks live in `az-bench/profiles/bench_*.py`, out of production code.
