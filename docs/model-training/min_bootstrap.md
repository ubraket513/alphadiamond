# Starting Min — what is settled, and what is not

> **Handoff document.** Work stopped here to continue on another cluster. §7
> is the state to resume from; §6 is what the next machine has to set up before
> anything builds.

Min had never been trained. This records why it could not be, what was fixed,
which bootstrap prior it should use, and how the scratch network is
initialised — with the measurements each decision rests on, and an explicit
account of what those measurements do *not* establish.

Companion documents: [soo_scratch_training.md](soo_scratch_training.md) for the
Soo run this borrows its method from, and
[repetition_trigger_config_gap.md](repetition_trigger_config_gap.md) for the
first instance of the failure mode in §1.

---

## 1. Why Min could not start: a setting nothing read

`self_play.bootstrap_prior` was parsed by `config.cpp`, validated, written into
the resolved config — and consumed by nothing. `vacancy_prior()` had exactly one
caller in the tree, `profile.cpp`, a timing harness. Self-play priors always came
from the network.

So **no from-scratch run in this tree could produce a single training sample.** A
randomly initialised network cannot steer pieces toward a target camp, a game
ends only when a camp is filled, every game therefore ran to `max_moves`, and an
aborted game contributes nothing. The iteration died two stages later on:

```
insufficient replay samples: requested 256, available 0
```

which reads like a replay bug and is not one.

Reproduced on both families — 0/36 completed for Min, 0/32 for Soo — so it was
never Min-specific. Soo escaped it in practice only because it has a shipped
v2.0.0 artifact to `--warm-start` from. Min has none. That is the whole reason
this was a hard blocker on Min and an invisible one on Soo.

This is the second setting to fail this exact way; the repetition trigger was
the first. Both were implemented in the engine, exposed in config, and dropped
by the wiring in between. The failure is quiet by construction: the search stays
correct and only the completion rate moves.

**Fixed**, with the prior computed on the search worker rather than the
evaluator thread — that thread is the serial resource, and the vacancy prior
costs ~7.5 µs per evaluation, so a batch of 32 would put 240 µs on the critical
path that the workers absorb in parallel. This is what `BatchEvaluator::prepare`
already existed for.

**Tested**, because that is what was missing. `cli_contract_test` asserts the
config string reaches `EpisodeConfig`; `selfplay_test` asserts it changes play,
with temperature and dirichlet both zero so the prior is the only term that can
move a trajectory. The assertion was verified by reverting the supply line —
`selfplay_test` fails, and passes again when it is restored. An assertion that
cannot fail is not coverage.

---

## 2. Which prior: `alphadiamond-min-probe`

With the field live, the choice of bootstrap prior became a real decision, so it
was measured rather than assumed.

The probe holds everything but the prior fixed: authoritative 3P rules, MCTS3P,
the same openings, the same per-game seeds, and a zero leaf value on every arm.
Zero values are the design, not an omission — a value signal would answer a
different question with a heuristic of its own, and pinned at zero each arm
differs in exactly one term. It is self-play only: it never touches the arena, a
run directory, or the ledger.

```bash
alphadiamond-min-probe --arm vacancy|uniform|soo-policy|vacancy-min-value \
    [--match min|soo] [--value-init zero|random] [--network-seed N] \
    [--artifact models/soo/2.0.0] --games N --simulations 64 --max-moves 2000 \
    --lanes 64 --threads 12 --batch 128 --out report.json
```

---

## 3. Soo as a Min teacher: rejected

The idea was reasonable. Soo and Min share the 73-hole topology, the
camp-relative canonicalisation, and the 5329-entry action space; Soo v2.0.0 is a
trained network that plausibly knows more about hops and openings than a
distance heuristic does. It was tested as **policy only**: Min's 3P rules,
MCTS3P and value semantics throughout, with Soo supplying only the prior through
a 3P→2P input fold, its scalar value discarded rather than mapped onto three
seats, and the softmax always taken over the authoritative 3P legal set.

The fold ORs the two opponent occupancy channels — lossless about occupancy,
since a hole holds at most one piece, and lossy about identity.

| arm | games | completion | median | revisit fraction | max revisits | foreign in target |
|---|---|---|---|---|---|---|
| vacancy | 288 | **100 %** | 112 | 0.004 | 1.25 | 0.00 |
| uniform | 288 | 0 % | — | 0.000 | 1.01 | 9.10 |
| soo-policy | 36 | 0 % | — | **0.938** | **400.3** | 0.42 |
| soo-policy on Soo 2P *(control)* | 36 | 91.7 % | 66 | 0.072 | 7.03 | 0.08 |

**The Soo policy does not transfer, and fails worse than no information at
all.** The two failures are different in kind: uniform random-walks and leaves
target camps blocked by foreign pieces (9.1 of them), while soo-policy locks
into a short-cycle attractor — 94 % of observations are repeats, one position
recurring ~400 times — with target camps largely *clear* (0.42). The failure is
repetition, not blocking.

The 2P control is what makes this believable. The same artifact loading, the
same legal-logit gather, the same softmax, the same zero-value search, run on
Soo's own game: 91.7 % completion. So the collapse is the transfer, not the
harness.

### What this does not establish

The experiment changes at least three things at once and cannot say which
dominates:

- **density OOD** — Soo trained against ~10 opponent pieces; the fold presents
  ~20 in a single channel, a board density it never saw;
- **state aliasing** — merging `next` and `last` opponent makes two positions
  with swapped opponent placements identical to Soo, though their 3P transitions
  differ. The folded observation is not a Markov state for a 3P decision, which
  is a stronger defect than "some identity information is missing";
- **objective mismatch** — Soo's policy is trained to win a two-player game,
  which need not help in a three-way finish-order race.

An earlier draft of this file named density OOD as the likeliest cause. That
overstated what was measured. The clean target-camp counts do argue against
blocker retention being the mechanism, but they do not separate the three above.

**Conclusion: closed as a teacher.** Not "wait for more games" — a larger run
would sharpen the confidence interval on a decision already made. A 288-game
arm at 0 % would put the 95 % upper bound on its completion near 1 %.

**Trunk transfer remains open and is a different question.** Soo's policy head
carries exactly the behavioural bias rejected here, but its trunk — adjacency,
directional residual blocks, output norm — may still hold useful board
geometry. §7 sketches that experiment; it must not copy the policy head.

---

## 4. The scratch value head: zero, not random

A fresh Min network answers every leaf with an arbitrary three-vector. During
the bootstrap phase, where the prior is the heuristic's, that noise is the only
thing competing with the heuristic's sense of direction — and it is a preference
between seats the network has no reason to hold.

`diamond_training::zero_value_head()` zeroes `value_linear2` only, **after**
construction rather than inside it, so both arms of an A/B consume the same RNG
draws and differ in that layer alone. It is applied in `train_main`'s model
factory, which every stage calls: zeroing in one stage only would make
INITIALIZE and TRAIN disagree about what the scratch network is, and the scratch
path identifies iteration 0 by the model's digest. A warm start or a checkpoint
overwrites these weights immediately, so neither changes meaning.

768 games, 64 simulations, vacancy prior, values from the network:

| value head | p50 | p90 | p99 | revisit fraction | repeat within 8 | max revisits |
|---|---|---|---|---|---|---|
| **zero** | **111** | **136** | **172** | **0.0031** | **0.0029** | **1.24** |
| random, seed 4242 | 120 | 160 | 222 | 0.0149 | 0.0129 | 1.79 |
| random, seed 7 | 124 | 165 | 279 | 0.0104 | 0.0089 | 1.62 |
| random, seed 99 | 115 | 141 | 191 | 0.0037 | 0.0030 | 1.26 |

Completion is 100 % on every arm, so it does not discriminate; the length and
cycle tails do. Zero is best or tied-best in every column, and the spread across
random seeds is itself the argument: a random head costs a seed-dependent amount
(p99 from 191 to 279), while zero is deterministic and sits at or below the
luckiest draw.

The zero arm is invariant to the network seed by construction — its values are
exactly zero and its priors are the heuristic's — so it is run once. That also
makes it a cross-check against the plain `vacancy` arm, which is the same
configuration reached by another route: p50 111 against 112.

**No config knob was added.** The A/B runs in the probe; the winner becomes the
Min scratch semantics. A `value_head_init` setting would put this on the
resolved config and into checkpoint compatibility for no benefit.

### The gradient trap, checked rather than assumed

`value_linear2 = 0` passes no gradient to `value_linear1` on the first backward,
so the layers below start moving only on the second step — and Min production
runs **one train step per iteration**. `min_value_head_init_test` pins the whole
sequence, asserting the no-movement step as a *premise* that fails loudly if it
ever stops holding:

| | |
|---|---|
| after step 1 | `value_linear2` has left zero; `value_linear1` has **not** moved |
| after step 2 | `value_linear1` has moved |
| across an iteration boundary | `value_linear1` has moved; `training_step == 2` |

The test also pins that same-seed arms are bit-identical outside the value head,
and that the zeroed head returns exactly `[0,0,0]`.

The iteration-boundary case keeps the optimizer and weights in process; it does
not round-trip through a checkpoint. The production path rebuilds the model each
stage, zeroes it, then loads the checkpoint over it — covered by the existing
checkpoint tests, but not by this one.

---

## 5. Not yet measured

- **Value loss under the zero head.** §4 is the search-side half. Whether the
  value target converges stably needs a learner run over real replay, which
  belongs to the B0 baseline in §7.
- **The 288-game soo-policy arm.** Started three times and killed by session
  teardown each time; the 36-game figures in §3 are what exist. Deliberately not
  retried — the decision does not turn on it.
- **Anything about the arena.** See §6.

---

## 6. What the next machine needs

### Build

The project requires a C++26 dialect (`-std=c++2c`), so **GCC 14 or newer** —
GCC 13 rejects the flag outright.

```bash
cmake --preset native-training -G Ninja \
    -DCMAKE_PREFIX_PATH=<libtorch-cmake-dir> \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
```

Using the pip `torch` wheel as LibTorch needs two workarounds:

- **`TORCH_CUDA_ARCH_LIST`** must be set (e.g. `8.9`). Torch's CMake probes the
  GPU arch by compiling a test program that demands the `CUDA23` dialect, which
  the installed `nvcc` may not support; setting the list skips the probe.
- **`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`** when running any
  binary linked against it. The wheel's `libtorch.so` exports `std::filesystem`
  symbols from an older libstdc++ that interpose on the system ones, and
  `remove_all` then segfaults. Without this, five checkpoint tests fail with
  SIGSEGV and look like real breakage.

### Known-red, not caused by this work

- **`native-core (windows-latest)`** fails on `replay_schema_test`. It was
  already failing on `main` at `c89f262` (run 33037688890), the base of this
  work, in code none of it touches.
- **The arena stage times out** and is a genuine blocker for the promotion
  loop, though not for self-play or training. `arena_episode_config` runs
  `lanes = 1, threads = 1, max_batch = 1`, so its games are played strictly one
  at a time; a Min arena is 36 games per opening (3! seat assignments × 3! turn
  orders) at `max_moves` 2000. A full Min iteration reached TRAIN in ~30 s and
  then exceeded a 10-minute budget in ARENA. Self-play and training are proven
  end to end; **the full iteration is not.** Fix before starting a promotion
  loop, separately from any teacher or initialisation experiment.

### Long runs need to survive the harness

Every long background run in this session was killed, including one detached
with `setsid nohup`. Prefer short `--run-dir` segments joined with
`alphadiamond-train resume`: a run directory is idempotent and ledgered by
operation id, so resume replays exactly the work that did not complete.

---

## 7. Where to pick up

In order, each step depending on the one before:

1. **B0 baseline.** Min from scratch, vacancy bootstrap, zero value head, at the
   intended production configuration. Record completion, move percentiles, the
   cycle metrics, and the value-loss trajectory — that last one is the §5 gap.
2. **λ-mix, as a small diagnostic only.** `p = (1-λ)·p_vacancy + λ·softmax(soo_logits / T)`,
   λ ∈ {0.05, 0.10}, T ∈ {2, 4}, 36–72 paired games, both priors normalised over
   the same 3P legal set before mixing. Adoption requires completion unchanged,
   no regression in repeat-within-8 or max revisits, an improvement in **p90 as
   well as median**, and no seat or turn-order regression. If λ = 0.05 and 0.10
   show no length improvement, close the branch — keeping 100 % completion is
   not by itself a reason to admit Soo.
3. **Trunk-only transfer, as an independent A/B.** Copy adjacency, the
   convertible input-projection columns and bias, the directional residual
   blocks, the output norm, optionally `value_linear1`. Reinitialise
   `policy_source`, `policy_destination`, and a zeroed `value_linear2`. For
   occupancy the input mapping is exact — `next_occ` and `last_occ` both take
   Soo's opponent column, and since a hole holds one piece they cannot both be
   1, so the node-level projection reproduces the union fold. The finished
   channels need their own rule, as two opponents *can* both be finished.
   Do not freeze the trunk. Both arms use the vacancy bootstrap and identical
   seeds and openings; the success criterion is **removing vacancy sooner**, not
   B0 completion. Keep the strict warm-start family check intact and add a
   distinct initialisation mode with provenance — `transfer_spec`,
   `source_family`, `source_version`, `source_runtime_sha256`, the copied and
   reinitialised tensor lists.
4. **The A0 gate — much stronger than the historical 8-of-10.** Soo's
   heuristics-off probe passed 20/20 and production completion then fell to
   ~64 %, so a single small probe must not be trusted again. Require: the
   intended production search; ≥ 288 games, 768 at the final switch; two
   consecutive checkpoints passing; completion ≥ 97 %; no opening or seat block
   far below the rest; median and p90 not materially worse than B0; median
   revisit fraction ≤ 0.1; a stable repeat-within-8 and max-revisit tail; and
   automatic rollback armed for several iterations after the switch. Falling
   loss belongs in this gate only as a last, auxiliary indicator.
