# Soo from scratch — training record

The first Soo training run on the native self-play backend, from a randomly
initialised network. Live document: configuration and reasoning are settled,
results accumulate.

Companion documents: [native_selfplay_handoff.md](native_selfplay_handoff.md)
for the backend, §10–§12 of
[native_selfplay_phase1_progress.md](native_selfplay_phase1_progress.md) for how
it was gated.

---

## 1. Status

| | |
|---|---|
| run id | `soo-scratch-20260822` |
| phase | **B0** — heuristics on. A0 attempted twice (steps 14,250 and 22,350) and reverted both times; §5.4 and §5.8 |
| backend | `native` |
| host | RTX 5090 / Ryzen 9 9950X3D (16 physical cores) |
| run root | `/workspace/alphadiamond-training/runs/soo/soo-scratch-20260822` |

### ⚠ This host does not persist

`vast-capabilities` reports `workspace_is_volume: false`. **Nothing on this
instance survives a recycle or destroy** — including the training run. A
stop/start is safe; a recycle is not. The run directory is deliberately *outside*
the repo (`/workspace/alphadiamond-training`, not `/workspace/alphadiamond`) so
that checkpoints and replay chunks cannot silently re-bloat git the way the
previous run's 213 MB did.

`tools/backup_training_run.py` takes the durable copy. Three artefacts —
`latest.pt`, the replay manifest and its retained chunks, and the run
config/ledger/provenance — validated before they are trusted and staged at
**23.9 MB** total (the replay store compresses 357 MB → 15 MB).

Two properties matter more than the copying:

- **It is validated, not assumed.** A checkpoint torn by a concurrent write
  still has a plausible size, so the script *loads* it. A replay archive whose
  manifest references a chunk it does not contain is a backup of nothing, so
  every reference is checked against the archive.
- **It does not pause the run.** Stopping a healthy trainer to back it up is the
  wrong trade, so the capture retries until provably consistent. That is sound
  because chunks are written before the manifest that references them, and
  pruning rewrites the manifest before unlinking — so any capture whose manifest
  is fully covered by its own chunks is a real point in time. (The first
  attempt did catch a live write; the check is not theoretical.)

Publishing is deliberately a separate step (`--publish`). The repository is
public, so uploading should never be a side effect of taking a backup.

**The heuristics-off transition checkpoint must be preserved separately.** That
is a reproducibility requirement, not an optimisation.

---

## 2. Why heuristics on, and what turns them off

**On (B0)** means `bootstrap_prior = canonical-target-vacancy-distance-v2`: the
policy prior comes from the vacancy heuristic, values from the real network.

This is not a nicety. A randomly initialised network has no idea how to make
progress toward the target camp, and Soo games only end when a camp is filled.
Without the prior, self-play games do not terminate, produce zero samples, and
the run learns nothing — which is the same phenomenon Gate F ran into from the
other direction (§12.3: *deterministic* self-play does not terminate either,
even with a trained network).

**The switch to A0 (`bootstrap_prior = none`) is gated, not scheduled.** The
criterion is the blueprint's own, implemented in `tools/cpu_off_probe.py`:

> at least **8 of 10** games complete with `bootstrap_prior = none`

It is an *operational* gate — can this network generate real terminal games on
its own — and explicitly not a claim about playing strength. Run it with
`tools/off_probe_soo_scratch.sh`; probes are archived under
`probes/`.

One throughput consequence to expect at the switch: heuristics-on is the native
`value_only` path, heuristics-off is `policy_value`, which reached only 53 % of
its roofline against value_only's 89 % (§10.3).

Part of that has now been removed ahead of the switch. The per-row Python
softmax loop is replaced by a device-side segmented softmax — worth **1.23x of
the whole callback at the production batch of 128**, and 1.75x at 256, agreeing
with the loop to 1.8e-7 on real logits.

It is worth being precise about what that does *not* fix, because the first
version of this note overclaimed it. The loop was **part** of the roofline gap,
not all of it. The remainder is work `value_only` simply does not do — the
gather over `[B, 5329]` logits and the device-to-host copy of the ragged priors
— and no amount of softmax tuning touches that. Expect `policy_value` to sit
below `value_only` after the switch regardless, and **re-sweep the batch knobs
then**: its callback latency and lane request cadence both change, so the B0
optimum of 256 lanes / cap 128 / 50 µs cannot be assumed to carry over.

---

## 3. Configuration, and why

`runtime/configs/soo-rtx5090-native.json`, driven by
`tools/train_soo_scratch.sh`.

| setting | value | reasoning |
|---|---|---|
| `simulations` | 64 | the production reference point every measurement in this project uses |
| `dirichlet_epsilon` | 0.25 | production; without it lanes lock-step and play one game (pitfall 7.9) |
| `temperature` / `temperature_moves` | 1.0 / 20 | production; the other half of what makes games terminate and diverge |
| `max_moves` | 500 | measured median is ~85; 500 leaves headroom while bounding the straggler tail. An aborted game contributes nothing, so this is a real cost either way |
| `games_per_iteration` | 768 | **must exceed `native_lanes`** — see below |
| `native_lanes` | 256 | 2 × batch cap, pitfall 7.6 |
| `max_batch_size` | 128 | the only knob that moves throughput (§10.2) |
| threads | 16 | 16 physical cores, leaving the batcher a core (pitfall 7.5) |
| `train_steps_per_iteration` | 300 | ~77k samples drawn against ~70k produced: roughly one pass over the new data |
| `replay.capacity` | 200,000 | ~3 iterations of data |
| `batch_size` / `lr` | 256 / 1e-3 | unchanged from the reference config |

**Checkpoint retention matters here.** An archive is 8.7 MB and was previously
written every iteration with no retention — ~8 GB over a long run, on a 16 GB
overlay with 13 GB free. `--archive-every 25 --keep-archives 20` bounds it at
~174 MB. `latest.pt` is still written every iteration, so resume safety does not
depend on the cadence.

---

## 4. Operating it

```bash
# a resumable block; re-running continues from latest.pt rather than restarting
tools/train_soo_scratch.sh 6                       # B0, 6 hours
tools/train_soo_scratch.sh 6 none A0               # after the gate passes

# the heuristics-off gate
tools/off_probe_soo_scratch.sh

# progress
grep -E '^\[i' /workspace/alphadiamond-training/train-b0.log | tail
```

The run imports from a **pinned source snapshot**
(`/workspace/alphadiamond-training/pinned-src`), asserted at startup, so repo
work — including rebuilding the extension — cannot reach a run already in
flight. `SIGTERM` finishes the current iteration and exits cleanly with state
durable.

---

## 5. Results

### 5.1 First block, 384 games/iteration (superseded)

| iteration | games done | median moves | samples | loss | self-play |
|---|---|---|---|---|---|
| 0 | 384/384 | 87 | 36,109 | 4.7209 | 71 s |
| 1 | 379/384 | 86 | 36,505 | 4.7427 | 111 s |
| 2 | 372/384 | 83 | 32,251 | 4.5157 | 99 s |
| 3 | 381/384 | 76 | 30,853 | 4.2844 | 109 s |

Learning from the first iteration, and games terminate: 97–100 % complete.

But throughput *degraded* across the block — `mean_batch` fell 46.4 → 34.5
against a cap of 64 and the evaluator thread went from 50 % busy to 40 %. That
is §12.2's straggler tail returning: as the network changes, game lengths spread,
and 384 jobs over 128 lanes leaves too little queued to refill the batcher.
Retuned to the measured 18.5x operating point (768 jobs / 256 lanes / cap 128 /
16 threads) and resumed; the checkpoint and replay carry over, so nothing was
lost but the tuning iterations.

### 5.2 Second block, 768 games/iteration

Retuning worked. Against the first block's 34.5 mean batch and 283 samples/s:

| iteration | games done | median moves | samples | loss | self-play | mean batch |
|---|---|---|---|---|---|---|
| 5 | 766/768 | 73 | 57,515 | 3.8004 | 84 s | 81.2 |
| 6 | 767/768 | 75 | 58,004 | 3.5880 | 84 s | — |
| 7 | 768/768 | 73 | 57,684 | 3.4383 | 64 s | — |
| 8 | 766/768 | 73 | 56,866 | 3.5766 | 97 s | 74.5 |
| 9 | 767/768 | 74 | 58,657 | 3.5801 | 98 s | 88.1 |
| 10 | 768/768 | 75 | 58,879 | 3.4445 | 74 s | — |

**685 samples/s**, against the Python backend's 55.8 on this host — **12.3x**.
Loss 4.72 → 3.44 over eleven iterations, 99–100 % of games terminating.

The gap to Gate F's 18.5x is the from-scratch network's game profile: shorter,
more variable games than the step-80 checkpoint the sweep used, so the batcher
still runs at ~50 % occupancy.

### 5.3 The batcher wait was the real constraint — 2.6x

By iteration 15 the picture did not add up. Worker threads sat at **0.6 of 16
thread-equivalents** — 4 % utilisation — and the evaluator at 46 %. Neither side
was saturated, so the time was going somewhere neither number covered: 507
batches/s against 0.91 ms of forward means **54 % of every cycle was batch
collection**.

Sweeping the candidates against the live checkpoint settled it in one pass:

| lanes | cap | wait µs | threads | samples/s | evaluator |
|---|---|---|---|---|---|
| 256 | 128 | 2000 | 16 | 355 | 39 % |
| 256 | 128 | 500 | 16 | 767 | 63 % |
| 256 | 128 | 200 | 16 | 955 | 78 % |
| 512 | 128 | 2000 | 16 | 355 | 39 % |
| 512 | 128 | 500 | 16 | 773 | 63 % |
| 384 | 64 | 500 | 8 | 773 | 63 % |

**Lane count changes nothing. Thread count changes nothing. Only the wait
matters.** Refining it:

| cap | wait µs | mean batch | samples/s | evaluator |
|---|---|---|---|---|
| 128 | 200 | 50.6 | 955 | 78 % |
| 128 | 100 | 49.9 | 1,024 | 84 % |
| **128** | **50** | **49.9** | **1,077** | **88 %** |
| 48 | 200 | 31.1 | 801 | 82 % |
| 64 | 100 | 36.8 | 900 | 87 % |

The mechanism, once seen, is obvious. With 256 lanes running 64-simulation
searches, only ~50 lanes are ready to submit at any instant, so **the batch never
reaches the cap** — which means the batcher always waits the full `max_wait`,
every cycle. At 2 ms against a 0.9 ms forward that is more dead time than work.
Lowering the *cap* to meet the supply is worse, because it shrinks the batches
without removing the wait; keeping the cap high and the wait short is what wins.

The 2 ms came from `inference.max_wait_ms`, the **Python coordinator's** knob,
where a batch genuinely takes milliseconds to assemble and milliseconds are the
right unit. Deriving the native wait from it was the mistake. The native backend
now has `native_max_wait_us`, separate by design.

Live effect at iteration 18:

| | before | after |
|---|---|---|
| self-play per iteration | 111 s | **41 s** |
| evaluator occupancy | 46 % | **93 %** |
| mean batch | 68 | 80 |
| samples/s | 685 | **1,444** |
| vs the Python backend | 12.3x | **25.9x** |

### 5.4 The heuristics-off switch, and the gate that got it wrong

At step 13,650 the off-probe passed convincingly and the switch looked overdue:

| | OFF (`prior=none`) | ON (heuristic) |
|---|---|---|
| completion | 20/20 (100 %) | 20/20 (100 %) |
| median moves | **63.5** | 71.5 |
| p90 moves | **73** | 113 |

Thirty games across two seeds, and the learned policy was beating the heuristic
at the heuristic's own job — shorter games, tighter tail. The B0-final
checkpoint was published as a release and the run switched to A0.

**Production self-play then fell apart over four iterations:**

| iteration | completed | median moves | self-play |
|---|---|---|---|
| 50 | 695/768 (90.5 %) | 80 | 83 s |
| 51 | 652/768 (84.9 %) | 77 | 91 s |
| 52 | 495/768 (64.5 %) | 105 | 142 s |
| 53 | 493/768 (64.2 %) | 117 | 147 s |

Loss kept falling — 3.06 → 2.40 — which is exactly why **loss is not a
phase-transition signal**. It measures how well the policy head reproduces the
targets the current search produced; it says nothing about whether that
search-policy loop generates good games. And since only completed games reach
replay, the dataset is progressively censored while the number improves.

Health metrics for this project, in priority order: **completion rate**,
move-count distribution (median / p90 / p99), abort rate and reasons, self-play
throughput, external strength — and only then loss, as a training sanity check. The policy head was fitting its own visit distributions
beautifully while the games those distributions produced got longer and more
often hit the move cap. And an aborted game contributes **zero** samples, so
replay was filling only from games that happened to finish: survivorship bias on
top of a third of the compute being wasted.

Reverted to B0 within four iterations. The checkpoint and replay carry over, so
the cost was the four iterations.

### 5.5 What the gate was actually measuring

The first diagnosis was wrong and worth recording as wrong: *"the probe is
deterministic and production has Dirichlet noise."*

`MCTSConfig.dirichlet_epsilon` defaults to **0.25**, not to zero. The probe
always had Dirichlet noise. What it did not have was **temperature sampling** —
it ran `temperature_moves=0`, so every move was the argmax of the visit counts,
while production samples its first 20 moves *from* the visit distribution.

Perturbing a prior and then taking the best move is a mild thing. Sampling the
opening from a distribution is not: it is what sends games wandering past the
cap. Confirmed by re-probing the same checkpoint with production's full
exploration:

| gate variant | reading | verdict |
|---|---|---|
| without temperature sampling | 100 % | PASS — and wrong |
| **with production exploration** | **55 %** | **FAIL — matches the observed 64 %** |

The fixed gate predicts production. It also nearly shipped a second bug: writing
"unset" as an explicit `0.0` would have turned Dirichlet noise *off* for every
existing caller, because the dataclass default is 0.25. The sentinel is `None`,
and a test pins it.

**The lesson generalises past this gate.** A readiness check has to run the
configuration it is clearing you for. Any part of production's behaviour the
check omits is a part it silently assumes is harmless.

### 5.6 What the fixed gate says, in retrospect and now

Re-measuring the preserved checkpoints with the fixed gate closes the loop:

| step | context | fixed gate | verdict |
|---|---|---|---|
| 13,650 | B0-final, **pre-switch** | **88 %** (90 / 85 per seed) | **FAIL** — narrowly |
| 16,350 | after 4 A0 iterations | 55 % | FAIL |
| 19,050 | after ~10 B0 recovery iterations | **95 %** (95 / 90) | PASS |

Three things fall out.

**The strengthened threshold is what does the work.** At the blueprint's original
80 % the pre-switch checkpoint passes at 88 % — and production then collapsed to
64 %. At 90 % it fails. The margin is not conservatism for its own sake; 88 % is
precisely the reading that did not hold up.

**The A0 excursion damaged the network**: 88 % → 55 % in four iterations, and
reverting to B0 recovered it past where it started. The censored dataset was not
merely wasteful, it was harmful — replay filled only from games that finished,
so the network was trained away from exactly the trajectories it was failing.

**The two engines agree.** On one pinned checkpoint the native gate and the
Python probe both read 19/20 with medians of 75 and 76, so the improvement is
real and not an artefact of changing measurement tools mid-investigation. Worth
checking, because the gate switched engines and step at the same time.

### 5.7 Raising `max_moves` would not have helped

Run as a diagnostic on one checkpoint under production exploration:

| `max_moves` | completion | median | p90 | p99 | longest |
|---|---|---|---|---|---|
| 500 | 95 % | 72 | 103 | 329 | 329 |
| 750 | 95 % | 72 | 103 | 329 | 329 |
| 1000 | 98 % | 73 | 107 | **949** | 949 |
| 1500 | 98 % | 73 | 107 | 949 | 949 |

500 → 750 changes **nothing at all** — not the completion, not a single
percentile. So the games aborting at the cap are not games that would have
finished at 600 or 700; the distribution is bimodal, with a healthy mass under
~110 moves and a tail that wanders indefinitely. Going to 1000 recovers exactly
one game in forty, and that game took 949 moves.

The cap is therefore **not too tight**, and raising it would buy a percentage
point by paying 949 moves of pathological trajectory into replay, at 60 % more
wall-clock for the same forty games. `max_moves = 500` stays.

### 5.8 A0 attempt 2 — the gate was right, and A0 still failed

The gate passed three times running (92 %, 92 %, 95 %; per-seed 95/90), the
transition checkpoint was published, and the run switched again.

| iteration | completed | % | median | loss |
|---|---|---|---|---|
| 77 | 709/768 | **92.3 %** | 74 | 2.94 |
| 78 | 685/768 | 89.2 % | 78 | 2.72 |
| 79 | 606/768 | **78.9 %** | 95 | 2.27 |
| 80 | 612/768 | 79.7 % | 91 | 2.18 |

Rolled back at iteration 79 on the < 85 % rule. Post-A0 gate: **80 %, FAIL**.

Two trials, side by side:

| trial | pre-switch gate | production trajectory | post-A0 gate |
|---|---|---|---|
| 1 | 88 % | 90.5 → 84.9 → 64.5 → 64.2 | 55 % |
| 2 | **92 %** | 92.3 → 89.2 → 78.9 → 79.7 | 80 % |

**The gate is accurate.** It predicted the first production iteration to within a
point both times — 88 → 90.5 and 92 → 92.3. The stricter threshold also did real
work: trial 2 degraded at half the rate and ended 25 points healthier.

But it did not prevent the failure, and that falsifies the hypothesis the whole
approach rested on. This is **not** "the network is not yet strong enough". At
92 % the network *plays* fine. What fails is **training on its own A0 data**.

### 5.9 The mechanism: a censoring spiral

1. Under production exploration ~8 % of games wander past the move cap.
2. An aborted game contributes **zero** samples.
3. Replay therefore holds only completed games — the network never sees the
   states in which it wandered, nor any recovery from them.
4. Trained on that censored set it drifts, and more games wander.
5. Return to 1.

B0 does not have this problem because the heuristic prior keeps the **search** on
track regardless of what the policy believes, so games complete and the replay is
not censored.

That suggested an uncomfortable corollary — *waiting longer in B0 cannot fix it,
because B0 never produces the training signal A0 needs* — and the corollary was
drawn too early. **The next measurement contradicted it** (§5.10).

It also explains §5.7. Raising `max_moves` did not help because the wanderers are
not near-misses — they are the tail that censoring is systematically deleting
from the dataset.

**How much is deleted.** An aborted game runs the full 500-move cap and then
contributes nothing, so the discarded volume is exactly `aborted × 500` moves:

| iteration | phase | completed | moves kept | moves discarded | **% thrown away** | median | p90 |
|---|---|---|---|---|---|---|---|
| 75 | B0 | 767/768 | 57,575 | 500 | **1 %** | 73 | 86 |
| 76 | B0 | 768/768 | 59,009 | 0 | **0 %** | 75 | 89 |
| 77 | A0 | 709/768 | 63,578 | 29,500 | 32 % | 74 | 125 |
| 78 | A0 | 685/768 | 65,274 | 41,500 | 39 % | 78 | 144 |
| 79 | A0 | 606/768 | 68,797 | 81,000 | **54 %** | 95 | 187 |
| 80 | A0 | 612/768 | 69,132 | 78,000 | **53 %** | 91 | 193 |

Trial 1 went further, reaching **74 %** discarded by its sixth A0 iteration.

So this is not a dataset missing a few games. Within four iterations **more than
half of every move played is thrown away**, and it is the long half — selected,
precisely, for being the experience the network most needs in order to stop
producing it. B0 discards 0–2 % by comparison.

### 5.10 B0 *does* improve A0 robustness — the corollary was wrong

The gate had read 92 % three times across 3,300 steps, and I took that flatness
as evidence of a structural ceiling: B0 keeps the search on rails, so the network
is never *required* to learn recovery, so it never does. That reading did not
survive the next data point.

| step | context | gate | verdict |
|---|---|---|---|
| 13,650 | pre-switch, trial 1 | 88 % | FAIL |
| 18,750 | B0 | 92 % | PASS |
| 21,750 | B0 | 92 % | PASS |
| 22,050 | pre-switch, trial 2 | 92 % | PASS |
| 23,550 | after 4 A0 iterations | **80 %** | FAIL |
| 28,350 | B0 recovery | **98 %** | PASS |

B0 recovery did not merely undo the damage, it went **past the plateau** — 80 %
to 98 %, six points above the previous best. So B0 *is* teaching A0 robustness,
just slowly enough that three readings 3,300 steps apart could not see it.

That matters for the strategy, because the two failed switches both began from
92 %: an 8 % wander rate, which censored a third of all moves in the first
iteration and half by the fourth. At 98 % the wander rate is 2 %, and the spiral
has four times less to feed on. Whether that is enough to keep the loop stable
is an empirical question — but "keep training B0 and switch from a much higher
gate reading" is now a live option that needs no code change and no semantic
change, where an hour earlier it looked ruled out.

The honest summary is that one flat stretch is not a ceiling, and I called it one
too confidently.

### 5.11 The replay store had to be bounded first

The run could not have finished a six-hour block. `PersistentReplayStore` is
append-only, and two costs compound:

- **Disk.** Chunks are never removed. Measured at ~79 MB and ~500 chunks per
  iteration — 13 GB of free space gone in about two and a half hours.
- **Time.** `load_buffer()` re-reads *every chunk ever written*, once per
  iteration, to rebuild a buffer bounded at `capacity`. Everything outside the
  newest `capacity` samples is read and immediately evicted. That is quadratic
  in the length of the run.
- **Manifest write amplification.** `ingest_episode` rewrites the whole manifest
  per call. At 768 games against a manifest already grown to 1.4 MB, one
  iteration wrote about a gigabyte of manifest to record what one write records.

Fixed by `prune_to_capacity()` (explicit, because it deletes durable evidence)
and `ingest_episodes()` (one manifest write per iteration). The safety property
the pruning tests assert is that **`load_buffer()` returns exactly what it
returned before** — the dropped chunks are provably unreachable, not merely old.

Effect on the live run, at the first pruned iteration:

| | before | after |
|---|---|---|
| replay chunks | 6,502 | 2,636 |
| run directory | 936 MB | 374 MB |
| training step time | 16 s | 12 s |

Disk is now bounded by the capacity window rather than by run length.

---

## 6. Why A0 fails: three measurements that settle it

### 6.1 The frozen-actor experiment — censoring is causal

Both A0 attempts confounded two things: the dataset was censored *and* the
learner became the next iteration's actor immediately. Pinning the actor
separates them. Two matched arms were cloned from the same checkpoint and the
same replay; the frozen arm's self-play always ran the pinned step-34,650 actor
while its learner trained normally for four iterations — four being the point at
which 200k replay has turned over once at the observed A0 sample rate.

The arm's *self-play* completion is stable by construction (same actor, same
distribution), so it measures nothing. What matters is whether the **learner**
degrades, measured at 768 games:

| | completion | discarded |
|---|---|---|
| actor, step 34,650 (frozen) | **93.2 %** | 28.7 % |
| learner after 4 frozen-actor iterations | **86.3 %** | 44.8 % |

A 6.9-point drop, about 4.5 standard errors at n=768. **The learner degrades even
when the state distribution is held completely still.** So the censored dataset
is causal on its own; the actor-refresh loop can only be an amplifier.

### 6.2 What the aborted games are actually doing

"Wandering" had been an assumption. Auditing 13 aborted games at 64 simulations,
using the encoded root features as position identity:

| metric | median | range |
|---|---|---|
| unique positions / moves | **0.316** | 0.116 – 0.928 |
| max revisits of a single position | **61** | 6 – 112 |
| returns within 2/4/6/8 ply | **68.4 %** | 6.8 – 88.4 % |

**Short-cycle shuffle**, not slow progress. One position was revisited 61 times.

One caveat on the identity used. These are the *encoded* features, which the
encoder canonicalises — the acting player's camp is rotated to a fixed
orientation and the player channels reordered to `(self, next)`. So symmetric
images hash together and the same position with the other side to move hashes
apart. For diagnosis that is acceptable, and arguably the right notion, since it
counts "the network saw this exact input again"; the 2/4/6/8-ply structure makes
genuine repetition overwhelmingly likely either way. It is **not** acceptable for
a control decision: anything that changes search behaviour on detecting a repeat
must key on the authoritative physical state — occupancy, side to move, status —
and exclude bookkeeping like `turn_number` that does not affect the dynamics.
That explains §5.7 — a repetition attractor is indifferent to the move cap, so
raising it changed nothing — and it predicts that deeper search should help,
because escaping needs to see past the cycle.

### 6.3 Search budget: 128 is the knee

Same frozen actor, same seeds, exploration untouched — only the budget moves:

| simulations | completion | **discarded** | terminal samples/s |
|---|---|---|---|
| 64 | 93.8 % | **28.6 %** | 176 |
| **128** | **97.7 %** | **12.2 %** | 97 |
| 256 | 97.7 % | 12.5 % | 30 |
| **adaptive 64 → 128 at move 100** | **97.7 %** | **12.6 %** | **106** |

Doubling to 128 **more than halves the censoring**; 256 adds nothing at three
times the cost. So 64 simulations is not enough for this network to correct its
own prior, and the deficiency has a threshold rather than a gradient.

Adaptive search reaches flat-128's quality for 106 samples/s against 97 — real
but modest, because although only ~15 % of *games* pass move 100, those games
are long and carry roughly a third of all moves played. A stagnation detector
would target the tail far more precisely than a move-number threshold; the audit
above says the signal to detect is short-cycle repetition, not slow progress.

### 6.4 What this does and does not settle

It settles that **64 simulations was under-searching**, and that the censored
dataset harms the learner **independently** of the actor-refresh loop.

It does not settle whether more search or a better network is the right cure.
128 simulations buys its improvement with test-time compute; a deeper network
might buy the same thing at 64. That comparison — `128×6 @ 128 sims` against a
depth-expanded `128×12 @ 64 sims`, at roughly matched wall-clock — is the next
experiment, and it has to be judged on completion, discarded fraction, samples/s
and head-to-head strength. Not on loss, which has been wrong at every step of
this investigation.

### 6.5 The flat-128 frozen control — A0 is stable at 128 simulations

The 64-simulation frozen arm showed the learner degrading with the actor held
still. Repeating it with self-play at 128 simulations — same pinned actor, same
cloned replay, same 300 steps/iteration, four iterations — separates "censoring
harms the learner" from "*this much* censoring harms the learner".

All four cells at 768 fresh games:

| checkpoint | @ 64 sims | @ 128 sims |
|---|---|---|
| actor, step 34,650 | 93.2 % / 28.7 % censored | **97.8 % / 11.6 %** |
| learner trained on **128-sim** data | 86.2 % / 48.7 % | **97.9 % / 13.1 %** |
| learner trained on **64-sim** data | 86.3 % / 44.8 % | — |

Three readings, and the third is the awkward one.

**At 128 simulations the training loop is stable.** The learner holds its actor's
97.8 % and even improves the shape of its games: median 71 → 65, p90 131 → 89,
p99 307 → 173. Shorter, more decisive. So ~12 % censoring is tolerable where
~29 % is not, and A0 is viable — at 128.

**The learner does not absorb the search advantage.** Trained entirely on
128-simulation targets, it is no better at 64 simulations than the learner
trained on 64-simulation targets: 86.2 % against 86.3 %. Whatever the deeper
search knows, four iterations of distillation did not move it into the prior.

**A0 training costs 64-simulation robustness regardless.** The B0 actor plays at
93.2 % on 64 sims; both A0 learners drop to ~86 %, even the one whose data was
only 13 % censored. So the 64-sim regression is not caused by censored data
quality — clean data produced it too.

That is the "learner@128 stable, learner@64 unchanged" row: **the search
dependency is real and remains**. The immediate answer is to keep the extra
search where it is needed rather than to hope the network learns to do without
it, and the longer-term answer is a better network — but as an Elo/hour
experiment, not as a patch over a broken loop, because the loop at 128 is not
broken.

### 6.6 The repetition trigger beats flat-128 on every axis

The audit said the failure is a short-cycle attractor, so the trigger keys on
repetition rather than lateness: base 64 simulations, 128 only when the current
position already occurred within the last 8 plies of that game.

Identity is the **physical** state — occupancy, side to move, status and finish
order — hashed by `dynamics_key`. Deliberately not `State::operator==`, which
includes `turn_number` and so can never report a repetition; and deliberately
not the encoded features, which canonicalise orientation and player channels
(§6.2).

All three configurations, 768 fresh games, same actor, same seeds:

| configuration | completion | **discarded** | terminal samples/s | moves boosted |
|---|---|---|---|---|
| flat 64 | 93.2 % | 28.7 % | 473 | — |
| flat 128 | 97.8 % | 11.6 % | 247 | 100 % |
| **repetition trigger** | **98.0 %** | **9.6 %** | **392** | **5.1 %** |

It matches flat-128's completion, **beats** it on censoring, and runs at **1.59x
its throughput** — while paying the extra search on one move in twenty. The
signal was worth targeting: the audit's short-cycle finding translated directly
into a trigger that spends 5 % of the compute flat-128 spends and gets more.

### 6.7 A correction: the earlier yield figures were measured wrong

§6.3's throughput column is not comparable and should not be used. Those runs
were configured with `games == lanes`, which disables the job queue and
reintroduces the straggler tail from §12.2 — the run finishes at the pace of its
slowest game while most lanes sit idle. The effect scales with game length, so
it penalised the higher simulation counts hardest and made the extra search look
far more expensive than it is:

| configuration | yield as first measured (`games == lanes`) | yield measured properly (768 games, 256 lanes) |
|---|---|---|
| flat 64 | 176/s | **473/s** |
| flat 128 | 97/s | **247/s** |

Completion and discarded fraction are unaffected — they are rates, and the
scheduling does not change which games finish. Only the throughput column was
wrong, and it was wrong by 2.5x.

The lesson is one this project has now paid for twice: **`games_per_iteration`
must exceed `native_lanes`**, in measurement harnesses exactly as in production.
The gate's default of 20 games per seed against 64 lanes has the same flaw, which
does not matter for the completion rate it exists to report but would matter the
moment anyone read a timing off it.

### 6.8 Step 3 overturns §6.5: it is target quality, not censoring

The repetition trigger was run as a *generator* for a third frozen-actor arm —
same pinned actor, same cloned replay, four iterations — and the learner was then
measured at every budget. Together with the other two arms:

**Actor, step 34,650:** 93.2 % @64 · 97.8 % @128 · 98.0 % @trigger

**Learner after four frozen-actor iterations, by generator:**

| generator | targets from 128-sim search | learner @64 | learner @128 | learner @trigger |
|---|---|---|---|---|
| flat 64 | 0 % | 86.3 % | — | — |
| flat 128 | **100 %** | 86.2 % | **97.9 %** | — |
| repetition trigger | 5 % | 84.9 % | 94.0 % | 91.5 % |

Judged where it matters — does the learner hold its actor's level on the budget
it was generated with?

| generator | actor | learner | |
|---|---|---|---|
| flat 128 | 97.8 % | **97.9 %** | **holds** |
| trigger | 98.0 % | 91.5 % | degrades 6.5 pt |
| flat 64 | 93.2 % | 86.3 % | degrades 6.9 pt |

**This contradicts §6.5.** That section concluded "~12 % censoring is tolerable
where ~29 % is not". The trigger generator censors **9.6 %** — the lowest of the
three — and its learner degrades anyway. Censoring is not the operative variable.

What tracks the outcome perfectly is the **fraction of training targets produced
by a 128-simulation search**: 100 % holds, 5 % degrades, 0 % degrades. And the
trigger's learner is worse than the flat-128 learner even when both are measured
at 128 sims (94.0 % against 97.9 %), so this is about the data it was trained on,
not the budget it is tested with.

The mechanism this points to is the one the few-simulation AlphaZero literature
warns about: at 64 simulations the root search does not reliably improve on the
network's own prior, so its visit distribution is not a policy-improvement
target. Training on it moves the network sideways at best. At 128 the search does
improve on the prior, and the loop becomes a genuine improvement operator.

It also retro-explains why **B0 was always healthy at 64 simulations**. B0 does
not use the neural prior at all — the vacancy heuristic replaces it — so the
search is anchored to an external, informative signal rather than to the
network's own weak prior. The self-referential loop that fails in A0 does not
exist in B0.

### 6.9 What the repetition trigger is, and is not, good for

It remains the best *generator* on record: 98.0 % completion, 9.6 % censoring,
392 samples/s, 5 % of moves boosted — beating flat-128 on all four. If the goal
were producing terminal-labelled games cheaply, it would be the answer.

But 95 % of its policy targets still come from 64-simulation search, and §6.8
says that is what decides whether training helps. **A cheap generator does not
make a cheap teacher.** The extra search has to be in the targets the network
learns from, not only in the moves that were going wrong.

So the production A0 configuration is **flat 128** — the only one with a
demonstrated stable loop — at 247 terminal samples/s against B0's ~1,444. That
is the price of heuristic-free training at this network size, and it is the
number a larger network would have to beat: the `128×12 @ 64 sims` experiment now
has a concrete target, which is `128×6 @ 128 sims` at 97.9 % learner stability
and 247 samples/s.

---

## 7. A0 in production, and the first strength measurement

### 7.1 Rolling A0 at 128 simulations holds

§6.8 said flat 128 was the only configuration with a demonstrated stable loop
under a *frozen* actor. Rolling A0 — the real thing, learner deployed as the next
iteration's actor — was then started from the B0 checkpoint at step 34,650:

| iteration | completed | median moves |
|---|---|---|
| 118 | 754/768 (98.2 %) | 71 |
| 121 | 755/768 (98.3 %) | 65 |
| 125 | 751/768 (97.8 %) | 64 |
| 130 | 754/768 (98.2 %) | 64 |
| 135 | 755/768 (98.3 %) | 64 |
| 137 | 750/768 (97.7 %) | 65 |

**Twenty iterations, no degradation.** Both previous A0 attempts had collapsed by
iteration three or four. Median game length also fell from 71 to ~64, which is
the network committing to lines rather than shuffling.

### 7.2 Strength, which health metrics cannot see

A network could hold 98 % completion and play no better than the checkpoint it
started from. `tools/arena_head_to_head.py` measures the difference: two
checkpoints under the existing `SooArena`, deterministic (temperature 0, and the
arena zeroes Dirichlet itself), across its balanced matchup cycle.

Candidate step 38,250 against the A0 starting point, step 34,650:

| search budget | result | win rate | implied Elo | verdict |
|---|---|---|---|---|
| 64 simulations | 30W–10L | 75.0 % (CI 60–86 %) | **+191** | **stronger** |
| 128 simulations | 30W–10L | 75.0 % (CI 60–86 %) | **+191** | **stronger** |

A0 training is producing a genuinely better network, not merely a healthy one.

**The gain is identical at both budgets**, which is worth stating because the
first, buggy run suggested otherwise — it showed 100 % at 64 and 50 % at 128, and
I had already written an explanation for that pattern (a better prior helps most
where search is scarce). With the harness fixed the pattern is not there. The
improvement is uniform, and the tidy story about it was a story about a bug.

It does still rebut the worry recorded in §6.8, that the network might stay
permanently search-dependent: it is 191 Elo better at 64 simulations than the
checkpoint A0 started from.

### 7.3 A harness bug that produced a perfect score

The first run of this arena returned **40W–0L**, which is what prompted checking
the harness rather than celebrating.

`SooArena` crosses every turn order with every candidate seat to get a four-fold
balanced cycle, and it hands the turn order to the caller's `game_factory`. My
factory ignored that argument and always built the default order, collapsing the
balance to two-fold. Corrected — `build_players(len(order), order=order)`, as
`orchestration/production.py` does — the same comparison gives **30W–10L**.

Still clearly stronger, and not remotely 100 %. A result that good should be
treated as evidence about the measurement until the measurement has been checked.
