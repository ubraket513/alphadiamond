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
| phase | **B0** — heuristics on |
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
`value_only` path, heuristics-off is `policy_value`, which reaches only 53 % of
its roofline against value_only's 89 % (§10.3). The gap is the per-row Python
softmax loop in `policy_value_callback`, and it becomes worth fixing at that
point.

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

### 5.4 The replay store had to be bounded first

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
