# The promotion arena: what it cost, what was fixed, what is still open

> **Continuation of [min_bootstrap.md](min_bootstrap.md).** That document's §6
> listed the arena stage as known-red and its §7 made the B0 baseline the next
> step. B0 cannot run without the arena, because every iteration runs it. This
> records the fix, the measurements, and the part that is still not good enough.

---

## 1. Two defects, one untested function

`arena_episode_config` built the arena's `EpisodeConfig` inside `train_main.cpp`,
where nothing could reach it from a test. It was wrong in two independent ways.

**It ran `lanes = 1, threads = 1, max_batch = 1`.** Synchronous MCTS allows one
outstanding request per lane, so a single lane caps the batch at one position: a
Min arena is 36 games per opening (3! seat assignments × 3! turn orders) and the
GPU answered them one position at a time.

**It never set `bootstrap_prior`.** While a run bootstraps -- exactly when Min
has no other way to steer a piece into a camp -- every arena game therefore ran
to `arena.max_moves` and the stage reported nothing but incomplete blocks,
having paid for a full-length game each.

Both are the failure mode `cli_contract_test` already exists to catch, and which
[min_bootstrap.md §1](min_bootstrap.md) documents for `self_play.bootstrap_prior`
and [repetition_trigger_config_gap.md](repetition_trigger_config_gap.md) for the
repetition trigger: a field the engine supports and the wiring drops leaves the
search correct and moves only the completion rate and the cost.

## 2. The fix

`arena_episode_config` became `diamond_orchestration::wire_arena_episode`, beside
`wire_training_iteration`, where it is reachable from `cli_contract_test`.

- **The arena runs the phase the config declares.** It takes
  `self_play.bootstrap_prior`. Both sides take the same heuristic prior, so the
  comparison stays symmetric and turns on the value head -- which is what
  differs between candidate and champion at that point. Removing the prior from
  the config removes it from the arena.
- **Games that share a turn order are played together.** A scheduler run fixes
  the match but not the position, since every job carries its own start state,
  so the grouping is by turn order *across the whole schedule* rather than
  within an opening: with ten openings the six Min turn orders become six runs
  of sixty concurrent games instead of sixty runs of one.
- **`BatchItem` carries its job index.** A lane takes the next unstarted job
  when its own game ends, so neither the lane nor the outcome pointer identifies
  a game; the arena router holds the candidate's seat per job rather than once
  for the run.

Threads and batch are bounded by the group size, because a group of four games
can never present a fifth position to evaluate.

### Tested, and each assertion verified by breaking what it covers

- `cli_contract_test` pins that the arena carries the bootstrap prior, that
  lanes are the group size while threads and batch saturate at it, and that the
  search stays greedy on the arena's own move budget. Pinning
  `.bootstrap_prior = false` fails it.
- `selfplay_test` pins the premise the grouping rests on -- that playing games
  together does not change them -- by replaying each job alone at
  `lanes = threads = max_batch = 1` and requiring the grouped run move for move.
  It also pins that a batch item names its job, using three jobs on two lanes so
  that a `job` wired from the lane id would report two distinct values for three
  games; hard-coding `job = 0` fails it.

Commit `62cd60c`, branch `min-arena-throughput`. 38/38 tests pass.

## 3. Measured: a Min iteration on one RTX 4090

`runtime/configs/min-smoke.json` -- CUDA, 64 simulations, 32 games, 24 lanes,
batch 64, vacancy bootstrap prior, from scratch. Deliberately small: this is a
throughput measurement, not a baseline.

| stage | wall | outcome |
|---|---|---|
| initialize | 0.3 s | |
| self-play | 54.7 s | **32/32 completed, 0 aborted**, 3640 samples |
| replay-ingest | 1.8 s | |
| train (8 steps) | 1.5 s | policy 8.87 → 6.07, value 0.667 → 0.647 |
| save-candidate | 0.4 s | |
| arena, `max_moves` 400 | **> 10 min, did not finish** | killed by the harness |
| arena, `max_moves` 120 | **3 m 53 s** | **31 of 36 games aborted**; not promoted, `incomplete_opening_blocks` |

Self-play and training are proven end to end on this machine, and the value head
moves off zero -- the gap [min_bootstrap.md §5](min_bootstrap.md) left open. The
first two training steps behave as `min_value_head_init_test` predicts.

**The arena is still the cost of an iteration**, at roughly 4× self-play for a
comparable amount of work. Two effects, both measurable in the numbers above:

- **Throughput.** Self-play did 3640 × 64 ≈ 233 k evaluations in 54.7 s, or
  4.3 k/s at 24 lanes. The arena did ≈ 270 k in 233 s, or 1.2 k/s. The smoke
  config has one opening, so a turn-order group is six games, and the router
  then splits every batch between the candidate and champion pools -- two GPU
  calls of about two and four items. Effective batch ≈ 3 against self-play's 24.
- **Game length.** Self-play completed every game at a median of ~114 moves. The
  arena aborted 31 of 36 at a 120-move cap. The arena is greedy -- temperature 0,
  no Dirichlet -- and at iteration 0 candidate and champion hold the *same*
  weights, which is the setup most likely to cycle.

## 4. What this does not establish

The 120-move cap is close to self-play's median, so the abort count does not
separate "greedy identical players cycle" from "the cap is simply tight". The
distributions were not compared: no move-percentile or repetition statistics
were collected for arena games, only the completion count. `alphadiamond-min-probe`
measures exactly those, but on self-play, not on an arena pairing.

More openings do **not** fix the throughput half. They make groups bigger and
multiply the number of games by the same factor, so wall time barely moves.

## 5. Where to pick up

1. **Separate the two effects.** Re-run the standalone arena
   (`alphadiamond-train evaluate`) at caps of 200, 400 and 800 and record the
   completion curve. If completion is still poor at 800, the cause is cycling,
   not the cap, and the arena needs a repetition remedy of its own rather than a
   bigger budget.
2. **Per-job matches, if throughput is the binding constraint.** The remaining
   serialisation is that `run_episodes` fixes one `Match` for the whole call, so
   the six turn-order groups run one after another. Giving `EpisodeJob` its own
   match would let the entire schedule run as one batch -- 36 lanes at one
   opening, 360 at ten. There are about eight uses of `match` in the episode
   loop (`native/src/selfplay.cpp` lines ~533-706), all of them per-lane-able.
   Cost this against step 1 first: if arena games are cycling to the cap, the
   cheaper win is making them end.
3. **Then B0.** Min from scratch, vacancy bootstrap, zero value head, at the
   intended production search, as [min_bootstrap.md §7](min_bootstrap.md)
   specifies. Run it in short `--run-dir` segments joined by
   `alphadiamond-train resume`; every long run in these sessions has been killed,
   including this document's first arena measurement.

## 6. Setup notes for the next machine

Both are already documented in [min_bootstrap.md §6](min_bootstrap.md) and both
bit again here:

- This host had only `gcc-14-base`, not the compiler. **Nothing builds without
  `g++-14`** -- GCC 13 rejects `-std=c++2c` outright.
- `TORCH_CUDA_ARCH_LIST=8.9` for the build, and
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for every binary. Without
  the preload, five checkpoint tests fail with SIGSEGV and look like real
  breakage.
- The changed-line format gate needs `clang-format-18`, the pinned major.
