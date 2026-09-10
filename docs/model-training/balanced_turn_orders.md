# Balanced self-play turn orders

Enable `self_play.balance_turn_orders: true` to allocate the same number of games
to every permutation of the players. Min requires games per iteration divisible
by six; Soo requires divisibility by two. Existing configurations default to false
and preserve their canonical configuration hashes.

The order changes both the initial acting player and subsequent game dynamics.
Player IDs, starting camps, and target camps stay attached to their original
players. Each queued job carries its own match; a reused lane resets its search
session to that match. Encoding, action conversion, rule application, and camp
diagnostics all use the job's match. Orders are interleaved deterministically in
lexicographic cycles, independent of batching and worker scheduling.

`selfplay.metrics.json` reports requested games, completed games, aborts, and
retained training samples for each order. Equal game counts do not imply equal
sample counts: game lengths differ, and aborted games are excluded from replay.
Check these counters before interpreting a training run as balanced in practice.

## Local smoke run

`configs/alphazero/min-balanced-orders-cpu-smoke.json` uses the verified step 20064
deployment artifact as a weight-only warm start: fresh optimizer and fresh replay,
six games, 32 simulations, eight learner updates, and one iteration. It tests the
pipeline and local cost; it is not a strength experiment or a releasable model.
The artifact version is retained for compatibility and outputs use an isolated
run directory. Arena is disabled for this smoke test.

```powershell
$env:PATH=(Resolve-Path dist/diamond-qt-soo).Path+';'+$env:PATH
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& build/native-package/native/alphadiamond-train.exe train `
  --run-dir artifacts/balanced-orders/runs/min/cpu-smoke-20064 `
  --config configs/alphazero/min-balanced-orders-cpu-smoke.json `
  --warm-start artifacts/min-step20064-verified/models/min/2.0.0-min-test.20064
```

For a subsequent learning experiment, retain the six-order schedule, use at least
the original 128 simulations, collect substantially more games, and evaluate each
order separately against a frozen baseline. Do not mix old fixed-order replay
into the first experiment. Changing rewards or network size simultaneously would
make the effect of order coverage difficult to determine.

## Historical distinction

The checked native `iteration_job` and the earlier Python
`build_authoritative_selfplay_jobs` at commit `5998bc6` shared one standard initial
state across jobs, including Soo. Collecting both players' moves in a game is not
the same as balancing who starts. Arena turn-order balancing is also distinct
from self-play balancing. This does not establish the settings of every historical
Soo training run; it documents the implementations inspected for this change.

## Observed local result — 2026-09-11

Built `alphadiamond-train`, `selfplay_test`, and `config_test`. Both tests passed,
including mixed-order versus isolated trajectories, input-channel/rules agreement,
lane reuse, rejection of incomplete order cycles, and old config round trips.

The smoke run finished one iteration and eight finite-loss learner updates.
Self-play took 155.17 seconds and produced 795 retained positions:

| Order | Requested | Completed | Move-limit aborts | Retained positions |
|---|---:|---:|---:|---:|
| 1-2-3 | 1 | 0 | 1 | 0 |
| 1-3-2 | 1 | 1 | 0 | 105 |
| 2-1-3 | 1 | 1 | 0 | 122 |
| 2-3-1 | 1 | 1 | 0 | 443 |
| 3-1-2 | 1 | 0 | 1 | 0 |
| 3-2-1 | 1 | 1 | 0 | 125 |

This verifies execution, not effective six-order learning: two orders supplied
no retained data. The per-order sidecar is authoritative here; the existing CLI
summary's `completed_games: 6` counts processed games, including these aborts.
Learning took 2.35 seconds. The checkpoint lineage records source step 20064 and
runtime digest `9fa49c9a18194a688c816cc4838174e3fa8f12e6d74b26a8abc91dcbdc8d39f5`,
with optimizer reset. Nothing was installed into the GUI or public model catalog.

The six-channel encoder still omits explicit opponent goal planes. Mixed-order
training is the first coverage experiment; it does not establish that the current
representation can distinguish every strategically different position. Investigate
explicit goals if failures persist after adequate per-order data and evaluation.
