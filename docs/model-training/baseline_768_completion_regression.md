# Resolved: the 768-game completion regression was a match-construction bug

**Status:** closed. Root cause found and fixed in `standard_soo_match()`.

## Resolution

The previously reported completion regression and entombment pathology were
caused by an **incorrect native match construction in the training and benchmark
paths**. The native rules engine itself remained parity-correct; production
orchestration supplied the wrong Soo seat/camp contract. Restoring the
Python-era golden geometry recovered step-44250 completion to **97.5 %** over 768
games with the fix applied inline, and **98.2 %** once every call site was routed
through the shared factory.

`game_match()` built the Soo match as `{1, camp 0, target 3}` and
`{2, camp 3, target 0}`, so each player's target camp *was the other player's
starting camp*. The normative fixture `tests/golden/rules-v1.txt` specifies
`1,2,5` and `2,0,3` — each targeting the camp opposite its own, starts 120°
apart, neither target overlapping the other's start.

Under the broken geometry every game began with all ten of the opponent's pieces
already occupying the camp that has to be filled to win, and `has_finished()`
requires all ten cells to hold the owner's own piece. Games could only end after
one side evacuated that camp entirely, and the last piece out was easily sealed
in.

| configuration | completion | aborted |
|---|---|---|
| broken geometry, step 44,250 | 77.1 % (592/768) | 176 |
| broken geometry, step 37,050 | 78.0 % (599/768) | 169 |
| **golden geometry, step 44,250** | **98.2 % (754/768)** | **14** |
| §7.1 historical reference | 97.7–98.3 % (750–755/768) | ~14 |

## What was wrong in the earlier analysis

This document previously concluded that the winning player *entombs* a loser's
piece, and that **a stronger network meets the rule more often** — offered as an
explanation for step 44,250 scoring 77 % where §7.1 recorded 98 %.

That was wrong, and the measurement that killed it was the step-37,050
comparison: converted from the `soo-v2.0.0-rc.1` release under an audited
conversion, it scored **78.0 %**, statistically indistinguishable from step
44,250's 77.1 % (Fisher exact p = 0.71). Two checkpoints seven thousand steps
apart cannot both have independently "discovered" a pathology at the same rate;
a fixed configuration fault can produce exactly that.

The blocker census had the answer in it all along and was misread: all 186
blocking pieces sat in camp 0 or camp 3 — the two *starting* camps, which under
the broken configuration were also the two *target* camps. They were never
wanderers that had to be sealed in over the course of a game. They were starting
pieces that had not finished evacuating.

## What survives from it

The entombment mechanism is real, just rare. With correct geometry the residual
14 aborts in 768 games are still entombments — all 15 blocking pieces sat on
non-corner cells, i.e. pieces that genuinely travelled into a target camp during
play and were sealed there. The geometry fault amplified a ~1.8 % phenomenon by
roughly thirteen times.

Adjacent camps deliberately share one corner hole (camp 0 with camp 5 at hole
12, camp 2 with camp 3 at hole 60, and so on), so each player legitimately starts
with one piece inside an opponent's target camp. That is by design and is part
of why correct geometry still aborts a small fraction of games rather than none.

The diagnostic tooling built during the investigation stays useful:
`iterations/N/selfplay.metrics.json` and `iterations/N/aborted-games.json`, and
the completed-game move percentiles that first showed the network was playing
*better* than the reference while completing far less often — the dissociation
that ruled out a degrading network and pointed at termination.

## Consequences for other measurements

**Everything measured before `1c57881` ran the wrong game.** In particular the
repetition-trigger comparison recorded in
[`repetition_trigger_config_gap.md`](repetition_trigger_config_gap.md) — trigger
off 4/32 against trigger on 8/32 — was collected under the broken geometry and
carries no information about the trigger. It was already too small to conclude
from; it is now also measuring the wrong game.

The GPU throughput figures are unaffected in kind but were measured on a
workload where roughly a fifth of games were pathological: ~98,000 evaluations/s
at 0.96 evaluator busy, batch mean 112 against a cap of 128, at 768 games / 256
lanes / 16 threads / cap 128 / 50 us. That operating point remains far better
than the acceptance configuration's 22,300 eval/s and is worth adopting; the
per-sample yield should be re-measured now that games complete.

## Prevention

The geometry was written out by hand at five call sites and drifted. The Qt
application held the correct triples while the trainer and both benchmarks held
broken ones, so the application and the trainer played different games and only
the trainer was wrong.

`soo::standard_soo_match()` and `soo::standard_min_match()` are now the single
authority, and `match_geometry_test` pins them to the golden fixture on four
independent properties: fixture equality, no seat targeting a camp another seat
starts in, each seat targeting the camp opposite its own, and target camps
occupied at move zero only through a shared corner. Reintroducing the original
broken pair fails three of the four.

Min was also corrected — it was `{1,0,3} {2,2,5} {3,4,1}` against the fixture's
`{1,2,5} {2,1,4} {3,0,3}`. Min had no target/start overlap so it was not broken
the way Soo was, but it was a different game from the one the fixture defines.
Min self-play has been exercised on CUDA through the corrected factory;
behavioural certification still needs a trained Min checkpoint, which does not
exist in this tree.
