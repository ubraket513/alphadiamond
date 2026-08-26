# The 768-game baseline does not reproduce: 77 % against a recorded 98 %

**Status:** open. Reproduced twice at 768 games. Cause not identified.
**Blocks:** systems tuning, and any generator experiment whose metric is completion.

## What was measured

The historical baseline in [`soo_scratch_training.md`](soo_scratch_training.md)
§7.1 -- rolling A0 at flat 128 simulations -- recorded **750-755 of 768 games
completing (97.7-98.3 %)** across twenty iterations, median game length 64-65.

Reproducing it against the promoted **step 44250** actor, on the same operating
point §6.7 prescribes (768 games / 256 lanes / 16 threads / cap 128 / 50 us),
`max_moves` 500, `bootstrap_prior` none, production exploration, training off:

| run | simulations | completion | aborted | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| A | 128 | **73.8 %** (567/768) | 201 | — | — | — | — |
| B | 128 | **77.2 %** (593/768) | 175 | 61 | 71 | 127 | 265 |
| C | 256 | **81.6 %** (627/768) | 141 | 63 | 73 | 139 | 345 |
| §7.1 reference | 128 | **97.7-98.3 %** | ~14 | 65 | 89 | 173 | — |

Every abort is `max_moves`. The accounting identity
`aborted * max_moves + new_samples == total_moves` holds exactly in all three.

## Why this is not the network degrading

**The completed games are better than the reference.** p50 61 against 65, p90 71
against 89, p99 127 against 173 -- shorter and tighter on every percentile.
§6.5 reads exactly that shape as improvement: "median 71 -> 65, p90 131 -> 89,
p99 307 -> 173. Shorter, more decisive."

**The recorded collapse looks nothing like this.** In §5 completion and game
length degraded *together* -- 90.5 % -> 84.9 % -> 64.5 % as median rose
80 -> 77 -> 105. Here completion is at collapse levels while length is at
best-recorded levels.

**The distribution is bimodal with no middle.** Completed games top out at 265
moves; every other game runs to exactly the 500 cap. Nothing sits between. A
network playing slightly worse produces a continuum of longer games, not a
cliff.

**It is not under-searching.** Doubling to 256 simulations buys 4.4 points
(77.2 -> 81.6) and plateaus 16 points short. §6.3 measured 128 -> 256 as adding
*nothing* on the step-34,650 actor, because that actor was already at 97.7 %.
Here the extra search cannot recover the deficit, so the deficit is not search
budget.

Together: the network plays *better* than the reference when it finishes, and a
fixed ~20 % of games cannot finish at all.

## Leading hypothesis, not yet confirmed

`has_finished` in [`native/src/rules.cpp`](../../native/src/rules.cpp) requires
**every** cell of a player's target camp to hold that player's own piece:

```cpp
for (int i = 0; i < kCampSize; ++i)
    if (state.occupancy[target[i]] != spec.id) return false;
```

This is the classic Chinese-checkers blocking problem. A single opponent piece
resting in a target camp makes that camp unfillable, and the game
unterminable -- regardless of how well either side plays. There is no
anti-blocking provision anywhere in the rules: no vacate obligation, no
"filled including opponent pieces" clause, no draw adjudication.

A stronger network is *more* likely to find this, because refusing to lose is
worth as much as winning when the alternative is a loss. That would produce
precisely the observed signature: sharper play in the games that resolve, and a
growing fraction that cannot resolve at all.

**This is a hypothesis.** It has not been confirmed, and confirming it means
inspecting the final positions of aborted games for opponent pieces in target
camps. The episode artifact is binary and there is no tool that dumps a final
position; that tool is the next step.

Note what it would *not* be: the rules are frozen and parity-gated against the
Python oracle (`tests/golden/`), so if this is the mechanism it is inherent game
semantics that the historical measurement shared, not something the native
migration introduced. In that case the difference between 98 % and 77 % has to
come from the actor's behaviour meeting an unchanged rule, which is consistent
with step 44250 being a later and stronger checkpoint than any measured in §7.

## What must happen before tuning resumes

1. **Dump the final position of aborted games** and count opponent pieces in
   target camps. Confirms or kills the hypothesis in one measurement.
2. **Measure an earlier checkpoint** on the identical harness. If step ~38,250
   reproduces 98 % where 44250 gives 77 %, the behaviour is the actor's and the
   rules gap is being newly exploited rather than newly present.
3. Only then decide whether the answer is a rules change (an anti-blocking
   provision, or draw adjudication), a training change, or an accepted property
   with `max_moves` sized accordingly.

Do not tune scheduler parameters against this workload in the meantime.
Roughly a fifth of its games are pathological, and any samples/hour figure
measured on it is measuring the pathology as much as the pipeline.

## Incidental: the operating point is far better than acceptance

The §6.7 operating point is worth adopting regardless: 768 games / 256 lanes /
cap 128 / 50 us sustains **~98,000 evaluations/s at 0.96 evaluator busy**, with
batch mean 112 against a cap of 128. The acceptance configuration
(16 lanes, cap 16) reaches 22,300 eval/s. That is a 4.4x difference and it costs
nothing but configuration.
