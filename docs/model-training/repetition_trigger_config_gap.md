# The repetition trigger is implemented but unreachable

**Status:** wiring fixed; **the trigger must stay off in production.**

## Conclusion first

**Production A0 is flat 128 simulations. Do not enable this trigger in
`soo-production.json` or `min-production.json`.**

The repetition trigger is the best *generator* on record -- 98.0 % completion,
9.6 % censoring, 392 samples/s, 5 % of moves boosted, beating flat-128 on all
four axes. It is also the wrong thing to train on.
[`soo_scratch_training.md`](soo_scratch_training.md) §6.8 measured the learner
each generator produces, and only flat-128 holds its actor's level:

| generator | targets from 128-sim search | actor | learner | |
|---|---|---|---|---|
| flat 128 | **100 %** | 97.8 % | **97.9 %** | **holds** |
| repetition trigger | 5 % | 98.0 % | 91.5 % | degrades 6.5 pt |
| flat 64 | 0 % | 93.2 % | 86.3 % | degrades 6.9 pt |

What tracks the outcome is the fraction of training targets produced by a
128-simulation search, not the censoring rate: the trigger censors the *least*
of the three and its learner degrades anyway, which overturned the earlier
"censoring is the operative variable" reading in §6.5. At 64 simulations the
root search does not reliably improve on the network's own prior, so its visit
distribution is not a policy-improvement target. **A cheap generator does not
make a cheap teacher** -- the extra search has to be in the targets the network
learns from, not only in the moves that were going wrong.

So this trigger is an instrument for generator-side experiments and for
diagnosing the aborted tail. It is not a production setting, and the wiring
below exists to make it *available*, not to make it *default*.

### Correction to an earlier draft of this document

An earlier version of this file, and the commit message of `13fc6b3`, described
the trigger as a measured improvement that every run had silently lost. That
overstated it: it read §6.6 (the generator result) without §6.8 and §6.9 (the
learner result that supersedes it). The wiring gap below is real and worth
fixing, but it was **not** costing production anything, because production
should be running flat-128 regardless.

## The gap

[`soo_scratch_training.md`](soo_scratch_training.md) §6.2 audited the aborted
self-play tail and found it is not slow progress but a **short-cycle attractor**:
median 31.6 % unique positions per move, one position revisited 61 times, 68.4 %
of moves returning within 8 ply. §6.6 then measured a mitigation -- boost the search
budget only on a move whose position already occurred within the last few plies,
keyed on the physical `dynamics_key` -- and found it dominates a flat budget as a
*generator* (see the conclusion above for why that does not make it a training
configuration):

| configuration | completion | discarded | terminal samples/s | moves boosted |
|---|---|---|---|---|
| flat 64 | 93.2 % | 28.7 % | 473 | — |
| flat 128 | 97.8 % | 11.6 % | 247 | 100 % |
| **repetition trigger** | **98.0 %** | **9.6 %** | **392** | **5.1 %** |

That trigger is implemented in the native engine. `EpisodeConfig::repeat_window`
and `EpisodeConfig::simulations_late` are declared in
[`native/include/soo/selfplay.hpp`](../../native/include/soo/selfplay.hpp), and
the live logic -- `lane.seen_recently(key)` selecting the boosted budget, and
`lane.remember(key, repeat_window)` maintaining the window -- is in
[`native/src/selfplay.cpp`](../../native/src/selfplay.cpp).

**Nothing ever set either field to a non-zero value.** `native/src/config.cpp`
parsed no such key, so no JSON config could reach them; `training_wiring.cpp`
did not forward them; `arena_episode_config` in `train_main.cpp` left them at
their defaults. Both default to `0`, which disables the trigger. Every
production, bootstrap and acceptance run therefore executed a flat budget, and
the knob was unavailable for experiments. For production runs that is the
correct budget anyway (§6.8); the cost of the gap is that the generator-side
experiment could not be run at all, not that production regressed.

This is a migration gap, not a design decision. The engine kept the capability
across the Python retirement; the configuration surface did not.

## Why it went unnoticed

The trigger degrades silently. With both fields zero the search is correct, the
pipeline is correct, and every gate passes -- games simply abort more often, and
an aborted game contributes zero samples, so the loss appears as a slightly
smaller replay rather than as a failure. `EpisodeMetrics::boosted_moves` exists
precisely so the trigger's firing rate is reported rather than inferred, but
with the trigger unreachable it was always zero, which is indistinguishable from
"configured and never needed".

## What changed here

`mcts.simulations_late` and `mcts.repeat_window` are now parsed, validated,
serialised and forwarded to self-play.

Both are **optional on read and always written**. The asymmetry is deliberate.
`resume` parses a run's immutable `resolved-config.json` back through
`ProductionConfig::from_json` ([`native/src/report.cpp`](../../native/src/report.cpp)),
so making the keys required would have made every run directory created before
this change unresumable. A new `require_keys` helper accepts a named optional
set while still rejecting unknown keys, so strictness is preserved.

Setting only one of the pair is rejected. A window with no boosted budget never
changes the search and a boosted budget with no window can never fire, so a
half-set pair is inert while looking enabled -- the failure mode this whole
document is about.

## What the first measurement showed

One two-iteration run per arm against step 44250 on an RTX 5090, identical in
every other respect (128 simulations, 16 games/iteration, `max_moves` 2000,
production exploration, `run_seed` 7):

| arm | iter 0 | iter 1 | pooled aborted |
|---|---|---|---|
| trigger off | 1/16 | 3/16 | 4/32 = 12.5 % |
| trigger on (`simulations_late` 256, `repeat_window` 8) | 4/16 | 4/16 | 8/32 = 25.0 % |

Fisher exact p = 0.34. **This measures nothing.** The direction is opposite to
the hypothesis and the result is not significant, but more importantly the
sample cannot support either reading: iteration-0 abort counts for one
*unchanged* configuration have been 7, 3, 1 and 4 out of 16 across four runs on
this host. The between-run spread is larger than the difference being tested.

Recorded here so the next person does not repeat it at this scale, and does not
mistake it for evidence that the trigger does not work.

## An observability gap this exposed

`EpisodeMetrics::boosted_moves` exists so the trigger's firing rate is
*reported rather than inferred*, but nothing surfaces it. The self-play stage
report in `train_main.cpp` is rebuilt from the reloaded episode artifact -- which
is what lets it survive `resume` -- and `SelfPlayResult` carries no metrics, so
the count is computed and dropped.

The practical consequence is that a run cannot currently distinguish "the
trigger was configured and never needed" from "the trigger was configured and
silently did nothing". Any measurement of the operating point needs this
surfaced first; threading it through touches the persisted episode artifact and
belongs in its own change.

Note for whoever writes that: the trigger cannot be exercised with
`DummyBatchEvaluator`. A near-uniform evaluator diffuses through the state space
and never repeats a full position -- measured zero repeats in 400 moves with an
unbounded window. The attractor is a property of a *trained* network's
preferences, so a behavioural test needs an evaluator that forces a cycle. The
wiring itself is asserted in `cli_contract_test`.

## What is still open

1. **Reproduce the historical baseline before anything else.** §7.1 recorded
   rolling A0 at flat 128 holding 750-755 of 768 games (97.7-98.3 %) over twenty
   iterations, with median game length 64-65. That is the number to reproduce
   against step 44250 before any trigger experiment is worth running: at 768
   games, `max_moves` 500, prior none, training off. If it reproduces at 97-98 %
   the pipeline is healthy and the next step is systems tuning. If it comes back
   at 70-80 %, there is a regression to find, and every abort-rate observation
   in this document is measuring that regression rather than A0.

2. **Sixteen games cannot answer it.** Iteration-0 abort counts for one
   unchanged configuration on one RTX 5090 were 7, 3 and 1 out of 16 across
   three runs. Any comparison at this sample size measures noise; §6.6 used 768
   games per arm, and a decision here needs a comparable scale.

3. **The arena is untouched.** `arena_episode_config` still leaves the trigger
   disabled, and arena games abort by the same mechanism -- an aborted arena
   game produces an incomplete opening block, which is barred from contributing
   promotion evidence. Extending the trigger there would reduce discarded
   evidence, but it changes evaluation behaviour and belongs in its own change
   with its own measurement.

4. **`max_moves` is inconsistent with the recorded finding.** §5.7 measured
   500 -> 750 as changing nothing at all -- not completion, not a single
   percentile -- and concluded `max_moves = 500` stays, because a repetition
   attractor is indifferent to the cap. All five shipped configs use `2000`. If
   that is deliberate it deserves a note; if not, every aborted game is paying
   four times the wall clock for no recovered games.
