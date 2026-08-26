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

## Confirmed: the winner entombs a loser's piece in its own target camp

All 176 aborted games of a 768-game run were inspected at the move cap
(`iterations/N/aborted-games.json`, written by the self-play stage).

```
aborted games                                176   (all at the 500-move cap)
target camp contains a foreign piece         176   (100.0 %)
  exactly 1 blocker                          166   ( 94.3 %)
  >= 2 blockers                               10   (  5.7 %)
no foreign blocker                             0   (  0.0 %)

blocker had zero legal moves                 176   (100.0 %)
blocker pieces total 186: immobile 186, mobile 0
short-cycle repetition in last 64 ply        176   (100.0 %)
max revisits of one position   p50 46   p90 96   max 112
unique positions / moves       p50 0.358   p10 0.252
```

The last two lines reproduce §6.2's audit almost exactly (median 0.316 unique,
one position revisited 61 times), so the repetition that audit found is this
same phenomenon seen from the other end.

The composition of the blocked camps says what is happening:

| own | foreign | empty | camps |
|---|---|---|---|
| 8 | 1 | 1 | 149 |
| 9 | 1 | 0 | **17** |
| 7 | 2 | 1 | 10 |

**The blocked camp belongs to the player who is about to win.** It already holds
eight or nine of that player's ten pieces. One enemy piece sits in a remaining
cell, and in 17 games the camp is completely full -- nine own pieces plus one
enemy, no empty cell at all.

**The blocker cannot leave.** Not once in 186 cases did it have a legal move. It
is not being retained by a losing player choosing to hold ground; it is
*entombed*, surrounded by the pieces of the player whose camp it is sitting in.
Blockers cluster in eight cells of the board, with 58, 48 and 57 accounting for
162 of 186 -- the deep corners of the target camps, filled last and sealed by
the owner's own advance.

So the mechanism is the exact opposite of a losing side stonewalling. The
winning side advances its pieces home, seals an enemy piece that wandered in
early into a corner it can no longer step or jump out of, and `has_finished`
then requires all ten cells to hold the owner's own piece:

```cpp
for (int i = 0; i < kCampSize; ++i)
    if (state.occupancy[target[i]] != spec.id) return false;
```

The winner cannot satisfy it, the loser cannot move the piece that would let
them, and no other seat can finish either. The game has no reachable terminal
state, and the remaining pieces shuffle until the cap -- which is what produces
the short-cycle repetition in every one of these games.

This also explains why search budget does not help. 128 -> 256 simulations
recovered 4.4 points and plateaued because deeper search cannot reach a terminal
state that does not exist. Whatever those 4.4 points are, they are games that
avoided the trap, not games that escaped it.

An earlier draft of this document, and the framing that prompted the
investigation, both supposed a *strategic* block held by the losing side. The
data rules that out: a retained block requires a mobile blocker, and there were
none.

## What this is, and what to do about it

This is not a native-migration regression. The rules are frozen and parity-gated
against the Python oracle, so the historical measurement ran the same
`has_finished`. What changed is the actor: step 44250 finishes its games -- p50
61 moves against the reference's 65 -- and driving that many pieces home that
fast is precisely what seals an enemy piece into the camp. **A stronger network
meets this rule more often.** That is consistent with 97.7-98.3 % at step
~34,650-38,250 and 77 % at step 44,250 with no code change between them.

Three readings, and they need separating before anything is changed:

1. **Intended Diamond strategy.** If leaving a piece in an opponent's camp is a
   legitimate spoiling tactic in this game, then the rules are right and the
   training configuration has to tolerate it.
2. **A long-standing design flaw.** Standard Chinese-checkers rule sets
   generally do address this -- a camp counts as complete when full if at least
   one piece is the owner's, or a piece may not remain in a foreign camp. This
   rule set has no such provision, and the failure is invisible until a player
   is strong enough to cause it.
3. **A missing termination/draw definition.** Even if the position is legal, a
   game with no reachable terminal state needs an adjudicated outcome rather
   than a move cap, because a capped game contributes zero training samples and
   is deleted from the replay.

Reading 3 is the one that bites training regardless of how 1 and 2 are decided:
23 % of games currently produce no data at all, and they are systematically the
games where one side had nearly won.

**Do not change the rules from this evidence alone.** Changing `has_finished`
changes game semantics, invalidates `tests/golden/`, and breaks parity with the
oracle those fixtures came from. The decision belongs to whoever owns the game
definition.

## What must happen before tuning resumes

1. **Decide which of the three readings above applies.** That is a game-design
   decision, not an engineering one.
2. **Measure an earlier checkpoint** on this identical harness. If step ~38,250
   reproduces 97-98 % where 44,250 gives 77 %, it confirms the rule was always
   there and strength is what exposes it.
3. Only then size `max_moves`, or add adjudication, or change the rule.

Do not tune scheduler parameters against this workload in the meantime. Roughly
a fifth of its games are pathological, and any samples/hour figure measured on
it is measuring the pathology as much as the pipeline.

## Incidental: the operating point is far better than acceptance

The §6.7 operating point is worth adopting regardless: 768 games / 256 lanes /
cap 128 / 50 us sustains **~98,000 evaluations/s at 0.96 evaluator busy**, with
batch mean 112 against a cap of 128. The acceptance configuration
(16 lanes, cap 16) reaches 22,300 eval/s. That is a 4.4x difference and it costs
nothing but configuration.
