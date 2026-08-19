# AlphaDiamond Bootstrap Heuristic Design

Date: 2026-08-19

## Goal

Allow cold-start Soo and Min self-play to reach real terminal Diamond outcomes often enough to populate replay and begin learning, without changing Diamond rules, terminal value semantics, MCTS backup semantics, replay targets, rating semantics, or deployment behavior.

The heuristic is temporary bootstrap scaffolding. It must be simple, deterministic, framework-neutral, easy to test, and removable once a learned policy can finish games reliably.

## Problem

With an untrained network and very small search budgets, especially `mcts.simulations=1`, legal self-play can wander for `self_play.max_moves` without completing a game. The current self-play contract correctly discards incomplete episodes, so repeated `max_game_moves_exceeded` outcomes leave replay empty and training cannot start.

Increasing `max_moves` does not solve the information problem. Fabricating winners or dense value targets at the move limit would change the learning objective and is explicitly out of scope.

## Decision

Introduce one bootstrap-only policy prior named:

`canonical-target-distance-v1`

For the acting player, the existing canonicalization always rotates the player's home camp to canonical `Z_POS`, so the player's target camp is canonical `Z_NEG`.

Precompute a graph-distance table over the authoritative 73-hole board:

`distance[position] = shortest neighbour-graph distance from position to any hole in canonical Z_NEG target camp`

Use multi-source BFS from all canonical `Z_NEG` camp positions. This table is board-topology derived and independent of learned weights.

For each authoritative legal action `source -> destination`, compute:

`progress(action) = distance[source] - distance[destination]`

Positive progress means the moved piece ends closer to the canonical target camp. Negative progress is allowed; no legal action is removed.

Convert progress scores to a normalized prior with a fixed softmax:

`P_h(a) = exp(progress(a) / temperature_h) / sum_b exp(progress(b) / temperature_h)`

Version 1 uses `temperature_h = 1.0`. This is part of the heuristic identity and should not become a broad tuning surface in the first implementation.

## Integration Boundary

The heuristic must not be implemented inside PUCT selection, MCTS backup, game rules, or neural-network code.

Implement a framework-neutral `Evaluator` decorator/wrapper for bootstrap self-play:

1. Receive the same batch of `EvalRequest` objects as the base evaluator.
2. Call the base evaluator normally.
3. Preserve each returned `value` exactly.
4. Replace only `priors` with `canonical-target-distance-v1` priors over exactly the legal action IDs in the request.
5. Return normal `EvalResult` objects.

The base evaluator may be local Torch or the existing remote/central evaluator. Therefore centralized batching, model routing, GPU inference, and MCTS remain unchanged.

## Scope of Use

Heuristic prior is allowed only for explicitly configured bootstrap self-play.

Allowed:

- bootstrap Soo self-play
- bootstrap Min self-play
- CPU bootstrap probes
- CUDA bootstrap generation when explicitly enabled

Forbidden:

- promotion arena
- Soo Elo benchmark
- Min TrueSkill benchmark
- leaderboard/rating matches
- normal post-bootstrap self-play
- GUI/deployment agents
- neural value targets
- terminal adjudication

If bootstrap mode is disabled, behavior must be byte-for-byte/semantically equivalent to the existing evaluator path aside from irrelevant object identity.

## Training Semantics

The bootstrap heuristic changes only the MCTS prior used to generate trajectories.

Policy targets remain ordinary MCTS visit distributions.

Value targets remain ordinary real terminal Diamond outcomes:

- Soo: current-player scalar win/loss `+1/-1`
- Min: canonical placement utility `[self,next,previous]` with first `+1`, second `0`, third `-1`

An episode that exceeds `self_play.max_moves` remains aborted and contributes zero training samples.

No draw, truncated-game value, distance reward, pseudo-winner, or heuristic terminal value may be introduced.

## Why Prior Replacement, Not Blending

Version 1 deliberately avoids a blend such as:

`P = lambda * P_h + (1-lambda) * P_net`

At cold start, `P_net` contains little useful information, while introducing `lambda` creates an unnecessary schedule/tuning problem.

Bootstrap v1 therefore has two explicit modes:

- bootstrap: heuristic prior replaces neural prior; neural value is preserved
- normal: neural prior and neural value are used unchanged

If experiments later show a need for a gradual handoff, that is a separate design decision backed by evidence.

## Exploration

Existing root Dirichlet noise remains authoritative and may be applied after the heuristic prior exactly as it is applied after neural priors today.

Do not add a second exploration mechanism.

The heuristic must not forbid backward, lateral, or tactically unusual legal moves.

## Configuration

Add the smallest explicit bootstrap configuration surface possible. A recommended shape is a focused config value such as:

- `bootstrap_prior: "none" | "canonical-target-distance-v1"`

Do not add adaptive controllers, automatic completion thresholds, lambda schedules, heuristic weights, per-model hand tuning, or a general plugin registry in v1.

The selected bootstrap prior identity must be recorded in run/stage metadata for reproducibility. It is not part of model checkpoint compatibility because it does not alter network architecture or value semantics, but it must be visible in self-play provenance.

## Bootstrap Workflow

Phase B0 — Bootstrap generation/training:

- start from the untrained/bootstrap checkpoint
- explicitly enable `canonical-target-distance-v1`
- use the existing real self-play pipeline
- keep ordinary authoritative legality and terminal semantics
- generate completed games and replay
- train normally
- save a bootstrap checkpoint

Phase A0 — Normal AlphaZero:

- start from the bootstrap checkpoint
- disable heuristic prior
- use learned neural priors normally
- run ordinary production self-play/training/promotion/rating

No automatic phase transition is required in v1.

## Scientific Evaluation

Before using the heuristic for a long training run, perform a fixed-seed A/B probe with identical settings except the prior mode.

Baseline:

- `mcts.simulations=1`
- bootstrap prior disabled

Candidate:

- `mcts.simulations=1`
- `canonical-target-distance-v1` enabled

Run a small, deterministic set for Soo and Min separately. Twenty episodes per condition is a reasonable initial probe, but the test harness should make the count configurable.

Report at minimum:

- completion rate before `max_moves`
- median move count among completed games
- p90 move count among completed games when enough completions exist
- replay samples generated per attempted episode
- abort count/reason

The heuristic is successful if it materially increases real terminal completion and creates stable non-empty replay without changing terminal semantics.

Do not use Elo or TrueSkill to evaluate the bootstrap heuristic itself. This experiment measures data-generation viability, not final playing strength.

## Architecture

Recommended dependency direction:

```text
MCTS
  |
  v
Evaluator
  |
  +--> BootstrapPriorEvaluator (optional, self-play only)
          |
          v
      Base Evaluator
          |
          +--> local TorchEvaluator
          or
          +--> RemoteEvaluator -> central inference coordinator
```

The wrapper may need access to action source/destination decoding and the precomputed canonical distance table. It must not import Torch.

MCTS must remain unaware that the returned prior is heuristic.

## Implementation Shape

Follow existing repository patterns after inspection. A likely minimal shape is:

- one small heuristic module responsible for distance-table construction and legal-action prior calculation
- one small evaluator decorator responsible for replacing priors while preserving values
- narrow production/self-play configuration wiring to enable the decorator only in bootstrap mode
- focused unit and integration tests

Do not create a generic heuristic framework, policy-composition DSL, curriculum subsystem, or new trainer.

## Required Tests

### Geometry / heuristic

- canonical `Z_NEG` camp positions have distance zero
- every board position has a finite non-negative distance
- distance table is deterministic
- each neighbour edge differs in distance by at most one where applicable to shortest-path distance
- a legal action ending closer to target has greater progress than the same source ending farther away when such fixture exists
- backward actions remain present with non-zero prior
- priors cover exactly the supplied legal action IDs
- priors are finite, positive, normalized to approximately 1
- action ordering does not change semantic probabilities

### Evaluator wrapper

- preserves scalar Soo value exactly
- preserves vector Min value exactly
- replaces priors only
- preserves batch order
- rejects malformed/mismatched legal-action results consistently with existing evaluator contracts
- imports no Torch
- works over a deterministic dummy evaluator
- works over the same public `Evaluator` interface used by `RemoteEvaluator`

### Self-play integration

- bootstrap disabled follows the existing path
- bootstrap enabled can produce training samples only after a real terminal state
- aborted `max_game_moves_exceeded` episodes still produce zero samples
- Soo value semantics unchanged
- Min value semantics unchanged
- no arena/rating path enables the bootstrap evaluator
- self-play provenance records heuristic identity

### Scientific probe

Add a bounded test/probe command or function that can compare `none` vs `canonical-target-distance-v1` with fixed seeds and return the required metrics. It must not fabricate success criteria or hard-code expected production completion percentages.

## Non-Goals

Do not implement in this change:

- dense reward shaping
- heuristic value head targets
- pseudo-terminal adjudication at `max_moves`
- forced target-camp moves
- backward-move prohibition
- cycle penalties
- opponent blocking heuristics
- Hungarian piece-to-target assignment
- handcrafted endgame solver
- supervised imitation dataset pipeline
- lambda blending/decay
- adaptive bootstrap scheduler
- C++ heuristic/search code
- changes to Elo, TrueSkill, arena, or checkpoint value semantics

These may be reconsidered only if measured `canonical-target-distance-v1` results are insufficient.

## Acceptance Criteria

The design is complete when implementation can demonstrate all of the following:

1. Existing Milestone 1 and Milestone 2 tests remain green.
2. MCTS remains framework-neutral and unchanged in learning semantics.
3. Bootstrap prior is opt-in and self-play-only.
4. Network values and terminal training targets are unchanged.
5. All authoritative legal actions remain available.
6. Incomplete episodes still contribute zero samples.
7. Fixed-seed CPU A/B probe shows whether the heuristic materially improves completion/replay generation.
8. Disabling bootstrap restores normal neural-prior AlphaZero without checkpoint conversion or architecture changes.
9. Arena, Elo, TrueSkill, benchmark, and deployment paths never use the bootstrap prior.
10. No additional heuristic complexity is added without evidence from the v1 probe.

## Architectural Principle

The heuristic exists to help AlphaZero reach its first useful terminal trajectories, not to become the game-playing intelligence.

The strongest version of this design is the smallest one that reliably gets replay started and can then be switched off.

---

## Addendum: v1 measured result and the v2 successor

Date: 2026-08-19 (implementation)

### What v1 does well

`canonical-target-distance-v1` was implemented exactly as specified above and
drives the opening and midgame effectively: total target-distance for a Soo
player falls from 70 to roughly 2 within about 100 moves.

### The measured limitation

Under deterministic greedy play (`dirichlet_epsilon=0.0`, argmax on the prior)
v1 never terminates. It reaches a target-camp packing position where eight of ten
pieces are home, the last two sit one step out, and *every* legal action scores
`progress == 0`. The remaining target holes are reachable only from inside the
camp, so the fixed distance table cannot distinguish a useful camp-internal
rearrangement from an idle shuffle. Greedy play then enters a short cycle.

This is a limitation of the metric, not of the implementation: `distance[]` is
state-independent and cannot see which target holes are already filled.

Root Dirichlet noise does break the plateau in practice, so v1 does complete
games under ordinary MCTS settings. The stall is nonetheless real, and the
regression is recorded in `tests/alphazero/bootstrap/test_vacancy_prior.py`.

v1 is preserved unchanged as an experimental result. It was deliberately not
patched.

### v2: `canonical-target-vacancy-distance-v2`

The minimal next experiment scores the whole position instead of the single
moved piece:

```text
U   = canonical Z_NEG target holes the acting player does NOT yet occupy
O   = the acting player's pieces outside the target camp
Phi = sum over O of min over U of graph_distance(piece, target)
```

Each legal action is scored `Phi(before) - Phi(after)`, updating only the acting
player's own occupancy, then passed through the same fixed softmax
(`temperature = 1.0`).

`U` means "not yet mine", never "physically empty": a target hole held by an
opponent is still a slot this player must eventually fill. Whether a move is
playable right now remains entirely the authoritative rules' business.

Opponent pieces, legality, rules, MCTS backup, neural value and terminal targets
are all unchanged, every legal action keeps a strictly positive prior, and no
Torch dependency is added.


### Probe results

Fixed-seed A/B probe, `mcts.simulations=1`, `max_moves=2000`, 20 episodes per
condition, seeds 0-19, untrained (dummy) evaluator:

| model | prior | completion | median | p90 | samples/episode | aborts |
|---|---|---|---|---|---|---|
| Soo | `none` | 0.0% | - | - | 0.0 | 20 x `max_game_moves_exceeded` |
| Soo | `canonical-target-distance-v1` | 100.0% | 84.5 | 126 | 88.6 | none |
| Soo | `canonical-target-vacancy-distance-v2` | 100.0% | 74.0 | 93 | 76.4 | none |
| Min | `none` | 0.0% | - | - | 0.0 | 20 x `max_game_moves_exceeded` |
| Min | `canonical-target-distance-v1` | 100.0% | 131.0 | 172 | 135.3 | none |
| Min | `canonical-target-vacancy-distance-v2` | 100.0% | 101.0 | 130 | 103.3 | none |

Both priors take completion from 0% to 100% and turn an empty replay into a
populated one, which is the question the probe exists to answer. v2 additionally
finishes games in fewer moves (Soo p90 93 vs 126; Min p90 130 vs 172), consistent
with it seeing the target-camp endgame that v1 is blind to.

Since v2 produces stable terminals well inside 2000 moves for both models,
assignment matching and a separate endgame heuristic remain unimplemented, as
the Non-Goals require.

### Scope

Both priors remain opt-in, self-play-only, and selected by the single
`bootstrap_prior` config value. Every Non-Goal above still stands: assignment
matching, forced moves, camp locking, cycle penalties and reward shaping remain
unimplemented, and are to be reconsidered only if v2 fails to produce stable
terminals.
