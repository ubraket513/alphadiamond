# Min next intervention — measured decision

## Selected branch

Select branch D: **vacancy-prior annealing (Phase 3B)**. Reject parallel MCTS,
immediate adoption of adaptive serial search, optimizer changes, and direct A0
promotion.

Min remains B0 until the final two-checkpoint A0 acceptance gate passes.

## Evidence and gate values

- Repository base: `589d6b452242f2a7d96a3277af76fec54582326d`
- Model SHA-256: `847007a72b0a283a1789c7cb160f9fd93864117ad1cdbf5bbf3541c92f5ef59a`
- Effective config SHA-256: `524b888adcba2b0bbdd5aa87f4c82a80be57c6098e31da8ec70cb092f2dbc89b`
- Replay manifest SHA-256: `dc0ce2efea60fc4b4b3ce1a21fe36f60bf7ff5b0fb916489a127e1988ae3ee59`
- Learner: finite for 256 steps; policy source/destination gradient means
  `8.97e-5`/`8.36e-5`; relative-update means `3.18e-6`/`2.88e-6`;
  held-out full KL `3.07686`; start-to-end policy KL `2.83e-11`.
- B0 full-sweep completion at 128/256/400 simulations: `256/256` for every arm.
  Median wall times were `83.436`, `136.960`, and `217.188` seconds.
- A0 full-sweep completion at 128/256/400 simulations: `0/256` for every arm.
  A0 adaptive-256/adaptive-400 completion was also `0/256`.
- A0 aborts: 128 and both adaptive arms hit the 800-move limit in all 256
  games; constant 256 and 400 hit the 180-second deadline in all 256 games.
- Adaptive boosted fraction was `0.00000976563` for both arms. The repetition
  trigger affected only two moves across 204,800 moves per arm and therefore
  cannot be adopted.
- A0 legal KL: `0.104071` at 128, `0.0864906` at 256, `0.0808324` at 400.
- A0 legal probability mass: `0.125646`, `0.124227`, `0.122423`.
- A0 full KL exceeds legal KL by more than `0.50` nat at every arm; the 256
  and 400 arms satisfy the complete `ILLEGAL_MASS_DOMINATED` gate. The existing
  full policy-loss plateau must not be described as proof that legal policy fit
  is stalled.

The full sweep used 256 games, 128 lanes, seed `20260901`, FP32, and the same
checkpoint and config for every arm. Its eight JSON files are under
`$MIN_A0_EXP/results/serial-search/full`; the deterministic repository report
contains the complete metric table and comparisons.

## Rejected branches

- **Adaptive serial search:** rejected because `boosted_fraction=0.00000976563`
  and completion stayed 0%; it was behaviorally identical to A0-128.
- **Stronger constant serial search:** rejected because 256/400 reduced legal KL
  but did not complete any game and converted move-limit aborts into deadline
  aborts. `DEEPER_SEARCH_HELPS` is false.
- **Parallel MCTS:** rejected because its prerequisite—measured benefit from
  deeper serial search—is false.
- **Immediate A0 acceptance:** rejected because completion is 0%, far below 97%.
- **Optimizer/LR intervention:** rejected for this branch because gradients and
  parameter updates are finite and non-zero. The nearly invariant policy
  distribution remains a tracked learner concern, not authorization to combine
  optimizer changes with the prior intervention.

## Authorized change

Implement a legal-set blend between the normalized vacancy prior and normalized
network prior. The only production configuration field authorized to change is:

```text
self_play.bootstrap_prior_weight
```

`self_play.bootstrap_prior` remains
`canonical-target-vacancy-distance-v2`. Search simulations remain 128. Replay
capacity, learner steps, FP32 precision, seeds, MCTS exploration fields, move and
deadline limits, optimizer fields, and checkpoint/replay schemas remain unchanged.

The fixed reversible schedule is alpha `1.00` confirmation, then `0.75`, `0.50`,
`0.25` for bounded segments, followed by an alpha `0.00` probe only after every
prior gate passes.

## Expected improvement and rollback

Expected improvement is monotonic transfer from B0's 100% completion behavior
toward network-prior play while preserving completion, bounded move length,
cycling safety, seat balance, finite inference, and measurable policy learning.

Before decreasing alpha, a 256-game probe must satisfy the roadmap's complete
stage gate. On any failure, restore the prior checkpoint and prior durable config
transition, record the failed gate, and stop. Do not increase max moves or add
parallel search as compensation. Increasing alpha is allowed only through that
explicit rollback record.
