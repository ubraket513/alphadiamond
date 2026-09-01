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
- B0 smoke completion at 128/256/400 simulations: `32/32`, `32/32`, `32/32`.
- A0 smoke completion at 128/256/400 simulations: `0/32`, `0/32`, `0/32`.
- A0 adaptive-256/adaptive-400 completion: `0/32`, `0/32`.
- A0 aborts: 128 and both adaptive arms hit the 800-move limit in all 32
  games; constant 256 and 400 hit the 180-second deadline in all 32 games.
- Adaptive boosted fraction: `0` for both arms. The repetition trigger captured
  no extra search and therefore cannot be adopted.
- A0 legal KL: `0.103219` at 128, `0.0871268` at 256, `0.0814674` at 400.
- A0 legal probability mass: `0.124412`, `0.126187`, `0.123776`.
- A0 full KL exceeds legal KL by more than `0.50` nat at every arm; the 256
  and 400 arms satisfy the complete `ILLEGAL_MASS_DOMINATED` gate. The existing
  full policy-loss plateau must not be described as proof that legal policy fit
  is stalled.

The smoke invocation requested 32 lanes for 32 games, but the benchmark
correctly rejects `games <= lanes` because no job queue is exercised. The runner
used 16 lanes for all arms, preserving identical games and seeds while producing
two production-style queue waves.

## Rejected branches

- **Adaptive serial search:** rejected because `boosted_fraction=0` and completion
  stayed 0%; it was behaviorally identical to A0-128.
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
