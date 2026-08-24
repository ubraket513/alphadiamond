# What one Python callback per node costs

Measured 2026-08-24 on the development machine (Windows, CPython 3.12) with
`az-bench/profiles/bench_bridge_search.py`, two-seat opening position, a
scripted evaluator that does no arithmetic worth speaking of.

The question this answers: the native search suspends on every node and asks
Python for that node's answer. Does the boundary crossing cost more than the C++
tree saves? For the arena it plainly does not — a network forward pass dwarfs
the crossing. For a *cheap* evaluator, which is the self-play runners' shape in
tests, it is not obvious.

| Simulations | Python `MCTS2P` | Native + callback | Ratio |
|---|---|---|---|
| 64 | 58.0 ms | 35.3 ms | 1.65× |
| 400 | 406.1 ms | 196.1 ms | 2.07× |
| 1500 | 1679.3 ms | 753.1 ms | 2.23× |

**Warm up before measuring.** The first native search in a process pays for the
extension's one-time setup, and an unwarmed 64-simulation measurement reported
native as 3.5× *slower* — the opposite of the truth, and a number that would
have justified abandoning the bridge. The table above takes the best of three
after a warm-up search.

So the crossing is real but small: even against an evaluator that returns a
constant, the C++ tree wins by roughly 2×, and the margin grows with the
simulation count because the per-node Python cost is amortised over more tree
work.

## What this does and does not license

It licenses the arena and the GUI agent, which is where it is used: one search
per move, one evaluator that is a real network.

It does not make the per-node bridge the right vehicle for self-play. Self-play
wants many games in flight so one evaluator call answers a *batch*, which is
what the native pool (`play_episodes`) already does; a runner on the per-node
bridge would give up batching to gain 2× on the tree. The self-play runners
therefore stay on the Python search until they move to the pool, not to this.

There is also a test-shaped consequence worth recording:
`test_a_lane_that_finishes_early_takes_pending_work` makes one job slow by
giving it many simulations. Its margin assumes the search cost it was tuned
against; anything that changes that constant by 2× needs the fixture re-tuned,
not the assertion relaxed.
