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

It does not make the per-node bridge the right vehicle for *training* self-play.
Training wants many games in flight so one evaluator call answers a *batch*,
which is what the native pool (`play_episodes`) already does; a runner on the
per-node bridge gives up batching to gain 2× on the tree. Training self-play
therefore runs on the pool.

**Updated 2026-08-24.** This section used to end "the self-play runners
therefore stay on the Python search until they move to the pool". They did not
move to the pool and they did not stay on the Python search — there is no
Python search. `runner_2p` and `runner_3p` take their search from
`search_factory`, which is the per-node bridge, because what they serve is the
*single-episode* caller that holds its own `Evaluator` (`bootstrap/probe`
above all), and the pool's shape is one model in one process. The paragraph
above is still the reason training does not use them.

There was also a test-shaped consequence, and it came true in the worst way:
`test_a_lane_that_finishes_early_takes_pending_work` made one job slow by
giving it many simulations, and this section warned that anything changing that
constant by 2× needs the fixture re-tuned. What happened was larger than 2×.
Once the runner reached the native search, the near-terminal position's tree
was exhausted long before the simulation budget — 60,000 simulations ran no
longer than 1,500 — and the lanes split evenly. The fixture now buys its
slowness with a 300-move game: one round trip per move, engine-independent.
**Slowness measured in one engine's units is not slowness.**
