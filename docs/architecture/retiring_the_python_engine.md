# Retiring the Python engine

Written 2026-08-24. **The C++ core is the authority** for rules, encoding,
search and self-play. `src/diamond/game` and `src/diamond/alphazero/mcts` are no
longer the source of truth; they are kept for two jobs and are scheduled for
deletion once neither job needs them.

## The two jobs it still has

1. **Oracle.** `tools/build_golden.py` regenerates `tests/golden/` from the
   Python implementation. The native tests then run with no interpreter at all.
   A C++ test that compared one C++ function with another would prove nothing
   about the port, so *something* has to be the independent answer, and today
   that something is Python.
2. **The bridge.** `tests/native/` no longer compares the two engines -- those
   five gates were retired once `tools/mutation_check.py` showed their C++
   replacements catching the same mistakes. What remains tests the pybind
   bridge itself: the callback ABI, the GIL contract, the self-play pool, and
   the search paths the arena and the GUI agent drive through Python. Those
   retire when the bridge does.

Everything else that imports it is migration debt, counted and frozen by
`tests/test_engine_retirement.py`. It is counted in two piles, because one
number could not distinguish a module that *runs the rules* from one that names
a dataclass in a type hint:

* **behaviour** (7) -- runs `legal_moves`, `GameSession`, `MCTS2P`. This must
  reach zero before `diamond.game` can go, and it is the work queue.
* **definitions and types** (14) -- `standard_board`, `build_players`,
  `GameState`. Not rules; C++ receives the same tables through the topology
  export. These survive until the trainer speaks the native `State`.

The criterion is whether rules are applied. `initial_state` is a definition by
that test: it fills each seat's home camp and applies nothing.

## Both decisions are taken

`docs/architecture/decisions.md` records them: training game execution requires
the native extension, and the golden corpus is frozen as the normative game
contract. What follows is the queue those decisions drain, kept for the record
of why each entry was there.

Phase A is under way. `search_factory` has already left the behaviour queue --
it no longer names a Python search to fall back to -- taking the count from 7
to 6.

## The behaviour queue was two decisions, not seven tasks

Reading the seven, none is independent work:

| Module | Waits on |
|---|---|
| `game_adapter` | the corpus generator: it is what `tools/build_golden.py` drives |
| `search_factory` | the Python search existing at all |
| `selfplay/runner_2p`, `runner_3p` | the Python self-play backend |
| `orchestration/benchmark` | the Python search, which it benchmarks on purpose |
| `smoke`, `milestone2_smoke` | scripting games with `find_legal_move` |

So the queue drains on two decisions, both of which are product calls rather
than refactors:

1. **Does the trainer keep a fallback that runs without the extension?** Today
   `selfplay_backend` resolves to `python` on a host with no compiled backend.
   Dropping that removes the runners, the selector's fallback and the
   benchmark's comparison in one go -- and makes a compiler a hard requirement
   for training.
2. **Is the golden corpus frozen?** Freezing it retires the generator, and with
   it `game_adapter`'s oracle role. Safe now in a way it was not before: the C++
   side is pinned by the golden files, the Python side by its own unit tests
   (`tests/test_rules.py`, `test_moves.py`, `test_session.py`), and
   `tests/native/test_callback.py` still compares a native game against the
   Python backend move for move.

Neither is blocked on engineering. Both change what the project promises, so
they are recorded here rather than taken unilaterally.

## What has to be true before deletion

Deletion is not one commit. In rough dependency order:

1. **The runtime path is native.** `selfplay_backend` accepts `native` only;
   `python` and `auto` are recognised and refused with the reason, because a
   config that named an engine should not silently receive another. The Soo arena and the GUI agent's two-seat path now go through
   `diamond.alphazero.search_factory.two_player_search()`, which prefers the
   native search and falls back per game.

   Still on the Python search, each for a reason:
   - `selfplay/runner_2p` and `runner_3p`. Not for want of a native search any
     more -- both sessions take a wall-clock budget now, and the selector
     accepts a `deadline`. The reason is batching: self-play wants many games in
     flight so one evaluator call answers a batch, which is what the native pool
     already does. A runner on the per-node bridge would trade that away for
     about 2x on the tree (measured:
     docs/performance-profiling/bridge_search_findings.md). They move to the
     pool, not to the bridge.
   - `selfplay/runner_3p` passes a deadline for the same reason as
     `runner_2p`. `MinArena` and the agent's three-seat path now run on
     `SearchSession3P`, the native vector search.
   - `orchestration/benchmark` measures the Python search deliberately.
2. **Arena runs on the native core.** Done for Soo: `Game.search_with_callback`
   suspends the C++ tree on every node and asks the Python evaluator for that
   node's answer, so two different networks can alternate moves inside one game
   without the tree leaving C++. `tests/native/test_arena_search_parity.py`
   holds both engines to the same selected action, the same visit distribution
   and the same evaluator request sequence; the q values agree to float32,
   which is the precision of the callback ABI and of the network itself.
   Statistics and reports stay in Python -- the review is explicit that
   rewriting them buys nothing. `MinArena` is three-player and stays on the
   Python search until a 3P native search exists.
3. **The golden corpus has an independent generator.** Today the oracle *is*
   the Python engine. Options, in order of preference:
   - freeze the corpus and its expected answers permanently: the file is the
     oracle, generated once by a Python implementation preserved in history and
     regenerated never;
   - or keep a minimal Python reference implementation whose only purpose is
     regeneration, in `tools/`, with no production code importing it.
   The first is cheaper and is what the deletion plan assumes. It requires the
   corpus to be complete enough that no future change wants regeneration --
   which is exactly what freezing it means.
4. **Each retired parity gate has a demonstrated C++ replacement.** Done for
   all five: `tools/mutation_check.py` runs thirteen behavioural mutations and
   requires the named gate to fail on each. What is left in `tests/native/`
   tests the bridge, not parity.
5. **The trainer no longer needs a Python `GameState`.** Replay samples carry
   encoded features rather than states, so this is mostly about
   `selfplay_workers` and the episode conversion around it.

## What is deliberately *not* being ported

From the review, and unchanged: the training loop and autograd, optimizer and
checkpoint surgery, experiment orchestration, rating and statistics, and the
Hugging Face synchronisation. Python is the better control-plane language and
these are not on the shipped application's runtime path.

## Order of operations

```text
native self-play default            done (auto)
dependent ratchet                   done (tests/test_engine_retirement.py)
arena on the native core            done (Soo and Min)
GUI agent on native                 done (both seat counts)
three-player native search          done (SearchSession3P)
native wall-clock deadline          done (set_budget on both sessions)
Min self-play on the native pool    done (EpisodeSearch, per-seat targets)
Soo/Min training on native by default  done (selfplay_backend auto)
freeze the golden corpus            then
retire the Python parity gates      one at a time, each with its C++ replacement
delete src/diamond/game             last
```

Nothing in that list is a rewrite of the training system. It is a change of
which implementation is authoritative, and the Python evidence goes only after
its replacement has proved the same things.
