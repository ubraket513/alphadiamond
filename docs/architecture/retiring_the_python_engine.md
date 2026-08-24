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
2. **Bridge parity.** `tests/native/` compares the two implementations position
   by position. That comparison is the only thing that would catch a native
   regression that the frozen golden files happen not to cover, and it is what
   keeps the frozen files trustworthy in the first place.

Everything else that still imports it is migration debt, counted and frozen by
`tests/test_engine_retirement.py`: the list of dependents may shrink, never
grow. Adding a module to it fails; removing one is the progress.

## What has to be true before deletion

Deletion is not one commit. In rough dependency order:

1. **The runtime path is native everywhere it matters.** Self-play defaults to
   `auto`, which resolves to the native backend wherever the extension is
   importable. The Soo arena and the GUI agent's two-seat path now go through
   `diamond.alphazero.search_factory.two_player_search()`, which prefers the
   native search and falls back per game.

   Still on the Python search, each for a reason:
   - `selfplay/runner_2p` passes a wall-clock `deadline` to the search, which
     the native side does not implement. Losing a game's time bound silently
     would be worse than not using the native search, so the selector declines
     any call carrying extras it cannot honour. Implementing a deadline in the
     C++ session is what unblocks this.
   - `selfplay/runner_3p`, `MinArena` and the agent's three-seat path: the
     native search is two-player. `MCTS3P` has no native counterpart at all.
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
4. **Each retired parity gate has a demonstrated C++ replacement.** The
   standing rule from `products.md`: a Python gate goes only after its C++
   counterpart has been shown to catch the mutations that gate catches --
   demonstrated by mutating the expectation and watching it go red, not
   asserted.
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
arena on the native core            done (Soo; Min is 3P and stays)
GUI agent on native                 done (two-seat path)
self-play runners on native         blocked: the native search has no deadline
freeze the golden corpus            then
retire the Python parity gates      one at a time, each with its C++ replacement
delete src/diamond/game             last
```

Nothing in that list is a rewrite of the training system. It is a change of
which implementation is authoritative, and the Python evidence goes only after
its replacement has proved the same things.
