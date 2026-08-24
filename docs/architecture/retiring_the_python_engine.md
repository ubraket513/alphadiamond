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
   that something is Python. The generator now carries its own adapter
   (`PythonRulesGame`, in that script): `AlphaZeroGameAdapter` asks the native
   core, and a corpus regenerated from the implementation it exists to pin
   would prove nothing. Verified: the tools-local oracle reproduces every
   frozen file byte for byte.
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

* **behaviour** (0) -- runs `legal_moves`, `GameSession`, `MCTS2P`. This must
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
to 6. `orchestration/benchmark` followed it: its rated matches now go through
`two_player_search()` / `three_player_search()`, the same selector the arena and
the GUI agent use. 6 to 5.

That entry was mis-described in decision 1 and in the table below as a
*comparison* against native. It was not: the stage played its rated games on
`MCTS2P`/`MCTS3P` because that was the only search when it was written. So it
did not need to leave the shipped package -- it needed the selector, and the
matches it rates are now rated by the engine that plays them in production,
which is the point of rating them at all.

The two self-play runners followed, for the same reason and by the same one-line
change. Decision 1 said to delete them; that turned out to be the wrong unit.
What ran the Python engine in `runner_2p` was the hard-coded `MCTS2P`, not the
episode loop around it -- the temperature schedule, the sample construction and
the wall-clock abort are control plane, and the review keeps control plane in
Python. They now take a `search_factory` defaulting to the native search, so the
runtime path is native without a second episode driver having to exist. The
training loop still runs on the pool, which batches; what the runners keep
serving is the single-episode caller with its own `Evaluator` -- `bootstrap/probe`
measuring a checkpoint's data-generation viability, where the pool's
model-in-one-process shape does not fit. 5 to 3.

Then the two smokes. Both build the same near-terminal fixture and asked
`find_legal_move` whether the finishing move they had constructed was real. The
geometry -- camp positions, neighbours -- is board definition and stays; the
legality answer now comes from the native core, asked with the finishing seat to
move rather than the state's current one. 3 to 1.

Last was `game_adapter` itself, and it was the only one that was a port rather
than a one-line swap: it *is* what arena, benchmark, probe and the worker path
apply moves through. It keeps `GameState` -- that type crosses process
boundaries in `SelfPlayJob` and is what the Qt agent and the arena speak, and
replacing it is Phase B -- and asks the native `Game` for legality and for
successors, converting by field copy at the boundary.
`tests/native/test_game_adapter_parity.py` holds both implementations to the
same answer for every legal action of all 1,327 fixture positions: 61,139
successors, identical in occupancy, seat to move, turn number, status and
podium. `resolve_action` went with it -- the GUI agent needs a `Move` with its
jump path, and that path is already in the request's own legal moves, so the
agent matches the chosen action against them instead of re-resolving through a
second implementation.

**The behaviour queue is empty. Phase A is done.**

## Phase B: the definitions moved rather than being rewritten

The fourteen definition dependents never wanted the engine. They wanted
`GameState`, `PlayerSpec`, `Board`, `Move` -- what a position *is*. Those were
in the same package as the rules only because that is where they were written.

So they moved, unchanged, to `diamond.contract`: `board.py`, `coordinates.py`,
`move.py` and `state.py`, by `git mv`. `next_player_id` did not go with them --
it decides whose turn it is, which is a rule, and it now lives in
`diamond.game.rules` where the rest of the turn order is. `IllegalMoveError`
did go: the type is vocabulary, and the C++ core raises it now too.

`diamond.game` imports from `diamond.contract` and re-exports nothing from it.
The dependency points away from the package being deleted, which is what makes
the deletion a deletion rather than a rewrite. Both ledgers are now empty, and
`src/diamond/game` is reachable from `tools/build_golden.py` and the bridge
gates in `tests/` -- from nothing that ships.

What is left before `src/diamond/game` can go:

* **The board is still generated in Python.** `standard_board()` produces the
  topology tables the extension is configured with at import and the deployment
  artifact ships; `native/src/topology_io.cpp` only reads them. That generator
  is in `diamond.contract` now, not in the engine, so it does not block the
  deletion -- but it does mean the C++ core is not yet self-sufficient in
  geometry, and that is the next thing to settle.
* **The rules themselves.** `rules.py`, `session.py` and `history.py` are the
  oracle. Decision 2's precedent applies: the generator moves to `tools/`, where
  no shipped code imports it, and `src/diamond/game` is deleted.

## The behaviour queue was two decisions, not seven tasks

Reading the seven, none was independent work (`search_factory` and
`orchestration/benchmark` have since left):

| Module | Waits on |
|---|---|
| `game_adapter` | the corpus generator: it is what `tools/build_golden.py` drives |
| `search_factory` | the Python search existing at all |

So the rest of the queue drains on two decisions, both of which are product
calls rather than refactors:

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

   Nothing shipped is left on the Python search. `selfplay/runner_2p` and
   `runner_3p` take their search from the selector, with the `deadline` it
   already accepted; training self-play runs on the native pool, which batches,
   and the runners serve the single-episode callers that hold their own
   `Evaluator`.
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
rated benchmark matches on native   done (ProductionBenchmarkStage)
episode runners on the selector     done (runner_2p, runner_3p)
smoke fixtures on native legality   done (both smokes)
game_adapter on the native Game     done (behaviour queue empty)
freeze the golden corpus            then
retire the Python parity gates      one at a time, each with its C++ replacement
delete src/diamond/game             last
```

Nothing in that list is a rewrite of the training system. It is a change of
which implementation is authoritative, and the Python evidence goes only after
its replacement has proved the same things.
