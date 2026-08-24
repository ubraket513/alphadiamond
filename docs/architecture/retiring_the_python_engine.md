# Retiring the Python engine

**Done, 2026-08-24.** `src/diamond/game`, `src/diamond/alphazero/mcts` and the
Python board are deleted. The C++ core is the only implementation of Diamond's
rules, encoding, search, self-play and geometry.

This file is the record of how, kept because the *reasons* outlive the code and
because someone will eventually ask why the corpus cannot be regenerated.

## What was deleted, and what replaced it

| Deleted | Replaced by |
|---|---|
| `game/rules.py`, `game/session.py`, `game/history.py` | `native/src/rules.cpp`, reached through `AlphaZeroGameAdapter` |
| `alphazero/mcts/` (the Python MCTS) | `SearchSession`, `SearchSession3P`, selected by `search_factory` |
| `contract/board.py`, `contract/coordinates.py` | `native/src/topology_gen.cpp`, read back through `native/topology.py` |
| `tools/build_golden.py` (the oracle) | nothing: `tests/golden/` is frozen and normative |
| `tools/build_native_corpus.py` | nothing: `tests/native/fixtures/positions.jsonl` is frozen |
| `src/diamond/agents/` | `native/qt/ai_worker.cpp` -- the shipped app's agent is C++ |

## The order it happened in, and why that order

Deleting the engine first would have broken every module that merely described
a position. So the work went outside-in:

1. **Nothing shipped runs the Python engine.** The benchmark, both self-play
   runners, the two smokes and finally `game_adapter` each stopped naming a
   Python implementation. `game_adapter` was the only real port; the rest were
   one-line swaps to `search_factory`.
2. **The definitions moved out of the engine.** `GameState`, `PlayerSpec`,
   `Move` and the seats went to `diamond.contract` by `git mv`, so the
   dependency pointed away from the package being deleted.
3. **The core generated its own geometry.** Until then the board was built in
   Python and handed to the extension at import.
4. **The trainer read geometry from the core.** Encoder rotation, prior
   distances, network adjacency, the opening.
5. **Then, and only then, the deletion.**

Each step ended with a ratchet line struck from `tests/test_engine_retirement.py`
-- a test that itself no longer exists, because both of its lists reached zero
and a ledger of nothing is not worth running.

## The evidence collected before deleting

Deletion removes the ability to re-check, so the checks were run first:

* **61,139 successors.** Every legal action of all 1,327 fixture positions,
  through the native adapter and the Python engine, compared on occupancy, seat
  to move, turn number, status and podium. Identical.
* **The frozen corpus reproduced byte for byte** from the Python oracle, at
  every step, up to and including the commit that deleted it.
* **Five parity gates retired on mutation evidence** (`tools/mutation_check.py`,
  thirteen mutations) before their Python halves went.
* **The geometry checked table by table** against the implementation it
  replaced -- distances, camps, the opening, and the network's adjacency buffer.
* **`topology_test` proven non-vacuous by mutation:** reordering two lattice
  directions fails it.

## What cannot be done any more, on purpose

**The golden corpus cannot be regenerated from the source tree.** That is what
freezing it means (decision 2), and it is why the oracle could be deleted at
all. A deliberate rules change does not mean "regenerate from whatever Python
does now" -- there is no Python to ask. It means restoring the oracle from Git
history, raising `game_contract_version`, and creating a new versioned corpus on
purpose.

`RULESET_FINGERPRINT` is frozen for the same kind of reason and a sharper one:
it hashed cube coordinates that no longer exist anywhere, and every checkpoint
in the bucket carries the value it was trained under.

## What was never going to be ported

Unchanged from the review that started this: the training loop and autograd,
optimizer and checkpoint surgery, experiment orchestration, rating and
statistics, and the Hugging Face synchronisation. Python is the better
control-plane language and none of it is on the shipped application's runtime
path.
