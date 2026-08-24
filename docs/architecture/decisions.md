# Decisions

Short, dated, and binding. Each one changes what the project promises, so it is
recorded here rather than inferred from the code.

---

## 1. Training game execution requires the native extension

*2026-08-24.*

The pure-Python search and self-play backend are no longer supported fallbacks.
Python remains the training and control-plane language.

**What is required is the compiled extension, not a compiler.** A source
checkout builds it today; a wheel or prebuilt artifact would let a user train
without a toolchain. The contract is "native extension present", and the
documentation says that rather than "compiler required".

**Why.** Keeping a second implementation of rules, search and self-play alive
for the case where the extension is missing costs more than it returns. Both
model families already train on the native pool; every Python parity gate has
been replaced by a C++ gate proven on mutation evidence. What is left is not
resilience, it is a duplicate engine kept warm.

**What it removes.** The Python-search fallback in `search_factory`, and the `python` value of `selfplay_backend`. `selfplay/runner_2p` and
`runner_3p` were listed for deletion here and were not deleted: what ran the
Python engine in them was the hard-coded search class, and the episode loop
around it is control plane. They take the selector instead -- see
`retiring_the_python_engine.md`. The
Python-versus-native comparison in `orchestration/benchmark` leaves the shipped
package for `az-bench/`, where research and historical benchmarks live. The two
smokes stop scripting games through the Python rules and exercise the native
path instead.

**What it does not remove.** The training loop, the optimizer and checkpoint
tooling, the experiment orchestration, the arena statistics and the rating
code. Those are the control plane and stay in Python.

---

## 2. The golden corpus is frozen as the normative game contract

*2026-08-24.*

`tests/golden/` is no longer a fixture regenerated from the Python engine on
every CI run. It is the normative artifact that defines the game contract, and
it carries its own provenance: format version, game-contract version, corpus
hash, the oracle commit that produced it, and a payload hash per file.

CI verifies that provenance. It does not regenerate.

**Why.** Regenerating from Python each run means declaring C++ the authority
while the Python implementation still holds the back door to the specification.
The evidence to stop is in place: five parity gates were retired against
thirteen mutations, and the C++ side has independent golden tests.

**What it removes.** `tests/test_golden_is_current.py`, and with it the reason
for `core` to run the Python oracle on every commit.

**Changing the contract later.** A deliberate rules change does not mean
"regenerate the golden files from whatever Python does now". It means raising
the game-contract version and creating a new, versioned golden corpus on
purpose. The implementation that produced the current one is preserved in Git
history; it does not have to stay in the shipped source to remain recoverable.

---

---

## 3. The Python engine and its oracle are deleted

*2026-08-24.*

`src/diamond/game`, `src/diamond/alphazero/mcts`, the Python board
(`contract/board.py`, `contract/coordinates.py`), `tools/build_golden.py` and
`tools/build_native_corpus.py` are gone. So is `src/diamond/agents`, the Python
agent layer for the PySide GUI that was deleted upstream -- the shipped
application's agent is `native/qt/ai_worker.cpp`.

**Why now.** Decisions 1 and 2 left the engine with two jobs: oracle and bridge
half. Both ended. The corpus is frozen and normative, so the oracle has nothing
to generate; the bridge gates that compared two rule implementations were
retired against the evidence they existed to produce.

**What it costs, stated plainly.** The corpus cannot be regenerated from the
source tree, and there is no second implementation to check the first against.
That is the point of freezing a contract, but it is a real loss and it is
one-way: restoring the oracle means restoring it from Git history, deliberately,
as part of a `game_contract_version` change.

**What replaces the coverage.** CTest, with no interpreter: `rules_golden_test`,
`mcts_golden_test`, `mcts3p_golden_test`, `mcts_stochastic_test`,
`topology_test`, `budget_test`, `batcher_test`, `selfplay_test`,
`action_codec_test`. The Python tests that remain test Python.

---

## Phases these two open

Both phases are complete. Recorded as planned, with what actually happened:

```text
Phase A -- eliminate duplicated behaviour                        done
    remove the no-extension fallback                             done
    freeze the golden corpus                                     done
    behaviour ledger 7 -> 0                                      done
    delete the Python MCTS and self-play implementations         done, except
        the episode runners: what ran the engine in them was the hard-coded
        search class, not the loop. They take the selector instead.

Phase B -- eliminate the game-definitions dependency             done
    standard_board / build_players -> a neutral source           done, and the
        source turned out to be the core itself: it generates the board now
        (native/src/topology_gen.cpp) instead of being handed it
    GameState -> a trainer-owned representation                  done, by
        `git mv` into diamond.contract -- the same classes, a package nothing
        plans to delete
    definitions ledger 14 -> 0                                   done
    delete src/diamond/game                                      done, with the
        oracle and the Python board (decision 3)
```

Keeping the two ledgers apart is what made the work legible: "30 dependents"
could not distinguish a module that ran the rules from one that named a
dataclass in a type hint, and the two retired at completely different times.

See [`retiring_the_python_engine.md`](retiring_the_python_engine.md) for the
order it happened in and the evidence collected before each deletion.
