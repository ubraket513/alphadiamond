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

## Phases these two open

```text
Phase A -- eliminate duplicated behaviour
    remove the no-extension fallback
    freeze the golden corpus
    behaviour ledger 7 -> 0
    delete the Python MCTS and self-play implementations

Phase B -- eliminate the game-definitions dependency
    standard_board / build_players -> a neutral topology and contract source
    GameState -> the native State, or a trainer-owned representation
    definitions ledger 14 -> 0
    delete src/diamond/game
```

`behaviour -> 0` removes the *behavioural* blocker on deleting
`src/diamond/game`. It does not by itself make the directory deletable: the
fourteen definition dependents still import `standard_board`, `build_players`
and `GameState`. That is Phase B, and keeping the two apart is what makes the
ledger mean something.

See [`retiring_the_python_engine.md`](retiring_the_python_engine.md) for the
running state.
