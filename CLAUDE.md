# alphadiamond

3-player Diamond (Chinese-checkers-family) engine, Qt console, and an AlphaZero
training stack. The two-player configuration is called **Soo**.

## Working on the native self-play backend

Active project. **Read [docs/native_selfplay_handoff.md](docs/native_selfplay_handoff.md) first** —
it covers gate status, environment setup, how to run the benchmarks, the
invariants that must not break, and the pitfalls already paid for.

Two rules that are easy to get wrong and expensive to rediscover:

- **The Python implementation is the oracle and must not be deleted.** The C++
  in `native/` duplicates it on purpose; every gate is defined as equality
  against the Python side and CI re-runs that comparison on every change.
- **Do not micro-optimise before measuring.** The plan is explicit that a
  simple native representation and exact parity come first.

## Layout

| path | what |
|---|---|
| `src/diamond/game/` | authoritative rules, board, state — the oracle |
| `src/diamond/alphazero/` | encoder, MCTS, evaluators, orchestration |
| `src/diamond/alphazero/native/` | optional native backend: import guard, topology export, callback |
| `native/` | the C++ extension (`_diamond_native`) |
| `tests/native/` | parity gates A–D |
| `az-bench/profiles/` | benchmarks, out of production code |
| `docs/` | design records and measured findings |

## Build and test

```bash
python tools/build_native.py     # optional extension; absence must stay harmless
pytest -m "not gui"              # engine + AlphaZero + native gates
pytest -m gui                    # needs Qt and QT_QPA_PLATFORM=offscreen
```
