"""Decompose the worker-side cost of one Soo MCTS evaluation.

The scaling work established that self-play spends ~5.3 ms of *worker* CPU per
neural evaluation against ~1.4 ms in the serialized parent -- 79 % of all CPU the
system burns per evaluation -- and that this has never been measured. It sets
the ceiling once the parent's serialization is removed, and it is the thing a
native port would actually replace, so the port needs to know what it is made of.

Method: run real Soo searches over real game states with a deterministic
in-process evaluator, so no GPU, no IPC and no inference latency enter the
measurement. Every ``DiamondSearchAdapter`` call the search makes is wrapped in a
counter/timer, which splits the cost into game-side components; whatever the
wrappers do not account for is MCTS's own bookkeeping (selection, backup, tree
allocation).

Two passes are reported because they answer different questions:

  * wall-clock attribution via wrappers -- absolute ms, low distortion, but only
    resolves the game/MCTS boundary;
  * ``cProfile`` -- function-level detail inside that boundary, at the cost of
    inflating absolute numbers, so its percentages are used and its totals are not.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalRequest, EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.contract.state import build_players, initial_state


class DeterministicEvaluator:
    """Instant, reproducible priors/value derived from the request itself.

    Uniform priors would be a degenerate search (every edge tied, so selection
    always falls to the action-id tie-break); these vary with the position while
    staying a pure function of it, so the search shape is realistic and the run
    is repeatable.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        self.calls += 1
        self.rows += len(requests)
        results = []
        for request in requests:
            actions = request.legal_action_ids
            weights = [((action * 2654435761) % 1000 + 1) / 1000.0 for action in actions]
            total = sum(weights)
            results.append(
                EvalResult(
                    priors={a: w / total for a, w in zip(actions, weights, strict=True)},
                    value=((hash(actions) % 2000) - 1000) / 1000.0,
                )
            )
        return tuple(results)


@dataclass
class Timings:
    seconds: Counter = field(default_factory=Counter)
    calls: Counter = field(default_factory=Counter)

    def record(self, name: str, elapsed: float) -> None:
        self.seconds[name] += elapsed
        self.calls[name] += 1


class InstrumentedAdapter:
    """Wrap the real adapter, timing every call the search makes into it.

    Delegates rather than subclasses so the wrapped object is the genuine
    production adapter and no method can be accidentally bypassed.
    """

    def __init__(self, inner: DiamondSearchAdapter, timings: Timings) -> None:
        self._inner = inner
        self._timings = timings

    def _timed(self, name: str, function, *args):
        start = perf_counter()
        try:
            return function(*args)
        finally:
            self._timings.record(name, perf_counter() - start)

    def current_player_id(self, state):
        return self._timed("current_player_id", self._inner.current_player_id, state)

    def legal_action_ids(self, state):
        return self._timed("legal_action_ids", self._inner.legal_action_ids, state)

    def apply_action(self, state, action_id):
        return self._timed("apply_action", self._inner.apply_action, state, action_id)

    def is_terminal(self, state):
        return self._timed("is_terminal", self._inner.is_terminal, state)

    def terminal_scalar_value(self, state, player_id):
        return self._timed(
            "terminal_scalar_value", self._inner.terminal_scalar_value, state, player_id
        )

    def evaluation_request(self, state):
        return self._timed("evaluation_request", self._inner.evaluation_request, state)


def build_search(simulations: int, timings: Timings | None, evaluator):
    players = build_players(2)
    state = initial_state(players)
    inner = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state))
    game = InstrumentedAdapter(inner, timings) if timings is not None else inner
    config = MCTSConfig(
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        seed=7,
        simulations=simulations,
    )
    return inner, game, MCTS2P(game, evaluator, config), state


def play(search, inner, state, *, moves: int, rng: random.Random):
    """Advance a real game, searching at each ply, as self-play does."""
    played = 0
    for _ in range(moves):
        if inner.is_terminal(state):
            break
        search.run(state, temperature=1.0)
        actions = inner.legal_action_ids(state)
        if not actions:
            break
        state = inner.apply_action(state, rng.choice(actions))
        played += 1
    return state, played


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--moves", type=int, default=30)
    parser.add_argument("--warmup-moves", type=int, default=3)
    args = parser.parse_args()

    # ---- pass 1: wall-clock attribution -----------------------------------
    timings = Timings()
    evaluator = DeterministicEvaluator()
    inner, _game, search, state = build_search(args.simulations, timings, evaluator)

    rng = random.Random(11)
    state, _ = play(search, inner, state, moves=args.warmup_moves, rng=rng)
    timings.seconds.clear()
    timings.calls.clear()
    evaluator.calls = evaluator.rows = 0

    start = perf_counter()
    state, moves_played = play(search, inner, state, moves=args.moves, rng=rng)
    total_s = perf_counter() - start

    evaluations = evaluator.rows
    print(f"simulations={args.simulations}  moves={moves_played}  evaluations={evaluations}")
    print(f"total worker time {total_s * 1000:.1f} ms   "
          f"{total_s / evaluations * 1000:.3f} ms/eval   "
          f"{total_s / moves_played * 1000:.1f} ms/move")
    print()

    game_total = sum(timings.seconds.values())
    mcts_own = total_s - game_total
    print(f"{'component':<26}{'total_ms':>10}{'ms/eval':>10}{'share':>8}{'calls':>10}{'us/call':>9}")
    print("-" * 73)
    rows = [(name, seconds, timings.calls[name]) for name, seconds in timings.seconds.items()]
    rows.append(("MCTS own (select/backup)", mcts_own, 0))
    for name, seconds, calls in sorted(rows, key=lambda r: -r[1]):
        per_call = f"{seconds / calls * 1e6:>9.1f}" if calls else f"{'-':>9}"
        print(
            f"{name:<26}{seconds * 1000:>10.1f}{seconds / evaluations * 1000:>10.3f}"
            f"{seconds / total_s * 100:>7.1f}%{calls if calls else '-':>10}{per_call}"
        )
    print("-" * 73)
    print(f"{'TOTAL':<26}{total_s * 1000:>10.1f}{total_s / evaluations * 1000:>10.3f}{100.0:>7.1f}%")

    # ---- pass 2: function-level detail ------------------------------------
    print()
    print("--- cProfile (percentages only; absolute times are inflated) ---")
    evaluator2 = DeterministicEvaluator()
    inner2, _g2, search2, state2 = build_search(args.simulations, None, evaluator2)
    rng2 = random.Random(11)
    state2, _ = play(search2, inner2, state2, moves=args.warmup_moves, rng=rng2)

    profiler = cProfile.Profile()
    profiler.enable()
    play(search2, inner2, state2, moves=args.moves, rng=rng2)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(18)
    for line in stream.getvalue().splitlines():
        if line.strip():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
