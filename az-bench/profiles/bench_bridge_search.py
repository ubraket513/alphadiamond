"""How much does one Python callback per node cost?

The native search suspends on every node and asks Python for that node's
answer. Against a network the crossing is noise next to a forward pass; against
a scripted evaluator it is the whole cost, which is the case this measures.

Warm up first: the first native search in a process pays the extension's
one-time setup, and an unwarmed 64-simulation run reports native as slower than
Python -- the opposite of the truth.

Findings: docs/performance-profiling/bridge_search_findings.md

Usage::

    python az-bench/profiles/bench_bridge_search.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\dzk55\alphadiamond")
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.config import MCTSConfig
from diamond.alphazero.evaluator.base import EvalResult
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.mcts.search_2p import MCTS2P
from diamond.alphazero.native.search import NativeSearch2P
from diamond.contract.state import build_players


class Cheap:
    """A scripted evaluator: the shape a self-play runner test uses."""
    def evaluate(self, requests):
        out = []
        for request in requests:
            n = len(request.legal_action_ids)
            out.append(EvalResult(priors={a: 1.0 / n for a in request.legal_action_ids}, value=0.25))
        return tuple(out)


def _time(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


adapter = DiamondSearchAdapter(AlphaZeroGameAdapter(build_players(2)))
state = adapter.initial_state()

# Warm up: the first native search pays for the extension's one-time setup.
warm = MCTSConfig(simulations=8, c_puct=1.5, dirichlet_epsilon=0.0, seed=0)
NativeSearch2P(adapter, Cheap(), warm).run(state, temperature=0.0)
MCTS2P(adapter, Cheap(), warm).run(state, temperature=0.0)

for simulations in (64, 400, 1500):
    config = MCTSConfig(simulations=simulations, c_puct=1.5, dirichlet_epsilon=0.0, seed=0)
    def run_python(config=config):
        return MCTS2P(adapter, Cheap(), config).run(state, temperature=0.0)

    def run_native(config=config):
        return NativeSearch2P(adapter, Cheap(), config).run(state, temperature=0.0)

    python_s = min(_time(run_python) for _ in range(3))
    native_s = min(_time(run_native) for _ in range(3))

    print(f"sims={simulations:4d}  python={python_s*1000:7.1f} ms  "
          f"native+callback={native_s*1000:7.1f} ms  ratio={python_s/native_s:.2f}x")
