"""Where does the replay path actually spend its time?

Milestone 5 of the native migration says to port only what is measured to be
expensive. Replay ingestion and batch collation are the usual suspects -- they
touch every sample of every game -- so this measures them before anyone ports
them. The number that decides it is not "is this fast" but "what share of a
training step is this", because a step also runs a forward and a backward pass.

Usage::

    python az-bench/profiles/bench_replay_pipeline.py [--samples 20000] [--profile]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.replay import ReplayBuffer, TrainingSample

BOARD = 73
ACTION_SIZE = 5329
VISITED_ACTIONS = 24


def _sample(rng: random.Random, compatibility: CheckpointCompatibilitySpec) -> TrainingSample:
    player_count = compatibility.identity.player_count
    actions = sorted(rng.sample(range(ACTION_SIZE), VISITED_ACTIONS))
    weights = [rng.random() + 1e-6 for _ in actions]
    total = sum(weights)
    return TrainingSample(
        compatibility=compatibility,
        node_features=tuple(
            tuple(float(rng.getrandbits(1)) for _ in range(player_count * 2))
            for _ in range(BOARD)
        ),
        canonical_player_ids=tuple(range(1, player_count + 1)),
        sparse_policy=tuple(
            (action, weight / total) for action, weight in zip(actions, weights)
        ),
        # Soo value semantics: the outcome, not an estimate.
        value_target=(1.0 if rng.getrandbits(1) else -1.0,),
    )


def _time(label: str, function, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    print(f"{label}: {best * 1000:.1f} ms")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--profile", action="store_true", help="print a cProfile breakdown")
    arguments = parser.parse_args()

    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig()
    )
    rng = random.Random(20260824)
    print(f"building {arguments.samples} samples")
    samples = [_sample(rng, compatibility) for _ in range(arguments.samples)]

    buffer = ReplayBuffer(compatibility, capacity=arguments.samples)
    ingest = _time("ingest (extend)", lambda: buffer.extend(samples), repeats=1)
    draw = _time("draw a batch", lambda: buffer.sample(arguments.batch))
    batch = buffer.sample(arguments.batch)
    collate = _time(
        "collate a batch", lambda: buffer.collate(batch, action_size=ACTION_SIZE)
    )

    print(f"\ningest: {ingest / max(len(samples), 1) * 1e6:.2f} us/sample")
    print(f"batch pipeline (draw + collate): {(draw + collate) * 1000:.1f} ms per step")

    if arguments.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(20):
            buffer.collate(buffer.sample(arguments.batch), action_size=ACTION_SIZE)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(10)
        print(stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
