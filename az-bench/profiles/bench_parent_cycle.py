"""Decompose the parent's serialized per-evaluation cost, stage by stage.

The worker-scaling sweep established that the parent is pegged at ~100 % of one
core while the GPU idles at ~5 %, and that the marginal cost converges to roughly
1.5 ms per evaluation.  That figure is a *critical-path* cost, not a proven
Python-computation cost, so this benchmark takes the same code the parent runs
and times each stage of it in isolation on real request data.

Deliberately isolated rather than instrumented in place:

  * it perturbs nothing -- the production path is not touched at all;
  * each stage can be repeated enough times to be statistically stable;
  * it separates *inherent* stage cost from *contention* cost.  Whatever this
    reports is a floor: the real parent runs these stages while 48 queue-feeder
    threads contend for the same GIL, so real costs are equal or higher.

CUDA timing: the forward is measured both without synchronisation (CPU launch
cost, which is what the parent's thread actually pays and cannot overlap) and
with an explicit synchronise (true device execution time).  The production path
does not synchronise after the forward, so the sync is confined to this
benchmark and reported as a separate number rather than folded into the cycle.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.game_adapter import AlphaZeroGameAdapter, DiamondSearchAdapter
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.config import NetworkConfig
from diamond.alphazero.inference.coordinator import (
    InferenceConfig,
    InferenceCoordinator,
    InferenceMetrics,
    _QueuedRequest,
)
from diamond.alphazero.inference.model_pool import InferenceModelPool
from diamond.alphazero.inference.protocol import (
    InferenceRequest,
    InferenceResponse,
    ModelKey,
)
from diamond.game.state import build_players, initial_state


def build_requests(count: int, *, model_key: ModelKey, seed: int = 7) -> list[InferenceRequest]:
    """Real encoder output from real game states, not synthetic filler.

    Legal-action counts vary strongly with position (18-75 observed), and several
    stages are O(legal actions), so synthetic fixed-width requests would bias the
    result.
    """
    players = build_players(2)
    state = initial_state(players)
    game = DiamondSearchAdapter(AlphaZeroGameAdapter(players, initial=state))
    rng = random.Random(seed)
    requests: list[InferenceRequest] = []
    index = 0
    while len(requests) < count:
        if game.is_terminal(state):
            state = initial_state(players)
        eval_request = game.evaluation_request(state)
        requests.append(
            InferenceRequest.from_eval_request(
                client_id="bench-client",
                request_id=f"bench:{index}",
                model_key=model_key,
                request=eval_request,
            )
        )
        index += 1
        actions = game.legal_action_ids(state)
        if not actions:
            state = initial_state(players)
            continue
        state = game.apply_action(state, rng.choice(actions))
    return requests


class Stage:
    """One timed stage, reported per batch and amortized per evaluation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[float] = []

    def record(self, seconds: float) -> None:
        self.samples.append(seconds)

    def summary(self, *, batches: int, evaluations: int) -> dict[str, object]:
        total = sum(self.samples)
        return {
            "stage": self.name,
            "total_s": total,
            "per_batch_ms": total / batches * 1000.0 if batches else None,
            "per_eval_ms": total / evaluations * 1000.0 if evaluations else None,
            "p50_batch_ms": statistics.median(self.samples) * 1000.0 if self.samples else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch", type=int, default=12, help="batch size; 12 ~ the measured mean at 48-64 workers."
    )
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} device {args.device}")
    print(f"[env] diamond {Path(__import__('diamond').__file__).resolve()}")

    compatibility = CheckpointCompatibilitySpec.soo(
        model_version="2.0.0", network_config=NetworkConfig(residual_blocks=6, width=128)
    )
    pool = InferenceModelPool(device=args.device)
    model_key = pool.activate_checkpoint(args.checkpoint, expected=compatibility)
    evaluator = pool.evaluator(model_key)
    print(f"[env] model_key {model_key.checkpoint_sha256[:16]}…")

    total_requests = args.batch * (args.batches + args.warmup)
    requests = build_requests(total_requests, model_key=model_key)
    widths = [len(r.legal_action_ids) for r in requests]
    print(
        f"[data] {len(requests)} real requests, "
        f"features {len(requests[0].node_features)}x{len(requests[0].node_features[0])}, "
        f"legal actions min/mean/max {min(widths)}/{sum(widths) / len(widths):.1f}/{max(widths)}"
    )

    stages = {
        name: Stage(name)
        for name in (
            "1_unpickle_request",
            "2_submit_bookkeeping",
            "3_validated_request",
            "4_pool_grouping",
            "5_tensor_construction",
            "6_forward_launch",
            "7_policy_gather_softmax",
            "8_validity_sync",
            "9_d2h_tolist",
            "10_priors_dict_build",
            "11_response_envelope",
            "12_pickle_response",
            "13_metrics_record",
        )
    }
    forward_device_ms: list[float] = []

    config = InferenceConfig(max_batch_size=32, max_wait_ms=2, request_queue_capacity=1024)
    coordinator = InferenceCoordinator(pool, config)
    device = torch.device(args.device)

    cursor = 0
    for iteration in range(args.batches + args.warmup):
        batch = requests[cursor : cursor + args.batch]
        cursor += args.batch
        measuring = iteration >= args.warmup

        def timed(name: str, function):
            """Run ``function``; record its duration only past warm-up."""
            start = perf_counter()
            result = function()
            elapsed = perf_counter() - start
            if measuring:
                stages[name].record(elapsed)
            return result

        # -- 1. transport: the parent's bridge thread unpickles every request.
        wire = [pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL) for request in batch]
        decoded = timed("1_unpickle_request", lambda: [pickle.loads(blob) for blob in wire])

        # -- 2. submit(): identity extraction + queue item construction.
        def submit_bookkeeping():
            items = []
            for request in decoded:
                client_id, request_id, key = coordinator._extract_identity(request)
                items.append(
                    _QueuedRequest(
                        request=request,
                        reply_queue=None,  # type: ignore[arg-type]
                        submitted_at=perf_counter(),
                        admitted_at=perf_counter(),
                        admitted=True,
                        client_id=client_id,
                        request_id=request_id,
                        model_key=key,
                    )
                )
            return items

        items = timed("2_submit_bookkeeping", submit_bookkeeping)

        # -- 3. the coordinator rebuilds each already-valid request, which
        #       re-runs InferenceRequest.__post_init__ over every feature float.
        validated = timed(
            "3_validated_request",
            lambda: [coordinator._validated_request(item) for item in items],
        )
        group = tuple(validated)

        # -- 4. InferenceModelPool.evaluate's grouping/validation preamble.
        def pool_grouping():
            correlations = set()
            converted = []
            for request in group:
                correlations.add(request.correlation_id)
                converted.append(request.to_eval_request())
            return tuple(converted)

        eval_requests: tuple[EvalRequest, ...] = timed("4_pool_grouping", pool_grouping)

        # -- 5..9 reproduce TorchEvaluator.evaluate stage by stage.
        features = timed(
            "5_tensor_construction",
            lambda: torch.tensor(
                [r.node_features for r in eval_requests], dtype=torch.float32, device=device
            ),
        )

        def forward_launch():
            with torch.inference_mode():
                return evaluator.model(features)

        if measuring:
            torch.cuda.synchronize(device)
            device_start = torch.cuda.Event(enable_timing=True)
            device_end = torch.cuda.Event(enable_timing=True)
            device_start.record()
        policy_logits, values = timed("6_forward_launch", forward_launch)
        if measuring:
            device_end.record()
            torch.cuda.synchronize(device)
            forward_device_ms.append(device_start.elapsed_time(device_end))

        def gather_softmax():
            counts = [len(r.legal_action_ids) for r in eval_requests]
            widest = max(counts)
            index = torch.tensor(
                [list(r.legal_action_ids) + [0] * (widest - c)
                 for r, c in zip(eval_requests, counts, strict=True)],
                dtype=torch.long,
                device=device,
            )
            legal_logits = policy_logits.gather(1, index)
            if widest > min(counts):
                lengths = torch.tensor(counts, dtype=torch.long, device=device).unsqueeze(1)
                positions = torch.arange(widest, device=device).unsqueeze(0)
                legal_logits = legal_logits.masked_fill(positions >= lengths, float("-inf"))
            return counts, torch.softmax(legal_logits, dim=1)

        counts, probabilities = timed("7_policy_gather_softmax", gather_softmax)

        # The first device read in the real path; this is where the pipeline
        # actually stalls on the GPU, so it absorbs any queued device work.
        timed(
            "8_validity_sync",
            lambda: bool(
                torch.isfinite(probabilities).all() & (probabilities.sum(dim=1) > 0).all()
            ),
        )

        rows = timed(
            "9_d2h_tolist",
            lambda: (probabilities.cpu().tolist(), values.cpu().tolist()),
        )
        probability_rows, value_rows = rows

        def priors_build():
            results = []
            for request, count, row, raw in zip(
                eval_requests, counts, probability_rows, value_rows, strict=True
            ):
                priors = {
                    action: float(probability)
                    for action, probability in zip(
                        request.legal_action_ids, row[:count], strict=True
                    )
                }
                results.append((priors, float(raw[0])))
            return results

        built = timed("10_priors_dict_build", priors_build)

        # -- 11. InferenceResponse.__post_init__ re-validates every prior pair.
        from diamond.alphazero.evaluator.base import EvalResult

        def response_envelope():
            return [
                InferenceResponse.from_eval_result(
                    request, EvalResult(priors=priors, value=value)
                )
                for request, (priors, value) in zip(group, built, strict=True)
            ]

        responses = timed("11_response_envelope", response_envelope)

        # -- 12. the per-worker mp.Queue feeder thread pickles each response.
        timed(
            "12_pickle_response",
            lambda: [
                pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                for response in responses
            ],
        )

        # -- 13. coordinator metrics bookkeeping for the batch.
        metrics = InferenceMetrics()
        rng = random.Random(0)
        timed(
            "13_metrics_record",
            lambda: metrics.record_batch(
                batch_size=len(group),
                queue_to_dispatch_s=[0.001] * len(group),
                inference_duration_s=0.005,
                response_latencies_s=[0.01] * len(group),
                admission_latencies_s=[0.0001] * len(group),
                rng=rng,
            ),
        )

    evaluations = args.batches * args.batch
    rows = [stage.summary(batches=args.batches, evaluations=evaluations) for stage in stages.values()]
    cycle_ms = sum(row["per_eval_ms"] for row in rows)

    print()
    print(f"batch size {args.batch}, {args.batches} measured batches, {evaluations} evaluations")
    print(f"{'stage':<26} {'per_batch_ms':>12} {'per_eval_ms':>12} {'share':>8}")
    print("-" * 62)
    for row in sorted(rows, key=lambda r: -r["per_eval_ms"]):
        share = row["per_eval_ms"] / cycle_ms * 100.0
        print(
            f"{row['stage']:<26} {row['per_batch_ms']:>12.3f} "
            f"{row['per_eval_ms']:>12.4f} {share:>7.1f}%"
        )
    print("-" * 62)
    print(f"{'TOTAL serialized':<26} {cycle_ms * args.batch:>12.3f} {cycle_ms:>12.4f} {100.0:>7.1f}%")
    print()
    device_mean = sum(forward_device_ms) / len(forward_device_ms) if forward_device_ms else 0.0
    print(
        f"forward: CPU launch {stages['6_forward_launch'].summary(batches=args.batches, evaluations=evaluations)['per_batch_ms']:.3f} ms/batch, "
        f"GPU execution {device_mean:.3f} ms/batch (cuda events)"
    )
    print(f"implied ceiling from this cycle: {1000.0 / cycle_ms:.0f} evals/s single-threaded")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "batch": args.batch,
                    "batches": args.batches,
                    "evaluations": evaluations,
                    "cycle_per_eval_ms": cycle_ms,
                    "forward_device_ms_mean": device_mean,
                    "stages": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
