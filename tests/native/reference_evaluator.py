"""The Gate B reference evaluator, in Python.

Deliberately *not* a constant.  A constant value makes every PUCT key tie, so
selection would be decided entirely by the action-id tie-break and a genuinely
divergent traversal could still produce matching visit counts.  This evaluator
is a pure function of the request, which is what
``docs/native_selfplay_phase0.md`` section 7 asks for.

It is specified in integer arithmetic (FNV-1a, then an exact 53-bit mantissa
division) so that this file and ``native/src/evaluator.cpp`` agree
**bit for bit**, not merely to a tolerance.  ``test_mcts_parity`` checks that
claim directly before it compares any search.
"""

from __future__ import annotations

from diamond.alphazero.evaluator.base import EvalRequest, EvalResult

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF

MANTISSA_SCALE = 9007199254740992.0
"""2^53.  Values below it are exact as doubles, so int -> float never rounds."""


def _mix(hash_value: int, byte: int) -> int:
    return ((hash_value ^ byte) * FNV_PRIME) & MASK64


def _mix_u32(hash_value: int, value: int) -> int:
    value &= 0xFFFFFFFF
    for shift in (0, 8, 16, 24):
        hash_value = _mix(hash_value, (value >> shift) & 0xFF)
    return hash_value


def _mix_u64(hash_value: int, value: int) -> int:
    for shift in range(0, 64, 8):
        hash_value = _mix(hash_value, (value >> shift) & 0xFF)
    return hash_value


def _unit(hash_value: int) -> float:
    """The top 53 bits of ``hash_value`` mapped into [0, 1) without rounding."""
    return (hash_value >> 11) / MANTISSA_SCALE


def request_hash(request: EvalRequest) -> int:
    """FNV-1a over the request's canonical byte serialization."""
    hash_value = FNV_OFFSET
    # Features are exactly 0.0 or 1.0 (Gate A proves it), so one byte each.
    for row in request.node_features:
        for feature in row:
            hash_value = _mix(hash_value, 1 if feature == 1.0 else 0)
    hash_value = _mix(hash_value, 0xFF)
    for action in request.legal_action_ids:
        hash_value = _mix_u32(hash_value, action)
    hash_value = _mix(hash_value, 0xFE)
    for player_id in request.canonical_player_ids:
        hash_value = _mix(hash_value, player_id)
    return hash_value


class ReferenceEvaluator:
    """A deterministic, request-dependent evaluator; records what it was asked.

    ``trace`` is the evaluator request sequence Gate B compares.  Matching root
    visit counts alone would not prove the two backends walked the same tree.

    ``uniform`` ties every prior while keeping the value request-dependent.
    That is the only way to exercise PUCT's ``(-score, action)`` tie-break:
    with distinct priors no two selection keys are ever exactly equal, so a
    mutation reversing the tie-break direction passes unnoticed.  Production is
    not so lucky -- the vacancy prior gives equal probability to every action
    sharing an integer progress score.
    """

    def __init__(self, *, uniform: bool = False) -> None:
        self.uniform = uniform
        self.trace: list[tuple[int, tuple[int, ...]]] = []
        self.calls = 0

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        results = []
        for request in requests:
            hash_value = request_hash(request)
            self.calls += 1
            self.trace.append((hash_value, tuple(request.legal_action_ids)))

            # Sequential left-to-right accumulation, matching C++'s loop.
            if self.uniform:
                prior = 1.0 / len(request.legal_action_ids)
                priors = {action: prior for action in request.legal_action_ids}
            else:
                weights: list[float] = []
                total = 0.0
                for action in request.legal_action_ids:
                    action_hash = _mix_u32(_mix_u64(FNV_OFFSET, hash_value), action)
                    # Shifted into [0.5, 1.5): strictly positive, and spread
                    # widely enough to order the PUCT keys.
                    weight = _unit(action_hash) + 0.5
                    weights.append(weight)
                    total += weight
                priors = {
                    action: weight / total
                    for action, weight in zip(request.legal_action_ids, weights)
                }

            results.append(
                EvalResult(priors=priors, value=_unit(hash_value) * 2.0 - 1.0)
            )
        return tuple(results)


__all__ = ["ReferenceEvaluator", "request_hash"]
