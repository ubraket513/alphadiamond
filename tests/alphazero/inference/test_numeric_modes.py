from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.inference.profile import assert_evaluation_agreement


class _Model(nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        return torch.tensor([[0.0, 1.0, 2.0]], device=features.device).repeat(batch, 1), torch.zeros(
            (batch, 1), device=features.device
        )


def _request() -> EvalRequest:
    return EvalRequest(
        node_features=((0.0, 0.0),),
        legal_action_ids=(0, 1, 2),
        canonical_player_ids=(1, 2),
    )


def test_eager_fp32_remains_the_default_numeric_mode() -> None:
    evaluator = TorchEvaluator(_Model(), value_size=1)

    assert evaluator.precision == "fp32"


def test_bf16_requires_cuda_bf16_support() -> None:
    with pytest.raises(ValueError, match="CUDA BF16 support"):
        TorchEvaluator(_Model(), value_size=1, precision="bf16")


def test_numeric_agreement_checks_finite_legal_normalized_outputs() -> None:
    evaluator = TorchEvaluator(_Model(), value_size=1)
    reference = evaluator.evaluate((_request(),))
    candidate = evaluator.evaluate((_request(),))

    assert_evaluation_agreement(reference, candidate, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires CUDA BF16 support",
)
def test_bf16_uses_the_same_requests_and_agrees_with_eager_fp32() -> None:
    model = _Model()
    request = _request()
    eager = TorchEvaluator(model, value_size=1, device="cuda")
    bf16 = TorchEvaluator(model, value_size=1, device="cuda", precision="bf16")

    reference = eager.evaluate((request,))
    candidate = bf16.evaluate((request,))

    assert sum(candidate[0].priors.values()) == pytest.approx(1.0)
    assert_evaluation_agreement(reference, candidate, rtol=1e-2, atol=1e-3)
