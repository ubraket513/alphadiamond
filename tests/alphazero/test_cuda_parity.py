"""CPU and CUDA must agree before any GPU training result is believed.

These skip cleanly on a CPU-only host, which is where ordinary CI runs; they are
the gate that must pass on the RTX 3060 box before a GPU run means anything.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diamond.alphazero.checkpoint import load_inference_checkpoint, save_checkpoint
from diamond.alphazero.config import NetworkConfig, TrainingConfig
from diamond.alphazero.evaluator.base import EvalRequest
from diamond.alphazero.evaluator.torch import TorchEvaluator
from diamond.alphazero.identity import CheckpointCompatibilitySpec
from diamond.alphazero.inference.profile import assert_evaluation_agreement
from diamond.alphazero.network import SooModel
from diamond.alphazero.trainer import AlphaZeroTrainer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

# FP32 across two devices: the same arithmetic, but not necessarily the same
# kernel or reduction order, so agreement is tight rather than exact.
RTOL = 1e-4
ATOL = 1e-5

NETWORK = NetworkConfig(width=32, residual_blocks=2)


def _spec() -> CheckpointCompatibilitySpec:
    return CheckpointCompatibilitySpec.soo(model_version="0.4.0", network_config=NETWORK)


def _requests() -> tuple[EvalRequest, ...]:
    """A spread of legal-action counts, including the single-action edge case."""
    legal_sets = (
        (7,),
        (7, 19),
        (0, 1, 2, 3),
        (7, 19, 100, 512, 1024),
        tuple(range(0, 40)),
        tuple(range(100, 130)),
        (5328,),
        (0, 5328),
    )
    return tuple(
        EvalRequest(
            node_features=tuple(
                tuple(float((index + row) % 3) for _ in range(4)) for row in range(73)
            ),
            legal_action_ids=legal,
            canonical_player_ids=(1, 2),
        )
        for index, legal in enumerate(legal_sets)
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """One immutable artifact, loaded separately onto each device."""
    torch.manual_seed(1234)
    spec = _spec()
    trainer = AlphaZeroTrainer(
        SooModel(NETWORK, model_version="0.4.0"),
        spec,
        TrainingConfig(batch_size=1, learning_rate=1e-3, weight_decay=1e-4, seed=11),
    )
    path = tmp_path_factory.mktemp("cuda-parity") / "latest.pt"
    save_checkpoint(path, trainer, operation_id="cuda-parity")
    return path


def _evaluator(path, device: str) -> TorchEvaluator:
    model = SooModel(NETWORK, model_version="0.4.0")
    load_inference_checkpoint(path, model, expected=_spec(), device=device)
    return TorchEvaluator(model, value_size=1, device=device)


def test_cpu_and_cuda_agree_on_priors_and_values(checkpoint) -> None:
    requests = _requests()
    cpu = _evaluator(checkpoint, "cpu").evaluate(requests)
    cuda = _evaluator(checkpoint, "cuda:0").evaluate(requests)

    for expected, actual in zip(cpu, cuda, strict=True):
        # Legal-action identity is exact: a tolerance here would hide a real bug.
        assert set(expected.priors) == set(actual.priors)
    assert_evaluation_agreement(cpu, cuda, rtol=RTOL, atol=ATOL)


def test_cuda_priors_stay_normalized_over_legal_actions(checkpoint) -> None:
    for result in _evaluator(checkpoint, "cuda:0").evaluate(_requests()):
        assert sum(result.priors.values()) == pytest.approx(1.0, abs=1e-5)
        assert all(probability > 0 for probability in result.priors.values())


def test_batched_matches_single_request_evaluation(checkpoint) -> None:
    """Central batching depends on this; a batching bug would surface here."""
    evaluator = _evaluator(checkpoint, "cuda:0")
    requests = _requests()

    batched = evaluator.evaluate(requests)
    one_at_a_time = tuple(evaluator.evaluate((request,))[0] for request in requests)

    assert_evaluation_agreement(batched, one_at_a_time, rtol=RTOL, atol=ATOL)


def test_a_cuda_resident_model_saves_a_cpu_loadable_checkpoint(tmp_path) -> None:
    """Checkpoints must stay portable in both directions."""
    spec = _spec()
    trainer = AlphaZeroTrainer(
        SooModel(NETWORK, model_version="0.4.0"),
        spec,
        TrainingConfig(
            batch_size=1, learning_rate=1e-3, weight_decay=1e-4, device="cuda:0", seed=11
        ),
    )
    path = save_checkpoint(tmp_path / "cuda.pt", trainer, operation_id="cuda-save")

    cuda_results = TorchEvaluator(
        trainer.model, value_size=1, device="cuda:0"
    ).evaluate(_requests())
    cpu_model = SooModel(NETWORK, model_version="0.4.0")
    load_inference_checkpoint(path, cpu_model, expected=spec, device="cpu")
    cpu_results = TorchEvaluator(cpu_model, value_size=1, device="cpu").evaluate(
        _requests()
    )

    assert_evaluation_agreement(cuda_results, cpu_results, rtol=RTOL, atol=ATOL)
