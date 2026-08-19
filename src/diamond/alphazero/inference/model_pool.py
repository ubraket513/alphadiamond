"""Compatibility-checked resident eager evaluators keyed by checkpoint artifact."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from ..checkpoint import load_inference_checkpoint
from ..evaluator.torch import TorchEvaluator
from ..identity import MIN_MODEL_NAME, SOO_MODEL_NAME, CheckpointCompatibilitySpec
from ..network import MinModel, SooModel
from .protocol import InferenceRequest, InferenceResponse, ModelKey


class InferenceModelPool:
    """Own eager evaluators (FP32 by default); never share forwards across keys."""

    def __init__(
        self, *, device: str = "cpu", precision: Literal["fp32", "bf16"] = "fp32"
    ) -> None:
        self.device = device
        self.precision = precision
        self._evaluators: dict[ModelKey, TorchEvaluator] = {}
        self._compatibility: dict[ModelKey, CheckpointCompatibilitySpec] = {}

    @property
    def model_keys(self) -> tuple[ModelKey, ...]:
        return tuple(self._evaluators)

    def activate_checkpoint(
        self, path: str | Path, *, expected: CheckpointCompatibilitySpec
    ) -> ModelKey:
        """Load one exact artifact after semantic validation and return its key."""
        if expected.identity.model_name == SOO_MODEL_NAME:
            model = SooModel(
                expected.network_config, model_version=expected.identity.model_version
            )
            value_size = 1
        elif expected.identity.model_name == MIN_MODEL_NAME:
            model = MinModel(
                expected.network_config, model_version=expected.identity.model_version
            )
            value_size = 3
        else:  # CheckpointCompatibilitySpec prevents this; retain a clear boundary error.
            raise ValueError(f"unsupported inference model {expected.identity.model_name!r}")

        info = load_inference_checkpoint(path, model, expected=expected, device=self.device)
        key = ModelKey(
            model_name=expected.identity.model_name,
            model_version=expected.identity.model_version,
            checkpoint_sha256=info.checkpoint_sha256,
        )
        if key not in self._evaluators:
            self._evaluators[key] = TorchEvaluator(
                model, value_size=value_size, device=self.device, precision=self.precision
            )
            self._compatibility[key] = expected
        return key

    def evaluate(self, requests: tuple[InferenceRequest, ...]) -> tuple[InferenceResponse, ...]:
        """Evaluate requests in per-artifact batches while preserving input order."""
        if not isinstance(requests, tuple):
            raise ValueError("inference requests must be a tuple")
        if not requests:
            return ()
        grouped: dict[ModelKey, list[tuple[int, InferenceRequest]]] = defaultdict(list)
        correlations: set[tuple[str, str]] = set()
        for index, request in enumerate(requests):
            if not isinstance(request, InferenceRequest):
                raise ValueError("inference requests must contain InferenceRequest values")
            if request.correlation_id in correlations:
                raise ValueError("duplicate inference request correlation ID")
            correlations.add(request.correlation_id)
            if request.model_key not in self._evaluators:
                raise KeyError(f"unknown inference model key: {request.model_key}")
            compatibility = self._compatibility[request.model_key]
            if len(request.node_features[0]) != compatibility.identity.player_count * 2:
                raise ValueError("request feature width does not match the activated model")
            grouped[request.model_key].append((index, request))

        responses: list[InferenceResponse | None] = [None] * len(requests)
        for key, group in grouped.items():
            results = self._evaluators[key].evaluate(
                tuple(request.to_eval_request() for _, request in group)
            )
            for (index, request), result in zip(group, results, strict=True):
                responses[index] = InferenceResponse.from_eval_result(request, result)
        return tuple(response for response in responses if response is not None)


__all__ = ["InferenceModelPool"]
