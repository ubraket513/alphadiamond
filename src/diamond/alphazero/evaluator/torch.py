"""PyTorch eager evaluator with legal-only policy normalization."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Literal

import torch
from torch import nn

from .base import EvalRequest, EvalResult


class TorchEvaluator:
    def __init__(
        self,
        model: nn.Module,
        *,
        value_size: int,
        device: str = "cpu",
        precision: Literal["fp32", "bf16"] = "fp32",
    ) -> None:
        if value_size not in (1, 3):
            raise ValueError("value_size must be 1 or 3")
        self.model = model.to(device)
        self.value_size = value_size
        self.device = torch.device(device)
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if precision == "bf16" and (
            self.device.type != "cuda" or not torch.cuda.is_bf16_supported(self.device)
        ):
            raise ValueError("bf16 evaluation requires CUDA BF16 support")
        self.precision = precision
        self.model.eval()

    def evaluate(self, requests: tuple[EvalRequest, ...]) -> tuple[EvalResult, ...]:
        if not requests:
            return ()
        for request in requests:
            if not request.legal_action_ids:
                raise ValueError("evaluation requires at least one legal action")
        feature_shapes = {
            (len(request.node_features), len(request.node_features[0]))
            for request in requests
        }
        if len(feature_shapes) != 1:
            raise ValueError("all evaluation requests must have the same feature shape")

        features = torch.tensor(
            [request.node_features for request in requests],
            dtype=torch.float32,
            device=self.device,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            policy_logits, values = self.model(features)

        if values.shape != (len(requests), self.value_size):
            raise ValueError(
                f"evaluator model value shape {tuple(values.shape)} does not match "
                f"({len(requests)}, {self.value_size})"
            )
        if policy_logits.ndim != 2:
            raise ValueError("evaluator model policy output must have shape [batch, actions]")

        results: list[EvalResult] = []
        for row, request in enumerate(requests):
            legal = torch.tensor(request.legal_action_ids, dtype=torch.long, device=self.device)
            if int(legal.min()) < 0 or int(legal.max()) >= policy_logits.shape[1]:
                raise ValueError("legal action is outside the model policy space")
            probabilities = torch.softmax(policy_logits[row].index_select(0, legal), dim=0)
            if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
                raise ValueError("model produced invalid legal policy probabilities")
            priors = {
                action: float(probability)
                for action, probability in zip(request.legal_action_ids, probabilities.cpu().tolist())
            }
            raw_value = values[row].cpu().tolist()
            value: float | tuple[float, ...]
            value = float(raw_value[0]) if self.value_size == 1 else tuple(map(float, raw_value))
            if not all(math.isfinite(component) for component in raw_value):
                raise ValueError("model produced a non-finite value")
            results.append(EvalResult(priors=priors, value=value))
        return tuple(results)


__all__ = ["TorchEvaluator"]
