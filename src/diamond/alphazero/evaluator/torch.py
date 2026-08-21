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

        # Everything below is batched deliberately. The per-row form this
        # replaced issued ~6 device synchronisations per request -- two bounds
        # reads, two validity reads and two host transfers -- so a batch of 30
        # serialised ~180 of them and spent 69% of the call outside the network
        # forward, which is flat from batch 1 to batch 32. See Finding 4 in
        # docs/gpu_profile_findings.md.
        action_space = policy_logits.shape[1]
        # The ids are already Python tuples, so bounds are checked on the host
        # rather than by reading two scalars back off the device per row.
        if (
            min(min(request.legal_action_ids) for request in requests) < 0
            or max(max(request.legal_action_ids) for request in requests) >= action_space
        ):
            raise ValueError("legal action is outside the model policy space")

        # One padded gather over the widest legal set, masked back to -inf so the
        # padding contributes exactly zero to each row's softmax.
        counts = [len(request.legal_action_ids) for request in requests]
        widest = max(counts)
        index = torch.tensor(
            [
                list(request.legal_action_ids) + [0] * (widest - count)
                for request, count in zip(requests, counts, strict=True)
            ],
            dtype=torch.long,
            device=self.device,
        )
        legal_logits = policy_logits.gather(1, index)
        if widest > min(counts):
            lengths = torch.tensor(counts, dtype=torch.long, device=self.device).unsqueeze(1)
            positions = torch.arange(widest, device=self.device).unsqueeze(0)
            legal_logits = legal_logits.masked_fill(positions >= lengths, float("-inf"))
        probabilities = torch.softmax(legal_logits, dim=1)

        # One scalar read for the whole batch instead of two per row.
        valid = torch.isfinite(probabilities).all() & (probabilities.sum(dim=1) > 0).all()
        if not bool(valid):
            raise ValueError("model produced invalid legal policy probabilities")

        # One host transfer each; the remaining work is pure Python.
        probability_rows = probabilities.cpu().tolist()
        value_rows = values.cpu().tolist()

        results: list[EvalResult] = []
        for request, count, row_probabilities, raw_value in zip(
            requests, counts, probability_rows, value_rows, strict=True
        ):
            priors = {
                action: float(probability)
                for action, probability in zip(
                    request.legal_action_ids, row_probabilities[:count], strict=True
                )
            }
            value: float | tuple[float, ...]
            value = float(raw_value[0]) if self.value_size == 1 else tuple(map(float, raw_value))
            if not all(math.isfinite(component) for component in raw_value):
                raise ValueError("model produced a non-finite value")
            results.append(EvalResult(priors=priors, value=value))
        return tuple(results)


__all__ = ["TorchEvaluator"]
