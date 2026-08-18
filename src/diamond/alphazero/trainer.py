"""FP32 eager AdamW trainer shared by independently identified Soo and Min."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .config import TrainingConfig
from .identity import CheckpointCompatibilitySpec, ModelIdentity
from .replay import ReplayBatch, validate_value_target


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        compatibility: CheckpointCompatibilitySpec,
        config: TrainingConfig,
    ) -> None:
        identity = getattr(model, "identity", None)
        if not isinstance(identity, ModelIdentity) or identity != compatibility.identity:
            raise ValueError("model identity does not match trainer compatibility metadata")
        if getattr(model, "config", None) != compatibility.network_config:
            raise ValueError("model network config does not match compatibility metadata")
        if config.learning_rate <= 0 or config.batch_size <= 0 or config.weight_decay < 0:
            raise ValueError("invalid training configuration")
        self.model = model.to(config.device)
        self.compatibility = compatibility
        self.config = config
        self.device = torch.device(config.device)
        self.value_size = 1 if identity.player_count == 2 else 3
        torch.manual_seed(config.seed)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.training_step = 0

    def train_batch(self, batch: ReplayBatch) -> TrainingMetrics:
        if batch.compatibility != self.compatibility:
            raise ValueError("training batch compatibility does not match trainer")
        if not batch.node_features:
            raise ValueError("training batch must not be empty")
        batch_size = len(batch.node_features)
        if len(batch.policy_targets) != batch_size or len(batch.value_targets) != batch_size:
            raise ValueError("training batch components have different sizes")
        for target in batch.value_targets:
            validate_value_target(target, self.compatibility.identity.player_count)

        features = torch.tensor(batch.node_features, dtype=torch.float32, device=self.device)
        policy_targets = torch.tensor(
            batch.policy_targets, dtype=torch.float32, device=self.device
        )
        value_targets = torch.tensor(
            batch.value_targets, dtype=torch.float32, device=self.device
        )
        if not torch.isfinite(features).all() or not torch.isfinite(value_targets).all():
            raise ValueError("training batch contains non-finite values")
        if torch.any(policy_targets < 0) or not torch.isfinite(policy_targets).all():
            raise ValueError("policy target contains invalid probabilities")
        if not torch.allclose(
            policy_targets.sum(dim=1),
            torch.ones(batch_size, device=self.device),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("each policy target must sum to one")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        policy_logits, predicted_values = self.model(features)
        if policy_logits.shape != policy_targets.shape:
            raise ValueError("policy target shape does not match model output")
        if predicted_values.shape != value_targets.shape:
            raise ValueError("value target shape does not match model output")

        policy_loss = -(
            policy_targets * functional.log_softmax(policy_logits, dim=1)
        ).sum(dim=1).mean()
        value_loss = functional.mse_loss(predicted_values, value_targets)
        total_loss = policy_loss + value_loss
        if not torch.isfinite(total_loss):
            raise ValueError("training produced a non-finite loss")
        total_loss.backward()
        self.optimizer.step()
        self.training_step += 1

        metrics = TrainingMetrics(
            total_loss=float(total_loss.detach().cpu()),
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
        )
        if not all(
            math.isfinite(value)
            for value in (metrics.total_loss, metrics.policy_loss, metrics.value_loss)
        ):
            raise ValueError("training metrics are non-finite")
        return metrics


__all__ = ["AlphaZeroTrainer", "TrainingMetrics"]
