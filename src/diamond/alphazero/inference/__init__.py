"""Central, compatibility-checked AlphaZero inference primitives."""

from .protocol import InferenceFailure, InferenceRequest, InferenceResponse, ModelKey
from .coordinator import InferenceConfig, InferenceCoordinator, InferenceMetrics
from .remote import RemoteEvaluator


def __getattr__(name: str):
    if name == "InferenceModelPool":
        from .model_pool import InferenceModelPool

        return InferenceModelPool
    raise AttributeError(name)

__all__ = [
    "InferenceFailure",
    "InferenceConfig",
    "InferenceCoordinator",
    "InferenceMetrics",
    "InferenceModelPool",
    "InferenceRequest",
    "InferenceResponse",
    "ModelKey",
    "RemoteEvaluator",
]
