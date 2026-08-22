"""Python side of the native batch callback (section 4 of the Phase 0 design).

Two modes, because B0 discards the neural policy but later phases will not.

``ValueOnly`` is the B0 production path. The vacancy prior is computed natively
-- Gate A proves it equals the Python oracle within 1e-12 -- so priors never
cross the boundary and the entire policy tail disappears: no bounds check, no
padded index build, no gather, no mask, no softmax, no validity sync, no
``[B, 5329]`` device-to-host copy, no priors dict, no response envelope.
Section 0.4 measured that tail at 25 % of the evaluator path at batch 32.

``PolicyValue`` keeps the neural policy and is the reference path.

The features array handed to a callback is a **view onto a buffer the native
side reuses**. It is valid for the duration of the call and no longer; a
callback that stores it will see the next batch's data. Copy if you need to
keep it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch and numpy are optional; this module imports without them
    import numpy as np


def value_only_callback(
    model: Any,
    *,
    device: str = "cpu",
    trunk_only: bool = True,
) -> Callable[[Any], Any]:
    """A ``value_only`` callback over a Soo model.

    ``trunk_only`` skips the policy head entirely rather than computing and
    discarding it. Section 0.4 measured that at 7 % of the forward (6 of 61
    aten operations) -- small, but it is free to skip and there is no reason to
    compute a tensor nothing reads.
    """
    import torch

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    def callback(features: np.ndarray) -> np.ndarray:
        # from_numpy shares the native staging buffer; the tensor lives only
        # inside this call, which is exactly the contract above.
        batch = torch.from_numpy(features).to(torch_device, non_blocking=False)
        with torch.inference_mode():
            if trunk_only:
                nodes = model.trunk(batch)
                values = model.value_head(nodes.mean(dim=1))
            else:
                _, values = model(batch)
        return values.reshape(-1).to(torch.float32).cpu().numpy()

    return callback


def policy_value_callback(
    model: Any,
    *,
    device: str = "cpu",
) -> Callable[[Any, Any, Any], Any]:
    """A ``policy_value`` callback: ragged legal sets in, flat priors out.

    Legal sets arrive flat plus offsets rather than padded, so the gather here
    never materialises ``[B, max_legal]``.
    """
    import numpy as np
    import torch

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    def callback(
        features: np.ndarray, actions: np.ndarray, offsets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = torch.from_numpy(features).to(torch_device)
        with torch.inference_mode():
            logits, values = model(batch)

        action_space = logits.shape[1]
        if actions.size and (actions.min() < 0 or actions.max() >= action_space):
            raise ValueError("legal action is outside the model policy space")

        # One flat gather over the ragged set, then a segmented softmax. Row
        # membership comes from the offsets, so no padding is built.
        row_of = np.repeat(
            np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets).astype(np.int64)
        )
        rows = torch.from_numpy(row_of).to(torch_device)
        columns = torch.from_numpy(actions.astype(np.int64)).to(torch_device)
        selected = logits[rows, columns]

        priors = torch.empty_like(selected)
        for index in range(len(offsets) - 1):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            priors[begin:end] = torch.softmax(selected[begin:end], dim=0)

        return (
            priors.to(torch.float32).cpu().numpy(),
            values.reshape(-1).to(torch.float32).cpu().numpy(),
        )

    return callback


__all__ = ["policy_value_callback", "value_only_callback"]
