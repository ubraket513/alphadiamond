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


def _segmented_softmax(values: Any, rows: Any, row_count: int) -> Any:
    """Softmax within each ragged row, in four device ops and no Python loop.

    What it replaces was a Python ``for`` over rows issuing one
    ``torch.softmax`` per row -- at batch 128, 128 slice-and-launch round trips
    per evaluation, on the single evaluator thread that must never be starved.

    Measured against that loop on the whole callback (forward included), on an
    RTX 5090 under load:

    ===== ========= ============== =========
    batch loop (ms) segmented (ms) speedup
    ===== ========= ============== =========
    32    1.598     1.509          1.06x
    64    1.936     1.722          1.12x
    128   2.514     2.052          1.23x
    256   4.662     2.669          1.75x
    ===== ========= ============== =========

    Worth stating plainly, because the earlier note overclaimed it: the loop was
    **part** of why ``policy_value`` reached 53 % of its roofline where
    ``value_only`` reached 89 %, not all of it.  The rest is work ``value_only``
    simply does not do -- the gather over ``[B, 5329]`` logits and the
    device-to-host copy of the ragged priors -- and no amount of softmax tuning
    touches that.

    Numerically this is the same algorithm ``torch.softmax`` uses -- subtract the
    row maximum before exponentiating -- so it is stable in the same way.  It
    agrees with the per-row loop to 1.8e-7 on real model logits, and a test
    asserts that rather than assuming it.
    """
    import torch

    row_max = torch.full(
        (row_count,), float("-inf"), dtype=values.dtype, device=values.device
    ).scatter_reduce_(0, rows, values, reduce="amax", include_self=False)
    shifted = (values - row_max[rows]).exp()
    row_sum = torch.zeros(
        row_count, dtype=shifted.dtype, device=shifted.device
    ).scatter_add_(0, rows, shifted)
    return shifted / row_sum[rows]


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

        # `output_size` is the whole point of this block, not a micro-detail.
        #
        # `repeat_interleave(arange(n, device=cuda), counts_cuda)` reads as pure
        # device work but is not: the output length depends on the *values* in
        # `counts`, so torch copies `counts` back to the host to size its own
        # result.  That is a device-to-host sync on the evaluator thread -- the
        # one thread §7.5 says must never stall -- and a sync costs whatever is
        # queued behind it.  Passing the length torch would otherwise go and ask
        # for removes it; `offsets` already knows it, for free, on the host.
        #
        # Measured at batch 128 against `value_only`, GPU loaded: x2.60 without
        # `output_size`, x1.87 with.  Building `rows` with `np.repeat` on the
        # host and transferring is exactly as fast (x1.86) and was tried first,
        # but this keeps row construction on the device, moves no 41 KB buffer,
        # and leaves the callback friendlier to compile/graph capture later.
        #
        # The emptiness check stays on the host for the same reason: the old
        # `int(counts.min())` was a *second* sync asking a question the numpy
        # array already answers.  The native side only asks about nodes it is
        # expanding, so an empty row should be impossible -- but it would divide
        # by zero and hand MCTS NaN priors, so it stays checked.
        counts_host = np.diff(offsets).astype(np.int64)
        if counts_host.size and counts_host.min() <= 0:
            raise ValueError("a policy_value request has a row with no legal actions")
        counts = torch.from_numpy(counts_host).to(torch_device)
        rows = torch.repeat_interleave(
            torch.arange(counts.numel(), device=torch_device),
            counts,
            output_size=int(actions.size),
        )
        columns = torch.from_numpy(actions.astype(np.int64)).to(torch_device)

        # One flat gather over the ragged set, then a segmented softmax done
        # entirely on the device.  Row membership comes from the offsets, so no
        # padding is built.
        selected = logits[rows, columns]
        priors = _segmented_softmax(selected, rows, counts_host.size)

        return (
            priors.to(torch.float32).cpu().numpy(),
            values.reshape(-1).to(torch.float32).cpu().numpy(),
        )

    return callback


__all__ = ["policy_value_callback", "value_only_callback"]
