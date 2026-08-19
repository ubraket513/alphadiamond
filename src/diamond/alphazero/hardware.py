"""CPU capacity detection for the self-play worker pool.

Standard library only, deliberately: spawn workers import this module, and the
worker import path is asserted to stay free of Torch.
"""

from __future__ import annotations

import os

RESERVED_CPUS = 2
"""CPUs held back for the parent loop and centralized GPU inference."""


def available_cpu_count() -> int:
    """Return the CPUs this process may actually run on.

    ``os.cpu_count()`` reports the machine, not the process, so a cpuset or
    container restriction would silently oversubscribe the pool.  Affinity is
    the authoritative answer where the platform provides it.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            affinity = getaffinity(0)
        except OSError:
            # Some platforms expose the symbol but refuse the call.
            affinity = None
        if affinity:
            return len(affinity)
    count = os.cpu_count()
    if isinstance(count, int) and count > 0:
        return count
    return 1


def resolve_worker_count(
    configured: int | None = None,
    *,
    available: int | None = None,
    reserved: int = RESERVED_CPUS,
) -> int:
    """Return the self-play worker count, explicit override winning.

    Without an override the pool takes every available CPU except ``reserved``,
    and never falls below one worker however small the machine is.
    """
    if configured is not None:
        if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
            raise ValueError("worker_count must be a positive integer")
        return configured
    if available is None:
        available = available_cpu_count()
    if not isinstance(available, int) or isinstance(available, bool) or available < 1:
        raise ValueError("available CPU count must be a positive integer")
    if not isinstance(reserved, int) or isinstance(reserved, bool) or reserved < 0:
        raise ValueError("reserved CPU count must be a non-negative integer")
    return max(1, available - reserved)


__all__ = ["RESERVED_CPUS", "available_cpu_count", "resolve_worker_count"]
