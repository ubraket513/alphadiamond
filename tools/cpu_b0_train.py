"""Backward-compatible alias for ``tools/az_train.py``.

The training loop became hardware-neutral -- the device comes from the
configuration -- but this entry point is a documented CPU command with runbooks
and shell history pointing at it, so the name stays.  There is one
implementation; this module only re-exports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from az_train import (  # noqa: E402,F401
    ACTION_SIZE,
    LoopState,
    append_ledger,
    build_compatibility,
    describe_environment,
    load_config,
    main,
    new_model,
    percentile,
    throughput_summary,
)

if __name__ == "__main__":
    raise SystemExit(main())
