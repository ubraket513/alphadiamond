"""Versioned file boundary for the transitional native training pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class NativePipelineError(RuntimeError):
    """The native pipeline rejected or failed the requested stage."""


def run_native_pipeline(executable: Path, request_path: Path, result_path: Path, request: dict[str, object]) -> dict[str, object]:
    """Invoke only the schema-v2 native command and validate its file result."""
    if request.get("schema_version") != 2 or not isinstance(request.get("operation_id"), str):
        raise ValueError("native pipeline requests require schema_version=2 and operation_id")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    completed = subprocess.run(
        [str(executable), str(request_path), str(result_path)], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise NativePipelineError(completed.stderr.strip() or f"native pipeline exited {completed.returncode}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativePipelineError("native pipeline did not produce JSON result") from error
    if not isinstance(result, dict) or result.get("schema_version") != 2 or result.get("operation_id") != request["operation_id"]:
        raise NativePipelineError("native pipeline returned incompatible result")
    return result
