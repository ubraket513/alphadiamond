"""Headless command line entry points for AlphaZero training operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, TextIO

EXIT_OK = 0
EXIT_ARGUMENT_ERROR = 2
EXIT_RUNTIME_ERROR = 3
EXIT_INTERNAL_ERROR = 4


class CommandServices(Protocol):
    """The small headless service boundary used by command dispatch."""

    def train(self, *, model_name: str, run_id: str) -> Mapping[str, object]: ...

    def resume(self, *, model_name: str, run_id: str) -> Mapping[str, object]: ...

    def benchmark(self, *, model_name: str, run_id: str) -> Mapping[str, object]: ...

    def leaderboard(self, *, model_name: str, run_id: str) -> Mapping[str, object]: ...

    def profile(self, *, model_name: str, max_seconds: int) -> object: ...


ServicesFactory = Callable[[Path, str, Path, Path], CommandServices]


class _ArgumentError(ValueError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="alphadiamond-train", add_help=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "resume", "benchmark", "leaderboard"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--runtime-dir", required=True, type=Path)
        subparser.add_argument("--model", required=True, choices=("Soo", "Min"))
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument("--config", required=True, type=Path)
        subparser.add_argument("--checkpoint", required=True, type=Path)
    profile = subcommands.add_parser("profile")
    profile.add_argument("--seconds", type=int, default=1)
    profile.add_argument("--runtime-dir", required=True, type=Path)
    profile.add_argument("--model", required=True, choices=("Soo", "Min"))
    profile.add_argument("--config", required=True, type=Path)
    profile.add_argument("--checkpoint", required=True, type=Path)
    return parser


def _default_services(
    root: Path, model_name: str, config_path: Path, checkpoint_path: Path
) -> CommandServices:
    # Torch and all orchestration dependencies remain lazy so this module is
    # importable in environments that intentionally do not initialize a GUI.
    from .production import build_production_services

    return build_production_services(root, model_name, config_path, checkpoint_path)


def _emit(payload: Mapping[str, object], stream: TextIO) -> None:
    print(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")), file=stream)


def main(
    argv: list[str] | None = None,
    *,
    services_factory: ServicesFactory = _default_services,
    stdout: TextIO | None = None,
) -> int:
    """Dispatch one machine-readable headless command and return an exit code."""
    stream = stdout or sys.stdout
    parser = build_parser()
    command_hint = argv[0] if argv else None
    try:
        args = parser.parse_args(argv)
        if hasattr(args, "run_id"):
            from .run_state import validate_run_id

            validate_run_id(args.run_id)
        if args.command == "profile":
            if args.seconds <= 0:
                raise ValueError("--seconds must be positive")
            services = services_factory(
                args.runtime_dir,
                args.model,
                getattr(args, "config", Path()),
                getattr(args, "checkpoint", Path()),
            )
            result = services.profile(model_name=args.model, max_seconds=args.seconds)
            to_dict = getattr(result, "to_dict", None)
            if not callable(to_dict):
                raise ValueError("profile service must return a ProfileReport")
            _emit(
                {
                    "command": "profile",
                    "status": "ok",
                    **dict(to_dict()),
                },
                stream,
            )
            return EXIT_OK

        services = services_factory(
            args.runtime_dir,
            args.model,
            getattr(args, "config", Path()),
            getattr(args, "checkpoint", Path()),
        )
        operation = getattr(services, args.command)
        result = operation(model_name=args.model, run_id=args.run_id)
        _emit(
            {"command": args.command, "status": "ok", **dict(result)},
            stream,
        )
        return EXIT_OK
    except _ArgumentError as error:
        command = getattr(locals().get("args", None), "command", command_hint)
        _emit(
            {
                "command": command,
                "error": str(error),
                "status": "error",
            },
            stream,
        )
        return EXIT_ARGUMENT_ERROR
    except ValueError as error:
        command = getattr(locals().get("args", None), "command", command_hint)
        _emit(
            {
                "command": command,
                "error": str(error),
                "status": "error",
            },
            stream,
        )
        return EXIT_RUNTIME_ERROR
    except Exception as error:  # Keep automation failures structured and stable.
        command = getattr(locals().get("args", None), "command", command_hint)
        _emit(
            {
                "command": command,
                "error": f"{type(error).__name__}: {error}",
                "status": "error",
            },
            stream,
        )
        return EXIT_INTERNAL_ERROR


__all__ = [
    "EXIT_ARGUMENT_ERROR",
    "EXIT_INTERNAL_ERROR",
    "EXIT_OK",
    "EXIT_RUNTIME_ERROR",
    "CommandServices",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
