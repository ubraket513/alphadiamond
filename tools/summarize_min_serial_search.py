#!/usr/bin/env python3
"""Validate and summarize Min serial-search benchmark JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def percent(value: float, baseline: float) -> str:
    if baseline == 0:
        return "n/a (baseline=0)"
    return f"{100.0 * (value - baseline) / baseline:+.1f}% (denominator={baseline:.6g})"


def load_scale(directory: Path) -> dict[str, dict]:
    rows = {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}
    if not rows:
        return rows
    checks = {
        "source commit": lambda row: row["provenance"]["source_commit"],
        "model digest": lambda row: row["domain"]["model_sha256"],
        "seed": lambda row: row["workload"]["seed"],
        "precision": lambda row: row["environment"]["precision"],
        "game count": lambda row: row["workload"]["games"],
    }
    for label, getter in checks.items():
        values = {getter(row) for row in rows.values()}
        if len(values) != 1:
            raise ValueError(f"mixed {label} in {directory}: {sorted(values)!r}")
    return rows


def metric(row: dict, *path: str) -> float:
    value = row
    for name in path:
        value = value[name]
    return float(value)


def render_scale(name: str, rows: dict[str, dict]) -> list[str]:
    lines = [f"## {name}", ""]
    lines += [
        "| arm | completion | aborts max/deadline/other | moves p50/p90/p99 | revisit | repeat<=8 | max revisits | cycling | samples/h | eval/s | batch mean/p50/p90 | target H | norm H | full/legal KL | legal mass | top1 | boosted | wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in rows.items():
        d = row["domain"]
        attempted = d["attempted_episodes"]
        completion = d["completed_episodes"] / attempted if attempted else 0.0
        aborts = d["abort_reasons"]
        repetition = d["repetition"]
        target = d["search_targets"]
        policy = d["policy_fit"]
        lines.append(
            f"| {arm} | {completion:.1%} | {aborts['max_moves']}/{aborts['max_game_seconds']}/{aborts['other']} | "
            f"{d['moves_p50']:.0f}/{d['moves_p90']:.0f}/{d['moves_p99']:.0f} | "
            f"{repetition['revisit_fraction_mean']:.6g} | {repetition['repeat_within_8_fraction_mean']:.6g} | "
            f"{repetition['max_revisits_mean']:.6g} | {repetition['cycling_games']} | {d['samples_per_hour']:.6g} | "
            f"{d['evaluations_per_second']:.6g} | {d['batch_mean']:.3f}/{d['batch_p50']:.0f}/{d['batch_p90']:.0f} | "
            f"{target['entropy_mean']:.6g} | {target['normalized_entropy_mean']:.6g} | "
            f"{policy['full_kl_mean']:.6g}/{policy['legal_kl_mean']:.6g} | "
            f"{policy['legal_probability_mass_mean']:.6g} | {policy['top1_agreement']:.6g} | "
            f"{d['boosted_fraction']:.6g} | {row['summary_seconds']['median']:.3f} |"
        )
    lines += ["", "### Comparisons", ""]
    if "b0-128" in rows:
        base = rows["b0-128"]
        for arm, row in rows.items():
            if arm == "b0-128":
                continue
            lines.append(
                f"- `{arm}` vs `b0-128`: completion "
                f"{percent(metric(row, 'domain', 'completed_episodes'), metric(base, 'domain', 'completed_episodes'))}; "
                f"wall {percent(metric(row, 'summary_seconds', 'median'), metric(base, 'summary_seconds', 'median'))}."
            )
    if "a0-128" in rows:
        base = rows["a0-128"]
        for arm, row in rows.items():
            if not arm.startswith("a0-") or arm == "a0-128":
                continue
            lines.append(
                f"- `{arm}` vs `a0-128`: completion "
                f"{percent(metric(row, 'domain', 'completed_episodes'), metric(base, 'domain', 'completed_episodes'))}; "
                f"repeat<=8 {percent(metric(row, 'domain', 'repetition', 'repeat_within_8_fraction_mean'), metric(base, 'domain', 'repetition', 'repeat_within_8_fraction_mean'))}; "
                f"wall {percent(metric(row, 'summary_seconds', 'median'), metric(base, 'summary_seconds', 'median'))}."
            )
    return lines + [""]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_min_serial_search.py INPUT_DIR OUTPUT_MD")
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    scales = {path.name: load_scale(path) for path in sorted(source.iterdir()) if path.is_dir()}
    scales = {name: rows for name, rows in scales.items() if rows}
    if not scales:
        raise ValueError(f"no benchmark JSON found under {source}")
    lines = ["# Min serial-search sweep — 2026-09-01", ""]
    for name, rows in scales.items():
        lines.extend(render_scale(name, rows))
    smoke = scales.get("smoke", {})
    a0_rows = {name: row for name, row in smoke.items() if name.startswith("a0-")}
    any_a0_completed = any(row["domain"]["completed_episodes"] for row in a0_rows.values())
    adaptive_triggered = any(
        row["domain"]["boosted_fraction"] > 0 for name, row in a0_rows.items() if "adaptive" in name
    )
    decision = "FULL_SWEEP_REQUIRED" if any_a0_completed else "NO_DEEPER_SEARCH_BENEFIT"
    lines += [
        "## Decision classification",
        "",
        f"`{decision}`",
        "",
        f"Smoke A0 completion observed: `{str(any_a0_completed).lower()}`. Adaptive trigger observed: `{str(adaptive_triggered).lower()}`.",
        "The summarizer reports evidence only and does not alter code or configuration.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
