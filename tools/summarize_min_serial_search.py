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


def completion(row: dict) -> float:
    attempted = metric(row, "domain", "attempted_episodes")
    return metric(row, "domain", "completed_episodes") / attempted if attempted else 0.0


def smoke_qualifiers(rows: dict[str, dict]) -> list[str]:
    base = rows["a0-128"]
    base_completion = completion(base)
    base_repeat = metric(base, "domain", "repetition", "repeat_within_8_fraction_mean")
    base_cycles = metric(base, "domain", "repetition", "cycling_games")
    qualifiers = []
    for name, row in rows.items():
        if not name.startswith("a0-"):
            continue
        qualifies = (
            completion(row) >= 0.50
            or completion(row) >= base_completion + 0.10
            or metric(row, "domain", "repetition", "repeat_within_8_fraction_mean")
            <= 0.50 * base_repeat
            or metric(row, "domain", "repetition", "cycling_games") <= 0.50 * base_cycles
        )
        if qualifies:
            qualifiers.append(name)
    return sorted(qualifiers)


def illegal_mass_dominated(row: dict) -> bool:
    policy = row["domain"]["policy_fit"]
    return (
        policy["legal_kl_mean"] <= 0.10
        and policy["full_kl_mean"] >= policy["legal_kl_mean"] + 0.50
        and policy["legal_probability_mass_mean"] <= 0.6065306597
    )


def deeper_search_helps(rows: dict[str, dict]) -> bool:
    base = rows["a0-128"]
    for name in ("a0-256", "a0-400"):
        if name not in rows:
            continue
        row = rows[name]
        completion_gate = completion(row) >= 0.90 or completion(row) >= completion(base) + 0.10
        repeat_gate = metric(row, "domain", "repetition", "repeat_within_8_fraction_mean") <= (
            0.70 * metric(base, "domain", "repetition", "repeat_within_8_fraction_mean")
        )
        cycling_gate = metric(row, "domain", "repetition", "cycling_games") <= metric(
            base, "domain", "repetition", "cycling_games"
        )
        legal_gate = metric(row, "domain", "policy_fit", "legal_kl_mean") <= (
            1.10 * metric(base, "domain", "policy_fit", "legal_kl_mean")
        )
        deadline_gate = metric(row, "domain", "abort_reasons", "max_game_seconds") <= metric(
            base, "domain", "abort_reasons", "max_game_seconds"
        )
        if completion_gate and repeat_gate and cycling_gate and legal_gate and deadline_gate:
            return True
    return False


def adaptive_search_wins(rows: dict[str, dict]) -> bool:
    for suffix in ("256", "400"):
        adaptive = rows.get(f"a0-adaptive-{suffix}")
        constant = rows.get(f"a0-{suffix}")
        if adaptive is None or constant is None:
            continue
        constant_completion = completion(constant)
        completion_gate = completion(adaptive) >= max(0.0, constant_completion - 0.05)
        moves_gate = metric(adaptive, "domain", "moves_p90") <= 1.10 * metric(
            constant, "domain", "moves_p90"
        )
        repeat_gate = metric(adaptive, "domain", "repetition", "repeat_within_8_fraction_mean") <= (
            1.10 * metric(constant, "domain", "repetition", "repeat_within_8_fraction_mean")
        )
        boost_gate = metric(adaptive, "domain", "boosted_fraction") <= 0.15
        constant_throughput = metric(constant, "domain", "samples_per_hour")
        adaptive_throughput = metric(adaptive, "domain", "samples_per_hour")
        throughput_gate = (
            constant_throughput > 0.0 and adaptive_throughput >= 1.25 * constant_throughput
        )
        if completion_gate and moves_gate and repeat_gate and boost_gate and throughput_gate:
            return True
    return False


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
    if sys.argv[1:] == ["--self-test"]:
        fixture = {
            "domain": {
                "attempted_episodes": 32,
                "completed_episodes": 0,
                "repetition": {"repeat_within_8_fraction_mean": 0, "cycling_games": 0},
            }
        }
        assert smoke_qualifiers({"a0-128": fixture, "a0-256": fixture}) == ["a0-128", "a0-256"]
        adaptive_fixture = {
            "domain": {
                "attempted_episodes": 32,
                "completed_episodes": 0,
                "moves_p90": 800,
                "samples_per_hour": 0,
                "boosted_fraction": 0,
                "repetition": {"repeat_within_8_fraction_mean": 0, "cycling_games": 0},
            }
        }
        assert not adaptive_search_wins(
            {"a0-256": adaptive_fixture, "a0-adaptive-256": adaptive_fixture}
        )
        return 0
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
    qualifiers = smoke_qualifiers(smoke) if smoke else []
    full = scales.get("full", {})
    full_required = bool(qualifiers)
    full_complete = bool(full) and set(smoke) <= set(full)
    dominated = sorted(name for name, row in (full or smoke).items() if illegal_mass_dominated(row))
    deeper = deeper_search_helps(full) if full_complete else False
    adaptive = adaptive_search_wins(full) if full_complete else False
    if full_required and not full_complete:
        decision = "FULL_SWEEP_REQUIRED"
        selected = "PENDING_FULL_SWEEP"
    elif adaptive:
        decision = "ADAPTIVE_SEARCH_WINS"
        selected = "B_ADOPT_ADAPTIVE_SERIAL_SEARCH"
    elif deeper:
        expensive = any(
            metric(full[name], "summary_seconds", "median")
            >= 1.5 * metric(full["a0-128"], "summary_seconds", "median")
            for name in ("a0-256", "a0-400")
        )
        selected = "C_AUTHORIZE_PARALLEL_MCTS" if expensive else "B_ADOPT_STRONGER_SERIAL_SEARCH"
        decision = "DEEPER_SEARCH_HELPS"
    else:
        decision = "NO_DEEPER_SEARCH_BENEFIT"
        selected = "D_AUTHORIZE_VACANCY_PRIOR_ANNEALING"
    lines += [
        "## Decision classification",
        "",
        f"`{decision}`",
        "",
        f"Smoke qualifiers (literal gate, including zero baselines): `{', '.join(qualifiers) or 'none'}`.",
        f"Full matrix complete: `{str(full_complete).lower()}`.",
        f"Illegal-mass-dominated arms: `{', '.join(dominated) or 'none'}`.",
        f"Deeper search helps: `{str(deeper).lower()}`. Adaptive search wins: `{str(adaptive).lower()}`.",
        f"Selected ordered branch: `{selected}`.",
        "Min remains B0 until the final acceptance gate passes.",
        "The summarizer reports evidence only and does not alter code or configuration.",
        "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
