"""Summarize the JSONL emitted by diamond_qt_min_position_audit (stdlib only)."""
import json
import statistics
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/min-local-audit")
    summary = {}
    for path in sorted(root.glob("*-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        groups = {}
        for budget in sorted({row["simulations"] for row in rows}):
            subset = [row for row in rows if row["simulations"] == budget]
            groups[str(budget)] = {
                "positions": len(subset),
                "median_ms": statistics.median(row["total_ms"] for row in subset),
                "mean_absolute_nn_mcts_gap": statistics.mean(abs(row["nn"] - row["mcts"]) for row in subset),
                "last_position": {key: subset[-1][key] for key in ("ply", "nn", "mcts", "selected_action")},
            }
        summary[path.name] = groups
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
