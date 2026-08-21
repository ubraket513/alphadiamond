import json, sys
from pathlib import Path

BENCH = Path("/tmp/claude-0/-workspace-alphadiamond/3fe1188f-824d-4745-af77-883f74376ad2/scratchpad/az-bench/soo")
HIST = Path("/workspace/alphadiamond/az-bench/soo")

def row(path, label):
    rec = None
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "iteration":
            rec = r
            break
    if rec is None:
        return None
    i = rec["inference"]
    t = rec["throughput"]
    return dict(
        run=label,
        wall=round(rec["selfplay_s"], 1),
        samp_h=round(t["samples_per_hour"]),
        games=f"{rec['completed']}/{rec['attempted']}",
        aborts=rec["aborted"],
        med_mv=rec["median_moves"],
        p90_mv=rec["p90_moves"],
        mean_b=round(i["mean_batch_size"], 2),
        max_b=i["max_batch_size"],
        qd50=i["queue_to_dispatch_p50_ms"],
        qd90=i["queue_to_dispatch_p90_ms"],
        inf50=i["inference_p50_ms"],
        resp50=i["response_p50_ms"],
        evals_s=round(i["evaluations_per_second"]),
        batches=i["batches_completed"],
        requests=i["requests_completed"],
    )

rows = []
for label, base in (("hist s64-w30", HIST/"s64-w30"), ("hist s64-prof", HIST/"s64-prof"), ("hist s64-prof2", HIST/"s64-prof2")):
    p = base/"ledger.jsonl"
    if p.exists():
        rows.append(row(p, label))
for name in ("pre-1","pre-2","pre-3","post-1","post-2","post-3"):
    p = BENCH/name/"ledger.jsonl"
    if p.exists():
        rows.append(row(p, name))

rows = [r for r in rows if r]
cols = list(rows[0])
w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print(" | ".join(c.ljust(w[c]) for c in cols))
print("-|-".join("-"*w[c] for c in cols))
for r in rows:
    print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))
