import json,glob,os,sys
rows=[]
for p in sorted(glob.glob(sys.argv[1])):
    n=os.path.basename(os.path.dirname(p))
    for l in open(p):
        r=json.loads(l)
        if r.get("event")=="iteration":
            i=r["inference"]
            rows.append((n,r["selfplay_s"],round(r["throughput"]["samples_per_hour"]),
                f"{r['completed']}/{r['attempted']}",round(i["mean_batch_size"],2),i["max_batch_size"],
                i["queue_to_dispatch_p50_ms"],i["inference_p50_ms"],i["response_p50_ms"],
                round(i["evaluations_per_second"]),i["batches_completed"]))
            break
print(f"{'run':18s} {'wall':>7s} {'samp/h':>7s} {'games':>6s} {'meanb':>6s} {'maxb':>5s} {'qd50':>7s} {'inf50':>6s} {'resp50':>7s} {'ev/s':>5s} {'batches':>8s}")
for r in rows: print(f"{r[0]:18s} {r[1]:7.1f} {r[2]:7d} {r[3]:>6s} {r[4]:6.2f} {r[5]:5d} {r[6]:7.2f} {r[7]:6.2f} {r[8]:7.2f} {r[9]:5d} {r[10]:8d}")
