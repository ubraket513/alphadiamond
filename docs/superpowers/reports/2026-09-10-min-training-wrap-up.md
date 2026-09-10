# Min training stopped for user testing — 2026-09-10

Final learner step: 47,520 (iteration134). Iteration135 SELF_PLAY was interrupted.
Training and monitor are stopped; no task GPU processes remain. Resume from the
full run, preserving replay, optimizer, RNG, ledger and configuration. Unfinished
self-play must be regenerated. Do not reuse the old budget/deadline blindly.

Training source:79793fedb6d2ba77b366bb0502cd3530126f7731. Main incorporates that
source plus Qt integration372511afb2509dcfdc8f627bde956107c1ccb004 and wrap-up docs.
Run root:/workspace/alphadiamond-training/runs/min/min-newhost-20260909.
Restore archive under /workspace to preserve recorded absolute paths. Build
against local LibTorch. Validate iterations/134/candidate-checkpoint first.
Set a NEW explicit runtime budget, then run under supervisor:

    alphadiamond-train resume --run-dir /workspace/alphadiamond-training/runs/min/min-newhost-20260909 --config /workspace/alphadiamond-training/runs/min/min-newhost-20260909/active-config.json

Do not invoke train --scratch. Existing wrappers contain the old absolute deadline;
update budget.json and review the wrapper before enabling any service. Autostart
is disabled. Production arena remains disabled; bootstrap_prior none, weight0.
Width128,6blocks,FP32 learner,batch1024,352steps,LR0.001,WD0.0001;1024games,
128simulations,500move cap,512lanes32searchthreads,FP16actors.

Qt test ZIP contains BOTH Min20064 (validated reference) and Min47520 (latest,
strength untested). Extract beside compatible diamond_qt executable, refresh
Models and explicitly select the desired Min version. Not a public catalog release.
50 configured tests passed; latest export exactly matched12positions. Linux tested;
Windows packaging untested.20k beat10k in a narrow comparison;40kvs20k was
inconclusive. No general strength/Elo claim.

Private full backup location:
hf://buckets/ubraket513/alphadiamond-min-private-backups/min-newhost-20260909/final/min-final-backup-20260910.tar.zst
Use hf buckets cp with your authenticated account; verify SHA256 sidecar, extract
with tar --zstd -xf, then verify final/manifest-sha256.json entries. All checkpoints,
replay, failures, config transitions and evidence are retained. Storage on this
instance is NOT durable through destruction/recycle.
