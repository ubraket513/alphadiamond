# `cpu8h-soo-20260819` — a pinned fixture, not a live run

This directory is **no longer a training run**. What remains is the artefact the
project's measurements are defined against:

| file | why it is here |
|---|---|
| `latest.pt` | The **immutable step-80 Soo checkpoint**, `sha256:1634b901e213b065c107eea734b8c172c14babb1c2565352203961e86ea165af`. Hash-asserted by `tests/native/test_callback.py` and by CI, and loaded by every `az-bench/profiles/bench_native_*.py`. **Do not replace it** — every number in `docs/native_selfplay_phase1_progress.md` and `docs/rtx5060_bottleneck_findings.md` is relative to this exact file. |
| `config.json`, `ledger.jsonl`, `loop_state.json`, `off-probe.json` | The provenance of that checkpoint: the configuration and per-iteration record of the 8-hour CPU run that produced it. Small, and cited by `blueprint/cpu_train_runbook.md` and `docs/cpu_profile_findings.md`. |

Deleted, because they were large, derived and superseded:

- `checkpoints/` — 19 intermediate `B0-i*.pt` snapshots (166 MB). The run they
  belonged to is finished and `latest.pt` is its terminal state.
- `replay/` — 47 MB of replay chunks from that run. Training from scratch builds
  its own replay store; these could only pollute it.
- `latest.pt.cpu-backup` — a device-migration backup of a *different*
  (`sha256:4b2a32ff…`) checkpoint, left over from the CUDA/CPU migration test.

The path stays where it is deliberately. A dozen historical documents and every
`az-bench/soo/*/provenance.txt` cite it by name; moving the file would make
those records wrong.
