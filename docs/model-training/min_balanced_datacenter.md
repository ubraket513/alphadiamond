# Min: six-order datacenter experiment

This branch adds equal **requested game counts** for all six Min turn orders.
It does not claim equal retained samples or improved playing strength. The local
CPU smoke run completed four of six games and eight learner updates; see
[the implementation and results](balanced_turn_orders.md).

## Start on a Linux GPU host

Requirements: a C++26-capable compiler (the previous training host used GCC 14),
CMake 3.21+, Ninja, Git, unzip, sha256sum, an NVIDIA driver, CUDA toolkit, and a
CUDA-enabled LibTorch distribution compatible with that toolkit. `LIBTORCH_ROOT`
may also point to an existing CUDA PyTorch installation's `torch` package folder.
The trainer itself is native C++. No RTX 5090-specific requirement is imposed.

The Hugging Face CLI must be installed and authenticated with access to the
existing private bucket `ubraket513/alphadiamond-min-private-backups`.

```bash
git clone --branch codex/min-balanced-orders-datacenter \
  https://github.com/ubraket513/alphadiamond.git
cd alphadiamond
hf auth login

# Set these to the installed CUDA LibTorch and compiler on this host.
export LIBTORCH_ROOT=/path/to/libtorch
export CXX=g++-14
export OUTPUT_ROOT=/workspace/min-balanced-results

bash tools/run_min_balanced_datacenter.sh smoke
```

The launcher checks the CUDA library, records the GPU and source commit, derives
the GPU architecture list (override `TORCH_CUDA_ARCH_LIST` if necessary), downloads
the verified step 20064 artifact, checks its pinned SHA-256, builds the trainer,
runs the self-play/config tests, and starts one bounded iteration. Use a host or
container with the intended GPU exposed; each run uses `cuda:0`, not distributed
training. The script does not install packages or change drivers.

| Mode | Orders | Games | Simulations | Concurrent games | Learner batch / updates |
|---|---|---:|---:|---:|---|
| `smoke` | six, equally requested | 12 | 128 | 6 | 32 / 8 |
| `pilot` | six, equally requested | 192 | 128 | 32 | 256 / 32 |
| `control` | fixed 1-2-3 | 192 | 128 | 32 | 256 / 32 |

All modes use step 20064 weights, a fresh replay, a fresh optimizer, learning rate
0.0001, unchanged six-block/128-width architecture, unchanged +1/0/−1 utilities,
500-ply limit, and one iteration. `pilot` and `control` differ only in order
balancing. These are test configurations, not a full retraining prescription.
The smoke's 12 games are not enough to measure strength.

## Run the comparison

After inspecting smoke completion and memory use:

```bash
bash tools/run_min_balanced_datacenter.sh pilot
bash tools/run_min_balanced_datacenter.sh control
```

Each command prints its run directory. The launcher refuses an existing run ID;
use a unique `RUN_ID` or the default timestamp. To resume an interrupted run, use
`build/min-balanced-datacenter/native/alphadiamond-train resume --run-dir RUN_DIR`.
Preserve its config and replay; do not turn a fixed-order run into a balanced run
by editing its saved configuration.

Inspect `iterations/0/selfplay.metrics.json` in both runs. Its `turn_orders`
section records requested/completed/aborted games and retained samples. A large
completion imbalance means the learner still lacks balanced data. In particular,
the CLI's existing top-level `completed_games` field counts processed games and
includes aborts: use the sidecar for actual completion.

For each pilot/control run, compare its candidate with its frozen starting model:

```bash
run=/workspace/min-balanced-results/runs/min/REPLACE_WITH_PRINTED_RUN_ID
build/min-balanced-datacenter/native/alphadiamond-train evaluate \
  --run-dir "$run" \
  --candidate "$run/iterations/0/candidate-checkpoint" \
  --champion "$run/initial-champion-checkpoint" \
  --opening-suite production-openings-v1
```

The saved configuration specifies 216 arena games across 12 opening blocks,
balanced across candidate seats and turn orders. Evaluation is explicit, rather
than automatic during the smoke/pilot. Results go to `arena.json` and the rating
registry. Retain those records to examine results by turn order, not just pooled
utility. A confidence interval crossing zero does not establish improvement.
No public release or GUI model promotion is performed by the launcher.

## Hugging Face handoff

Private asset prefix:

```text
hf://buckets/ubraket513/alphadiamond-min-private-backups/min-balanced-orders-20260911/v2
```

It contains the original `min-step20064.zip`, `MANIFEST.json` with the exact source
commit, `SHA256SUMS`, this handoff, and local audit/smoke evidence. It deliberately
does not use the eight-update smoke candidate as the starting model. To retrieve
the complete handoff and verify file hashes:

```bash
hf buckets sync \
  hf://buckets/ubraket513/alphadiamond-min-private-backups/min-balanced-orders-20260911/v2 \
  ./artifacts/min-balanced-handoff
(cd artifacts/min-balanced-handoff && sha256sum --check SHA256SUMS)
```

For an exact source checkout, use `source_commit` in `MANIFEST.json` after cloning.
Upload new datacenter results to a **new run-specific prefix**, for example:

```bash
hf buckets sync /workspace/min-balanced-results/runs/min/ACTUAL_RUN_ID \
  hf://buckets/ubraket513/alphadiamond-min-private-backups/min-balanced-results/ACTUAL_RUN_ID
```

Use persistent storage for `OUTPUT_ROOT`. Keep the immutable handoff prefix intact.
CUDA execution remains to be verified on that host: this preparation was tested
on Windows CPU, with shell syntax and artifact/hash checks locally.
