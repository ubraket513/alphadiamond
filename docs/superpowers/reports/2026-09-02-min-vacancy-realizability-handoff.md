# Min Vacancy Realizability Diagnostic Handoff — 2026-09-02

## Status

The offline vacancy-prior realizability diagnostic is implemented on
`codex/min-a0-realizability`. It never opens a production run for mutation: the input checkpoint,
replay, and config are read-only, while reports and disposable checkpoints require explicit output
paths.

The authoritative experiment requested by the design is **not yet run**. This instance and the
latest GitHub release contain only the iteration-25 checkpoint at training step 27,776. The required
iteration-100 checkpoint at training step 133,376 and its one-million-row replay are not locally or
remotely available as release assets.

## Implemented contracts

- `configs/alphazero/min-anneal-alpha050-v1.json` pins vacancy prior weight `0.5` and legal loss.
- Replay channel 0 reconstructs canonical self occupancy.
- Every authoritative legal action in `sparse_policy`, including zero-visit actions, is retained.
- Arm `head` trains only `policy_source` and `policy_destination`.
- Arm `full` trains trunk plus policy and freezes both value modules.
- Reports include legal KL, expected progress and teacher ratio, top-1 agreement, network top-3
  teacher mass, before/after policy KL, trainable gradient/update L2, digests, device, seed, and
  clean source provenance.
- Optional checkpoint-v3 output advances training step and saves the updated model and optimizer
  together with source checkpoint lineage.
- Source commit, dirty state, canonical config digest, and input model digest fail closed before
  training.

## Non-authoritative iteration-25 smoke

Inputs:

- release: `min-v1.0.1`
- checkpoint step: 27,776
- input model digest: `847007a72b0a283a1789c7cb160f9fd93864117ad1cdbf5bbf3541c92f5ef59a`
- replay manifest digest: `dc0ce2efea60fc4b4b3ce1a21fe36f60bf7ff5b0fb916489a127e1988ae3ee59`
- seed: 20,260,902
- device: `cuda:0`
- steps: 100
- batch: 256
- held out: 1,024
- evaluation batch: 256

| arm | initial legal KL | final legal KL | initial/final progress ratio | policy KL | gradient L2 | update L2 |
|---|---:|---:|---:|---:|---:|---:|
| head | 0.547243 | 0.547243 | -0.211124 / -0.211124 | 0 | 0.00002154 | 0.00059355 |
| full | 0.547243 | 0.547243 | -0.211124 / -0.211124 | 0 | 0.00006934 | 0.00726020 |

Both arms updated trainable parameters but left the held-out legal policy exactly unchanged over
this bounded smoke. This is not an iteration-100 realizability verdict. It is consistent with the
earlier learning diagnostic, where policy parameters moved and raw logits had RMS delta `24.5961`,
while policy KL remained only `2.83092e-11`: most movement was in a softmax-invariant common-logit
direction.

Smoke reports are outside the repository:

- `/workspace/alphadiamond-experiments/min-vacancy-realizability-smoke/head-100.json`
- `/workspace/alphadiamond-experiments/min-vacancy-realizability-smoke/full-100.json`

## Authoritative execution commands

After placing read-only iteration-100 artifacts under an experiment root, configure from a clean
commit and derive the three expected identities independently:

```bash
source /venv/main/bin/activate
TORCH_CUDA_ARCH_LIST=8.9 \
CMAKE_PREFIX_PATH=/venv/main/lib/python3.12/site-packages/torch/share/cmake \
  cmake --preset native-training
cmake --build build/native-training --target min_vacancy_realizability -j2
git rev-parse HEAD
jq -j -cS . "$CONFIG" | sha256sum
jq -r .model_digest "$CHECKPOINT/generations/$(tr -d '\r\n' < "$CHECKPOINT/CURRENT")/manifest.json"
```

Run both arms with identical seeds and counts, changing only `--arm` and output paths:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
build/native-training/native/min_vacancy_realizability \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --replay "$REPLAY_NAMESPACE" \
  --arm head \
  --device cuda:0 \
  --steps 1408 \
  --batch-size 256 \
  --eval-samples 16384 \
  --eval-batch 256 \
  --seed 20260902 \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-model-sha256 "$MODEL_SHA256" \
  --out "$EXPERIMENT/head.json" \
  --checkpoint-out "$EXPERIMENT/head-checkpoint"
```

Repeat with `--arm full`, `full.json`, and `full-checkpoint`. Only after those checkpoints exist
should the existing no-prior evaluation path run 256 games and report completion, p50/p90/p99,
seat balance, target-camp progress, and stagnation diagnostics. Production acceptance remains 97%.

## Verification

The focused suite covers config parsing, teacher reconstruction, arm freezing, optimizer-step
accounting, legal KL reduction on a realizable fixture, checkpoint/replay compatibility, and CLI
gates. Changed C++ files pass clang-format-18 and `git diff --check`.
