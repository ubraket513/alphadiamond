# Min iteration-25 learning diagnostic — 2026-09-01

## Result

Classification: `LEARNING_BUT_NOT_BEHAVIORALLY_READY`.

All losses, gradients, updates, KL values, and outputs were finite. Both policy
heads had non-zero gradients and updates at every reported step. Their mean
relative updates were `3.18e-6` (source) and `2.88e-6` (destination), well above
the `1e-8` effective-step gate, while held-out full KL remained `3.07686` nat.
The learner therefore is neither disconnected nor numerically broken.

The behavioral movement is nevertheless negligible: start-to-end policy KL was
only `2.83e-11`, full cross-entropy and top-1 agreement were unchanged, while
raw logit RMS moved by `24.5961`. This is consistent with updates dominated by a
softmax-invariant common-logit direction. The raw value is retained here because
the roadmap's ordered rules do not set a minimum for “held-out policy KL moves.”
It is not evidence that the current checkpoint is A0-ready.

## Inputs and immutability

- Checkpoint step: `27776`
- Model SHA-256: `847007a72b0a283a1789c7cb160f9fd93864117ad1cdbf5bbf3541c92f5ef59a`
- Replay samples: `1000000`
- Replay manifest SHA-256: `dc0ce2efea60fc4b4b3ce1a21fe36f60bf7ff5b0fb916489a127e1988ae3ee59`
- Config: `/workspace/alphadiamond-experiments/min-v1.0.1/extracted/run-state/active-config.json`
- Release archives after both runs: `latest.tar.gz` `61189bc2...f405b`,
  `replay.tar.gz` `da2c3531...88d0e`, `run-state.tar.gz` `6f11693f...81d27`

The smoke exposed a CLI path-contract mismatch: the release environment points
directly at the replay namespace, while `ReplayStore` expects its parent root.
The CLI now normalizes either form before construction. The empty nested replay
created by the failed attempt was removed; it contained only a new 601-byte
manifest and no samples. The source archives and authoritative namespace were
unchanged.

## Commands

Both commands used `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, the
release checkpoint/config/replay environment variables, `--device cuda`,
`--iteration 26`, `--batch-size 256`, and `--seed 20260901`.

```text
min_learning_diagnostic --steps 2 --eval-samples 512 --eval-batch 256 \
  --log-every 1 --out $MIN_A0_EXP/results/learning-smoke.json

min_learning_diagnostic --steps 256 --eval-samples 4096 --eval-batch 256 \
  --log-every 32 --out $MIN_A0_EXP/results/learning.json
```

## Held-out metrics

| metric | initial | final | delta |
|---|---:|---:|---:|
| target entropy | 2.986928 | 2.986928 | 0 |
| full cross-entropy | 6.063785 | 6.063785 | 0 |
| full KL | 3.076857 | 3.076857 | 0 |
| top-1 agreement | 0.017334 | 0.017334 | 0 |
| value MSE | 0.597459 | 0.597581 | 0.000122 |

Additional drift: policy KL `2.83092e-11`, logit RMS delta `24.5961`, and value
RMS delta `0.0104188`.

## Parameter-group summaries

The table summarizes steps 1, 2, every 32 steps, and step 256.

| group | grad min | grad mean | grad max | update min | update mean | update max | relative-update mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| input_projection | 0.00049169 | 0.00147893 | 0.00314692 | 0.000756468 | 0.001376 | 0.00247296 | 0.000180719 |
| residual_trunk | 0.000895688 | 0.00209723 | 0.00459066 | 0.0000894897 | 0.000189251 | 0.000309039 | 0.0000038883 |
| last_residual_block | 0.000234809 | 0.000726616 | 0.00174877 | 0.0000435607 | 0.0000827037 | 0.000132885 | 0.00000379125 |
| output_norm | 0.000499314 | 0.00112323 | 0.00255677 | 0.0000260358 | 0.0000470725 | 0.0000815309 | 0.00000357533 |
| policy_source | 0.0000746834 | 0.0000896939 | 0.000120386 | 0.0000722785 | 0.0000830182 | 0.000088499 | 0.00000317653 |
| policy_destination | 0.0000686831 | 0.0000835923 | 0.0000980232 | 0.0000642419 | 0.0000750256 | 0.000082236 | 0.00000288038 |
| value_hidden | 0.00166017 | 0.00526631 | 0.0130208 | 0.00186353 | 0.00553655 | 0.0200592 | 0.000395829 |
| value_output | 0.0234781 | 0.0557344 | 0.0813525 | 0.000320577 | 0.000771567 | 0.00123715 | 0.000429013 |

Reported policy loss stayed in `[6.0637846, 6.0637851]`; reported value loss
ranged from `0.589556` to `0.621530` across retained steps. The next gate remains
the paired serial-search sweep; Min remains B0.
