# Deployment artifact format v3

Written 2026-08-24, implemented the same day. v2 hardcoded Soo model version
`2.0.0`, width 128 and six residual blocks, so a second model family could not
be described without changing the validator. v3 declares the family and the
architecture, and the loader checks the weights against the declaration --
which is what made adding Min a model port rather than a format redesign.

Written by `tools/export_deployment.py --family soo|min`; validated by
`diamond_model::validate_deployment_artifact` and
`diamond.alphazero.deployment.validate_metadata`.

## Training checkpoint is not a deployment artifact

| | Training checkpoint | Deployment artifact |
|---|---|---|
| Holds | weights, optimizer, scheduler, RNG, replay bookkeeping | weights and metadata only |
| Lives in | the bucket, at an immutable path | a release package, beside the executable |
| Consumed by | the trainer | the native application |

The application never bundles a training checkpoint.

## Metadata

```json
{
  "format_version": 3,
  "model_family": "soo",
  "model_version": "2.0.0",
  "architecture": {
    "type": "directional_residual",
    "width": 128,
    "residual_blocks": 6
  },
  "game_contract": {
    "topology": "diamond73-v1",
    "encoder": "diamond-camp-relative-v1",
    "action_space": "diamond73-srcdst-v1"
  },
  "source": {
    "checkpoint_sha256": "...",
    "training_commit": "...",
    "training_step": 44250
  },
  "runtime_sha256": "..."
}
```

Validation stays as strict as v2 is today — the artifact contract test
deliberately corrupts metadata and tensors and requires rejection — but the
strictness moves from *hardcoded Soo constants* to *the declared architecture*:
the loader checks the weights against `architecture`, and checks
`game_contract` against what the binary implements.

## Packaging

Bundled beside the executable, not compiled into it, so a model can be replaced
without relinking:

```text
AlphaDiamond/
  bin/diamond_qt
  models/
    index.json          default model per family, and what else is present
    soo/2.0.0/{metadata.json,weights/,...}
  licenses/
```

The application reads `models/index.json`, validates the chosen artifact and
verifies `runtime_sha256` before loading it.

## One representation, eventually

v2 writes both a TorchScript graph and raw weight tensors. That was right while
native inference was being proved against LibTorch; in a release package it
stores every model twice. v3 ships exactly one canonical runtime
representation, and keeps the second only as a conversion or diagnostic fixture
if it is still earning its place.

## Min

`DiamondModel` (the former `SooModelImpl`, generalised) takes the input-feature
count and value-head width from the artifact, so one implementation serves both
families: Soo is 4 features and a scalar value, Min is 6 features and one value
per seat. `model_parity_test` runs the artifact's own deterministic corpus
through the native model for each family and compares against the outputs
PyTorch produced at export time; both currently match exactly.

Still open: promotion policy -- preserve every checkpoint in the bucket,
convert release candidates during promotion, publish only accepted artifacts,
bundle one default per family.
