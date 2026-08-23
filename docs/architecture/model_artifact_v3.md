# Deployment artifact format v3 (proposed)

Written 2026-08-24. Not implemented: `tools/export_soo_deployment.py` still
writes v2, which hardcodes Soo model version `2.0.0`, width 128 and six
residual blocks. This record fixes the shape v3 must have *before* Min gets a
native inference path, so that adding Min is not also a format redesign.

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

## Open before Min can be bundled

* `native/src/soo_model.cpp` / `soo_evaluator.cpp` are Soo-specific; Min needs
  its own native inference and its own artifact contract test.
* Promotion policy: preserve every checkpoint in the bucket, convert release
  candidates during promotion, publish only accepted artifacts, bundle one
  default per family.
