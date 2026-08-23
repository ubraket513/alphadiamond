# TrainAlphaDiamond

Local synchronisation root for the Hugging Face storage bucket
`hf://buckets/ubraket513/AlphaDiamond`.

**Nothing under this directory is tracked by Git except this README and
`manifests/`.** The bucket is mutable, non-versioned object storage: it holds
checkpoints, replay/self-play data, logs and intermediate training data. Git
holds source, documentation and the manifest schema.

## Layout

```text
TrainAlphaDiamond/
  README.md                    tracked
  manifests/                   tracked  (schema + release index)
  checkpoints/                 ignored  soo/<run-id>/step-<n>/
  datasets/                    ignored  selfplay/ replay/ fixtures/
  releases/                    ignored  soo/ min/
  logs/                        ignored
  staging/                     ignored
```

## Sync

```bash
make data-push-dry-run    # show what a push would change
make data-push
make data-pull-dry-run
make data-pull
```

No target passes `--delete`. Deleting bucket content is a separate, manual
operation performed after reviewing a dry-run plan.

## Immutable paths

`hf sync` overwrites in place and the bucket keeps no history, so a path like
`latest.pt` is not an identity. Checkpoints live at immutable paths:

```text
checkpoints/soo/soo-scratch-20260822/step-00044250/checkpoint.pt
checkpoints/soo/soo-scratch-20260822/step-00044250/manifest.json
```

`latest.json` may *point at* one of those paths. Release and conversion tooling
always consumes the immutable path and verifies `checkpoint_sha256` first.

Manifest fields: see [manifests/checkpoint.schema.json](manifests/checkpoint.schema.json).
