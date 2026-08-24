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

## Promotion

```bash
python tools/promote_checkpoint.py checkpoints/soo/<run>/step-<n> --to candidate
python tools/promote_checkpoint.py checkpoints/soo/<run>/step-<n> --to promoted     --artifacts artifacts/soo-release
```

States move one step at a time, because `candidate` is where conversion is
gated. Promotion converts: it exports the deployment artifact and writes its
digests back into the manifest, so a `promoted` checkpoint whose artifact was
never built cannot exist. The checkpoint's own digest is verified first --
these paths are immutable, and one that was overwritten invalidates every
measurement recorded against it.

## CI fixtures

[manifests/ci-fixtures.json](manifests/ci-fixtures.json) names every large file
CI needs, with its SHA-256 and the bucket path it lives at. An entry carrying
`tracked_in_git` is a temporary exception -- the file still ships in the
repository -- and `tests/test_repo_hygiene.py` verifies its digest matches the
manifest, so the exception cannot rot into an accidental commit. When CI fetches
these from the bucket by path and digest, drop the `tracked_in_git` field and
`git rm` the file; the hygiene test then enforces its absence.
