# The replay store is a durable dataset with a stateless sampler

## What changed

`ReplayStore::sample()` used to be a durable transaction. Every minibatch
advanced a persisted RNG, wrote a `selection_transaction` record naming every
selected row, and rewrote `manifest.json` — once per training step. A TRAIN
stage of 1024 steps therefore performed 1024 manifest writes, each one
serialising and hashing every chunk descriptor in the store, and the manifest
digest changed under a stage that logically only reads.

Sampling is now a pure function:

```text
samples = replay.sample(batch, replay_sampling_seed(replay_seed, iteration, step))
```

It reads memory and copies rows. It opens no file, mutates no sampler state,
and can be called any number of times in any order.

## Why this is safe for resume

The old design persisted sampler position so that a resumed TRAIN could
continue the draw sequence. It does not need to: TRAIN has no partial-credit
resume. If the candidate checkpoint is absent, the stage re-runs from step 0.

The seed is derived from the run's `replay.seed`, the iteration index and the
*local* training step, so a stage re-run from step 0 against unchanged replay
contents reproduces the identical minibatch sequence, and the resumed run's
final model matches an uninterrupted one. Determinism comes from the seed
derivation, not from durable state — which is why the durable state was
removable without weakening the contract.

## The manifest is contents identity only

`manifest.json` is schema 4 and answers exactly one question: which samples
does this store hold, under which compatibility, at which capacity.

```text
aborted  capacity  chunks  compatibility  game_ids  schema_version
```

Gone: `rng`, `selection_transaction`, `ingest_transaction`. Training cannot
change the manifest, so its digest — which checkpoint provenance records and
`validate_checkpoint_context` enforces — is stable across a TRAIN stage by
construction rather than by convention.

Schema 2 and 3 manifests still load; their sampler and transaction fields are
read past and dropped on the next write. Schema 1 (`persistent-replay-v1`)
still loads and migrates, because the frozen fixtures under
`tests/golden/replay-v1/` are v1 and are normative — see the oracle rule in
`CLAUDE.md`. The CPython MT19937 sampler state those fixtures carry is now
ignored: it described a sampling stream that no longer exists.

## Consequence for tests

`replay_schema_test` used to pin the *order* in which the two fixture rows came
back, which was a property of the retired MT19937 stream rather than of the
stored contents. It now pins the set. The fixture bytes are untouched.
