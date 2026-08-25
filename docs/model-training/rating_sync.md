# Rating v2 Hugging Face synchronization

`alphadiamond-rating-sync` materializes `ratings/ratings.json` from immutable event files. It is deliberately separate from training: it reads no credentials and makes no network calls.

## Protocol input and output

The CLI accepts either a v2 `RatingRegistry::report_json()` document with `"schema_version": 2` added or the v2 envelope written by `save_rating_registry`. Its required registry fields are `family`, `protocol_id`, `protocol_config`, and `participants`. The rating-store loader validates the Elo/TrueSkill configuration, full participant identity, canonical identity hash, event protocol, event SHA-256 ID, and event semantics.

```sh
alphadiamond-rating-sync \
  --protocol protocol-v2.json \
  --events-dir rating-outbox/events \
  --output ratings.json
```

Every input event must be the canonical `SooRatingEvent::to_json()` or `MinRatingEvent::to_json()` payload and be named `<event_id-without-sha256-prefix>.json`. Distributed v2 events embed canonical participant identities; this lets a new device rebuild ratings for models and anonymous local opponents it has never seen before without trusting display text or a mutable participant list. The CLI rejects malformed names, duplicate/conflicting event IDs, corrupt IDs, identity/hash mismatches, unknown legacy participants, protocol mismatches, and invalid protocol configuration. Sortable game IDs define deterministic replay order. The output has `schema_version: 2`, an event-set SHA-256/count, protocol configuration, and one merged per-model row containing canonical full identity, display name, and Elo or TrueSkill values. Output promotion uses a same-directory temporary and atomic rename.

## Bucket wrapper

`tools/sync_ratings.sh` requires `hf auth whoami` to succeed. It uploads only absent immutable `ratings/events/<event_id>.json` files and byte-compares an existing remote file before accepting it. It never calls a delete/remove/sync-with-delete command.

```sh
tools/sync_ratings.sh \
  --bucket namespace/ratings-bucket \
  --outbox path/to/rating-outbox \
  --protocol protocol-v2.json \
  --binary build/native-training/alphadiamond-rating-sync
```

It lists the complete remote event set, downloads it, rebuilds locally, re-lists before and after publishing `ratings/ratings.json`, and retries when that set changed. Local outbox receipts are atomically written only after a stable successful `ratings.json` upload. Authentication uses the existing `hf` login or `HF_TOKEN`; neither tokens nor command output are persisted by this script.

Use `--dry-run` with a fake `hf` executable first in offline CI. It still checks authentication and downloads/materializes events, but performs no uploads and writes no receipts.
