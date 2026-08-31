# Binary Replay Segments Design

## Goal

Remove JSON sample serialization and parsing from new replay writes while preserving existing schema 4 runs, atomic recovery, deterministic duplicate detection, and the 1,000,000-sample capacity.

## Chosen design

Schema 5 keeps the small manifest as canonical JSON and stores every completed episode as a content-addressed binary segment. A manifest descriptor gains `encoding: "binary-v1"`; schema 4 descriptors without an encoding remain `json-v1`. Readers accept schema 4 and 5, so an existing run may contain old `.json` chunks and new `.bin` segments. New writes use only binary-v1. No eager migration is required; ordinary replay pruning eventually removes unreachable JSON chunks.

Each segment uses explicitly little-endian integers and IEEE-754 float32 values. It contains:

1. A fixed header: eight-byte magic `ADRPBIN1`, format version, flags, header byte count, sample count, episode metadata byte count, payload byte count, and compatibility digest.
2. Length-prefixed UTF-8 episode metadata: game ID and retry ID, followed by seed and move count.
3. A fixed-size sample index containing the offset and byte length of every sample record.
4. Sample records containing length-prefixed canonical player IDs, node features, sparse-policy `(action, probability)` pairs, and value targets.
5. A 32-byte SHA-256 footer over all preceding bytes.

Lengths and offsets are checked before allocation or pointer arithmetic. Counts are bounded by the segment byte size and by existing model/action-space constraints. NaNs and infinities are rejected. Compatibility is checked against the namespace before decoded samples enter the replay pool.

## Transactions and identity

The digest of the complete binary segment is its filename and manifest identity. Writers build bytes once, compute the digest once, write `<digest>.bin.tmp`, flush/close, and atomically rename it to `<digest>.bin`. Only then is the manifest atomically replaced. A crash before manifest replacement leaves an unreachable segment, which startup cleanup removes. A failure before manifest commit rolls in-memory vectors back by size; it never snapshots the 1M-sample pool.

Duplicate game IDs compare the incoming encoded payload digest with the recorded digest. Old JSON chunks retain their canonical-JSON digest behavior. Cleanup resolves the file extension from descriptor encoding and must not delete reachable mixed-format chunks.

## Rollout

Phase 1 decodes binary-v1 into existing `TrainingSample` objects, minimizing training-path risk while eliminating JSON construction/parsing. Phase 2 may introduce mmap-backed sample views only after profiling shows decoding or resident copies are again dominant. The current Min run switches binaries only at a durable iteration boundary and resumes from its checkpoint; it never restarts from scratch.

## Acceptance

- Golden binary bytes are deterministic across repeated encodes.
- Encode/decode round trips every field exactly.
- Truncation, corrupt lengths, checksum mismatch, non-finite floats, and compatibility mismatch fail closed.
- Schema 4 JSON replay reopens unchanged.
- A schema 5 mixed JSON/binary replay reopens, samples deterministically, prunes correctly, and recovers orphan files.
- The complete native suite passes.
- Timed ingest/reopen benchmarks report JSON schema 4 versus binary-v1 on the same episode pool; binary becomes the default only with no correctness regression.
