# Native replay-store loading decision

The Python-zero replay baseline used a full manifest reload and then selected a
bounded in-memory replay window.  The native store keeps the same observable
contract: descriptors remain authoritative on disk, while the loaded sample
window is bounded by the persisted capacity.  This deliberately favors a
simple, validated manifest scan over a second persistent index; replay chunks
are immutable and the capacity window bounds training memory.

This decision is recorded with the PR06 migration because changing it would
alter restart, pruning, and deterministic sampling behavior.
