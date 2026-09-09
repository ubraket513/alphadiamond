# Min topology repair and fresh restart

The production model factory initialized the directional board adjacency buffer to zero. With the real Min encoder, legal self-to-empty moves consequently had identical logits. Scaling the network cannot recover board connections absent from its input computation.

Production now installs the generated authoritative board topology. Checkpoint continuation, warm start, and evaluation reject a mismatched topology rather than silently changing old weights. The bootstrap probe shares the same topology builder. Low-level model and checkpoint APIs retain support for synthetic test fixtures.

The integration regression runs the production CLI from scratch, inspects the archived topology, checks nonconstant legal logits and directional policy gradients on a real encoded Min opening, and verifies that exact resume rejects a checksum-valid zero-topology checkpoint. The test failed before the repair and passed after it. All 46 native tests passed with CUDA hidden; separate native CUDA teacher tests exercise the actual learner on the RTX 5090.

## Bounded parameter measurements

Measurements use native FP32 training on 512 training and 512 held-out positions from independent historical games, with native vacancy-prior teacher targets. Each candidate receives 51,200 examples. Selection requires held-out KL <= 0.05 and teacher expected-progress ratio >= 0.90 on two seeds. These are pressure-test criteria, not game strength or heuristic-removal acceptance criteria.

A 128-wide, six-block network with batch 1,024, eight Torch threads, and learning rate 0.001 was the fastest passing learner configuration tested. It averaged 7.85 seconds per fixed exposure budget across two seeds, versus 11.16 seconds for batch 256. Steady-step throughput was approximately 7,000 versus 4,800 samples/s. Batch 2,048 was slower and missed the quality threshold. Width 192 and eight blocks increased cost without a clear fit benefit. Production retains width 128 and six blocks; 352 steps of batch 1,024 preserve the previous 360,448 examples per iteration.

Full-game self-play concurrency measurements and final operational state are recorded in `/workspace/alphadiamond-experiments/min-restart-20260909/`. The underlying diagnosis, independent numerical audit, and native realizability pressure tests are in `/workspace/alphadiamond-experiments/min-policy-audit-20260909/`.

The four previous run directories were moved intact to `/workspace/alphadiamond-training/quarantine/20260909-zero-topology/`, with a manifest of original paths. Restart uses fresh weights and full vacancy-prior assistance. No claim of heuristic-free readiness follows from these measurements.
