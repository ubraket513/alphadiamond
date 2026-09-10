# Min topology repair and fresh restart

The production model factory initialized the directional board adjacency buffer to zero. With the real Min encoder, legal self-to-empty moves consequently had identical logits. Scaling the network cannot recover board connections absent from its input computation.

Production now installs the generated authoritative board topology. Checkpoint continuation, warm start, and evaluation reject a mismatched topology rather than silently changing old weights. The bootstrap probe shares the same topology builder. Low-level model and checkpoint APIs retain support for synthetic test fixtures.

The integration regression runs the production CLI from scratch, inspects the archived topology, checks nonconstant legal logits and directional policy gradients on a real encoded Min opening, and verifies that exact resume rejects a checksum-valid zero-topology checkpoint. The test failed before the repair and passed after it. All 46 native tests passed with CUDA hidden; separate native CUDA teacher tests exercise the actual learner on the RTX 5090.

## Bounded parameter measurements

Measurements use native FP32 training on 512 training and 512 held-out positions from independent historical games, with native vacancy-prior teacher targets. Each candidate receives 51,200 examples. Selection requires held-out KL <= 0.05 and teacher expected-progress ratio >= 0.90 on two seeds. These are pressure-test criteria, not game strength or heuristic-removal acceptance criteria.

A 128-wide, six-block network with batch 1,024, eight Torch threads, and learning rate 0.001 was the fastest passing learner configuration tested. It averaged 7.85 seconds per fixed exposure budget across two seeds, versus 11.16 seconds for batch 256. Steady-step throughput was approximately 7,000 versus 4,800 samples/s. Batch 2,048 was slower and missed the quality threshold. Width 192 and eight blocks increased cost without a clear fit benefit. Production retains width 128 and six blocks; 352 steps of batch 1,024 preserve the previous 360,448 examples per iteration.

Full-game self-play concurrency measurements and final operational state are recorded in `/workspace/alphadiamond-experiments/min-restart-20260909/`. The underlying diagnosis, independent numerical audit, and native realizability pressure tests are in `/workspace/alphadiamond-experiments/min-policy-audit-20260909/`.

The four previous run directories were moved intact to `/workspace/alphadiamond-training/quarantine/20260909-zero-topology/`, with a manifest of original paths. Restart uses fresh weights and full vacancy-prior assistance. No claim of heuristic-free readiness follows from these measurements.

## Self-play selection and startup gate

A full fresh-model trial completed all 1,024 games at 128 simulations per move: 122,659 samples in 1,063.21 seconds, or 115.37 samples/s. No games aborted. Its 17.7-minute duration was excessive for initial tuning, so subsequent full-game trials were cancelled and replaced with eight-move timing trials.

The short trials compared 16, 64, and 88 search threads, inference batches of 256 and 512, wait limits of 100 and 500 microseconds, and FP32/BF16/FP16 actors. More search threads alone changed throughput by approximately 1%. The fastest tested two-seed configuration used 512 lanes, 88 search threads, batch 512, a 500-microsecond wait, and FP16 actors: 192.61 moves/s. BF16 reached 187.62 moves/s. Learner parameters and updates remain FP32. The config `min-topology-5090-v1.json` records this provisional selection. Short opening throughput does not establish full-game quality with trained values.

Historical production was faster: retained iteration 25 generated 125,542 samples in 276.37 seconds, about 454 samples/s, and its learner reported a median 11,129 samples/s. The 46% learner improvement above compares batch sizes on the current environment; it is not an improvement over historical production.

A sustained independent matrix-multiplication probe stayed at 217–225 MHz GPU clocks despite 100% kernel activity. Resetting GPU clocks was denied inside the container. No GPU settings or host drivers were changed. Restricting work to the GPU's local NUMA node did not materially improve the short trial. Exact host clock configuration remains unknown.

The user requires resolving this slowdown before actual training starts. Production training therefore remains paused. The supervisor wrapper and config are prepared, but the service is not registered. Arena is disabled unless a specific decision makes it necessary. The absolute goal deadline is 2026-09-10 03:18:15 UTC; the wrapper refuses to start after it and stops a running child at the deadline.

One million samples is the replay capacity, not a startup fill threshold. The learner requires only one full batch to be available. Self-play timing does not load replay data. Inference stage timings use host clocks unless `DIAMOND_EVAL_STAGE_TIMING` is enabled; the reported device-to-host span can therefore include waiting for prior GPU work. It must not be interpreted as transfer bandwidth alone.
