# Min A0 Baseline — 2026-09-01

- Base commit: `589d6b452242f2a7d96a3277af76fec54582326d`
- Release: `min-v1.0.1`
- Checkpoint root: `/workspace/alphadiamond-experiments/min-v1.0.1/extracted/checkpoint/candidate-checkpoint`
- Replay root: `/workspace/alphadiamond-experiments/min-v1.0.1/extracted/replay/replay/persistent-replay-v2/min/d77681134fe31d231ff1002ef95c2d653bc0899046c5f2dbeccaaae3a5e83971`
- Effective config: `/workspace/alphadiamond-experiments/min-v1.0.1/extracted/run-state/active-config.json`
- Immutable source config: `/workspace/alphadiamond-experiments/min-v1.0.1/extracted/run-state/resolved-config.json`
- Effective config SHA-256: `524b888adcba2b0bbdd5aa87f4c82a80be57c6098e31da8ec70cb092f2dbc89b`

## Release artifact verification

The release `SHA256SUMS` file embeds its original absolute staging paths, so
`sha256sum -c SHA256SUMS` cannot resolve files downloaded elsewhere. Comparing
the recorded digests to the downloaded basenames produced exact matches:

```text
61189bc2b9307ab74a252c8343b463ebfd6a0379a27935defda8f205b19f405b  latest.tar.gz
da2c353160d5e0b5271b1f1487063aa78a549dfc6bff5dd2456bafeda4888d0e  replay.tar.gz
6f11693f9b03bc4d7fdbc24bbba91925662907905bf8ebdae7b0782598b81d27  run-state.tar.gz
```

## Effective iteration-25 config

The released `resolved-config.json` is deliberately immutable and contains the
run's original 768-game, 50-microsecond, 1,024-step settings. Transition records
23 and 25 resolve to `active-config.json`, which is the actual iteration-25 FP32
operating point used for diagnostics below.

```json
{"arena":{"enabled":false,"games":36,"max_moves":800,"promotion_threshold":0.55000000000000004,"seed":7},"inference":{"max_batch_size":256,"max_wait_us":100,"request_queue_capacity":1024,"response_timeout_s":600.0},"mcts":{"c_puct":1.5,"dirichlet_alpha":0.29999999999999999,"dirichlet_epsilon":0.25,"repeat_window":0,"seed":7,"simulations":128,"simulations_late":0},"model_name":"Min","model_version":"2.0.0","network":{"residual_blocks":6,"width":128},"opening_suite":{"count":1,"id":"production-openings-v1","max_depth":6,"seed":7,"version":1},"promotion_statistics":{"bootstrap_replicates":10000,"confidence_level":0.94999999999999996,"method":"opening-block-bootstrap-v1","resampling_unit":"opening_block","seed":7},"replay":{"capacity":1000000,"seed":7},"run_budget":{"checkpoint_every_iterations":1,"max_iterations":250,"max_wall_clock_seconds":21600.0},"run_seed":7,"runtime":{"device":"cuda:0","precision":"fp32"},"schema_version":2,"self_play":{"bootstrap_prior":"canonical-target-vacancy-distance-v2","max_game_seconds":null,"max_moves":500,"seed":7,"temperature":1.0,"temperature_moves":20},"training":{"batch_size":256,"learning_rate":0.001,"seed":7,"train_steps_per_iteration":1408,"weight_decay":0.0001},"workers":{"games_per_iteration":1024,"logical_lanes":512,"retry_id":"attempt-0","search_threads":16}}
```

## Checkpoint validation

```text
format_version=3
generation=/workspace/alphadiamond-experiments/min-v1.0.1/extracted/checkpoint/candidate-checkpoint/generations/generation-v3-27776-2
training_step=27776
run_id=min-b0-6h-20260831
iteration=25
model_step=27776
optimizer_restored=true
valid=true
```
