You are the Chief AI Architect and senior ML/systems engineer responsible for
Milestone 2 of the AlphaDiamond project.

You are starting this task from ZERO conversational context.

Do not assume access to any previous discussion, prompt, plan, or unstated
requirement.

Repository:

    ubraket513/alphadiamond

======================================================================
0. CURRENT PROJECT STATUS
======================================================================

The repository implements Diamond.

It is NOT standard Chinese Checkers.

The CURRENT repository game engine is authoritative for all game semantics.

The AlphaZero Milestone-1 Python/PyTorch reference implementation is already
implemented and review-hardened.

Before changing anything, inspect the current main branch and verify this
status yourself.

The most recently recorded Milestone-1 status was:

    AlphaZero tests:
        99 passed

    Existing Diamond game/GUI tests:
        134 passed, 5 skipped

    Combined:
        233 passed, 5 skipped

    AlphaZero executable smoke:
        exit code 0

The current repository also records that a local Windows PyTorch mamba
environment had a PySide6.QtCore DLL loading mismatch, preventing one unified
interpreter from collecting both suites, while both suites passed separately
in working interpreters.

======================================================================
IMPLEMENTATION STATUS / EVIDENCE — TASK 14, 2026-08-19
======================================================================

Milestone 2 implementation is complete against the requirements below. This
section records evidence only; it does not alter the authoritative design.

- Independently versioned Soo and Min checkpoint artifacts are compatibility
  gated. Rating events bind artifact, compatibility, and benchmark-protocol
  identities.
- Soo uses Elo. Min uses official TrueSkill with `tau=0` for immutable
  checkpoints; a Min rating event requires three distinct artifacts, and the
  conservative exposure is the leaderboard key.
- Promotion and historical rating are separate: the promotion arena is 18
  games, while the Min historical rating schedule is 36 games.
- The versioned opening suite includes the standard initial position and
  authoritative legal action sequences; fixed search configuration and its
  opening/protocol identity prevent cross-namespace rating mixtures.
- Headless `train`, `resume`, `benchmark`, `leaderboard`, and `profile`
  commands; atomic run/stage artifacts; and idempotent replay ingestion are
  implemented. The CPU smoke exercises real Soo and Min workflow stages.
- Task 14 verification recorded 287 passed, 1 skipped for AlphaZero; 134
  passed, 5 skipped for the Qt suite; both smoke modules exited 0; and the
  MCTS import guard passed. See `task-14-report.md` for commands and outputs.
- This host has no `nvidia-smi`, CUDA device, or A30. The CPU profile therefore
  records `gpu_verified=false` and no GPU rows. There are no production A30
  stage percentages, so no C++ implementation is recommended. Reconsider only
  with reproducible production A30 evidence of CPU search/game/tree dominance.

Deferred: C++ implementation, OpenVINO, GUI routing, and multi-node training.

Do not simply trust these numbers.

Re-run the relevant tests in the environment available to you and record the
actual baseline.

Milestone 1 is now an ORACLE.

Do not redesign or casually rewrite it.

======================================================================
1. OFFICIAL MODEL IDENTITIES
======================================================================

There are two independently versioned learned models.

    Soo = 2-player Diamond AlphaZero model / agent

    Min = 3-player Diamond AlphaZero model / agent

Soo and Min version numbers are unrelated.

Examples:

    Soo 0.3.0
    Min 0.1.2

does NOT imply either model should have a matching version.

Preserve the current ModelIdentity and checkpoint compatibility architecture.

Current value semantics are:

Soo:

    current-player-scalar-winloss-v1

    scalar value
    current-player perspective
    +1 win
    -1 loss

Min:

    canonical-placement-utility-1-0-minus1-v1

    vector:
        [self, next, previous]

    final placement:
        first  = +1
        second = 0
        third  = -1

Never replace Min with winner-probability classification.

Never apply scalar sign negation to Min MCTS values.

======================================================================
2. EXISTING COMPATIBILITY CONTRACT
======================================================================

The current Milestone-1 checkpoint system is expected to gate compatibility
using semantics equivalent to:

    model_name
    model_version
    player_count
    ruleset_version
    board_topology_version
    ruleset_fingerprint
    encoder_version
    action_space_version
    seat_layout_version
    value_semantics_version
    network_config

Do not weaken these checks.

The current Diamond topology is expected to be:

    73 holes

with action identity:

    source -> final destination

and:

    action_id = source * 73 + destination
    action_size = 5329

Verify this against the current repository.

If current repository semantics differ, the repository wins.

======================================================================
3. MILESTONE 2 NAME AND OBJECTIVE
======================================================================

Milestone 2 is:

    PRODUCTION TRAINING & STRENGTH EVALUATION

Milestone 1 proved mathematical and game correctness.

Milestone 2 must make Soo and Min practical to:

    generate substantial self-play
    train repeatedly
    save and resume safely
    evaluate objectively
    track historical model strength
    promote candidates
    use an NVIDIA A30 efficiently
    measure actual bottlenecks

Milestone 2 is NOT primarily a neural-network redesign.

Do not increase model width/depth merely because Milestone 2 has started.

The existing network is the baseline.

======================================================================
4. OFFICIAL STRENGTH METRICS
======================================================================

The official model-strength systems are now fixed:

    Soo -> Elo

    Min -> TrueSkill

These are separate rating universes.

Never compare a Soo Elo number numerically with a Min TrueSkill value.

Model version, training step, and strength rating are separate concepts.

Example:

    Soo 0.4.0
        checkpoint step: 1,500,000
        Elo: 1278

    Min 0.2.0
        checkpoint step: 900,000
        TrueSkill:
            mu: ...
            sigma: ...
            exposure: ...

The version identifies model lineage/compatibility.

The checkpoint identifies immutable learned weights at a training step.

The rating measures demonstrated playing strength.

======================================================================
5. TRUE SKILL IMPLEMENTATION FOR MIN
======================================================================

Use the official Python package:

    trueskill

documented by the official TrueSkill Python project.

Do NOT implement a home-grown approximation of TrueSkill.

Add the dependency only to the AlphaZero/training dependency surface.

Do not make GUI-only users install unnecessary rating/training dependencies
unless existing packaging conventions require it.

Use an explicit TrueSkill environment.

Do NOT rely on mutable process-global TrueSkill configuration.

Initial v1 environment should conceptually use:

    mu = 25.0
    sigma = 25.0 / 3.0
    beta = 25.0 / 6.0
    tau = 0.0
    draw_probability = 0.0
    backend = default/internal

CRITICAL:

    tau = 0.0

Reason:

A Min checkpoint is immutable.

Its skill does not drift over time like a human player's ability can.

A non-zero dynamic factor would incorrectly model an immutable checkpoint as
changing skill between matches.

Diamond also produces a strict full ranking:

    first
    second
    third

so:

    draw_probability = 0.0

for the current rating protocol.

Persist ALL TrueSkill environment parameters as part of the rating-system
identity.

Changing any parameter creates a NEW rating namespace.

Do not mix ratings produced by different TrueSkill configurations.

======================================================================
6. MIN TRUE SKILL MATCH SEMANTICS
======================================================================

A rated Min game is a three-player free-for-all.

Given distinct model checkpoints:

    A
    B
    C

and final order:

    B first
    C second
    A third

use TrueSkill free-for-all semantics equivalent to:

    groups:
        [(rating_A,), (rating_B,), (rating_C,)]

    ranks:
        [2, 0, 1]

Lower rank is better.

Use the package's native free-for-all rating update.

Do NOT decompose the match into:

    B beats C
    B beats A
    C beats A

for rating purposes.

That discards the native multiplayer structure and is not the approved design.

======================================================================
7. IMPORTANT MIN RATING RESTRICTION
======================================================================

A TrueSkill-rated Min match MUST contain THREE DISTINCT checkpoint identities.

For example:

    Min checkpoint A
    Min checkpoint B
    Min checkpoint C

is valid.

This is NOT valid for TrueSkill rating:

    candidate
    champion
    champion

because the same immutable model appears twice as two independent competitors.

The existing Min candidate-vs-champion-vs-champion arena may remain useful as
a PROMOTION evaluation.

However:

    candidate/champion/champion games MUST NOT be written into the TrueSkill
    rating event stream.

Keep:

    promotion evaluation

and:

    historical TrueSkill league rating

as separate concepts.

This distinction must be explicit in code and tests.

======================================================================
8. MIN LEADERBOARD VALUES
======================================================================

Persist for each rated Min checkpoint:

    mu
    sigma

and compute the official conservative leaderboard exposure using the
TrueSkill environment's own:

    expose(rating)

API.

Do not duplicate its formula unless required for serialization tests.

The primary Min leaderboard sort key is:

    exposure

while reports should show:

    mu
    sigma
    exposure
    rated games

Do not display mu alone as if it represented rating confidence.

======================================================================
9. SOO ELO
======================================================================

Implement Soo Elo directly.

No third-party Elo dependency is necessary.

Create a clean, independently tested implementation.

The default v1 protocol should be parameterized equivalent to:

    initial_rating = 1000.0
    k_factor = 32.0
    logistic_scale = 400.0

For players A and B:

    expected_A =
        1 / (1 + 10 ** ((rating_B - rating_A) / logistic_scale))

and:

    new_A =
        rating_A + K * (score_A - expected_A)

Use the PRE-MATCH ratings for both updates.

Do not update A and then calculate B using A's already-updated rating.

Soo current Diamond has no valid draw result.

Completed outcomes are:

    win  -> 1.0
    loss -> 0.0

Aborted games produce:

    NO Elo update

If Diamond rules later introduce a real draw, that is a ruleset change and
must be designed explicitly rather than silently treating an abort as a draw.

======================================================================
10. RATING SYSTEM VERSIONING
======================================================================

Model versions and rating-system versions are independent.

Introduce explicit identities similar to:

    soo-elo-v1
    min-trueskill-v1

A rating namespace must also bind to the benchmark protocol.

Changing important evaluation semantics must produce a new rating namespace.

Examples:

    search simulation budget changes
    opening suite changes
    MCTS decision temperature changes
    value/ruleset compatibility changes
    Elo K/scale changes
    TrueSkill environment changes

Historical numbers produced under different protocols must never silently mix.

======================================================================
11. BENCHMARK PROTOCOL IDENTITY
======================================================================

Define a serializable BenchmarkProtocol.

It should include enough information to determine whether two ratings are
comparable.

At minimum consider:

    model_name
    player_count

    ruleset_version
    ruleset_fingerprint
    board_topology_version
    encoder_version
    action_space_version
    seat_layout_version
    value_semantics_version

    MCTS simulations per move
    c_puct

    root Dirichlet noise:
        MUST be disabled for ratings

    decision temperature:
        normally 0.0 after benchmark opening

    max game moves

    opening_suite_version / opening_suite_hash

    rating_system_version

    rating-system parameters

    inference numeric mode if it can affect semantics

Derive a deterministic:

    benchmark_protocol_id

or fingerprint.

Do not mix match events produced by different benchmark protocol IDs.

======================================================================
12. FIXED COMPUTE IS MANDATORY FOR RATINGS
======================================================================

A strength rating is meaningless if different models receive different search
budgets.

All models in one rating namespace must use the same:

    simulations per move
    search constants
    root-noise policy
    benchmark opening protocol
    inference semantic mode

Do NOT rate:

    model A at 200 simulations

against:

    model B at 800 simulations

and call the result model strength.

Search compute belongs to the benchmark protocol.

======================================================================
13. BENCHMARK DIVERSITY / OPENING SUITE
======================================================================

The current deterministic arena starts from the standard initial position.

That is appropriate as an important promotion test, but insufficient by itself
for a long-lived rating system.

Repeated deterministic games from exactly the same:

    checkpoint pairing
    seats
    turn order
    initial state
    no-noise search

may produce identical games and therefore misleading repeated evidence.

Implement a small VERSIONED deterministic benchmark opening suite.

Do not modify Diamond rules.

An opening is represented as an authoritative sequence of legal action IDs
applied from the standard initial state.

The suite must be:

    deterministic
    ruleset-bound
    validated through the authoritative engine
    reconstructible
    versioned / hashed
    independent from candidate model behavior

A reasonable v1 implementation may generate a small collection of openings
using fixed RNG seeds and authoritative uniformly sampled legal actions at
small opening depths.

Do not hard-code serialized GameState internals if action sequences can safely
reconstruct the state.

Include the standard initial state as one benchmark opening.

Tests must verify every stored/generated opening remains legal under its
bound ruleset.

If the current repository structure makes a simpler equivalent design safer,
document and use it.

======================================================================
14. BALANCED SOO RATING SCHEDULE
======================================================================

Two distinct Soo checkpoint artifacts:

    A
    B

must be evaluated with balanced:

    physical seat assignment
    turn order

The minimum complete geometric/order balance cycle remains:

    2 seat assignments
        x
    2 turn orders

    = 4 games per opening

Do not accept partial balance cycles as rated evidence.

For N benchmark openings:

    complete pair cycle = 4 * N games

The event scheduler must be deterministic.

======================================================================
15. BALANCED MIN RATING SCHEDULE
======================================================================

For THREE DISTINCT Min checkpoints:

    A
    B
    C

rating must balance:

    all six assignments of A/B/C to the three physical seats

and:

    all six authoritative turn-order permutations

Therefore one complete balance cycle per opening is:

    6 seat assignments
        x
    6 turn orders

    = 36 rated games

This is different from the existing Min promotion arena.

The existing promotion arena can use:

    candidate seat 3
        x
    turn order 6
        =
    18 games

because the other two players use the same baseline model.

TrueSkill league rating has three distinct participants, so its full balance
cycle is 36.

Do not confuse these two numbers.

Tests must explicitly protect this distinction.

======================================================================
16. RATING EVENT LOG
======================================================================

Do not make rating state an opaque mutable number with no audit trail.

Implement an append-only semantic rating event log.

A rated match event should contain enough information to reproduce rating
state.

Conceptually include:

    event_id

    timestamp or deterministic sequence index

    benchmark_protocol_id

    model participant checkpoint IDs

    model names
    model versions
    training steps
    checkpoint hashes

    seat assignment
    turn order
    opening_id

    completed / aborted

    outcome:
        Soo winner/loser
        Min full final ranking

    relevant search configuration

Do not update rating for aborted games.

Events must have stable IDs.

Reprocessing the same event twice must not double-update ratings.

Use either:

    explicit duplicate rejection

or:

    idempotent processing

but test it.

======================================================================
17. RATING REGISTRY
======================================================================

Implement a persistent rating registry.

Do not use an external database for Milestone 2 unless the repository already
has one and there is a compelling reason.

Local structured files are sufficient.

The registry must support:

    add immutable checkpoint participant
    record rated match event
    rebuild ratings from events
    inspect leaderboard
    persist
    reload
    deterministic replay

The source of truth should preferably be:

    participant metadata
    +
    append-only event history

with current rating state treated as derived/cacheable data.

If the rating cache becomes corrupted, replaying events should recreate the
same result.

Tests must verify this.

======================================================================
18. CHECKPOINT PARTICIPANT IDENTITY
======================================================================

Do not assume:

    model_version alone

uniquely identifies a rated checkpoint.

A long training run may create multiple checkpoints with the same model
version at different training steps.

Use an immutable checkpoint artifact identity.

Include at least:

    model_name
    model_version
    training_step
    checkpoint content SHA-256

A rating participant can be keyed by the content hash or a deterministic ID
including that hash.

Display names may look like:

    Soo 0.4.0 @ 1500000
    Min 0.2.0 @ 900000

Do not silently allow the same checkpoint participant ID to refer to different
weights.

Do not invent automatic semantic-version bump policy.

Model release versioning remains a separate product decision.

======================================================================
19. PROMOTION VS RATING
======================================================================

Do not make rating and checkpoint promotion the same mechanism.

They are related but distinct.

Promotion answers:

    "Should candidate replace the current champion for the next training
     iteration?"

Rating answers:

    "How strong is this immutable checkpoint relative to historical
     checkpoints under a fixed benchmark protocol?"

Preserve/use the current arena promotion mechanism unless analysis finds a
real correctness issue.

Soo promotion may remain candidate vs champion under complete balanced cycles.

Min promotion may remain:

    candidate
    champion
    champion

under complete 18-game candidate-seat x turn-order balancing.

Rating events may be generated separately.

A rejected checkpoint may still be entered into the historical rating pool.

That is useful information.

======================================================================
20. MILESTONE 2 TRAINING LOOP
======================================================================

Implement a production-oriented iterative training coordinator.

The conceptual state machine is:

    INITIALIZE / RESUME
            |
            v
    SELF_PLAY
            |
            v
    REPLAY INGEST
            |
            v
    TRAIN
            |
            v
    SAVE CANDIDATE CHECKPOINT
            |
            v
    PROMOTION ARENA
            |
            v
    HISTORICAL RATING BENCHMARK
            |
            v
    PROMOTE or REJECT
            |
            v
    PERSIST RUN STATE
            |
            v
    NEXT ITERATION

Each stage must be resumable or safely repeatable.

Do not create a monolithic infinite while-loop with implicit state.

Persist explicit iteration/run state.

======================================================================
21. CHAMPION SEMANTICS
======================================================================

Maintain a champion checkpoint pointer separately for:

    Soo
    Min

A training run only operates on one model identity/player-count at a time.

A run must never accidentally use:

    Soo replay for Min
    Min checkpoint for Soo
    incompatible network configuration
    incompatible ruleset
    incompatible benchmark protocol

Use the existing compatibility system aggressively.

======================================================================
22. SELF-PLAY MODEL PINNING
======================================================================

An individual self-play episode must be generated using a well-defined
checkpoint identity.

Do not swap model weights halfway through one game.

Each self-play episode records:

    game_id
    model checkpoint ID
    compatibility identity
    seed
    completion/abort status

Model activation changes happen at explicit orchestration boundaries.

If the central inference service changes active weights, existing requests
must be drained or version-routed safely.

No episode may unknowingly mix two network versions.

======================================================================
23. MULTIPLE SELF-PLAY WORKERS
======================================================================

Milestone 1 self-play is single-process.

Milestone 2 must add multiple CPU self-play workers.

Use standard Python multiprocessing unless current repository evidence
justifies another approach.

The primary target environment includes Windows.

Therefore code must be spawn-safe.

Do not depend on fork-only behavior.

Use top-level process entry functions.

Protect executable entry points correctly.

Avoid passing unpicklable closures/process-local objects.

Use explicit process lifecycle management.

======================================================================
24. CENTRALIZED BATCHED GPU INFERENCE
======================================================================

Implement a central Python inference coordinator.

Target architecture:

    CPU Self-Play Worker 1 ----\
    CPU Self-Play Worker 2 -----\
    CPU Self-Play Worker 3 ------> Inference Request Queue
    ...                         /
                               /
                         Central Coordinator
                               |
                               v
                      batched Torch inference
                               |
                               v
                         NVIDIA A30

MCTS must remain framework-neutral.

Do not import torch into generic MCTS.

The existing Evaluator abstraction should remain the dependency boundary.

Introduce a remote/queued Evaluator implementation instead of rewriting MCTS.

======================================================================
25. REMOTE EVALUATOR CONTRACT
======================================================================

A worker-side evaluator should conceptually:

    receive EvalRequest from MCTS
    attach:
        request_id
        model/checkpoint key
    submit to inference coordinator
    wait for correlated EvalResult

The coordinator:

    collects pending requests
    groups compatible requests
    performs batched inference
    returns results to the correct requesters

The semantic MCTS API should remain unchanged as much as possible.

No torch.Tensor needs to cross the MCTS public boundary.

======================================================================
26. BATCHING POLICY
======================================================================

Make batching configuration explicit.

At minimum:

    max_batch_size
    max_wait_ms
    request_queue_capacity

Flush a batch when:

    max_batch_size reached

or:

    oldest request waited max_wait_ms

Do not busy-spin.

Provide queue backpressure.

Record actual batch-size distribution.

The production goal is not simply:

    "batching exists"

but:

    "the A30 receives useful batches and GPU inference throughput is measured."

======================================================================
27. MULTI-MODEL INFERENCE
======================================================================

Self-play normally needs one active model.

Arena/rating can require:

Soo:

    candidate
    champion

Min TrueSkill:

    checkpoint A
    checkpoint B
    checkpoint C

Therefore the central inference service must support model/checkpoint keys.

It should be possible to load multiple compatible models simultaneously when
memory allows.

Requests are routed by model key.

Different checkpoint weights cannot be put into the same neural forward batch.

Group batching by:

    model key
    compatible network semantics

Do not switch one model's weights every move during an arena game.

That would destroy throughput.

If multiple model residency is impossible for some configuration, implement a
clear bounded fallback rather than silently thrashing weights.

======================================================================
28. INFERENCE MODEL LOADING
======================================================================

Checkpoint loading into inference must use the existing semantic compatibility
checks.

Do not bypass checkpoint metadata because optimizer state is unnecessary.

If needed, introduce a read-only inference checkpoint loader that:

    validates compatibility first
    extracts model state
    loads strict state_dict
    does not require creating an AdamW trainer

Do not weaken training checkpoint validation to achieve this.

======================================================================
29. GPU / TRAINING SCHEDULING
======================================================================

Initial Milestone-2 implementation should prefer clear stage separation:

    self-play inference stage
    then
    training stage
    then
    evaluation stage

Do NOT attempt to overlap full training and self-play GPU inference on the
same A30 in the first implementation.

Correct orchestration and predictable memory behavior come first.

Pipeline overlap may be evaluated later with measurements.

======================================================================
30. A30 BASELINE
======================================================================

The intended training GPU is:

    NVIDIA A30

Milestone 2 must establish measured baselines for:

    eager FP32 inference
    eager FP32 training

before enabling alternative execution modes.

Record:

    evaluated states / second
    inference calls / second
    mean batch size
    p50/p95 inference request latency
    simulations / second
    games / hour
    training steps / second
    training step latency
    GPU memory usage when measurable
    aborted games
    replay size

If reliable GPU-utilization instrumentation is available, record it.

Do not add a heavy monitoring framework solely to display GPU utilization.

======================================================================
31. BF16 / AMP
======================================================================

After the FP32 production path is correct and measured, implement a controlled
BF16/AMP option suitable for the A30.

It must be CONFIGURABLE.

It must not silently replace FP32.

Compare with identical:

    checkpoint
    request set
    benchmark configuration

Check:

    finite outputs
    legal policy normalization
    reasonable policy/value numerical agreement
    no systematic illegal action behavior
    throughput improvement

Do not claim a performance improvement without measured results.

Do not change rating protocol midstream without creating a new protocol
identity when numerical semantics can affect playing results.

======================================================================
32. TORCH.COMPILE
======================================================================

After eager batching and BF16 evaluation, optionally benchmark:

    torch.compile

It is NOT mandatory to enable by default.

Measure:

    compile/startup cost
    steady-state inference throughput
    training throughput if relevant
    memory use
    stability

Keep it only if evidence supports it.

The reference eager path must remain available.

======================================================================
33. C++ POLICY FOR MILESTONE 2
======================================================================

Do NOT immediately rewrite MCTS or move generation in C++.

Milestone 1 exists as the correctness oracle.

Milestone 2 first needs:

    real multi-worker self-play
    central batching
    A30 utilization
    measured profiling

Only after those exist can we know whether the real bottleneck is:

    Python move generation
    state transition
    tree operations
    inference queue latency
    GPU inference
    training

At the END of Milestone 2, produce an evidence-based C++ acceleration proposal
if CPU search/game code is still the dominant bottleneck.

Do NOT implement C++ in this task without an explicit subsequent approval.

The future C++ implementation must be differential-tested against the current
Python oracle.

======================================================================
34. REPLAY PERSISTENCE
======================================================================

Milestone-1 replay is CPU resident.

Milestone 2 must support safe run resume.

Add persistence without destroying the current clean replay semantics.

Persist enough information to reconstruct the useful replay window.

Prefer:

    chunked/snapshotted local files
    atomic manifest updates
    compatibility metadata

over:

    one giant fragile pickle

Do not introduce Redis, PostgreSQL, or another service for this milestone.

Replay storage must preserve:

    checkpoint/ruleset compatibility
    sparse policy targets
    value semantics
    deterministic sampling state where practical

Do not mix Soo and Min replay.

======================================================================
35. EXACTLY-ONCE EPISODE INGESTION
======================================================================

Multiprocessing workers may fail or be retried.

Each self-play episode needs a stable:

    game_id

Replay ingestion must not duplicate a completed episode because a coordinator
retried a message.

Implement either:

    idempotent game_id-based ingestion

or an equivalently safe design.

Test retry behavior.

An aborted game:

    contributes metrics
    contributes NO fabricated final-value targets

Preserve Milestone-1 behavior.

======================================================================
36. TRAINING RUN STATE
======================================================================

Create explicit serializable run state.

A run should know enough to resume:

    run_id
    model identity
    active/champion checkpoint
    current iteration
    training step
    replay manifest
    completed self-play game IDs
    candidate checkpoint if present
    promotion stage/result
    rating stage/result
    RNG state or deterministic seed derivation scheme
    benchmark protocol IDs

Use atomic updates.

If the process dies after saving a candidate but before promotion, resuming
must not silently start a new unrelated candidate.

======================================================================
37. RUN ARTIFACT LAYOUT
======================================================================

Prefer a clean runtime directory similar to:

    runs/
        soo/
            <run-id>/
                config.json
                state.json
                checkpoints/
                replay/
                ratings/
                metrics/
                logs/

        min/
            <run-id>/
                ...

Do not commit runtime training artifacts to git.

Update .gitignore appropriately.

Exact directory names may follow existing repository conventions.

======================================================================
38. CONFIGURATION
======================================================================

Continue the project's typed, configuration-driven approach.

Add focused config types instead of one enormous configuration dataclass.

Possible boundaries:

    InferenceConfig
    WorkerConfig
    TrainingLoopConfig
    PersistenceConfig
    BenchmarkConfig
    EloConfig
    TrueSkillConfig
    RatingPoolConfig

Do not introduce Hydra or another large configuration framework.

Plain dataclasses + JSON/TOML-style serialization are sufficient.

All semantically important benchmark/rating values must be persisted.

======================================================================
39. MODEL VERSION VS CHECKPOINT VERSION
======================================================================

Do not invent an automatic semantic-version policy.

A training run may have:

    Soo model_version = 0.4.0

with checkpoint artifacts at:

    step 500000
    step 750000
    step 1000000

Each checkpoint may be rated independently.

Use:

    model version

as lineage/product identity.

Use:

    checkpoint artifact ID / hash + step

as the immutable rated participant.

Do not force every optimizer checkpoint to become a new semantic version.

======================================================================
40. HISTORICAL OPPONENT POOL
======================================================================

Ratings become meaningful only with historical comparison.

Maintain a historical checkpoint pool separately for Soo and Min.

The pool may contain:

    current champion
    previous champions
    selected rejected candidates
    anchor checkpoints

Do not keep every trivial intermediate checkpoint forever if this becomes
unbounded.

Make retention configurable.

Never mix incompatible compatibility namespaces.

======================================================================
41. SOO RATING MATCHMAKING
======================================================================

For a new Soo checkpoint:

evaluate against more than only the current champion when practical.

Prefer a deterministic mix such as:

    current champion
    recent historical checkpoints
    a few older stable anchors

Do not design a complex adaptive matchmaking service yet.

Deterministic/configurable scheduling is more important.

Each distinct pair must use complete balanced cycles.

======================================================================
42. MIN TRUE SKILL MATCHMAKING
======================================================================

Min requires triples of DISTINCT checkpoint artifacts.

For a candidate checkpoint, schedule triples such as:

    candidate
    current champion
    historical checkpoint A

    candidate
    historical checkpoint A
    historical checkpoint B

when enough compatible history exists.

Each rated triple must use the complete balanced schedule defined by the
benchmark protocol.

If fewer than three distinct compatible Min checkpoint artifacts exist:

    do NOT fake a TrueSkill match with duplicate models.

Simply report that there is insufficient league history for a rated triple.

Promotion arena may still run normally.

======================================================================
43. EARLY RATING BOOTSTRAP
======================================================================

Soo Elo can begin once two distinct Soo checkpoints exist.

Min TrueSkill league requires three distinct Min checkpoint artifacts.

Before then:

    Min rating can remain at the configured prior

and:

    rated_game_count = 0

Do not manufacture artificial games solely to move the number.

======================================================================
44. TRAINING METRICS VS STRENGTH METRICS
======================================================================

Do not confuse:

    loss

with:

    game strength.

Track training metrics:

    policy loss
    value loss
    total loss

Track system metrics:

    games/hour
    simulations/sec
    eval states/sec
    batch size
    latency

Track strength metrics:

Soo:

    Elo
    rated games
    arena win rate

Min:

    TrueSkill mu
    TrueSkill sigma
    TrueSkill exposure
    rated games
    1st/2nd/3rd rates
    mean placement utility

A decreasing loss is not proof of a stronger model.

======================================================================
45. RATING HISTORY REPORT
======================================================================

Provide an inspectable report/CLI for historical model strength.

Example conceptual output:

    Soo leaderboard
    ---------------------------------------------------------
    checkpoint                    Elo       games
    Soo 0.4.0 @ 1500000          1286       160
    Soo 0.4.0 @ 1000000          1232       188
    Soo 0.3.0 @ 2400000          1191       220

    Min leaderboard
    -----------------------------------------------------------------
    checkpoint              mu       sigma    exposure    games
    Min 0.2.0 @ 900000      ...      ...      ...         ...
    Min 0.1.0 @ 1600000     ...      ...      ...         ...

Do not imply Min exposure is Elo.

======================================================================
46. RATING EVENT REPRODUCIBILITY
======================================================================

Given:

    same initial registry
    same benchmark protocol
    same ordered event log

replaying rating history must produce the same:

    Soo Elo values

and:

    Min mu/sigma/exposure values

within deterministic floating-point expectations.

Provide tests.

This is particularly important for TrueSkill because updates are sequential.

The event sequence order is part of the rating history.

Do not reorder historical events during load.

======================================================================
47. EXISTING ARENA MUST REMAIN USEFUL
======================================================================

Do not delete the current Milestone-1 arena.

It already provides valuable deterministic correctness and promotion behavior.

Refactor only where needed to expose clean immutable match-result records.

Prefer:

    Arena produces results

then:

    rating subsystem consumes eligible results

over:

    rating code embedded inside MCTS or GameAdapter.

MCTS must know nothing about:

    Elo
    TrueSkill
    champion
    leaderboard

======================================================================
48. PROPOSED MODULE BOUNDARIES
======================================================================

Inspect current structure first.

A reasonable extension may resemble:

    src/diamond/alphazero/
        rating/
            __init__.py
            protocol.py
            participants.py
            events.py
            elo.py
            min_trueskill.py
            registry.py
            benchmark.py
            openings.py

        inference/
            __init__.py
            protocol.py
            remote.py
            coordinator.py
            model_pool.py

        orchestration/
            __init__.py
            run_state.py
            persistence.py
            selfplay_workers.py
            coordinator.py
            cli.py

Do not use these names mechanically if the current repository has a better
focused convention.

Prefer files with one clear responsibility.

Do not create a 2,000-line training_manager.py.

======================================================================
49. CLI / OPERATOR SURFACE
======================================================================

Provide a simple headless operator interface.

Use standard argparse or similarly lightweight mechanisms.

Conceptually support tasks such as:

    start/resume training run

    run benchmark

    inspect Soo Elo leaderboard

    inspect Min TrueSkill leaderboard

    profile centralized inference

Exact commands should follow current project conventions.

Do not require PySide/QML to operate training.

======================================================================
50. HEADLESS TRAINING
======================================================================

Training infrastructure must not depend on:

    QML
    GameController
    AiWorker
    window creation

The authoritative game engine may be imported.

GUI modules should not be imported in worker processes.

Fix any accidental dependency leakage if necessary with the smallest focused
change.

Do not redesign the GUI.

======================================================================
51. ENVIRONMENT REPRODUCIBILITY
======================================================================

The previous repository status documented a local Windows environment where
PyTorch and PySide6 Qt DLLs could not coexist correctly in one mamba
interpreter.

Milestone 2 should improve environment reproducibility.

Do not spend the entire milestone rewriting packaging.

At minimum:

    document a known-good training environment
    document a known-good GUI/test environment
    prefer a unified compatible environment when practical

Add clean CPU CI coverage if repository policy permits.

A clean CI environment should run:

    existing Diamond tests
    AlphaZero tests
    new Milestone-2 CPU tests

Do not require an NVIDIA GPU for ordinary CI.

GPU integration tests must be separable.

======================================================================
52. FAILURE HANDLING
======================================================================

A production training loop must fail explicitly.

Handle:

    inference worker death
    queue timeout
    malformed request
    incompatible checkpoint
    GPU OOM
    replay persistence failure
    candidate checkpoint failure
    rating registry corruption
    duplicate rating event
    process shutdown

Do not silently continue after semantic corruption.

For recoverable game-worker failure:

    abort/retry the episode safely

without:

    duplicated replay
    fabricated outcome
    mixed model identity

======================================================================
53. DETERMINISM
======================================================================

Preserve explicit deterministic seed derivation.

Avoid:

    process-global random state shared implicitly among workers.

A game seed should be derivable from stable inputs such as:

    run seed
    iteration
    worker/game index

A retry of the same game_id should either:

    reproduce the same game

or:

    explicitly receive a new retry identity

Do not accidentally count both as different completed games.

======================================================================
54. REQUIRED RATING TESTS — SOO
======================================================================

At minimum test:

- default Elo initial value
- known expected-rating calculation
- winner update
- loser update
- updates use both PRE-MATCH ratings
- aborted match leaves ratings unchanged
- deterministic event replay
- duplicate event rejection/idempotence
- complete 4-game balance schedule
- incomplete balance rejected for rated batch
- protocol namespace mismatch rejected
- incompatible checkpoint rejected
- two checkpoint artifacts from same model version can still be distinct
  participants when hashes/training steps differ
- leaderboard sorting

======================================================================
55. REQUIRED RATING TESTS — MIN
======================================================================

At minimum test:

- official trueskill package is used
- explicit environment parameters
- tau == 0.0
- draw_probability == 0.0
- three-player free-for-all update
- lower rank means better placement
- winner mu/exposure moves in expected direction
- last-place rating moves in expected direction
- sigma behavior remains finite
- expose() used for conservative leaderboard value
- aborted match produces no update
- deterministic event replay
- duplicate event rejection/idempotence
- three participants MUST be distinct checkpoint IDs
- candidate/champion/champion is rejected as a TrueSkill event
- 36-game complete distinct-triple balance schedule
- incomplete balance rejected
- protocol namespace mismatch rejected
- leaderboard sorting by exposure
- insufficient-history behavior with fewer than three checkpoints

Do not overfit tests to exact library floating point outputs unless necessary.

Prefer semantic invariants plus a few known sanity values.

======================================================================
56. REQUIRED OPENING-SUITE TESTS
======================================================================

Test:

- deterministic generation/reconstruction
- every action is authoritative legal
- same config gives same opening suite/hash
- invalid ruleset fingerprint rejected
- standard initial opening included
- opening action sequence reconstructs identical state
- Soo and Min suites respect correct player count
- benchmark protocol records opening suite identity

======================================================================
57. REQUIRED INFERENCE TESTS
======================================================================

CPU tests must verify:

- RemoteEvaluator satisfies Evaluator semantics
- request IDs correlate correctly
- multiple concurrent clients work
- batching actually forms batch size > 1 in a controlled test
- max_batch_size flush
- max_wait_ms flush
- queue backpressure behavior
- graceful coordinator shutdown
- malformed request failure
- worker/coordinator failure propagation
- model key routing
- two Soo model keys can coexist
- three Min model keys can coexist
- different model keys are not combined into one neural forward
- checkpoint compatibility checked before model activation
- local TorchEvaluator and central evaluator produce equivalent outputs for
  the same FP32 model/request within appropriate tolerance

MCTS must remain torch-independent.

Add an explicit test guarding this architectural dependency if practical.

======================================================================
58. REQUIRED MULTIPROCESS SELF-PLAY TESTS
======================================================================

Test:

- more than one worker can generate episodes
- worker processes are spawn-safe
- completed episode has stable game_id
- model identity is pinned for the whole episode
- legal actions only
- Soo values remain correct
- Min full placement values remain correct
- aborted episodes yield no samples
- duplicate episode result is not ingested twice
- worker failure is surfaced
- clean shutdown leaves no hanging processes

Do not require GPU for these tests.

Use DummyEvaluator/small test configurations where appropriate.

======================================================================
59. REQUIRED PERSISTENCE / RESUME TESTS
======================================================================

Simulate interruptions at least around:

    after self-play
    after replay save
    after training
    after candidate checkpoint
    after arena
    after rating update

Verify resume:

    does not duplicate completed work
    does not lose compatibility identity
    does not overwrite a different checkpoint
    does not double-rate matches
    preserves champion/candidate state

Atomicity matters more than cleverness.

======================================================================
60. REQUIRED END-TO-END CPU SMOKE
======================================================================

Provide a tiny CPU configuration that can complete:

    Soo:
        multi-worker self-play
        replay ingest
        one/small training update
        candidate checkpoint
        promotion arena
        Elo benchmark/event
        persistence
        resume

and:

    Min:
        multi-worker self-play
        full-ranking targets
        training update
        candidate checkpoint
        promotion arena
        TrueSkill handling where enough distinct fixture checkpoints exist
        persistence
        resume

Use tiny toy/near-terminal authoritative fixtures if needed for runtime.

Do not call a smoke test successful if it bypasses major orchestration stages.

======================================================================
61. OPTIONAL GPU INTEGRATION TESTS
======================================================================

If CUDA/A30 is available, add separately runnable GPU integration smoke.

It should verify:

    centralized inference on CUDA
    concurrent worker requests
    batch formation
    FP32 training
    checkpoint transition
    no CUDA OOM under tiny test config

These tests must be marked/separated so CPU CI remains usable.

If A30 is unavailable in the execution environment:

    do not fabricate results.

Report GPU verification as not executed.

======================================================================
62. PERFORMANCE PROFILING
======================================================================

Once the system is correct, profile a meaningful short run.

Record stage timing:

    legal move/state transition
    MCTS tree logic
    queue wait
    inference
    self-play episode
    replay collation
    training step

Report the major bottlenecks by measured percentage/time.

Do not decide that C++ is necessary from intuition alone.

======================================================================
63. PERFORMANCE COMPARISON TABLE
======================================================================

Where environment permits, produce a concise benchmark such as:

    mode                  eval/s   avg batch   games/h   train step/s
    -----------------------------------------------------------------
    local FP32
    central FP32
    central BF16
    central compiled FP32
    central compiled BF16

Only include rows actually measured.

Record hardware and important config with results.

Do not mix strength rating with throughput.

A model getting more eval/s is not automatically stronger.

======================================================================
64. RATING BENCHMARKS ARE NOT PERFORMANCE BENCHMARKS
======================================================================

Keep terminology clear.

Strength benchmark:

    Soo Elo
    Min TrueSkill

System performance benchmark:

    games/hour
    simulations/sec
    states/sec
    batch size
    GPU utilization
    latency

Do not call throughput "Elo performance."

======================================================================
65. DOCUMENTATION
======================================================================

Update AlphaZero developer documentation.

Document:

    Milestone 2 architecture

    Soo Elo semantics

    Min TrueSkill semantics

    why Min uses tau=0

    why Min rating requires 3 distinct checkpoint artifacts

    difference between:
        Min 18-game promotion cycle
        Min 36-game rated triple cycle

    benchmark protocol identity

    rating namespaces

    opening suite

    model version vs checkpoint artifact identity

    how to start training

    how to resume

    how to inspect rating leaderboards

    how to run CPU tests

    how to run GPU benchmark

    how to interpret throughput metrics

    deferred C++ work

Do not claim GPU measurements that were not executed.

======================================================================
66. DO NOT BREAK GUI CONTRACT
======================================================================

Milestone 2 is training infrastructure.

Do not route Soo/Min into the GUI yet.

Do not rewrite QML.

Do not auto-commit AI moves.

The existing future deployment contract remains:

    Agent
        ->
    MoveProposal
        ->
    GameController
        ->
    confirmation
        ->
    GameSession.commit()

GUI integration remains later work.

======================================================================
67. DO NOT IMPLEMENT OPENVINO
======================================================================

OpenVINO / Intel Iris Xe deployment is not Milestone 2.

Keep the network export-friendly.

Do not add OpenVINO dependency here.

======================================================================
68. DO NOT CHANGE MIN VALUE SEMANTICS
======================================================================

TrueSkill is an EXTERNAL CHECKPOINT STRENGTH RATING SYSTEM.

It does not replace Min's neural value head.

Min neural value remains:

    [self, next, previous]

with:

    first  +1
    second 0
    third -1

Do NOT train the neural network to predict TrueSkill mu/sigma.

Do NOT insert TrueSkill into MCTS backup.

TrueSkill exists outside gameplay/search/training targets.

======================================================================
69. DO NOT CHANGE SOO VALUE SEMANTICS
======================================================================

Elo is also an EXTERNAL CHECKPOINT STRENGTH RATING SYSTEM.

Soo neural value remains:

    scalar current-player win/loss

Do NOT train Soo to predict Elo.

Do NOT use Elo inside PUCT.

======================================================================
70. ARCHITECTURAL LAYERS
======================================================================

The intended dependency direction is:

    Diamond authoritative game
                |
                v
              MCTS
                |
                v
            Evaluator
                |
         ----------------
         |              |
    local Torch     Remote/Queued
                        |
                        v
                Inference Coordinator
                        |
                        v
                     PyTorch


Separate evaluation layer:

    Arena / Benchmark
            |
            v
       Match Records
            |
       -------------
       |           |
     Elo       TrueSkill
       |           |
       -------------
            |
            v
      Rating Registry


Separate training orchestration:

    Coordinator
        |
        +--> self-play workers
        +--> replay
        +--> trainer
        +--> checkpoint
        +--> arena
        +--> rating benchmark
        +--> persistence

Do not reverse these dependencies.

======================================================================
71. DEVELOPMENT WORKFLOW
======================================================================

Use the Superpowers development methodology available in Codex.

Before implementation:

    inspect repository
    run baseline
    write architecture delta
    write implementation plan

Use:

    superpowers:using-superpowers
    superpowers:using-git-worktrees where appropriate
    superpowers:writing-plans
    superpowers:test-driven-development
    superpowers:systematic-debugging
    superpowers:verification-before-completion
    superpowers:requesting-code-review

Save the Milestone-2 implementation plan under:

    docs/superpowers/plans/YYYY-MM-DD-alphazero-milestone2.md

Do not begin large implementation before the plan is internally coherent.

======================================================================
72. IMPLEMENTATION ORDER
======================================================================

Use the following high-level phase order unless repository dependencies prove
a small adjustment necessary.

PHASE 0
    inspect current main
    run baseline tests
    verify Milestone-1 status
    inspect existing docs
    create Milestone-2 implementation plan

PHASE 1
    rating/benchmark DTOs
    checkpoint participant identity
    benchmark protocol identity
    immutable rating events

PHASE 2
    Soo Elo
    Elo registry
    tests

PHASE 3
    Min official TrueSkill integration
    explicit TrueSkill environment
    Min registry
    tests

PHASE 4
    balanced rated scheduling
    Soo 4-game cycle
    Min 36-game distinct-checkpoint cycle
    opening suite
    rating replay/leaderboard

PHASE 5
    inference request protocol
    RemoteEvaluator
    central coordinator
    model-key routing
    batching
    CPU concurrency tests

PHASE 6
    spawn-safe multiple self-play workers
    stable game IDs
    exactly-once replay ingestion
    metrics

PHASE 7
    persistent replay
    run state
    atomic save/resume

PHASE 8
    iterative training coordinator
    self-play -> train -> candidate -> arena -> rating -> promote/reject

PHASE 9
    CLI/operator commands
    complete CPU end-to-end smoke

PHASE 10
    A30 eager FP32 baseline profiling

PHASE 11
    BF16/AMP controlled benchmark

PHASE 12
    torch.compile controlled benchmark if justified

PHASE 13
    full regression
    review hardening
    documentation
    evidence-based C++ recommendation

Do not move to optimization phases while correctness or resume semantics are
uncertain.

======================================================================
73. GIT HYGIENE
======================================================================

Keep commits focused.

Do not combine:

    GUI redesign
    unrelated formatting
    rating implementation
    multiprocessing implementation

into one giant commit.

Reasonable conceptual commit boundaries include:

    feat: add benchmark protocol identity
    feat: add Soo Elo registry
    feat: add Min TrueSkill registry
    feat: add balanced historical rating schedules
    feat: add centralized batched evaluator
    feat: add multiprocessing self-play
    feat: add persistent training run state
    feat: add production training coordinator
    perf: benchmark A30 inference modes

Use actual work boundaries rather than blindly copying these names.

======================================================================
74. NO PLACEHOLDERS
======================================================================

Do not finish Milestone 2 with central pieces represented by:

    TODO
    pass
    NotImplementedError
    fake metrics
    mock-only production paths

Mocks and toy games are appropriate for unit tests.

Production orchestration needs a real executable small path.

======================================================================
75. ACCEPTANCE CRITERIA
======================================================================

Do not call Milestone 2 complete unless all applicable criteria are satisfied.

A. REGRESSION

- Milestone-1 AlphaZero tests pass.
- Existing Diamond game/GUI tests pass or known environment limitation is
  documented with independent verification.
- Milestone-1 mathematical semantics are unchanged.

B. SOO RATING

- Elo implemented and tested.
- Rated games use fixed benchmark protocol.
- Complete seat/order cycles enforced.
- Aborts do not affect Elo.
- Historical leaderboard persists and reloads.
- Event replay reproduces Elo.

C. MIN RATING

- Official trueskill package used.
- tau=0.
- draw_probability=0.
- free-for-all ranks used directly.
- three distinct checkpoint artifacts required.
- candidate/champion/champion excluded from TrueSkill events.
- 36-game distinct-triple balance cycle implemented.
- mu/sigma/exposure persisted.
- event replay reproduces ratings.

D. CENTRAL INFERENCE

- Multiple worker requests can share one coordinator.
- Batches larger than one are demonstrated.
- Model-key routing works.
- MCTS remains PyTorch-independent.
- Local-vs-central evaluator parity is tested.
- graceful shutdown works.

E. MULTIPROCESS SELF-PLAY

- multiple workers execute successfully.
- spawn-safe.
- stable game IDs.
- no duplicate replay ingestion.
- model pinned per episode.
- abort semantics preserved.

F. TRAINING ORCHESTRATION

A tiny real run completes:

    self-play
    replay
    training
    checkpoint
    arena
    rating
    promote/reject
    persistence

and can resume after interruption.

G. PERFORMANCE

If A30 is available:

- eager FP32 baseline measured.
- batch statistics measured.
- GPU-oriented throughput reported.
- BF16 tested after FP32.
- torch.compile only reported if actually benchmarked.

If A30 is unavailable:

- GPU claims are explicitly marked unverified.

H. DOCUMENTATION

- operator instructions exist.
- rating semantics documented.
- model/checkpoint/rating identity distinction documented.
- future C++ scope documented from measurements.

======================================================================
76. FINAL REPORT FORMAT
======================================================================

At completion report:

A. Baseline
    current commit
    existing test result
    environment notes

B. Milestone-2 architecture
    files/modules added
    dependency boundaries

C. Soo Elo
    formula/config
    benchmark protocol
    leaderboard persistence
    tests

D. Min TrueSkill
    package integration
    environment parameters
    free-for-all semantics
    distinct-checkpoint restriction
    leaderboard exposure
    tests

E. Rating benchmark
    opening suite
    Soo balance cycle
    Min balance cycle
    event log
    historical pools

F. Production training
    worker architecture
    inference coordinator
    batching
    replay persistence
    resume behavior

G. End-to-end result
    tiny Soo run
    tiny Min run

H. Performance
    actual CPU/GPU measurements
    batch-size distribution
    bottlenecks

I. Verification
    exact commands executed
    exact test counts
    smoke exit codes

J. C++ recommendation
    whether C++ is justified
    measured bottleneck
    recommended boundary
    DO NOT implement C++ without separate approval

K. Deferred work
    C++ implementation
    OpenVINO
    Iris Xe
    GUI SooAgent/MinAgent routing
    distributed/multi-node training if still unnecessary

======================================================================
77. STOP CONDITIONS
======================================================================

Do not stop for ordinary engineering decisions that can be resolved from the
repository and this specification.

Proceed with senior engineering judgment.

Stop and clearly report if you discover a fundamental contradiction such as:

    Milestone-1 tests expose a mathematical correctness regression

    current main no longer matches the documented Diamond rules

    checkpoint compatibility cannot identify authoritative semantics

    Min no longer produces a strict full ranking

    multiprocessing would require violating the Evaluator abstraction

    current checkpoint format cannot be made safely resumable without a
    breaking compatibility decision

Minor refactors, filenames, or environment adjustments are not stop conditions.

======================================================================
78. ABSOLUTE INVARIANTS
======================================================================

INVARIANT 1

    The current Diamond engine is authoritative.

INVARIANT 2

    Milestone 1 is the correctness oracle.

INVARIANT 3

    Soo and Min are independently versioned.

INVARIANT 4

    Soo strength rating is Elo.

INVARIANT 5

    Min strength rating is official TrueSkill.

INVARIANT 6

    TrueSkill is external evaluation infrastructure, not Min's neural value.

INVARIANT 7

    Elo is external evaluation infrastructure, not Soo's neural value.

INVARIANT 8

    Min TrueSkill-rated matches require three DISTINCT checkpoint artifacts.

INVARIANT 9

    Min promotion candidate/champion/champion matches are not TrueSkill events.

INVARIANT 10

    Min TrueSkill tau is 0 for immutable checkpoints.

INVARIANT 11

    Aborted games never affect ratings.

INVARIANT 12

    Ratings from different benchmark protocols never mix.

INVARIANT 13

    Fixed search compute is required for comparable ratings.

INVARIANT 14

    Model version != checkpoint artifact != strength rating.

INVARIANT 15

    MCTS never depends on PyTorch, Elo, or TrueSkill.

INVARIANT 16

    Self-play episodes never mix checkpoint weights mid-game.

INVARIANT 17

    Central inference groups requests by model/checkpoint key.

INVARIANT 18

    Replay ingestion is idempotent/exactly-once at episode identity level.

INVARIANT 19

    Performance claims require measurements.

INVARIANT 20

    C++ is not implemented until profiling demonstrates the need and a
    separate approval is given.

======================================================================
79. FIRST ACTION
======================================================================

Begin now by:

1. inspecting the current main branch;
2. reading the Milestone-1 AlphaZero implementation and documentation;
3. running the baseline test/smoke commands that are available;
4. identifying any differences from the status stated above;
5. writing the detailed Milestone-2 implementation plan;
6. then implementing Milestone 2 task-by-task with TDD.

Do not redesign the game.

Do not rewrite Milestone 1 unnecessarily.

Build the production training and strength-evaluation system on top of the
verified reference implementation.
