You are acting as the senior ML systems engineer and training operator for:

Repository:
ubraket513/alphadiamond

Mission:
Run the first scientifically useful ~8-hour CPU-only AlphaZero training session
for both Diamond models:

- Soo = 2-player model
- Min = 3-player model

This is NOT a final-strength training run.
This is a bootstrap-to-normal-transition experiment whose purpose is to:

1. generate reliable real terminal self-play,
2. train the first useful neural checkpoints,
3. determine whether the heuristic can be removed,
4. if possible, continue normal AlphaZero training without the heuristic,
5. leave behind reproducible checkpoints, logs, metrics, and a concise ledger.

Do not overengineer.
Do not redesign AlphaZero.
Do not introduce new heuristics unless the existing v2 demonstrably fails.

======================================================================
1. FIRST: INSPECT THE ACTUAL REPOSITORY
======================================================================

Do not assume this prompt is newer than the repository.

Start by:

- inspect current `main`
- record exact HEAD SHA
- inspect recent commits
- inspect:
  - `docs/alphazero.md`
  - bootstrap design/spec
  - `src/diamond/alphazero/bootstrap/`
  - self-play runners
  - production/orchestration code
  - current Soo/Min bootstrap configs
  - current Soo/Min production configs
  - CLI train/resume semantics
  - run-state/replay/checkpoint persistence
- run relevant tests before training

At the time this prompt was written, the last observed HEAD was:

    7b0c94cce91f0b7b1e9c55bf2aa648cc3bc9a1d8

but use a newer HEAD if main has advanced.

Do not rewrite working systems merely because you would design them differently.

======================================================================
2. AUTHORITATIVE BOOTSTRAP STATUS
======================================================================

The cold-start problem has already been investigated.

Without a bootstrap prior, an untrained model with tiny search budgets wanders
until `max_game_moves_exceeded`, producing zero replay.

The accepted bootstrap heuristic is:

    canonical-target-vacancy-distance-v2

It is intentionally temporary.

Measured fixed-seed result:

    mcts.simulations = 1
    max_moves = 2000
    20 episodes per condition
    dummy/untrained evaluator

Soo:
    no heuristic:
        completion = 0%
        replay = 0

    v1:
        completion = 100%
        median = 84.5 moves
        p90 = 126

    v2:
        completion = 100%
        median = 74 moves
        p90 = 93

Min:
    no heuristic:
        completion = 0%
        replay = 0

    v1:
        completion = 100%
        median = 131 moves
        p90 = 172

    v2:
        completion = 100%
        median = 101 moves
        p90 = 130

Therefore:

DO NOT implement:
- v3 heuristic
- Hungarian assignment
- endgame solver
- reward shaping
- fake terminal outcomes
- forced forward moves
- camp locking
- cycle penalties
- imitation-learning subsystem
- heuristic value targets

unless new measured evidence shows v2 is inadequate.

======================================================================
3. IMMUTABLE LEARNING SEMANTICS
======================================================================

The bootstrap heuristic changes POLICY PRIOR ONLY.

It must never alter:

- Diamond legality
- terminal conditions
- MCTS backup semantics
- network architecture
- replay schema
- Soo value semantics
- Min value semantics

Soo terminal target remains:

    +1 / -1
    current-player scalar win/loss semantics

Min remains:

    first  = +1
    second =  0
    third  = -1

in canonical [self, next, previous] semantics.

Incomplete games still produce ZERO training samples.

Do not invent a value for an aborted game.

Arena, Elo, TrueSkill and benchmark matches must use:

    bootstrap_prior = none

======================================================================
4. CPU-ONLY CONSTRAINT
======================================================================

This run is intentionally CPU-only.

Do not attempt CUDA, BF16, A30 profiling, OpenVINO or C++ work.

Confirm at startup:

- Python executable
- PyTorch version
- torch device
- logical CPU count
- Torch CPU thread count
- available memory if easy to obtain

Record the information in the run ledger.

Do not spend time tuning every CPU parameter.

The current reference bootstrap configs are approximately:

    network:
        width = 128
        residual_blocks = 6

    self_play:
        max_moves = 2000
        temperature_moves = 20
        temperature = 1.0
        bootstrap_prior = canonical-target-vacancy-distance-v2

    workers:
        worker_count = 4
        games_per_iteration = 32

    replay:
        capacity = 50000

    training:
        batch_size = 256
        lr = 1e-3
        weight_decay = 1e-4
        device = cpu

The reference MCTS value is currently 128 simulations.

DO NOT blindly use 128 for this 8-hour CPU experiment.

Search throughput is more valuable than unnecessarily deep bootstrap search.

======================================================================
5. DO A SHORT SEARCH-BUDGET CALIBRATION
======================================================================

Spend no more than roughly 15-20 minutes on calibration.

Do not build a new benchmarking subsystem.

Use existing public runners/tools or a temporary local script.

Evaluate, at minimum:

    simulations = 8
    simulations = 16

Optionally test 32 only if 16 is surprisingly cheap.

You may measure simulations=1 as a throughput reference, but do not choose it
for the real training run merely because it finishes games: with one simulation
the visit target contains almost no search improvement.

For Soo and Min, run only a few fixed-seed bootstrap episodes.

Record:

- completion
- moves/game
- wall-clock sec/game
- replay samples/game
- approximate states/evaluations per second if readily available

Selection policy:

Prefer 16 simulations if its throughput is practical.

Prefer 8 if 16 would drastically reduce the number of games/updates obtainable
inside the 8-hour budget.

Use 32 only if measured evidence says its cost is acceptable.

Do not use 128 unless actual measurement gives a compelling reason.

Change only the runtime/training copy of the config.

Do not mutate the canonical checked-in reference config merely for this run.

======================================================================
6. 8-HOUR EXPERIMENT STRUCTURE
======================================================================

Total wall-clock target:

    approximately 8 hours

This includes:
- preflight
- calibration
- training
- transition probes
- final report

Do not intentionally run far beyond the budget.

Train models sequentially, not concurrently.
They should not fight over the same CPU.

Recommended allocation:

    ~20 min    preflight + calibration
    ~3h15m    Soo experiment
    ~4h00m    Min experiment
    ~25 min    final probes / report / verification

This is guidance, not a requirement to stop an atomic iteration halfway.

Min may receive somewhat more wall-clock because 3-player games and search are
more expensive.

======================================================================
7. PHASE B0 — BOOTSTRAP TRAINING
======================================================================

Start Soo first.

Then Min.

For B0 use:

    bootstrap_prior = canonical-target-vacancy-distance-v2

Use the selected CPU MCTS simulation count from calibration.

The normal data path must remain:

    real Diamond state
        -> MCTS using heuristic policy prior
        -> real completed game
        -> normal MCTS visit target
        -> real terminal value target
        -> persistent replay
        -> ordinary AlphaZero trainer
        -> checkpoint

The heuristic is NOT a reward.

The network value remains the real network value.

Keep:
- replay persistence
- atomic run state
- checkpoint persistence
- deterministic run IDs/seeds
- idempotent episode ingestion

Use distinct run IDs, e.g.:

    cpu8h-soo-20260819
    cpu8h-min-20260819

or equivalent collision-safe names.

Do not put checkpoints or runtime replay in Git.

======================================================================
8. DO NOT WASTE CPU ON RATING DURING B0
======================================================================

This session is about learning viability, not official strength measurement.

Do not spend a large fraction of the 8-hour CPU budget repeatedly running:

- historical Elo league
- historical TrueSkill league
- large benchmark suites
- large promotion arenas

unless the existing production coordinator inherently requires a small gate.

Do not change rating semantics to avoid the cost.

If the full `cli train` pipeline obligatorily performs expensive rating work
after every small bootstrap update, inspect existing public components and use
the narrowest existing training path that preserves:

    self-play
    replay
    trainer
    checkpoint
    resume safety

Do not redesign the coordinator merely for this experiment.

Record clearly if the experimental B0 loop bypasses promotion/rating.

======================================================================
9. CHECKPOINT FREQUENTLY
======================================================================

Do not make the 8-hour run all-or-nothing.

Persist after every normal completed training iteration/update unit supported
by the repository.

Record at each checkpoint:

- model: Soo / Min
- model version
- training step
- checkpoint path
- SHA-256 if already supported
- wall-clock elapsed
- self-play attempted/completed/aborted
- replay sample count
- replay size
- average / median completed-game moves if available
- policy loss
- value loss
- total loss
- bootstrap prior identity
- MCTS simulations
- worker count

Do not invent metrics that the system does not expose.

======================================================================
10. THE IMPORTANT PART: HEURISTIC-OFF GATE
======================================================================

The goal is NOT to leave the heuristic enabled forever.

Once a model has accumulated a meaningful bootstrap checkpoint, test whether
the learned network can stand on its own.

A sensible first probe point is:

- after at least a couple of successful training updates, OR
- once replay contains several thousand real samples, OR
- after roughly 60-90 minutes of successful B0 training

Use judgment based on actual iteration duration.

For the heuristic-OFF probe:

    load the actual trained checkpoint
    bootstrap_prior = none
    do NOT reinitialize the network

Run approximately 10 fixed-seed self-play episodes.

Use:
    max_moves = 2000

Start with the selected training MCTS simulation count.

Measure:

- completion rate
- abort reasons
- median completed moves
- p90 if meaningful
- replay samples/episode

Suggested operational gate:

    PASS:
        >= 8 / 10 games complete
        and replay is non-empty
        and there is no obvious deterministic shuffle/cycle pathology

This threshold is an operational bootstrap gate, not a claim of playing
strength.

======================================================================
11. IF HEURISTIC-OFF FAILS
======================================================================

Do NOT immediately invent a new heuristic.

First determine whether the problem is weak learned policy or insufficient
search.

Retry the SAME trained checkpoint with:

    heuristic = none
    simulations = 2 * selected training simulations

cap this diagnostic search at approximately 32 simulations for this CPU session.

Example:

    training simulations = 8

    OFF probe:
        sims 8  -> fails
        sims 16 -> test

or:

    training simulations = 16

    OFF probe:
        sims 16 -> fails
        sims 32 -> test

If the higher-search probe succeeds:

- use that higher simulation count for normal A0 self-play if CPU throughput
  remains acceptable.

If both fail:

- return to B0 for another bounded training block
- keep v2
- train more
- probe again later

Do not alter the learning objective.

======================================================================
12. PHASE A0 — NORMAL ALPHAZERO
======================================================================

As soon as a model passes the heuristic-OFF gate:

TURN THE HEURISTIC OFF.

Continue from THE SAME TRAINED CHECKPOINT.

Do not reset weights.
Do not create a new random model.

Use:

    bootstrap_prior = none

Normal path becomes:

    neural policy
    + neural value
        -> MCTS
        -> self-play
        -> replay
        -> training

Keep existing bootstrap-generated replay.

Do NOT flush it just because the heuristic has been disabled.

Let newer normal self-play gradually replace bootstrap-biased data through the
ordinary replay lifecycle.

Spend the remainder of that model's allotted CPU time in A0 if stable.

======================================================================
13. SOO AND MIN MUST REMAIN INDEPENDENT
======================================================================

Do not mix:

- checkpoints
- replay
- run state
- ratings
- model versions
- training steps

Soo and Min are separate models.

Soo is 2-player scalar value.

Min is 3-player vector value.

Do not compare Soo Elo to Min TrueSkill.

Do not synchronize model versions artificially.

For this CPU session, ratings are not the primary success metric.

======================================================================
14. SUCCESS METRICS FOR THIS 8-HOUR RUN
======================================================================

This experiment is successful even if the models are still weak.

Primary success criteria:

A. B0 viability
    - real terminal games generated
    - replay stays non-empty
    - training updates execute
    - checkpoints are recoverable

B. learning signal
    - losses are finite
    - no NaN/Inf
    - policy/value training proceeds
    - checkpoint restore works

C. heuristic handoff
    - determine empirically whether Soo can run with heuristic OFF
    - determine empirically whether Min can run with heuristic OFF

Best outcome:
    both pass and enter A0

Acceptable outcome:
    one passes and one remains B0

Still scientifically useful:
    neither passes, provided real terminal replay/training worked and the ledger
    clearly records how much more bootstrap learning appears necessary.

Do not call failure simply because Elo/TrueSkill did not rise during this short
CPU run.

======================================================================
15. FAILURE CONDITIONS
======================================================================

Stop or diagnose immediately if:

- replay unexpectedly remains empty with v2
- completed episodes are being discarded
- value targets change from authoritative semantics
- NaN/Inf appears
- checkpoint restore fails
- run-state resume repeats completed episode IDs
- self-play repeatedly hits 2000 moves despite v2
- process memory grows without bound
- training uses a different model/checkpoint than the self-play provenance says
- bootstrap heuristic accidentally enters arena/rating

Do not hide failures by increasing max_moves above 2000.

Do not fabricate draws/winners.

======================================================================
16. EXISTING GUI AGENT IS OUT OF SCOPE
======================================================================

There is a GUI-facing `AlphaZeroAgent`.

Do not spend this training session redesigning GUI integration.

Training uses the headless AlphaZero stack.

If you notice GUI/bootstrap-default behavior that should be changed later,
record it as a follow-up only unless it directly blocks this training run.

Do not consume the 8-hour training budget on UI work.

======================================================================
17. TESTING BEFORE AND AFTER
======================================================================

Before the long run, execute the relevant working test suites.

At minimum:

- bootstrap heuristic tests
- bootstrap evaluator tests
- self-play tests
- replay/persistence tests
- checkpoint tests
- trainer tests
- orchestration/resume tests

If the full non-GUI AlphaZero suite is reasonably fast, run it.

After the run, rerun focused smoke/restore tests.

Do not repeatedly run the entire suite inside every training iteration.

======================================================================
18. RUN LEDGER
======================================================================

Create a concise local ledger under the runtime/report area.

It must record:

- date/time
- Git HEAD
- dirty/clean status
- Python
- PyTorch
- CPU information
- selected MCTS simulation count and why
- exact config snapshots
- run IDs
- seeds
- start/end times
- number of self-play games
- completion/abort counts
- game length statistics
- replay counts
- training steps
- losses
- checkpoints
- heuristic-OFF probe results
- whether and when B0 -> A0 occurred
- any crashes/resumes
- final recommendation

Do not commit binary checkpoints or replay.

A small Markdown/JSON report may be committed only if repository practice and
permissions make that appropriate; runtime evidence itself should stay outside
Git.

======================================================================
19. FINAL REPORT FORMAT
======================================================================

When the CPU session ends, report Soo and Min separately.

For each model give:

MODEL
    Soo / Min

MODE
    B0 only
    or
    B0 -> A0

CPU TIME
    total elapsed

SEARCH
    chosen simulations
    worker count

SELF-PLAY
    attempted
    completed
    aborted
    completion rate
    median moves
    p90 moves

REPLAY
    final sample count
    samples generated

TRAINING
    training steps
    latest policy loss
    latest value loss
    latest total loss

CHECKPOINT
    path
    training step
    artifact hash if available

HEURISTIC-OFF PROBE
    simulations
    completed / attempted
    median moves
    abort reasons
    PASS / FAIL

FINAL STATE
    heuristic ON or OFF

Then conclude with exactly one of:

    READY FOR NORMAL A0 CPU TRAINING

    NEEDS MORE B0 BOOTSTRAP TRAINING

    BLOCKED BY A REAL SYSTEM DEFECT

Explain the conclusion with measured evidence.

======================================================================
20. ENGINEERING DISCIPLINE
======================================================================

This task is primarily an OPERATIONS/TRAINING task.

Do not refactor the repository for aesthetics.

Do not add generalized training frameworks.

Do not add Hydra.

Do not add C++.

Do not add new heuristic versions.

Do not tune dozens of hyperparameters.

The important scientific variables for this run are:

    bootstrap prior ON/OFF
    MCTS simulation budget
    elapsed bootstrap learning

Keep everything else as stable as practical.

If a genuine code defect blocks training:
- reproduce it
- write a focused regression test
- make the smallest fix
- rerun the relevant verification
- record the interruption in the ledger
- resume from persisted state

Never silently restart from scratch when a valid checkpoint/replay/run state
exists.

======================================================================
EXECUTE
======================================================================

Proceed now.

First print:

1. exact repository HEAD,
2. CPU/Python/PyTorch environment,
3. baseline test result,
4. current Soo/Min bootstrap config summary,
5. calibration results,
6. selected simulation budget,

then begin the actual CPU training session.

The central question to answer after ~8 hours is:

    "Have Soo and Min learned enough from v2 bootstrap self-play that we can
     remove the heuristic and continue with neural-policy AlphaZero?"

Everything you do should serve that question.