You are the Chief AI Architect and senior ML/systems engineer responsible for adding a production-quality AlphaZero subsystem to an existing game repository.

======================================================================
IMPLEMENTATION STATUS — 2026-08-19
======================================================================

Official learned-model naming/versioning contract:

    Soo = independently versioned 2-player Diamond model/agent
    Min = independently versioned 3-player Diamond model/agent

Soo and Min model versions never imply synchronization with each other.
Checkpoint compatibility is gated independently by:

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

Current value-semantics identities:

    Soo: current-player-scalar-winloss-v1
    Min: canonical-placement-utility-1-0-minus1-v1

Milestone 2 implementation status (Task 14, 2026-08-19):

- Completed: independent checkpoint artifact/compatibility identities; Soo Elo
  and Min TrueSkill (`tau=0`) historical ratings; distinct-artifact Min
  rating events; 18-game promotion and 36-game league schedules; protocol- and
  opening-suite-bound benchmarks; centralized inference, workers, atomic
  replay/run persistence, headless CLI, CPU smoke, and bounded profiling.
- Verification: AlphaZero suite 287 passed, 1 skipped; Qt/system suite 134
  passed, 5 skipped; both smoke modules exit 0; MCTS dependency guard passes.
- This host has no CUDA/A30 and the CPU profile reports `gpu_verified=false`.
  No production A30 stage percentages exist; C++ remains deferred pending
  measured production evidence that a CPU search/game/tree stage dominates.
- Deferred: C++ implementation, OpenVINO, GUI AlphaZero routing, and
  multi-node training. Requirements below remain authoritative.

Completed and verified:

- Phase 0 repository inspection and baseline: 134 passed, 5 skipped.
- Phase 1 version/config primitives, 73x73 ActionCodec, authoritative adapter.
- Phase 2 geometry-derived canonical encoder and action/player mappings.
- Phase 3 shared directional graph trunk, factorized policy head, SooModel,
  and MinModel.
- Phase 4 framework-neutral Evaluator, DummyEvaluator, and TorchEvaluator.
- Phase 5 scalar Soo MCTS with explicit child-to-parent sign alternation.
- Phase 6 vector Min MCTS with stable global player identity and no negation.
- Phase 7 single-process Soo/Min self-play, full Min ranking targets, and
  explicit abort-without-target behavior at the move safety limit.
- Phase 8 CPU sparse replay with deterministic sampling, bounded capacity,
  dense minibatch collation, and strict compatibility isolation.
- Phase 9 eager FP32 AdamW trainer shared by Soo/Min, with identity/config
  validation, separate policy/value loss metrics, and value-shape enforcement.
- Phase 10 atomic checkpoint save/load with strict pre-load compatibility
  gates and model, AdamW, training-step, and training-config round trips.
- Phase 11 deterministic Soo/Min arenas with training noise disabled, Soo
  seat/order balancing, Min candidate rotation, and all six 3P turn orders.
- Phase 12 executable authoritative smoke coverage, self-play metrics,
  developer documentation, and final regression hardening.
- Post-review hardening independently crosses candidate seat with turn order
  (4 Soo and 18 Min matchups), adds deterministic ruleset fingerprint and seat
  layout gates, makes failed checkpoint restore non-mutating, restores training
  configuration, preserves replay compatibility through collation, and enforces
  exact Soo/Min value-target domains.

Current full-suite verification before Phase 7/8 began:

    187 passed, 5 skipped

Current full-suite verification after Phase 8:

    194 passed, 5 skipped in 6.64s

Current full-suite verification after Phase 9:

    198 passed, 5 skipped in 8.93s

Current full-suite verification after Phase 11:

    205 passed, 5 skipped in 7.95s

Current Phase 12 verification:

    AlphaZero subsystem: 99 passed in 7.12s
    Existing game/GUI suite: 134 passed, 5 skipped in 4.21s
    Combined: 233 passed, 5 skipped
    Executable smoke: exit 0 with Soo/Min self-play, training, exact checkpoint
    restore, and balanced arena checks all successful

Follow-up review hardening also rejects partial arena balance cycles,
checkpoint training/runtime configuration disagreement, and incomplete smoke
result sets.

The current PyTorch mamba environment has a local PySide6.QtCore DLL loader
mismatch, so the two passing suites were run with their respective working
interpreters. This is an environment collection issue, not a repository test
failure.

Next implementation scope:

- Milestone 1 is complete. Milestone 2 scope remains intentionally deferred.

This status block records implemented behavior only. C++, central batching,
OpenVINO, distributed self-play, and GUI AlphaZero routing remain deferred.

You are starting from ZERO conversational context. Do not assume any previous prompt, design discussion, implementation plan, or unstated requirement exists.

Repository:
    GitHub: ubraket513/alphadiamond

Your job is to inspect the repository first, treat its current game engine and rules as authoritative, then implement the approved AlphaZero architecture described below.

======================================================================
0. OPERATING PRINCIPLE
======================================================================

The existing repository implements a game called Diamond.

It is NOT standard 121-hole Chinese Checkers.

The CURRENT repository implementation of Diamond is the source of truth for:

- board topology
- camp geometry
- move legality
- jump rules
- multi-hop canonical paths
- turn progression
- player seating
- player finishing/ranking
- terminal conditions
- GUI/controller behavior
- persistence semantics

Never rewrite those rules to match assumptions from conventional Chinese Checkers.

If this prompt and the actual current repository disagree about game rules,
the repository wins.

If the repository has evolved since this prompt was written:
1. inspect the current implementation and tests,
2. identify the discrepancy,
3. preserve the current authoritative rule behavior,
4. adapt the AlphaZero subsystem to it,
5. document the discrepancy in your implementation report.

Do not silently change existing game behavior.

======================================================================
1. REQUIRED DEVELOPMENT WORKFLOW
======================================================================

Before making substantive implementation changes:

1. Read the repository structure.
2. Read README and pyproject configuration.
3. Read the entire authoritative game layer relevant to:
   - board
   - coordinates
   - move
   - rules
   - state
   - session
4. Read the Agent / MoveRequest / MoveProposal abstractions.
5. Read the AI worker and GameController integration path.
6. Read the existing game and controller tests.
7. Run the existing test suite BEFORE adding AlphaZero code.
8. Record the baseline result.

Use the Superpowers development workflow available in Codex:

- superpowers:using-superpowers
- superpowers:using-git-worktrees where appropriate
- superpowers:writing-plans
- superpowers:test-driven-development
- superpowers:systematic-debugging for unexpected failures
- superpowers:verification-before-completion
- superpowers:requesting-code-review before considering a milestone complete

Create a detailed implementation plan before implementation.

Save the plan under:

    docs/superpowers/plans/YYYY-MM-DD-alphazero-milestone1.md

Use small, independently testable tasks and TDD.

Do not perform an unrelated repository refactor.

Do not rewrite working game or GUI code merely to make the new subsystem
look cleaner.

Prefer dependency inversion and adapters over invasive changes.

======================================================================
2. APPROVED SYSTEM ARCHITECTURE
======================================================================

Long-term system:

TRAINING

    Existing Diamond game semantics
              |
              v
    AlphaZero GameAdapter
              |
              v
    MCTS / self-play workers
              |
              v
    Evaluator abstraction
              |
              v
    Python inference coordinator
              |
              v
    PyTorch
              |
              v
    NVIDIA A30

Training pipeline eventually includes:

- self-play
- replay
- neural-network training
- evaluation / arena
- checkpoint promotion
- metrics
- multiple self-play workers
- batched centralized inference
- eventual C++ MCTS/game hot path

DEPLOYMENT

    PySide6 / QML
          |
          v
    GameController
          |
          v
    AlphaZero Agent
          |
          v
    C++ MCTS
          |
          v
    OpenVINO
          |
          v
    Intel Iris Xe

This long-term architecture is approved.

However:

    ONLY MILESTONE 1 IS TO BE IMPLEMENTED NOW.

Do not prematurely implement C++, OpenVINO, multiprocessing inference
servers, distributed self-play, or production GUI AlphaZero wiring.

Design clean interfaces so those components can be added later.

======================================================================
3. CURRENT REPOSITORY FACTS TO VERIFY
======================================================================

Verify these against the current repository before relying on them.

The repository has historically contained approximately:

    src/diamond/game/
        board.py
        coordinates.py
        move.py
        rules.py
        state.py
        session.py

    src/diamond/agents/
        base.py
        random_agent.py

    src/diamond/app/
        controller.py
        ai_worker.py

The current authoritative Diamond board has historically been:

    73 playable holes

with deterministic position IDs.

Do NOT assume 121 positions.

The expected Milestone-1 action space is therefore:

    N = number of authoritative board positions

    action_id = source * N + destination

For current Diamond:

    N = 73
    ACTION_SIZE = 73 * 73 = 5329

Do not hard-code a fictional 121-hole action universe around the current
73-hole engine.

The codec may be generic in N, but every model/checkpoint has a fixed
topology/action-space identity.

======================================================================
4. EXISTING GAME ENGINE MUST REMAIN AUTHORITATIVE
======================================================================

The AlphaZero subsystem must NOT become a second independent implementation
of Diamond rules at the Python reference stage.

Existing game code determines:

    legal moves
    legal jumps
    canonical multi-hop paths
    state transitions
    player finishing
    ranking
    match completion

For the correctness reference implementation, it is acceptable for
AlphaZeroGameAdapter to call existing authoritative Python rules/session
logic even if this is slower than an optimized implementation.

Correctness first.

Optimization comes after profiling.

For every selected AlphaZero action:

    action_id
        -> source, destination
        -> authoritative engine resolves canonical Move/path
        -> authoritative validation applies

Do not make the policy predict every intermediate jump in a multi-hop move.

The neural action is source -> FINAL destination.

The existing engine owns canonical path recovery.

======================================================================
5. MILESTONE 1 OBJECTIVE
======================================================================

Implement a fully testable Python/PyTorch AlphaZero correctness baseline
for BOTH:

    2-player Diamond
    3-player Diamond

Milestone 1 contains:

- configuration
- action codec
- canonical encoder
- game adapter
- shared graph-aware network trunk
- separate 2P and 3P neural models
- Evaluator abstraction
- deterministic DummyEvaluator
- TorchEvaluator
- scalar 2P MCTS
- vector-valued 3P MCTS
- single-process 2P self-play
- single-process 3P self-play
- CPU replay buffer
- trainer
- checkpoint save/load/compatibility validation
- basic arena
- metrics useful for correctness/debugging
- comprehensive unit and integration tests

Milestone 1 does NOT contain:

- C++ implementation
- pybind11
- OpenVINO
- Intel GPU deployment
- multiprocessing self-play
- centralized asynchronous GPU inference
- production GUI AlphaZero routing
- population-based self-play
- distributed training
- complex symmetry augmentation
- speculative optimization

======================================================================
6. PROPOSED MODULE BOUNDARY
======================================================================

Unless current repository conventions strongly justify a small adjustment,
prefer a focused package similar to:

    src/diamond/alphazero/
        __init__.py
        config.py
        action_codec.py
        encoder.py
        game_adapter.py
        replay.py
        trainer.py
        arena.py
        checkpoint.py
        metrics.py

        network/
            __init__.py
            trunk.py
            policy.py
            model_2p.py
            model_3p.py

        evaluator/
            __init__.py
            base.py
            dummy.py
            torch.py

        mcts/
            __init__.py
            tree.py
            puct.py
            search_2p.py
            search_3p.py

        selfplay/
            __init__.py
            common.py
            runner_2p.py
            runner_3p.py

Prefer small files with one responsibility.

Do NOT create giant modules containing encoder + model + MCTS + training.

Do NOT put PyTorch imports in generic MCTS code.

======================================================================
7. ACTION SPACE CONTRACT
======================================================================

Implement a versioned ActionCodec.

For a board with N positions:

    encode(source, destination):
        return source * N + destination

    decode(action_id):
        source = action_id // N
        destination = action_id % N

Current Diamond:

    N = 73
    action_size = 5329

Requirements:

- exact round-trip
- bounds validation
- deterministic
- independent from PyTorch
- no path encoded into action_id
- no duplicate semantic actions
- illegal source/destination pairs remain representable mathematically but
  are NEVER legal after masking

The authoritative engine provides the legal subset.

An ActionSpaceSpec should make version identity explicit.

At minimum preserve:

    board_size
    action_size
    action_space_version

Recommended current semantic version name:

    diamond73-srcdst-v1

Do not assume this version string if repository already has a compatible
versioning convention; integrate cleanly.

======================================================================
8. CANONICAL ENCODER
======================================================================

This is a correctness-critical component.

Never canonicalize using absolute player IDs or UI colors.

Canonicalization must derive from:

    PlayerSpec.camp
    actual match player order
    current player

The current player must always perceive their own home/target orientation
in one consistent canonical frame.

Use deterministic board-position permutations derived from Diamond geometry.

Do not manually maintain ad-hoc lookup tables if they can be derived safely
from authoritative board coordinates.

Precompute/cache immutable permutations where useful.

----------------------------------------------------------------------
8.1 PLAYER CHANNEL SEMANTICS
----------------------------------------------------------------------

For 2P:

    self
    opponent

For 3P:

preserve a stable player-relative ordering.

Use:

    self
    next
    previous

where "next" and "previous" are defined relative to the ORIGINAL match
turn-order tuple, not numeric player IDs.

Do NOT redefine channel identities merely because a finished player is
skipped by turn advancement.

Channel semantic identity must remain stable throughout a match.

----------------------------------------------------------------------
8.2 FINISH STATUS
----------------------------------------------------------------------

Diamond 3P can continue after one player has finished.

Therefore occupancy alone is not sufficient to describe competitive state.

Represent whether each canonicalized player has already placed/finished.

For example, include stable information equivalent to:

    self_finished
    next_finished
    previous_finished

This can be represented as broadcast node features or a clean global feature
mechanism compatible with the chosen network implementation.

Keep the implementation simple and export-friendly.

----------------------------------------------------------------------
8.3 REQUIRED ENCODER TESTS
----------------------------------------------------------------------

Test:

- every board position mapping
- inverse mapping
- source/destination action mapping under canonical transform
- all relevant camps
- every current-player seat
- 2P seat configurations
- 3P seat configurations
- multiple turn-order permutations
- opponent-channel identity
- finished-player features
- legal-action mask mapping
- canonical -> physical action recovery

A particularly important invariant:

    encode state canonically
    transform a legal physical action into canonical action
    inverse-transform it
    recover the exact original source/destination

Property-style exhaustive tests are preferred where practical because the
board only has 73 positions.

======================================================================
9. 2P AND 3P MUST NOT SHARE VALUE SEMANTICS
======================================================================

Shared infrastructure is encouraged.

Shared value semantics are NOT.

Use separate model classes/checkpoints for 2P and 3P.

Do not make one model dynamically switch player count at inference time.

======================================================================
10. 2-PLAYER VALUE SEMANTICS
======================================================================

For 2P:

    value is scalar
    value is from the CURRENT NODE PLAYER'S perspective
    range approximately [-1, +1]

Terminal training target:

    +1 if the sample's canonical current player eventually wins
    -1 otherwise

A tanh scalar head is appropriate.

Training loss:

    policy cross entropy
    +
    scalar value MSE or similarly justified bounded regression loss

MCTS backup must explicitly alternate perspective.

If a child leaf value is from child-to-play perspective, backing it into the
parent changes sign.

The implementation must not accidentally infer sign behavior from player IDs.

Node/edge perspective must be explicit.

Toy-tree tests MUST catch sign errors.

======================================================================
11. 3-PLAYER VALUE SEMANTICS — CRITICAL
======================================================================

Do NOT implement the 3-player value as:

    softmax winner probabilities [P1_win, P2_win, P3_win]

That is NOT the approved design.

Diamond continues after first place in order to determine ranking.

Use a FINAL PLACEMENT UTILITY VECTOR.

For final rank:

    1st place -> +1.0
    2nd place ->  0.0
    3rd place -> -1.0

For every 3P state, the model predicts one value component per player.

Use canonical relative ordering:

    [self, next, previous]

Therefore output shape:

    [B, 3]

The network may use tanh on each component or otherwise guarantee an
appropriate bounded prediction.

Training target is the final placement utility remapped into the sample's
canonical player ordering.

For example:

    final order = Player B, Player C, Player A

global utility:

    A = -1
    B = +1
    C = 0

If a training position's canonical order is:

    [C, A, B]

target becomes:

    [0, -1, +1]

This mapping must be explicitly tested.

Use MSE or Huber loss for this utility vector unless there is a compelling
repository-specific reason to choose another regression objective.

This is NOT classification.

Do not softmax these values.

======================================================================
12. 3-PLAYER MCTS — CRITICAL
======================================================================

3P MCTS is vector-valued.

Never convert it to a paranoid/minimax scalar formulation.

Never use scalar negation during 3P backup.

Never assume the two opponents form a coalition.

Each edge/node stores vector statistics logically equivalent to:

    P: prior
    N: visit count
    W: accumulated utility vector
    Q: mean utility vector

When evaluating a canonical network output:

    [self, next, previous]

remap the vector to an unambiguous node/global player representation BEFORE
backing it through the tree.

Back up the SAME semantic player-value vector through ancestors.

No sign negation.

Selection at a node uses:

    Q[current_player] + exploration_bonus

where Q[current_player] means the component belonging to the player who is
actually choosing at that node.

The current player maximizes THEIR OWN expected final placement utility.

The implementation must correctly handle finished players being skipped in
turn advancement.

Toy tests must prove:

- vector components retain player identity
- no accidental permutation occurs
- no scalar negation occurs
- each player selects according to its own Q component
- a player does not maximize another player's utility
- finished-player skipping does not corrupt component ownership

This is one of the highest-risk parts of the entire implementation.

======================================================================
13. 3P EPISODE TERMINATION
======================================================================

Do NOT terminate 3-player self-play when the first player finishes.

Follow the current authoritative Diamond match lifecycle.

Self-play must continue until the authoritative engine has determined the
full final placement/ranking required by the rules.

The final rank is then used to generate:

    +1 / 0 / -1

placement targets for ALL states collected during that match.

This means states occurring after the first player finishes are legitimate
training data.

Do not invent alternative terminal semantics.

======================================================================
14. GRAPH-AWARE NEURAL NETWORK
======================================================================

Build an OpenVINO-friendly graph-aware network using only standard tensor
operations.

Avoid:

- PyTorch Geometric dependency
- custom CUDA operators
- custom graph kernels
- exotic dynamic operators likely to make later OpenVINO export difficult

Initial model:

    width = 128
    residual directional blocks = 6

Treat board positions as graph nodes.

Input:

    [B, N, F]

Current N:

    73

Use the board's six lattice directions.

A suitable directional residual block may use fixed direction-specific
adjacency/connectivity tensors stored as buffers plus ordinary operations
such as:

- MatMul
- Linear
- Add
- GELU
- LayerNorm

Exact implementation can adapt to repository coding conventions, but must
remain:

- deterministic
- easy to unit test
- PyTorch eager compatible
- future OpenVINO export friendly
- free of PyG coupling

======================================================================
15. POLICY HEAD
======================================================================

Policy action:

    source -> destination

Policy output:

    [B, N * N]

Current Diamond:

    [B, 5329]

Use source/destination factorization rather than an unnecessarily huge
unstructured fully connected head.

Recommended conceptual structure:

    trunk -> source embeddings
    trunk -> destination embeddings

then pairwise compatibility:

    source @ destination^T

giving:

    [B, N, N]

then flatten:

    [B, N*N]

Network returns RAW POLICY LOGITS.

Do not bake legal masking into network weights.

Legal masking belongs in evaluator/search/training logic.

Illegal actions must receive zero probability after masking/normalization.

Tests must verify:

- exact output shape
- finite logits
- legal-only normalized probabilities
- no illegal action receives search visits

======================================================================
16. 2P MODEL
======================================================================

Separate class.

Conceptually:

    Shared Diamond graph trunk
    Policy head
    Scalar value head

Output:

    policy_logits: [B, 5329]
    value:         [B, 1]

Value should be bounded appropriately.

======================================================================
17. 3P MODEL
======================================================================

Separate class.

Conceptually:

    Shared Diamond graph trunk
    Policy head
    Placement utility head

Output:

    policy_logits: [B, 5329]
    value:         [B, 3]

The value vector is:

    [self, next, previous]

and predicts expected final placement utility.

Do NOT apply softmax to the value output.

======================================================================
18. EVALUATOR ABSTRACTION
======================================================================

MCTS must know NOTHING about PyTorch.

Define a clean evaluator interface.

Exact type names may be adjusted to established project style, but the
semantic contract should resemble:

    EvalRequest:
        state or encoded state reference
        player mapping
        legal actions

    EvalResult:
        policy probabilities or legal-action priors
        value

Prefer an interface that naturally admits BATCH evaluation even though
Milestone 1 may often evaluate one state at a time.

Implement:

    Evaluator protocol / ABC
    DummyEvaluator
    TorchEvaluator

DummyEvaluator:

- deterministic
- configurable
- no torch dependency required
- useful for toy MCTS tests

TorchEvaluator:

- PyTorch eager
- FP32 first
- configurable device
- eval mode
- no gradient during inference
- masks illegal actions safely
- handles degenerate numerical cases explicitly

Do not create a central asynchronous inference service yet.

But do not design the evaluator API in a way that prevents batching later.

======================================================================
19. MCTS COMMON REQUIREMENTS
======================================================================

Implement PUCT-style MCTS.

Expansion only creates authoritative legal actions.

At root:

- optionally apply Dirichlet exploration noise
- noise ONLY over legal root actions
- deterministic under configured RNG seed
- configurable alpha/epsilon

Search output must include at least:

- visit counts
- root policy from visits
- selected action
- useful root value/debug information

Temperature must be configurable by move number.

Examples:

    early moves: temperature > 0
    later moves: near/at deterministic argmax

Do not hard-code training-specific exploration into evaluation arena behavior.

Arena should normally use deterministic/no-noise settings.

======================================================================
20. RANDOMNESS AND REPRODUCIBILITY
======================================================================

Centralize/configure randomness.

Use explicit seeds for:

- Python RNG
- NumPy if introduced
- PyTorch
- root Dirichlet sampling
- action sampling
- replay sampling
- self-play

Tests must not depend on process-global uncontrolled RNG state.

A fixed configuration + fixed seed should provide reproducible reference
behavior as far as PyTorch/platform limitations reasonably permit.

======================================================================
21. SELF-PLAY
======================================================================

Implement separate entry points/runners for:

    2P
    3P

Common utilities can be shared.

Milestone 1 is SINGLE PROCESS.

3P baseline is:

    theta vs theta vs theta

using the same current network for all seats.

Do NOT implement population-based self-play yet.

Each training sample needs logically:

- encoded/canonical state information
- policy target derived from MCTS visit distribution
- final value target
- sufficient version/schema information for validation/debugging

For 2P:

    z in {-1, +1}

For 3P:

    z in R^3
    placement utility vector in canonical order
    values drawn from {+1, 0, -1}

Self-play should use the authoritative Diamond game engine.

======================================================================
22. LONG / PATHOLOGICAL GAMES
======================================================================

Inspect current Diamond rules for repetition/draw handling.

Do not invent a fake winner if a game exceeds a practical safety bound.

Add a configurable safety limit such as:

    max_game_moves

If exceeded:

- abort the self-play game
- do NOT label it with a false training winner/ranking
- discard invalid/incomplete value-target samples from that game
- increment/log an aborted-game metric

Do not silently treat an aborted match as draw unless Diamond rules explicitly
define that result.

======================================================================
23. REPLAY BUFFER
======================================================================

Replay remains CPU-resident in Milestone 1.

2P and 3P replay must be logically separate because value schemas differ.

Do NOT persist a dense float32 [5329] policy vector for every sample unless
measurement proves this is appropriate.

Prefer sparse search targets:

    legal/visited action IDs
    visit counts or normalized probabilities

At minibatch collation time, construct dense policy targets if the training
loss requires them.

This keeps replay representation efficient and future-friendly.

Replay requirements:

- bounded capacity
- deterministic seeded sampling
- schema validation
- correct 2P/3P isolation
- no silent mixing of incompatible encoder/action/ruleset versions

======================================================================
24. TRAINER
======================================================================

Use PyTorch.

Initial correctness configuration:

    eager execution
    FP32

Optimizer:

    AdamW

Configuration-driven:

- learning rate
- weight decay
- batch size
- training steps
- replay capacity
- MCTS simulations
- PUCT constant
- root noise
- temperature schedule
- model width
- residual block count
- device
- seed
- checkpoint interval
- arena settings
- safety move limit

Do NOT optimize before profiling.

After Milestone 1 correctness is established, future performance work may
evaluate:

    BF16 / AMP
    torch.compile
    larger inference batches

But those are not required to complete Milestone 1.

Trainer must support:

2P:
    policy loss + scalar value loss

3P:
    policy loss + placement-vector regression loss

Track the component losses separately.

======================================================================
25. CHECKPOINTS — STRICT COMPATIBILITY
======================================================================

Checkpoint format must contain enough metadata to reject incompatible
models BEFORE inference/training continues.

At minimum include:

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
    training_step
    model state
    optimizer state when training checkpoint
    training_config

Recommended value semantic identities:

2P:
    current-player-scalar-winloss-v1

3P:
    canonical-placement-utility-1-0-minus1-v1

Do not allow a 2P checkpoint to load as 3P.

Do not allow a checkpoint built for a different:

- board topology
- action space
- encoder
- ruleset
- value interpretation

to load silently.

Implement explicit compatibility exceptions with useful messages.

----------------------------------------------------------------------
25.1 RULESET FINGERPRINT
----------------------------------------------------------------------

Prefer a deterministic ruleset/topology fingerprint derived from immutable
semantics such as relevant canonicalized representations of:

- board positions / coordinates
- camp membership
- direction definitions
- seat layout semantics
- other static topology data that affects neural interpretation

Do not include runtime noise or irrelevant metadata.

The goal is not cryptographic security.

The goal is accidental incompatibility detection.

======================================================================
26. ARENA
======================================================================

Implement a basic evaluator/promotion arena.

No training root noise.

Use deterministic or appropriately low-temperature decision settings.

----------------------------------------------------------------------
26.1 2P ARENA
----------------------------------------------------------------------

Avoid seat / move-order bias.

Candidate and baseline/best model should play balanced seat/order assignments.

Track:

- wins
- losses
- win rate

----------------------------------------------------------------------
26.2 3P ARENA
----------------------------------------------------------------------

Baseline comparison concept:

    candidate / best / best

Rotate candidate placement.

Also account for turn-order permutations supported by the authoritative
engine.

Do not assume numeric player IDs define a fixed geometric seat or fixed turn
ordering.

A balanced subset is acceptable for early correctness testing.

Track at minimum:

    1st-place rate
    2nd-place rate
    3rd-place rate
    mean placement utility

with utility:

    +1, 0, -1

Promotion policy should be configuration-driven.

Do not implement population training in Milestone 1.

======================================================================
27. EXISTING GUI CONTRACT MUST BE PRESERVED
======================================================================

Do NOT rewrite QML.

Do NOT bypass GameController.

Do NOT auto-commit AlphaZero moves.

The expected future contract is:

    AlphaZeroAgent
        chooses
    MoveProposal
        ->
    GameController
        validates/displays
        ->
    user/controller confirmation
        ->
    authoritative GameSession.commit()

AI proposals must remain proposals.

They must not mutate game state by themselves.

The existing asynchronous AI worker / controller boundary should remain
intact.

Production AlphaZero Agent wiring belongs to a later deployment milestone.

If the current GameController currently recreates RandomAgent directly when
starting a new match, note that as a later AgentFactory/AgentRouter integration
point.

Do NOT modify it merely to finish Milestone 1 unless required by a focused
test or non-invasive preparatory abstraction.

======================================================================
28. FUTURE AGENT ROUTING
======================================================================

Long-term target:

    AgentRouter / AgentFactory

supporting conceptually:

    Random
    AlphaZero2P
    AlphaZero3P

This is NOT required for the Milestone-1 training implementation.

Do not couple training code to PySide6.

Training must be runnable headlessly.

======================================================================
29. FUTURE C++ BOUNDARY — DESIGN FOR IT, DO NOT IMPLEMENT IT
======================================================================

Milestone 2 will eventually move CPU hot paths such as:

- move generation
- state transition
- MCTS
- self-play search workers

into C++.

Python remains responsible for:

- training orchestration
- replay management
- PyTorch
- GPU inference coordination
- checkpoint lifecycle

Do NOT use libtorch in C++ MCTS.

MCTS should communicate with inference through plain semantic DTO/buffer
contracts.

Future C++ search does not need to carry full multi-hop Move.path as the
policy action identity.

Search action identity is:

    source
    destination

At the Agent/controller boundary the authoritative engine can recover the
canonical multi-hop path.

Future C++ work must have randomized differential tests against the Python
reference implementation:

    same state
        ->
    exact same legal source/destination action set

Python correctness baseline built now becomes the oracle for Milestone 2.

======================================================================
30. FUTURE CENTRAL INFERENCE — DESIGN FOR IT, DO NOT IMPLEMENT IT
======================================================================

Milestone 2 target:

    multiple CPU self-play workers
            |
            v
    inference request queue
            |
            v
    central Python inference coordinator
            |
            v
    batched PyTorch
            |
            v
    NVIDIA A30

Therefore evaluator/search boundaries created now must not assume:

    one search == one model call
    one model == one MCTS object
    torch.Tensor must cross the MCTS boundary

Keep semantic boundaries batch-friendly.

Do not build multiprocessing yet.

======================================================================
31. FUTURE PERFORMANCE OPTIMIZATION ORDER
======================================================================

Do not optimize in a different order without evidence.

The intended order is:

1. Python/PyTorch correctness
2. profiling
3. eager FP32 baseline metrics
4. batching
5. BF16/AMP evaluation
6. torch.compile benchmark
7. identify true CPU hot paths
8. C++ migration
9. OpenVINO deployment optimization

Collect enough timing information in Milestone 1 to make later profiling
possible, but do not create a large observability framework.

Useful future metrics include:

    games/hour
    simulations/second
    evaluated states/second
    mean inference batch size
    search latency
    training step latency
    aborted game count

======================================================================
32. FUTURE OPENVINO — DESIGN FOR IT, DO NOT IMPLEMENT IT
======================================================================

Milestone 3:

    trained PyTorch checkpoint
        ->
    exportable graph
        ->
    OpenVINO
        ->
    validation against PyTorch
        ->
    Intel Iris Xe
        ->
    AlphaZeroAgent
        ->
    existing GUI proposal/confirmation flow

Network design now should avoid making this unnecessarily difficult.

Do not add OpenVINO dependency in Milestone 1.

======================================================================
33. TESTING PHILOSOPHY
======================================================================

Correctness before speed.

Tests are part of the implementation, not an optional cleanup task.

Use TDD for new components.

Every important mathematical semantic must have a small deterministic test,
especially 3P backup behavior.

Do not rely exclusively on full self-play tests to catch MCTS logic bugs.

======================================================================
34. REQUIRED TEST GROUPS
======================================================================

Create focused tests for at least the following.

----------------------------------------------------------------------
34.1 ACTION CODEC
----------------------------------------------------------------------

- encode/decode round trip
- bounds
- N=73
- action_size=5329
- source/destination uniqueness

----------------------------------------------------------------------
34.2 ENCODER / CANONICAL MAPPING
----------------------------------------------------------------------

- shape
- occupancy preservation
- camp canonicalization
- all player seats
- 2P mapping
- 3P mapping
- turn-order permutations
- stable self/next/previous semantics
- finished-player feature
- canonical action mapping
- inverse action mapping
- legal-mask mapping

----------------------------------------------------------------------
34.3 NETWORK
----------------------------------------------------------------------

2P:
    input  [B,73,F]
    policy [B,5329]
    value  [B,1]

3P:
    input  [B,73,F]
    policy [B,5329]
    value  [B,3]

Verify:

- finite outputs
- gradients exist
- configurable batch sizes
- train/eval modes
- no illegal assumptions about absolute player ID

----------------------------------------------------------------------
34.4 EVALUATOR
----------------------------------------------------------------------

- DummyEvaluator deterministic behavior
- TorchEvaluator output shape
- legal masking
- probability normalization
- all-illegal/invalid situations produce explicit controlled failure rather
  than NaN
- no autograd during evaluation

----------------------------------------------------------------------
34.5 2P MCTS
----------------------------------------------------------------------

Toy games/trees must test:

- only legal actions expanded
- root visits sum correctly
- deterministic search with fixed seed/no noise
- winning move in one is preferred
- losing move is rejected when avoidable
- scalar backup sign alternates exactly
- terminal values have correct perspective
- legal policy normalization
- root Dirichlet touches legal actions only

A dedicated sign test is mandatory.

----------------------------------------------------------------------
34.6 3P MCTS
----------------------------------------------------------------------

Use tiny deterministic toy game trees independent of the full Diamond engine
where useful.

Tests MUST prove:

- value vector player ordering
- canonical -> global remapping
- backup preserves vector identity
- NO scalar negation
- player A selects according to A's component
- player B selects according to B's component
- player C selects according to C's component
- a choice beneficial to another player is not accidentally selected as
  self-beneficial
- finished-player skip does not rotate value identity
- terminal placement target [+1,0,-1] maps correctly

These tests are mandatory and should be easy to inspect manually.

----------------------------------------------------------------------
34.7 SELF-PLAY
----------------------------------------------------------------------

2P:
- produces valid samples
- only legal selected actions
- final targets correct

3P:
- theta vs theta vs theta
- continues after first finisher
- finishes according to authoritative engine
- rank utilities are correct
- every sample receives correctly canonicalized placement target
- incomplete/aborted games do not receive fabricated targets

----------------------------------------------------------------------
34.8 REPLAY
----------------------------------------------------------------------

- capacity behavior
- seeded sampling
- sparse policy storage
- correct batch collation
- incompatible sample schema rejected
- 2P/3P isolation

----------------------------------------------------------------------
34.9 TRAINER
----------------------------------------------------------------------

- one tiny 2P training step
- one tiny 3P training step
- parameters update
- losses finite
- value target shapes correct
- policy target correctly constructed
- optimizer state usable

Do not require a GPU for unit tests.

----------------------------------------------------------------------
34.10 CHECKPOINT
----------------------------------------------------------------------

- 2P round trip
- 3P round trip
- optimizer round trip
- training_step preserved
- config preserved

Reject:

- 2P -> 3P
- wrong ruleset
- wrong topology
- wrong encoder
- wrong action-space version
- wrong value-semantics version
- wrong network config when incompatible
- corrupted/incomplete metadata

----------------------------------------------------------------------
34.11 ARENA
----------------------------------------------------------------------

- seat rotation
- turn-order balancing
- statistics aggregation
- placement utility calculation
- deterministic dummy agents/evaluators

----------------------------------------------------------------------
34.12 EXISTING REGRESSION TESTS
----------------------------------------------------------------------

All pre-existing game/UI/controller tests must continue to pass.

Milestone 1 must not regress RandomAgent behavior or proposal/confirmation
workflow.

======================================================================
35. PYTORCH / DEPENDENCY POLICY
======================================================================

Inspect the current dependency management before editing it.

Add only dependencies actually needed for Milestone 1.

PyTorch is expected.

Do not introduce:

- PyTorch Geometric
- OpenVINO
- pybind11
- CMake
- CUDA extensions
- distributed frameworks

unless the current repository already has a compelling reason, which must be
documented.

Tests should run on CPU.

GPU-specific execution should be optional/configured.

======================================================================
36. CONFIGURATION
======================================================================

Do not scatter magic constants through search/training code.

Use structured configuration.

Separate concerns cleanly, for example:

    NetworkConfig
    MCTSConfig
    SelfPlayConfig
    ReplayConfig
    TrainingConfig
    ArenaConfig

Avoid unnecessary configuration frameworks.

Plain typed dataclasses are sufficient unless repository conventions dictate
otherwise.

Configs saved in checkpoints must be serializable.

======================================================================
37. ERROR HANDLING
======================================================================

Fail loudly on semantic incompatibility.

Examples:

- wrong player count
- unsupported board topology
- malformed action ID
- checkpoint/version mismatch
- evaluator returns wrong value shape
- legal mask has incorrect size
- 3P value vector has wrong player mapping
- training sample uses incompatible schema

Do not silently coerce incompatible objects.

Semantic corruption is worse than a hard failure.

======================================================================
38. CODE QUALITY
======================================================================

Use:

- type hints
- small interfaces
- immutable dataclasses where appropriate
- explicit perspective/player mapping types
- clear docstrings on mathematical contracts

Avoid vague names such as:

    value2
    arr
    tmp_player

Prefer names that expose semantics, such as:

    canonical_player_ids
    canonical_to_global_player
    placement_utilities
    legal_action_ids
    root_visit_counts

For MCTS value logic, comments should describe PERSPECTIVE precisely.

Do not write comments that merely restate syntax.

======================================================================
39. IMPORTANT DOMAIN INVARIANTS
======================================================================

These invariants must appear either in code-level documentation, tests, or
both:

INVARIANT A
    The current repository's Diamond rules are authoritative.

INVARIANT B
    Current action identity is source -> final destination.

INVARIANT C
    Current Diamond topology is 73 positions unless the repository itself
    has changed.

INVARIANT D
    Neural canonicalization is based on camp/geometry and actual match
    turn order, never arbitrary player IDs/colors.

INVARIANT E
    2P value is scalar current-player win/loss utility.

INVARIANT F
    2P MCTS alternates scalar perspective/sign.

INVARIANT G
    3P value is a vector of final placement utilities.

INVARIANT H
    3P utility is:
        first  = +1
        second = 0
        third  = -1

INVARIANT I
    3P MCTS NEVER negates the value vector.

INVARIANT J
    3P selection uses the Q component of the player making that decision.

INVARIANT K
    3P self-play continues until authoritative full ranking is determined.

INVARIANT L
    AlphaZero never auto-commits a GUI move.

INVARIANT M
    MCTS does not depend on PyTorch.

INVARIANT N
    C++ optimization is deferred until the Python reference is verified.

======================================================================
40. MILESTONE 1 ACCEPTANCE CRITERIA
======================================================================

Do not declare Milestone 1 complete unless ALL are true:

1. Existing repository tests pass.

2. New AlphaZero unit/integration tests pass on CPU.

3. Action codec exactly matches the current Diamond board size.

4. Canonical encoder passes exhaustive/strong mapping tests.

5. 2P model has correct policy/value shapes.

6. 3P model has correct policy/value shapes.

7. 2P toy MCTS proves scalar sign handling.

8. 3P toy MCTS proves vector backup without negation.

9. 3P MCTS selects the Q component belonging to the acting player.

10. Single-process 2P self-play can complete at least a small smoke run.

11. Single-process 3P self-play can complete at least a small smoke run
    through full ranking.

12. Replay can store/sample both modes with correct isolation.

13. Trainer can perform finite CPU smoke steps for 2P and 3P.

14. Checkpoints round-trip successfully.

15. Incompatible checkpoints are rejected.

16. Basic arena works with deterministic small configurations.

17. No PyTorch dependency leaks into generic MCTS interfaces.

18. Existing GUI/controller semantics remain unchanged.

19. No C++/OpenVINO/distributed scope creep was introduced.

20. Verification commands and their actual outputs have been inspected.

======================================================================
41. PERFORMANCE IS NOT A COMPLETION CRITERION YET
======================================================================

Do not reject a correct reference implementation because Python game
transitions are slow.

Do not replace authoritative Python semantics prematurely.

At Milestone 1, prioritize:

    correct
    deterministic
    understandable
    testable
    profileable

over:

    maximum games/second

The reference system is intended to become the oracle against which the
future C++ implementation is verified.

======================================================================
42. REQUIRED EXECUTION ORDER
======================================================================

Use this order unless repository inspection reveals a concrete dependency
reason to make a small adjustment:

Phase 0:
    repository inspection
    baseline tests
    architecture consistency report
    implementation plan

Phase 1:
    version/schema/config primitives
    ActionCodec
    GameAdapter skeleton

Phase 2:
    canonical encoder
    canonical player/action mapping
    exhaustive tests

Phase 3:
    graph trunk
    policy head
    2P model
    3P model
    model tests

Phase 4:
    Evaluator interface
    DummyEvaluator
    TorchEvaluator

Phase 5:
    common MCTS data structures / PUCT
    2P MCTS
    intensive 2P toy tests

Phase 6:
    3P vector MCTS
    intensive player-identity/vector-backup toy tests

Do not proceed past Phase 6 while any perspective/value-semantic test is
uncertain or failing.

Phase 7:
    self-play 2P
    self-play 3P
    aborted-game handling

Phase 8:
    sparse replay
    batch collation

Phase 9:
    trainer
    tiny CPU optimization smoke tests

Phase 10:
    checkpointing
    compatibility tests

Phase 11:
    arena

Phase 12:
    complete regression/verification
    documentation
    Milestone-1 final report

======================================================================
43. IMPLEMENTATION PLAN QUALITY
======================================================================

The implementation plan must identify exact files to create/modify.

Each task should include:

- files
- interfaces it consumes
- interfaces it produces
- failing test first
- command to demonstrate failure
- minimal implementation
- command to demonstrate pass
- focused commit

Do not use placeholders such as:

    TODO
    TBD
    "add tests"
    "handle errors appropriately"
    "implement the rest"

The plan must be executable by an engineer with no prior context.

======================================================================
44. GIT / CHANGE HYGIENE
======================================================================

Keep changes focused.

Do not mix unrelated GUI cleanup or formatting churn into AlphaZero commits.

Use small meaningful commits.

Examples of reasonable boundaries:

    test: specify Diamond action codec
    feat: add versioned action codec
    test: specify canonical player mapping
    feat: add Diamond canonical encoder
    feat: add graph policy-value models
    feat: add evaluator abstraction
    feat: add scalar two-player MCTS
    feat: add vector three-player MCTS
    feat: add Diamond self-play runners
    feat: add sparse replay buffer
    feat: add AlphaZero trainer
    feat: add compatible checkpoint format
    feat: add baseline arena

Do not blindly use these exact commit messages if the actual work decomposition
differs.

Never rewrite published history.

======================================================================
45. WHEN SOMETHING FAILS
======================================================================

Do not patch around unexplained failures.

Use systematic debugging.

For a failure:

1. reproduce it
2. isolate it
3. identify root cause
4. verify assumptions against authoritative game code
5. add/fix the smallest test exposing the bug
6. make the minimal correction
7. rerun focused tests
8. rerun broader regression tests

Especially do not "fix" game rules to satisfy an AlphaZero assumption.

Adapt AlphaZero to Diamond.

======================================================================
46. DOCUMENTATION
======================================================================

Add concise developer documentation explaining:

- AlphaZero package boundary
- 2P value semantics
- 3P placement utility semantics
- canonical player ordering
- action encoding
- checkpoint compatibility
- how to run unit tests
- how to run tiny self-play
- how to run a tiny CPU training smoke test
- how to run an arena smoke test

Do not write aspirational documentation claiming Milestone 2/3 features
already exist.

Clearly label C++, central batching and OpenVINO as future work.

======================================================================
47. FINAL VERIFICATION
======================================================================

Before claiming completion:

Run the existing full test suite using the repository's supported headless
Qt configuration if required.

Run the new AlphaZero test suite.

Run focused smoke commands for:

- 2P self-play
- 3P self-play
- 2P training
- 3P training
- checkpoint round trip
- arena

Inspect actual command output.

Do not state "tests pass" without having run them.

Do not state GPU/OpenVINO performance numbers that were not measured.

If something cannot be run in the available environment, state exactly:

- what could not run
- why
- what was verified instead

======================================================================
48. FINAL REPORT FORMAT
======================================================================

At the end, report:

A. Repository baseline
    - authoritative rules/topology discovered
    - baseline tests and result

B. Implemented architecture
    - package/file summary
    - key interfaces

C. Game semantics
    - board/action size
    - canonicalization
    - 2P value semantics
    - 3P placement semantics

D. MCTS correctness
    - 2P sign tests
    - 3P vector/player-identity tests

E. Training pipeline
    - self-play
    - replay
    - trainer
    - checkpoint
    - arena

F. Verification
    - commands actually executed
    - test counts/results
    - smoke results

G. Existing repository changes
    - explicitly list any modifications outside src/diamond/alphazero and
      AlphaZero-specific tests/config/docs
    - explain why each was necessary

H. Deferred milestones
    - C++ hot path
    - central batched inference
    - BF16 / torch.compile profiling
    - OpenVINO
    - AlphaZero GUI AgentRouter integration

I. Risks / follow-up
    - observed long-game behavior
    - training stability questions
    - performance bottlenecks supported by measurements only
    - any repository semantics that deserve additional differential tests

======================================================================
49. STOP CONDITIONS
======================================================================

Do NOT stop for ordinary implementation decisions that a senior engineer can
resolve from the repository and this specification.

Use sound judgment and proceed.

However, stop and clearly report instead of guessing if you discover a
fundamental contradiction such as:

- current repository no longer implements Diamond
- authoritative topology is no longer deterministically identifiable
- legal move semantics are internally inconsistent
- full 3P ranking cannot be determined from current engine behavior
- existing tests contradict the implementation on a core game rule

A minor filename difference, refactor, or changed internal API is NOT a stop
condition.

Adapt to it.

======================================================================
50. MOST IMPORTANT REMINDERS
======================================================================

Do not implement Chinese Checkers.

Implement AlphaZero for the CURRENT Diamond engine.

Do not assume 121 holes.

Use the authoritative current board size.

For the known current version, expect 73 holes and 5329 source/destination
actions.

Do not classify the 3P value as winner probability.

Use final placement utilities:

    first  +1
    second  0
    third  -1

Do not negate 3P values during backup.

Do not scalarize 3P search.

Do not infer canonical player semantics from numeric IDs.

Do not terminate 3P self-play after first place.

Do not let MCTS import or depend upon torch.

Do not auto-commit GUI moves.

Do not optimize in C++ before the Python reference is correct.

Tests demonstrating these facts are more important than comments claiming
them.

Begin by inspecting the current repository and running its baseline tests.
Then write the detailed implementation plan.
Then implement Milestone 1 task-by-task using TDD.
