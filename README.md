# Chinese Checkers — 3-Player Controller Console

A desktop **tournament/operator console** for running a real 3-player Chinese
Checkers match. It is not an online game: one human *Controller* sits at the
computer, records the moves that Players 1 and 2 make, asks the agent for
Player 3's move, and physically plays that move on the real board before
confirming it.

Player 3 is currently driven by a `RandomAgent`. The whole point of the
architecture is that it can be replaced by an AlphaZero agent later **without
touching the GUI or the controller** — see
[Future AlphaZero integration point](#future-alphazero-integration-point).

![board](guideline/board_sample.png)

---

## Project overview

| | |
|---|---|
| Board | Standard 121-hole six-pointed star, derived from lattice coordinates |
| Players | P1 human, P2 human, P3 agent — 10 pieces each, 30 total |
| Turn order | P1 → P2 → P3 → P1 … |
| Agent | `RandomAgent` (uniform over legal moves, seedable) |
| GUI | PySide6 / Qt 6 / Qt Quick / QML |
| Engine | Pure Python, no Qt imports, fully unit-tested headless |

Every move — human or agent — goes through the same four stages:

```
selection → proposal → confirmation → commit
```

Nothing reaches the authoritative game state until the Controller confirms.

---

## Architecture

```
                 ┌─────────────────┐
                 │ QML  (render,   │   Board.qml, Hole.qml, Piece.qml,
                 │ animate, input) │   SidePanel.qml, AiPanel.qml, …
                 └────────┬────────┘
                          │  properties / slots / list models
                 ┌────────┴────────┐
                 │ GameController  │   turn state machine, proposals,
                 │  (app/)         │   generation tokens, animation timing
                 └────────┬────────┘
             ┌────────────┴────────────┐
             │                         │
     ┌───────┴───────┐        ┌────────┴────────┐
     │ GameSession   │        │ AiWorker        │  QThreadPool, 1 thread
     │  (game/)      │        │  (app/)         │
     └───────┬───────┘        └────────┬────────┘
             │                         │
   rules / board / state /             │
   move / history / save-load     ┌────┴────────┐
                                  │ Agent       │  RandomAgent today,
                                  │ (agents/)   │  AlphaZeroAgent later
                                  └─────────────┘
```

**Dependency rule:** `game/` and `agents/` never import PySide6. QML never
computes legality. All rule decisions — including validation of moves an agent
proposes — happen in `game/rules.py`.

### Layout

```
src/chinese_checkers/
├── main.py              entry point: engine + QML wiring
├── app/
│   ├── controller.py    GameController: the turn state machine
│   ├── models.py        BoardModel, PieceModel, MoveHistoryModel,
│   │                    PlayerModel, BoardGeometry
│   └── ai_worker.py     async agent boundary + generation tokens
├── game/
│   ├── coordinates.py   cube lattice coordinates and the 6 directions
│   ├── board.py         121 holes, neighbours, camps (derived, not hard-coded)
│   ├── move.py          Move / MoveKind
│   ├── state.py         GameState (immutable), PlayerSpec, initial position
│   ├── rules.py         move generation, validation, win detection
│   ├── history.py       MoveRecord with a full state snapshot
│   └── session.py       authoritative game: commit / undo / save / load
├── agents/
│   ├── base.py          Agent protocol, MoveRequest, MoveProposal
│   └── random_agent.py  RandomAgent
└── qml/
    ├── Main.qml  Board.qml  Hole.qml  Piece.qml
    ├── SidePanel.qml  GamePanel.qml  PlayerPanel.qml
    ├── MovePanel.qml  AiPanel.qml  HistoryPanel.qml
    ├── StatusBar.qml  PanelSection.qml  ActionButton.qml
    └── Style/Theme.qml   ← every colour, size and duration lives here
```

`session.py` is an addition to the structure sketched in the blueprint: it keeps
commit/undo/persistence in the engine layer so those paths stay testable without
Qt, and leaves `GameController` a thin Qt wrapper.

### Board geometry

The 121 holes are **derived**, never hard-coded as pixels. Using cube
coordinates with `x + y + z == 0`:

| set | constraint | holes |
|---|---|---|
| triangle up | `x ≥ -4 ∧ y ≥ -4 ∧ z ≥ -4` | 91 |
| triangle down | `x ≤ 4 ∧ y ≤ 4 ∧ z ≤ 4` | 91 |
| central hexagon (overlap) | `-4 ≤ x, y, z ≤ 4` | 61 |
| **star** | union | **121** |

Each camp is one inequality (`x ≥ 5`, `z ≤ -5`, …) giving exactly 10 holes, and
a camp's opposite is the same axis with the sign flipped — which is precisely
the "move to the camp across the board" relation. Position IDs `0…120` come
from sorting by `(z, x)`, so they are stable and safe to persist.

QML receives logical coordinates and converts them to screen space itself; the
lattice, camp fills and move paths are drawn procedurally on a `Canvas`, so the
board stays sharp at any window size.

---

## Game rules implemented

* **Single step** — slide into an adjacent empty hole.
* **Jump** — hop over an occupied adjacent hole onto the empty hole immediately
  beyond it. Any colour may be jumped; there is no capture.
* Jumps **chain** within one turn; a turn is either one step or one jump chain.
* **Jump geometry** — each hop is built as `over = neighbour(from, d)` then
  `landing = neighbour(over, d)` with the *same* direction `d`, so origin,
  jumped-over hole and landing are always collinear on one of the six lattice
  directions, one step apart each. Bent or arbitrary-diagonal jumps are not
  representable. A chain may change direction *between* hops, never within one.
* **No cycles** — each hole is visited at most once per chain.
* **Win** — a player finishes when all ten of their pieces occupy their target
  camp. The first finisher ends the match; full 3-player ranking is
  deliberately left as a future rule option.

### Canonical path policy

Several jump chains can reach the same destination. Move generation runs a
breadth-first search with the six directions explored in a fixed order and each
hole visited once, keeping the **first** path found per destination. BFS gives
the *shortest* chain, and the fixed direction order makes the tie-break
deterministic. That single path is the canonical one, and `validate_move`
rejects any move whose path differs from it.

Step and jump destinations can never collide: a jump moves by an even
coordinate offset and a step by an odd one.

---

## Requirements

* Python **3.11+**
* PySide6 **6.6+** (Qt 6)
* pytest (only for the test suite)

Verified on Python 3.14.6 / PySide6 6.11.1.

## Installation

```bash
cd alphadiamond
pip install -e .            # runtime
pip install -e ".[dev]"     # runtime + pytest
```

## How to run

```bash
python -m chinese_checkers
```

or, after installation:

```bash
chinese-checkers
```

Options:

```bash
python -m chinese_checkers --seed 42            # reproducible RandomAgent
python -m chinese_checkers --thinking-delay 0   # skip the artificial pause
```

The window opens at 1440 × 900 and stays usable down to 980 × 640; the board
rescales with the window.

> **Headless machines / WSL without a display:** set
> `QT_QPA_PLATFORM=offscreen` to load and drive the app without a window (this
> is how the UI was verified here).

## How to run tests

```bash
QT_QPA_PLATFORM=offscreen python -m pytest
```

`QT_QPA_PLATFORM` is only needed where no display is available. The engine
tests need no Qt at all; the controller and integration tests run on a plain
`QCoreApplication` — no window is ever created.

The suite covers: board topology (121 holes, 10-hole camps, neighbour
symmetry), initial placement, move generation (steps, blocked steps, single and
chained jumps, jump collinearity, cycle prevention), turn cycling, proposal /
confirm / cancel, undo, save/load round-trips, win detection, RandomAgent
legality and seed determinism, stale AI results, and a full P1 → P2 → P3 → P1
integration loop.

---

## Controller workflow

**Human turn (P1 / P2)**

1. Click one of the current player's pieces — it is highlighted and every legal
   destination is marked (green ring = single step, amber ring = jump).
2. Click a destination. The move is shown as a **proposal**: path line on the
   board, numbered markers on intermediate jumps, and `source → destination`
   plus the full path in the MOVE panel.
3. **Confirm** commits it; **Cancel** discards it. Nothing changes until you
   confirm.

**Agent turn (P3)**

1. The turn starts automatically; the AI panel shows `Thinking…` while the
   agent runs on a worker thread. The GUI never blocks.
2. The proposal appears — `20 → 43`, with the full path if it is multi-hop.
   **It is not applied to the game state.**
3. Move that piece on the physical board, then press **Confirm AI Move**.
   **Think Again** asks for a different legal move without changing any state.

**Anytime**

| Action | Shortcut |
|---|---|
| Confirm current proposal | `Enter` |
| Cancel proposal / selection | `Esc` |
| Undo last committed move | `Ctrl+Z` |
| Save game | `Ctrl+S` |
| New game (with confirmation) | `Ctrl+N` |

Every shortcut re-checks the current phase, so none of them can confirm or undo
something the state machine does not allow.

### Turn state machine

```
WAITING_FOR_HUMAN_INPUT ──select+select──▶ HUMAN_MOVE_PROPOSED
        │                                        │ confirm
        │ (current player is an agent)           ▼
        ▼                                  ANIMATING_MOVE ──▶ next turn
    AI_THINKING ──proposal──▶ AI_MOVE_PROPOSED ──confirm──▶ ANIMATING_MOVE
                                   │ think again
                                   └──────────▶ AI_THINKING

    any ──win detected──▶ GAME_OVER
```

What the UI permits is derived entirely from this phase: the board only accepts
clicks in `WAITING_FOR_HUMAN_INPUT`, Confirm/Cancel only exist while a proposal
is pending, and the board is locked during `AI_THINKING`, `ANIMATING_MOVE` and
`GAME_OVER`.

### Error handling

Handled explicitly: selecting an empty hole, another player's piece, or an
illegal destination; clicking the board while a proposal is pending; undo or
New Game while the agent is still thinking; an agent returning an illegal move;
and closing the app with a worker in flight.

**Stale results** are prevented with a *generation token*. The controller bumps
it on every commit, undo, new game and load, and stamps it on each agent
request. A result whose token no longer matches is discarded, so a move
computed against an old board can never be applied to the current one.

---

## RandomAgent behaviour

* Chooses **uniformly at random among the legal moves** of the current player.
* Uses a private `random.Random` — never the global RNG — so `RandomAgent(seed=42)`
  reproduces the same sequence of proposals for the same sequence of states.
* Reports metadata: `agent`, `seed`, `legal_move_count`. The AI panel shows
  **only** what the agent actually reported; there are no placeholder
  evaluation, simulation or value numbers.
* **Think Again** passes the rejected move in `MoveRequest.avoid`; the agent
  picks something else, falling back to the same move only when it is the sole
  legal option.
* Runs on a worker thread with a small artificial delay (default 400 ms,
  `--thinking-delay`) purely so the "thinking" UX exists before a real search
  does. The delay lives in the worker, not in the agent, and never on the GUI
  thread.

---

## Save / load

JSON, written wherever you choose in the file dialog. The dialog opens at:

```
~/.alphadiamond/saves/
```

A save contains the schema version, the players, the current board, current
player, turn number, game status and the full move history. Pending proposals
are intentionally **not** saved.

Loading replays the saved history from the opening — which rebuilds the
per-move snapshots that undo needs — and then cross-checks the result against
the stored board, so a corrupted file is rejected rather than silently loaded.
Undo works normally after a load.

---

## Current limitations

* The agent is `RandomAgent`; there is no search, evaluation or learning.
* Only the **first finisher** is detected. Play does not continue to rank the
  remaining two players.
* No optional rules: pieces may leave a target camp, and may pass through or
  stop in camps that belong to other players.
* Save files always describe a match played from the standard opening; a
  session started from a set-up position cannot be saved and reloaded as such.
* No analysis features (policy heatmaps, candidate distributions, evaluation
  graphs). The AI panel has room reserved for them but shows nothing invented.
* Undo has no redo.

---

## Future AlphaZero integration point

The single seam is `agents/base.py`:

```python
class Agent(Protocol):
    @property
    def name(self) -> str: ...
    def choose_move(self, request: MoveRequest) -> MoveProposal | None: ...
```

`MoveRequest` carries the board, the state, the precomputed legal moves, moves
to avoid, and an optional seed. `MoveProposal` carries the move plus a free-form
`metadata` dict — which is where `search_time_ms`, `simulation_count`, `value`,
`visit_count` and `policy_probability` will go. The AI panel already renders
whatever real keys it finds, so those values appear with no QML change.

Target structure:

```
GameController → AlphaZeroAgent → MCTS → OpenVINO Runtime → Intel Iris Xe
```

To swap the agent, change one line in `main.py`:

```python
agents = {spec.id: AlphaZeroAgent(model_path=...) for spec in DEFAULT_PLAYERS
          if spec.kind is PlayerKind.AI}
```

Nothing else has to change:

* the async boundary (`AiWorker`) already runs the agent off the GUI thread and
  already tolerates searches lasting seconds;
* generation tokens already discard results from superseded positions;
* `GameState` is immutable with a flat 121-entry occupancy tuple, cheap to copy
  and ready for state hashing;
* the engine validates every agent proposal, so a buggy search cannot corrupt
  the game.
