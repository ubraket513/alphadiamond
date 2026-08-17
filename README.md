# Diamond — Controller Console

A desktop **tournament/operator console** for running a real 2- or 3-player
Diamond match. It is not an online game: one human *Controller* sits at the
computer, records the moves the human players make, asks the agent for its
seat's move, and physically plays that move on the real board before
confirming it.

The agent seat is currently driven by a `RandomAgent`. The whole point of the
architecture is that it can be replaced by an AlphaZero agent later **without
touching the GUI or the controller** — see
[Future AlphaZero integration point](#future-alphazero-integration-point).

![board](guideline/board_sample.png)

---

## Project overview

| | |
|---|---|
| Board | 73-hole six-pointed star (Diamond geometry), derived from lattice coordinates |
| Players | 2 or 3 seats, chosen at match setup — 10 pieces each |
| Turn order | Chosen at match setup; any permutation of the seats |
| Agent | `RandomAgent` (uniform over legal moves, seedable) on any one seat, or none |
| Typeface | Google Sans Flex, bundled under the OFL |
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
src/diamond/
├── main.py              entry point: engine + QML wiring
├── app/
│   ├── controller.py    GameController: the turn state machine
│   ├── models.py        BoardModel, PieceModel, MoveHistoryModel,
│   │                    PlayerModel, BoardGeometry
│   ├── fonts.py         registers the bundled typeface with Qt
│   └── ai_worker.py     async agent boundary + generation tokens
├── game/
│   ├── coordinates.py   cube lattice coordinates and the 6 directions
│   ├── board.py         73 holes, neighbours, camps (derived, not hard-coded)
│   ├── move.py          Move / MoveKind
│   ├── state.py         GameState (immutable), PlayerSpec, initial position
│   ├── rules.py         move generation, validation, win detection
│   ├── history.py       MoveRecord with a full state snapshot
│   └── session.py       authoritative game: commit / undo / save / load
├── agents/
│   ├── base.py          Agent protocol, MoveRequest, MoveProposal
│   └── random_agent.py  RandomAgent
├── assets/fonts/        bundled Google Sans Flex (OFL) + licence
└── qml/
    ├── Main.qml  Board.qml  Hole.qml  Piece.qml
    ├── SidePanel.qml  GamePanel.qml  PlayerPanel.qml
    ├── MovePanel.qml  AiPanel.qml  HistoryPanel.qml
    ├── StatusBar.qml  PanelSection.qml  ActionButton.qml
    ├── AppDialog.qml        the one styled shell every pop-up uses
    ├── NewMatchDialog.qml   seat count, turn order, agent seat
    ├── ResultDialog.qml     final standings
    ├── SegmentedControl.qml ReorderButton.qml
    └── Style/Theme.qml   ← every colour, size and duration lives here
```

`session.py` is an addition to the structure sketched in the blueprint: it keeps
commit/undo/persistence in the engine layer so those paths stay testable without
Qt, and leaves `GameController` a thin Qt wrapper.

### Board geometry

The 73 holes are **derived**, never hard-coded as pixels. Using cube
coordinates with `x + y + z == 0`:

| set | constraint | holes |
|---|---|---|
| triangle up | `x ≥ -3 ∧ y ≥ -3 ∧ z ≥ -3` | 55 |
| triangle down | `x ≤ 3 ∧ y ≤ 3 ∧ z ≤ 3` | 55 |
| central hexagon (overlap) | `-3 ≤ x, y, z ≤ 3` | 37 |
| **star** | union | **73** |

The hexagon is **7 holes across**, so each of its six sides is **4 holes** long.

#### Camps — the Diamond rule

This is where Diamond departs from traditional Chinese Checkers. A camp is not
just the star point sticking out past the hexagon: it is the **10-hole triangle
formed by that point plus the hexagon side it stands on**. The triangle's
4-hole base edge and the hexagon's 4-hole side are the *same row of holes*, so
at the opening **3 of the 6 hexagon sides are lined with pieces**.

| camp | constraint | rows | holes |
|---|---|---|---|
| `x+` | `x ≥ 3`, clipped to triangle up | 4 + 3 + 2 + 1 | 10 |

A camp's opposite is the same axis with the sign flipped — precisely the "move
to the camp across the board" relation. Position IDs `0…72` come from sorting
by `(z, x)`, so they are stable and safe to persist.

Two consequences the rules and the renderer both live with:

* The three starting camps `x+, y+, z+` are the corners of triangle *up*, so
  they sit on **alternating** hexagon sides, are mutually disjoint, and the 30
  opening pieces all fit. The three targets `x-, y-, z-` likewise.
* A `+` camp and a `-` camp still meet at a single hexagon **corner** hole.
  Camps are therefore **not** globally disjoint, and every target camp opens
  with two of its ten holes held by opponents — they clear as those opponents
  move out. Nothing in the code may assume camps are disjoint.

QML receives logical coordinates and converts them to screen space itself; the
lattice, camp fills and move paths are drawn procedurally on a `Canvas`, so the
board stays sharp at any window size.

---

## Match setup

**New Game (Ctrl+N)** opens the setup dialog: seat count, turn order and which
seat the agent drives.

### Seat layouts

Which camps are in play depends on the seat count, because every player must
aim at the camp *directly across* the board:

| Players | Camps | Why |
|---|---|---|
| 2 | `z+` vs `z-` | One camp and its literal opposite — head to head |
| 3 | `z+`, `y+`, `x+` | The corners of triangle *up*, 120° apart |

The three `+` camps are **not** opposite each other, so a 2-player match cannot
simply take two of the 3-player seats — it needs its own layout. Both layouts
keep the starting camps mutually disjoint, so the pieces always fit.

Seat ids stay tied to a board position and colour, so reordering turns never
changes where a player sits or what colour they are.

### Turn order

The seat list *is* the turn order — `next_player_id` walks it directly, and the
first entry moves first. Any permutation is allowed. The setup dialog reorders
with explicit up/down controls rather than drag-and-drop: with at most three
rows the target is always one click away, and it stays keyboard-reachable.

### Design references

UI patterns were taken from comparable production interfaces on
[Mobbin](https://mobbin.com):

* **Segmented control for the seat count** — the closed-set picker used by
  [HelloFresh's box-size dialog](https://mobbin.com/screens/7b1c1a2e-8d38-447e-8cee-54df649bdc73)
  and [Cursor's team setup](https://mobbin.com/screens/16e397ce-db38-447e-92a2-994d341a83cf):
  all options visible at once with the active one filled, so the choice and its
  alternatives read in a single glance.
* **Reorderable list in a modal** — the shape of
  [Circle's "Re-order courses"](https://mobbin.com/screens/a0684ada-0fe1-410b-a8e6-f4109d3c33a7)
  and [Behance's "Reorder Content"](https://mobbin.com/screens/c2e38307-e570-44b8-b96a-95b5709351e7):
  a numbered list with per-row controls and one confirming action.
* **Ranked result rows** — the compact leaderboard row (rank chip, identity,
  result) from [Binance's ranking screen](https://mobbin.com/screens/921a90b8-c664-4fba-9c94-f00d0b6d1748)
  and [Transit's contributor board](https://mobbin.com/screens/d1aa00b6-6263-41d9-9ac3-2c9a1a881f0d),
  rather than a podium graphic — this is an operator console, so density beats
  celebration furniture.

### Pop-ups

Every dialog uses `AppDialog.qml`, which paints its own surface, title, body
text and buttons from `Theme`. The Basic Qt Quick style ships an unstyled
`Dialog` that inherits the platform palette, which is what made the earlier
pop-ups hard to read; nothing visible is left to inherit now, message text
wraps instead of clipping, and the modal scrim is opaque enough to separate the
dialog from the board lattice behind it.

### Typography

The UI is set in **Google Sans Flex**, bundled in `src/diamond/assets/fonts/`
under the SIL Open Font License (Regular/Medium/Bold). It is registered with Qt
at startup by `app/fonts.py`, which hands the resolved family name to QML — so
the UI never asks for a family that might not exist. The font ships with the
app rather than being assumed installed: no desktop OS provides it, and a
missing family would silently fall back to a system default and shift every
metric the layout was tuned against.

Google Sans Flex is a Latin face with no arrow glyphs. Qt substitutes those
from the system font database, so move notation (`12 → 34`) still renders; UI
icons are drawn as shapes rather than typed, so they never depend on it.

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
* **Finishing** — a player is home when all ten of their pieces occupy their
  target camp. They take the next place on the podium and drop out of the turn
  rotation; the others keep playing.
* **End of match** — play stops once every place *but the last* is decided,
  because the final seat has nobody left to overtake them and takes the
  remaining place implicitly. In a 2-player match that means the first finisher
  ends it; in a 3-player match play continues past first place to settle
  **second**, and third falls out for free.

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
python -m diamond
```

or, after installation:

```bash
diamond
```

Options:

```bash
python -m diamond --seed 42            # reproducible RandomAgent
python -m diamond --thinking-delay 0   # skip the artificial pause
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

The suite covers: board topology (73 holes, 10-hole camps, neighbour
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
player, turn number, game status, the finishing order so far, and the full move
history. Pending proposals are intentionally **not** saved.

The seat list is part of the save, so a 2-player match reloads as one, in its
original turn order, whatever the session happened to be set up with. Schema
version **2** added `finish_order` and the variable seat list; version 1 files
are rejected rather than silently misread.

Loading replays the saved history from the opening — which rebuilds the
per-move snapshots that undo needs — and then cross-checks the result against
the stored board, so a corrupted file is rejected rather than silently loaded.
Undo works normally after a load.

---

## Current limitations

* The agent is `RandomAgent`; there is no search, evaluation or learning.
* At most **one** seat can be driven by an agent; there is no agent-vs-agent mode.
* A player with no legal move is reported as an error rather than being passed
  over, since the position cannot arise in normal play.
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
* `GameState` is immutable with a flat 73-entry occupancy tuple, cheap to copy
  and ready for state hashing;
* the engine validates every agent proposal, so a buggy search cannot corrupt
  the game.
