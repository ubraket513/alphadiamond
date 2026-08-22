# Diamond — Controller Console

[![CI](https://github.com/ubraket513/alphadiamond/actions/workflows/ci.yml/badge.svg)](https://github.com/ubraket513/alphadiamond/actions/workflows/ci.yml)

A desktop **tournament/operator console** for running a real 2- or 3-player
Diamond match. It is not an online game: one human *Controller* sits at the
computer, records the moves the human players make, asks the agent for its
seat's move, and physically plays that move on the real board before
confirming it.

The primary Windows application is a native Qt 6 executable. Two-player human
play uses the exported Soo AlphaZero model through LibTorch and the existing
native C++ MCTS; Python remains the training and model-export environment only.

![board](guideline/board_sample.png)

---

## Project overview

| | |
|---|---|
| Board | 73-hole six-pointed star (Diamond geometry), derived from lattice coordinates |
| Players | 2 or 3 seats, chosen at match setup — 10 pieces each |
| Turn order | Chosen at match setup; any permutation of the seats |
| Agent | Soo AlphaZero (LibTorch CPU + native MCTS) for two-player human play |
| Typeface | Google Sans Flex, bundled under the OFL |
| Palette | Apple HIG system colours — red/yellow/green players, blue as the only UI accent |
| Audio | Move sound per hop; volume under **View ▸ Sounds** |
| Chrome | Frameless window with a custom title bar (menus + caption buttons) |
| Icons | Native Qt image provider; Codicon-compatible chrome and diamond app icon |
| GUI | Native C++ / Qt 6 / Qt Quick / existing QML |
| Engine | Native C++ rules, controller, MCTS, and LibTorch evaluator |

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
                 │NativeController │   turn state machine, proposals,
                 │  (native/qt)    │   generation tokens, animation timing
                 └────────┬────────┘
             ┌────────────┴────────────┐
             │                         │
     ┌───────┴───────┐        ┌────────┴────────┐
     │ soo_core      │        │ NativeAiWorker  │  dedicated QThread
     │  (native/)    │        │  (native/qt)    │
     └───────┬───────┘        └────────┬────────┘
             │                         │
   rules / board / state /             │
   move / history / save-load     ┌────┴─────────────┐
                                  │ MCTS2P + SooModel│
                                  │ (LibTorch CPU)   │
                                  └──────────────────┘
```

**Dependency rule:** QML never computes legality. All authoritative rule
decisions and AI proposal validation happen in the native C++ engine. The
Python GUI is retained only as the explicit `legacy-gui` oracle during the
migration tail; it is not a dependency of the native application.

### Layout

```
native/
├── include/soo/         engine, state, rules, MCTS interfaces
├── include/diamond_model/ LibTorch Soo model/evaluator
├── src/                 reusable native engine/model implementation
└── qt/                  NativeController, worker, audio, native chrome, host
src/diamond/
├── alphazero/           authoritative Python training/export pipeline
├── app/ + game/         legacy Python GUI oracle (optional)
├── assets/              bundled fonts and move sound
└── qml/                 shared visual source used unchanged by native Qt
    ├── Main.qml  Board.qml  Hole.qml  Piece.qml
    ├── SidePanel.qml  GamePanel.qml
    ├── AiPanel.qml  HistoryPanel.qml
    ├── PanelSection.qml  ActionButton.qml
    ├── AppDialog.qml        the one styled shell every pop-up uses
    ├── NewMatchDialog.qml   seat count, turn order, agent seat
    ├── ResultDialog.qml     final standings
    ├── TitleBar.qml         custom window chrome: menus + caption buttons
    ├── TitleMenu.qml        one flat menu in the title bar
    ├── WindowButton.qml     minimise / maximise / close
    ├── SoundDialog.qml      mute + volume, from View ▸ Sounds
    ├── SegmentedControl.qml ReorderButton.qml PanelScrollBar.qml
    └── Style/Theme.qml   ← every colour, size and duration lives here
```

The app and native contract test both link the reusable `diamond_qt_backend`
library. The QML files are embedded as Qt resources; no Python package-resource
API is used by the shipped executable.

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
lattice, camp regions and move paths are drawn procedurally on a `Canvas`, so
the board stays sharp at any window size. How the camp regions are shaped — and
why they are not triangles — is covered under [Board rendering](#board-rendering).

---

## Match setup

**New Game (Ctrl+N)** opens the setup dialog: seat count, turn order and which
seat the agent drives.

### Seat layouts

Every layout draws from the same three seats — the corners of triangle *up*,
120° apart, sitting on alternating hexagon sides so the starting camps stay
disjoint:

| Players | Camps | Colours |
|---|---|---|
| 2 | `z+`, `x+` | Red and green; the yellow seat sits out |
| 3 | `z+`, `y+`, `x+` | Red, yellow, green |

A 2-player match simply leaves one seat empty rather than moving anyone, so the
board's geometry is identical whatever the seat count. Each player still aims
at the camp directly across from their own, which means in a 2-player match
both target camps start empty rather than being the opponent's home.

Seat ids stay tied to a board position and colour, so reordering turns never
changes where a player sits or what colour they are. Note that seat 2 is *not*
the same camp in both layouts — head-to-head puts the second player in the
green seat, not the yellow one.

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

### Screen layout

```
┌──────────────────────────────────────────────────────────┐
│ ☰  File Edit View Window Help   Diamond · Game #001  ─ □ ✕ │
├───────────────┬──────────────────────────────────────────┤
│          GAME │                                          │
│      AI AGENT │                board                     │
│  MOVE HISTORY │                                          │
└───────────────┴──────────────────────────────────────────┘
```

The side panel sits on the **left**; the title bar's ☰ collapses it. Section
titles are right-aligned while their content stays left, so the titles form a
quiet rail down the panel's inner edge instead of competing with the content
for the same starting edge. There is no status bar.

**GAME** answers one question — whose turn it is — and nothing else. Phase, the
proposed move and the status/error line were all removed from it: the board
already shows the proposal as a highlighted path with numbered hop markers.

> *Trade-off worth knowing:* error text (e.g. "Player 2's piece — it is Player
> 1's turn.") now has nowhere to appear. Illegal clicks are still refused, but
> silently. If that turns out to matter, a transient toast over the board is
> the natural home for it.

Two sections were folded away as redundant:

* **PLAYERS** — the board carries seat colour and GAME names the current
  player. Per-seat progress (`7/10` home) and the finishing-place chips are no
  longer visible during play; final standings still appear in the results
  dialog.
* **MOVE** — its Confirm/Cancel buttons are gone; confirming is `Enter` / `Esc`
  or **Edit ▸ Confirm move / Cancel**, and undo is **Edit ▸ Undo move**
  (`Ctrl+Z`).

#### Proposing a move

Click a piece to see its destinations, then click one to propose it. A pending
proposal is a **draft, not a commitment**: clicking a different destination
re-aims it, and clicking another of your own pieces starts over on that piece.
Neither needs a cancel first, because nothing reaches the game state until the
proposal is confirmed.

Only the *human* draft is editable this way. An agent's proposal still locks
the board — the operator has to physically play that move before confirming it,
so letting a stray click re-aim it would desynchronise the console from the
real board.

#### Collapsing the panel

The laid-out width is animated, so the board grows into the space over the same
frames rather than snapping once at the end. The content is pinned to the full
panel width and the Flickable's `contentX` keeps its *right* edge against the
panel's, so the panel slides out of view instead of being squeezed — pinning
the width also stops every frame of the animation rewrapping the text.

Pieces animate their **lattice** position, not their screen position — see
[Board rendering](#board-rendering).

### Motion

Two cubic-bezier curves, defined once in `Theme` and used everywhere, so
movement across the app reads as one system rather than a set of unrelated
tweens:

| Curve | Shape | Used for |
|---|---|---|
| `easeStandard` | `0.4, 0, 0.2, 1` | Things moving between two states — hops, colour changes, hover |
| `easeEmphasized` | `0.2, 0, 0, 1` | Things arriving or taking over — dialogs, menus, the side panel |

Both are beziers rather than Qt's named easing types, whose curves stop short
of the long soft tail that reads as considered rather than merely animated.

Dialogs scale up from 0.96 while fading in, and leave faster than they arrive —
a dismissed dialog should get out of the way rather than be admired on the way
out. Menus drop in from a few pixels above. A plain fade reads as a slideshow;
a large scale reads as a cartoon.

One hard constraint: a piece's hop animation must finish inside the 140 ms
native controller timer interval, which paces the multi-hop sequence.
Overrun it and the piece lags its own tick.

### Colour

The palette is the [Apple HIG system colours](https://developer.apple.com/design/human-interface-guidelines/color),
light appearance, pinned in `qml/Style/Theme.qml`. Apple asks apps to read
these through system APIs rather than hard-code them, since the values move
between releases — a Qt desktop app has no such API, so they are written down
once, in that one file, and nowhere else.

Two rules do most of the work:

* **Blue is the interface.** `systemBlue` is the *only* accent: selection,
  focus, legal destinations, the proposed move path, the confirming button. No
  player is blue, so "blue means the app is talking to you" never collides with
  "this colour is a player".
* **Colour is reserved for meaning.** Greys carry all structure — the lattice,
  empty holes, borders, panel chrome. The only saturated things on the board
  are pieces, camp washes and the accent.

Player identity is red / yellow / green, defined once in `SEAT_LAYOUTS`
(`game/state.py`) and delivered to QML through the models — QML never restates
a player colour, because seat 2 is yellow in a 3-player match but green in a
head-to-head one.

> **Accessibility caveat.** Red and green are the classic confusion pair for
> red–green colour vision deficiency (~8% of men), and on the board colour is
> currently the *only* thing separating two players' pieces. The HIG is
> explicit about this: *"provide the same information in alternative ways… use
> text labels or glyph shapes."* Off-board the information is redundant already
> — seat number, name, colour swatch and `P1/P2/P3` tags in the move history —
> but the board itself is not. Mitigations, in increasing cost: swap green for
> `systemTeal` or `systemIndigo`; or mark each player's pieces with a distinct
> glyph. Neither is implemented.

### Board rendering

The camp regions are the part most worth explaining, because the obvious
implementation looks wrong in two specific ways.

Each camp is a **sharp triangle whose three vertices sit exactly on the camp's
corner holes**, drawn at the very back and composited once through the layer's
own `opacity` — everything else on the board is painted over it.

The vertex placement is what makes the mitred triangle workable. Adjacent camps
share precisely one hexagon-corner hole, so their triangles meet at that single
vertex and never overlap by *area*. No colour has to win over another at the
junction, and no alpha doubles up, so the region needs no rounding, offsetting
or gap to resolve it.

Because the edges pass through hole centres, they cut across the sockets on a
camp's boundary. That stays invisible because both pieces *and* empty sockets
are opaque: a hole is painted in the board's colour rather than left clear, so
the triangle never shows through one.

#### Holes, pieces and move affordances

A hole is a **hollow socket**, and a piece is that socket filled. Both use one
radius (`Theme.socketRatio`, 0.32 lattice units against a spacing of 1.0), and
the socket's ring is stroked *inside* its bounds, so an occupied hole is a
clean disc of the owner's colour with no ring peeking out from under it.

Lattice segments are **trimmed back by a socket radius at each end**, so they
run between sockets rather than through them. Untrimmed, every empty socket
reads as a hole with an X drawn across it.

| State | Treatment |
|---|---|
| Empty hole | Grey ring filled with the board's own colour |
| Occupied | Socket filled in the owner's colour |
| Selected piece | The same fill, fully saturated |
| Any other piece | The same fill, blended halfway to the board |
| Legal destination | Socket filled with translucent accent — a *ghost* of the piece that would land there |

Two deliberate simplifications:

* **Step and jump look identical.** The engine still distinguishes them, but a
  second colour on the board bought nothing: either way the hole is somewhere
  this piece can go, and the operator does not act differently. One uniform
  destination marker is the pattern
  [Duolingo's chess puzzle](https://mobbin.com/screens/0e70c6d4-d220-44c4-89d1-4ae085a2a145)
  uses too.
* **Selection is weight of colour alone** — no ring around the selected piece,
  and no rings marking the last move or the proposal's endpoints either. A
  selected piece is the only fully saturated thing on the board.

The washed-back state is a **blend against the board colour, not `opacity`**.
A genuinely translucent piece lets whatever is behind it show through, and two
things are: the camp triangle, whose sharp edge cuts across the boundary
sockets and split those pieces diagonally, and the proposed move path, which
showed straight through the pieces it jumps over. Blending to an opaque colour
makes a piece look identical wherever it stands.

> *Trade-off:* opacity used to encode *whose turn it is* (the current player's
> pieces were brighter). It now encodes selection, so the board no longer shows
> which colour is live — only the GAME panel does. Note also that seat-picker
> UIs ([Expedia](https://mobbin.com/screens/c4c5deeb-c279-4ad1-8a10-b9ba1851e86a),
> [Shopee](https://mobbin.com/screens/923eb86e-2fa7-4e8b-a51a-93b7cca63411))
> generally signal selection with fill *plus* a second channel rather than one
> alone. A third opacity level, or a slight scale-up on the selected piece,
> would restore either cue without bringing a ring back.

#### Layer order

The board is six explicit layers, and the order is load-bearing rather than
incidental:

| z | Layer |
|---|---|
| 0 | Camp triangles |
| 1 | Lattice lines |
| 2 | Holes |
| 3 | Move path |
| 4 | Pieces |
| 5 | Hop numbers |

The move path deliberately sits **between holes and pieces**. Hopping over a
piece should read as passing behind it, while the same line crossing an empty
hole should stay visible — and since a hole is now opaque, a path drawn beneath
one would simply vanish. The hop numbers sit above the path for the same
reason: the line would otherwise cut straight through them.

The destination is left unnumbered. It is already the end of the line, and the
ghost fill marks it.

#### Why pieces animate in lattice space

`Piece.qml` eases its **lattice** coordinates, not its `x`/`y`. Animating
screen position conflates two different kinds of movement: a piece hopping to a
new hole (which should ease over `hopDuration`) and the board being rescaled
because the window or the side panel changed width (which should be instant).
A `Behavior on x` interpolates the second case too, so the pieces lag behind
the lattice they are standing on and the board visibly sloshes whenever it
resizes.

Easing the lattice position instead means a resize only changes `originX` /
`unitScale`, which feed straight through to `x`/`y` with no Behavior attached —
the pieces track the board frame for frame — while a hop still eases, because
that is the only thing that moves the lattice position.

### Sound

`assets/sounds/move.m4a` plays **once per hop** — a single step ticks once, a
chain of jumps ticks once per landing, so the audio tracks what the piece is
actually doing rather than firing once per turn. The ticks are driven by the
animation, so they stay in step with the piece on screen; with animation off
the whole move lands at once and gets a single sound.

It fires for human and agent moves alike, from the single commit path in
`GameController._commit`, so there is no way to advance the game silently.

Audio is best-effort: a machine with no device, no codec or a locked-down
backend degrades to silence rather than raising, and the match plays on. Every
failure is recorded in `MovePlayer.status` and surfaced by the controller —
*silently* best-effort was a trap, because a broken backend and a missing
feature looked identical from the outside.

`setSource()` loads asynchronously, so a move confirmed in the first moments of
a match would be dropped; the request is held and replayed once the media is
ready rather than discarded.

Sound is **opt-in** at construction (`GameController(sounds=True)`, which only
`main.py` passes). Each `QMediaPlayer` reserves an audio backend, and a process
that builds many controllers — the test suite builds dozens — otherwise piles
them up until it stalls.

**Volume** lives under **View ▸ Sounds**, which opens a dialog holding a mute
button, a slider and a numeric readout on one row — the toolbar audio control
from [Canva](https://mobbin.com/screens/ccf4d1b7-b786-402a-8b41-c91250e1bcc7)
and [Adobe Express](https://mobbin.com/screens/06749d6e-7bfe-41ca-8301-024d647fd619),
given a dialog now that it is reached from a menu rather than a bar button.
Releasing the slider previews the sound once; raising the volume above zero
also unmutes. Muting lives only in that dialog — a second entry in the menu
would be a second place to change one setting.

The dialog is fully keyboard-driven: **arrow keys** adjust the volume in 5%
steps (previewing as they go), **Space** toggles mute, and **Enter** closes it.
Those are `Shortcut`s rather than `Keys` handlers because a `Popup` is not in
the focus chain and never sees the key itself.

### Window chrome

The window is **frameless** (`Qt.Window | Qt.FramelessWindowHint`) and draws
its own title bar, so the chrome matches the app rather than the platform.

`TitleBar.qml` owns everything the OS would normally provide:

| Region | Contents |
|---|---|
| Left | Panel toggle (hides the left side panel) |
| Menus | **File** — New Game, Save, Load, Exit · **Edit** — Undo, Confirm, Cancel · **View** — panel, Sounds · **Window** — minimise, maximise, close · **Help** — About |
| Centre | Game context, and the drag region that moves the window |
| Right | Minimise / maximise / close |

Dragging the bar calls `startSystemMove()` and double-clicking toggles
maximise, so native behaviours (aero snap, multi-monitor) survive. A frameless
window also loses the OS resize border, so `Main.qml` re-creates all eight
edges and corners, each handing off to `startSystemResize()` — which keeps
snapping and the declared minimum size working.

Menus are built from a plain model rather than `QtQuick.Controls.Menu`, for the
same reason the dialogs are: the Basic style's menu inherits the platform
palette instead of the app's. The same rule covers the scroll bars
(`PanelScrollBar.qml`): the Basic default picks its handle colour from
`palette.mid`/`palette.dark` and branches on
`Qt.styleHints.accessibility.contrastPreference`, none of which this app
controls. **No Basic-style internals are left unstyled anywhere in the UI.**

### Diagnosing runtime QML problems

Set `QT_LOGGING_RULES="qt.qml.*=true"` and run the app; every Qt/QML message is
printed with its source location.

#### Icons

Icons are rendered by the native Qt image provider in `native/qt/main.cpp`.
It implements the exact icon names already requested by the shared QML, so the
visual files did not need a migration-only fork and QtAwesome is no longer a
runtime dependency.

The caption buttons use Microsoft's own **Codicons** (`msc.chrome-minimize`,
`msc.chrome-maximize`, `msc.chrome-restore`, `msc.chrome-close`), so the title
bar draws the same shapes the shell does.

QML reaches them through the native `qta` image provider:

```qml
Icon { name: "msc.chevron-up"; size: 14; color: Theme.text }
```

which resolves to `image://qta/msc.chevron-up/1B1B1B`. The colour is passed
without its leading `#`, since that character would terminate the URL, and the
image is requested at device resolution so it stays crisp on a scaled display.

The **app and taskbar icon** is `fa6s.diamond` in systemRed, rendered at every
size the shell asks for (16 through 256). A plain rhombus was chosen over the
detailed gem cuts (`mdi6.diamond-stone`, `fa6s.gem`) because those lose all
definition at 16px.

#### Rejoining the shell

`Qt.FramelessWindowHint` makes the window a `WS_POPUP`, which the shell treats
as a transient thing rather than an application window. Several behaviours go
with it, and `native/qt/native_chrome.cpp` puts them back:

| Behaviour | Restored by |
|---|---|
| Rounded corners | `DWMWA_WINDOW_CORNER_PREFERENCE = DWMWCP_ROUND` |
| Minimise / restore animation | `WS_MINIMIZEBOX` + `WS_SYSMENU` |
| Taskbar click-to-minimise | the same style bits |
| No accent band across the top | `DWMWA_BORDER_COLOR = DWMWA_COLOR_NONE` |
| Maximise / restore **size animation** | `WS_THICKFRAME`, made safe by `WM_NCCALCSIZE` |
| **Snap Layouts** flyout | answering `WM_NCHITTEST` with `HTMAXBUTTON` |
| Its own taskbar icon | `SetCurrentProcessExplicitAppUserModelID` |

The accent band is worth explaining. With *show accent colour on title bars and
window borders* enabled — the Windows 11 default — DWM paints the active
window's outline in the accent colour. On an ordinary window that reads as a
highlight framing the title bar; here there is no native title bar to frame, so
it lands as a coloured band across the top of our own chrome. Measured, a 2px
strip above the content. Switching the border colour off removes it; the 1px
border the app draws in `Main.qml` is the one that should be visible.

`WS_THICKFRAME` is what DWM wants before it will animate a resize, but on its
own it makes Windows reserve non-client space — which is what put that band
there and clipped the caption buttons at the edge. Answering `WM_NCCALCSIZE`
with the full window rectangle reclaims that space, so the bit is only ever set
when the native filter is running to do so. Measured on both states, the
client inset is 0 on all four sides, and a maximised window matches the
monitor's work area exactly.

The **taskbar icon** needs an explicit AppUserModelID. The native host sets it
before the first window exists so Windows groups the process as Diamond and
uses the bundled icon.

**Snap Layouts** is the flyout Windows 11 shows when you hover a maximise
button. Windows offers it only to a window whose hit test answers
`HTMAXBUTTON`, which a Qt window never does, because Qt reports its whole
surface as client area. `NativeChrome` answers it for the button's rectangle
and `HTCLIENT` everywhere else. Two consequences follow from Windows then
owning that pointer: the button stops receiving Qt hover events, so the filter
publishes a `maximiseHovered` property for it to bind to, and the click arrives
as `WM_NCLBUTTONUP` rather than a `TapHandler`, so it is re-emitted as a
signal. The title bar reports the button's rectangle back through the same
object.

`NativeChrome` is constructed *before* the QML loads so the context property is
a real object from the outset. Bound to a placeholder, the title bar skipped
reporting the rectangle at `Component.onCompleted` and never retried, leaving
the hit test with nothing to match — the whole feature silently dead.

Asking DWM for the rounding beats clipping the corners in QML: the radius then
matches the platform's own, follows it if the user changes preferences, and
keeps the drop shadow a self-clipped translucent window loses.

The style bits imply a frame that is never drawn — Qt is still painting the
window frameless, and `WS_CAPTION` is deliberately *not* among them, so no
native title bar reappears. Only the behaviour comes back.

Everything degrades to a no-op off Windows, on Windows 10 (where the corner
attribute predates the OS), or if the APIs cannot be reached.

The reference console's back/forward arrows are deliberately **not** copied:
there is nothing to navigate in a single-screen console, and a permanently
dead control is worse than an absent one.

### Pop-ups

Every dialog uses `AppDialog.qml`, which paints its own surface, title, body
text and buttons from `Theme`. The Basic Qt Quick style ships an unstyled
`Dialog` that inherits the platform palette, which is what made the earlier
pop-ups hard to read; nothing visible is left to inherit now, message text
wraps instead of clipping, and the modal scrim is opaque enough to separate the
dialog from the board lattice behind it.

### Typography

The UI is set in **Google Sans Flex**, bundled in `src/diamond/assets/fonts/`
under the SIL Open Font License (Regular/Medium/Bold). It is embedded in the
native Qt resources and registered at startup, which hands the resolved family name to QML — so
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

For the primary Windows GUI:

* Windows 10/11 x64
* Visual Studio C++ build tools
* CMake/Ninja, Qt 6 (Core, Gui, Qml, Quick, QuickControls2, Multimedia)
* CPU LibTorch for the Soo-enabled build

This checkout uses the mamba environment at
`C:\ProgramData\miniforge3\envs\alphadiamond`. Python 3.11+ is still required
for training/export and for the optional legacy GUI oracle, but not by the
packaged native executable.

## Installation

```powershell
mamba activate C:\ProgramData\miniforge3\envs\alphadiamond
cmake -S . -B build-qt-soo-clean -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DDIAMOND_BUILD_QT_SOO=ON `
  -DDIAMOND_BUILD_LIBTORCH_PROBE=ON
cmake --build build-qt-soo-clean --parallel 1
powershell -ExecutionPolicy Bypass -File .\tools\deploy_native_qt.ps1 `
  -BuildDir build-qt-soo-clean -OutputDir dist\diamond-qt-soo `
  -WithSoo -EnvironmentRoot $env:CONDA_PREFIX
```

## How to run

Run the packaged Soo application through the launcher, which clears stale Qt
platform settings and exposes the simulation count:

```powershell
.\tools\run_native_qt.ps1 -Soo
.\tools\run_native_qt.ps1 -Soo -Simulations 256
```

The window opens at 1440 × 900 and stays usable down to 980 × 640; the board
rescales with the window.

The native executable can also be launched directly from
`dist\diamond-qt-soo\diamond_qt.exe`. The deployment script verifies its DLL,
Qt plugin, QML, sound, engine, worker, and Soo model closure before reporting
success.

The old PySide6 application remains an explicit development oracle only:

```powershell
python -m pip install -e ".[legacy-gui]"
diamond-legacy
```

## How to run tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
ctest --test-dir build-qt-soo-clean --output-on-failure
python -m pytest -m "not gui"
```

See [docs/native_windows_runtime.md](docs/native_windows_runtime.md) for the
complete build matrix, parity manifest, benchmark command, and compatibility
notes.

`QT_QPA_PLATFORM` is only needed where no display is available. The engine
tests need no Qt at all; controller contracts use an offscreen
`QGuiApplication` and do not create the main window.

The suite covers board topology, native move generation, proposal/confirmation,
per-landing animation and sound requests, history, undo, schema-v2 save/load,
terminal state, QML-visible model roles, stale AI results, Think Again, and
canonical-to-physical Soo actions for the second seat.

---

## Controller workflow

**Human turn (any human-controlled seat)**

1. Click one of the current player's pieces — it is highlighted and every legal
   destination is marked (green ring = single step, amber ring = jump).
2. Click a destination. The move is shown as a **proposal**: a path line,
   ghost destination, and numbered markers on intermediate landings.
3. Press **Enter** (or Edit ▸ Confirm move) to commit it; **Escape** cancels.
   Nothing changes until confirmation.

**Agent turn (the configured Soo seat)**

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

## Soo agent behaviour

* Uses the existing native `MCTS2P` with the exported 128-wide, six-block Soo
  network loaded by LibTorch on CPU.
* Uses no root Dirichlet noise, temperature zero, and a deterministic
  visit-count/tie-break choice for human play.
* Defaults to 128 simulations; use `-Simulations` or
  `DIAMOND_MCTS_SIMULATIONS` to configure it.
* Uses one Torch intra-op and one inter-op thread by default;
  `DIAMOND_TORCH_THREADS` changes the intra-op count.
* Converts MCTS canonical actions back to the active seat's physical board
  coordinates before validating or displaying a proposal.
* **Think Again** excludes the displayed physical action and searches again.
* Runs on a dedicated native worker thread. Generation tokens discard results
  after undo, load, new game, or shutdown; the QML/GUI thread never performs
  inference.

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

* The release AI path is two-player human-vs-Soo. Three-player native rules and
  controller play work, but native Min/MCTS3P model search is not part of this
  migration gate.
* At most **one** seat can be driven by an agent; there is no agent-vs-agent mode.
* A player with no legal move is reported as an error rather than being passed
  over, since the position cannot arise in normal play.
* No optional rules: pieces may leave a target camp, and may pass through or
  stop in camps that belong to other players.
* Save files always describe a match played from the standard opening; a
  session started from a set-up position cannot be saved and reloaded as such.
* No analysis features (policy heatmaps, candidate distributions, evaluation
  graphs). The AI panel shows actual agent, legal-move, simulation, and search
  timing metadata only.
* Undo has no redo.

---

## Training and deployment boundary

Python/PyTorch remains authoritative for training and checkpoint creation; see
[docs/alphazero.md](docs/alphazero.md). `tools/export_soo_deployment.py`
produces the versioned portable artifact consumed by the Windows LibTorch
runtime. The native application neither imports Python nor starts a Python or
WSL subprocess.
