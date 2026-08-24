"""Diamond rules: move generation, validation and win detection.

Implemented rules
-----------------
* **Single step** — slide into an adjacent empty hole.
* **Jump** — if the adjacent hole in some direction is occupied (by *any*
  player; there is no capture) and the next hole in that same direction is on
  the board and empty, the piece may hop there.  Jumps chain within one turn.
* A turn is *either* one single step *or* one chain of jumps, never both.

Jump collinearity
-----------------
A hop is only ever built as ``over = neighbour(from, d)`` followed by
``landing = neighbour(over, d)`` using the **same** direction index ``d``.  So
for every hop the origin, the jumped-over hole and the landing hole lie on one
straight lattice line in one of the six legal directions, with the jumped-over
hole exactly one lattice step from the origin and the landing exactly one
further step in that same direction.  Bent or arbitrary-diagonal single jumps
are not representable at all.  The direction is re-chosen independently at each
node of the search, so a multi-hop chain may change direction *between* hops
while every individual hop stays straight.

Canonical path policy
---------------------
Several jump chains can reach the same destination.  Move generation runs a
breadth-first search over jump landings with :data:`DIRECTIONS` explored in
fixed order and each hole visited at most once, then keeps the **first** path
found for each destination.  BFS therefore yields the *shortest* chain, and the
fixed direction order makes the tie-break deterministic.  Visiting each hole at
most once is also what prevents cycles inside a jump sequence.

If a destination is reachable both by a single step and by a jump chain, the
single step wins (it is the shorter path), and the move is reported as
:attr:`MoveKind.STEP`.

The moving piece's own source hole is treated as **empty** for the whole search
— the piece has conceptually already left it — so it can neither be jumped over
nor landed on again.
"""

from __future__ import annotations

from collections import deque

from ..contract.board import Board, standard_board
from ..contract.coordinates import NUM_DIRECTIONS
from ..contract.move import IllegalMoveError, Move, MoveKind
from ..contract.state import CAMP_SIZE, EMPTY, GameState, GameStatus, PlayerSpec


def moves_from(
    board: Board,
    state: GameState,
    source: int,
    *,
    player_id: int | None = None,
) -> dict[int, Move]:
    """Every legal move for the piece on ``source``, keyed by destination."""
    owner = state.occupant(source)
    if owner == EMPTY:
        return {}
    if player_id is not None and owner != player_id:
        return {}

    occupancy = state.occupancy
    moves: dict[int, Move] = {}

    # Single steps first: a step is always the shortest path to its target.
    for direction in range(NUM_DIRECTIONS):
        adjacent = board.neighbour(source, direction)
        if adjacent is not None and occupancy[adjacent] == EMPTY:
            moves[adjacent] = Move(owner, source, adjacent, (source, adjacent), MoveKind.STEP)

    # Then chained jumps, breadth-first so the first path to a hole is shortest.
    visited = {source}
    queue: deque[tuple[int, tuple[int, ...]]] = deque([(source, (source,))])
    while queue:
        current, path = queue.popleft()
        for direction in range(NUM_DIRECTIONS):
            over = board.neighbour(current, direction)
            if over is None or over == source or occupancy[over] == EMPTY:
                continue  # nothing to jump over (source counts as vacated)
            landing = board.neighbour(over, direction)
            if landing is None or landing in visited or occupancy[landing] != EMPTY:
                continue  # `visited` contains `source`, so cycles are impossible
            visited.add(landing)
            landing_path = path + (landing,)
            queue.append((landing, landing_path))
            if landing not in moves:
                moves[landing] = Move(owner, source, landing, landing_path, MoveKind.JUMP)

    return moves


def legal_moves(board: Board, state: GameState, player_id: int | None = None) -> tuple[Move, ...]:
    """Every legal move for ``player_id`` (default: the player to act)."""
    player_id = player_id if player_id is not None else state.current_player_id
    result: list[Move] = []
    for source in state.positions_of(player_id):
        result.extend(moves_from(board, state, source, player_id=player_id).values())
    return tuple(result)


def find_legal_move(
    board: Board,
    state: GameState,
    source: int,
    destination: int,
    *,
    player_id: int | None = None,
) -> Move | None:
    """Return the canonical legal move ``source -> destination``, or ``None``."""
    return moves_from(board, state, source, player_id=player_id).get(destination)


def validate_move(board: Board, state: GameState, move: Move) -> None:
    """Raise :class:`IllegalMoveError` unless ``move`` is legal right now.

    The engine — not the UI — is the authority on legality, so every commit
    path goes through this, including moves that came from an agent.
    """
    if move.player_id != state.current_player_id:
        raise IllegalMoveError(f"it is player {state.current_player_id}'s turn")
    canonical = find_legal_move(
        board, state, move.source, move.destination, player_id=move.player_id
    )
    if canonical is None:
        raise IllegalMoveError(f"{move.source} → {move.destination} is not a legal move")
    if move.path != canonical.path:
        raise IllegalMoveError("move path is not the canonical path for this destination")


def has_finished(board: Board, state: GameState, spec: PlayerSpec) -> bool:
    """True when all ten of a player's pieces occupy their target camp."""
    target = board.camp_positions(spec.target_camp)
    return len(target) == CAMP_SIZE and all(state.occupant(pid) == spec.id for pid in target)


def find_winner(board: Board, state: GameState, players: tuple[PlayerSpec, ...]) -> int | None:
    """The player in first place, or ``None`` while nobody is home yet."""
    ranked = update_ranking(board, state, players)
    return ranked.winner_id


def newly_placed(
    board: Board, state: GameState, players: tuple[PlayerSpec, ...]
) -> tuple[int, ...]:
    """Players who are home but not yet on the podium, in seat order."""
    return tuple(
        spec.id
        for spec in players
        if not state.has_placed(spec.id) and has_finished(board, state, spec)
    )


def match_is_over(state: GameState, players: tuple[PlayerSpec, ...]) -> bool:
    """True once every place but the last has been decided.

    The final player's place is implied -- there is nobody left to overtake them
    -- so play stops one short of a full podium.  With two players that means
    the match ends on the first finisher; with three it plays on to settle
    second, and third falls out for free.
    """
    return len(state.finish_order) >= len(players) - 1


def update_ranking(
    board: Board, state: GameState, players: tuple[PlayerSpec, ...]
) -> GameState:
    """Record any newly-finished players, then close the match if it is decided.

    Returns ``state`` untouched when nothing changed, so callers can use the
    result unconditionally.
    """
    for player_id in newly_placed(board, state, players):
        state = state.placed(player_id)

    if match_is_over(state, players) and state.status is not GameStatus.FINISHED:
        # Whoever is left never finished; they take the last place implicitly.
        remaining = [p.id for p in players if not state.has_placed(p.id)]
        for player_id in remaining:
            state = state.placed(player_id)
        state = state.finished()
    return state


__all__ = [
    "find_legal_move",
    "find_winner",
    "has_finished",
    "legal_moves",
    "match_is_over",
    "moves_from",
    "newly_placed",
    "standard_board",
    "update_ranking",
    "validate_move",
]


def next_player_id(
    players: tuple[PlayerSpec, ...],
    current_id: int,
    *,
    skip: tuple[int, ...] = (),
) -> int:
    """The next seat to act, skipping any player already on the podium.

    Seat order in ``players`` *is* the turn order.  ``skip`` normally comes from
    :attr:`GameState.finish_order`: a player who is home stops taking turns
    while the rest play on for the remaining places.  If everyone is skipped the
    current player is returned unchanged, which only happens once the match is
    already over.
    """
    ids = [p.id for p in players]
    start = ids.index(current_id)
    for offset in range(1, len(ids) + 1):
        candidate = ids[(start + offset) % len(ids)]
        if candidate not in skip:
            return candidate
    return current_id
