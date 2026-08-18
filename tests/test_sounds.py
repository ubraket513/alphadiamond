"""The move sound effect.

Audio is best-effort by design, so these check the contract around it rather
than that anything is audible: the asset ships, muting works, and a controller
built without sound never touches an audio backend.
"""

from __future__ import annotations

import pytest
from conftest import pump

from diamond.agents.random_agent import RandomAgent
from diamond.app.controller import GameController, Phase
from diamond.app.sounds import MOVE_SOUND, MovePlayer
from diamond.game.state import DEFAULT_PLAYERS


def test_the_move_sound_ships_with_the_package():
    assert MOVE_SOUND.is_file()
    assert MOVE_SOUND.stat().st_size > 0


def _loaded(qapp, player):
    """Wait for the asynchronous setSource() to finish."""
    assert pump(qapp, lambda: player._loaded, timeout=5.0), player.status
    return player


def test_player_loads_without_error(qapp):
    player = _loaded(qapp, MovePlayer())
    assert player.available
    assert player.status == ""


def test_play_before_the_media_loads_is_queued_not_dropped(qapp):
    """setSource() is asynchronous. A move confirmed in the first moments of a
    match used to be swallowed; the request is now replayed once loaded."""
    player = MovePlayer()
    states = []
    player._player.playbackStateChanged.connect(lambda s: states.append(str(s)))

    assert player.play() is False  # queued, not started, and not an error
    assert player.status == ""

    assert pump(qapp, lambda: any("Playing" in s for s in states), timeout=5.0)


def test_muting_silences_playback(qapp):
    player = _loaded(qapp, MovePlayer())
    player.set_muted(True)
    assert player.muted
    assert player.play() is False
    player.set_muted(False)
    assert player.play() is True


def test_controllers_are_silent_unless_sound_is_requested(qapp):
    """Default off: each QMediaPlayer reserves an audio backend, and a process
    building many controllers would otherwise pile them up until it stalls."""
    controller = GameController(
        DEFAULT_PLAYERS, agents={3: RandomAgent(seed=1)}, thinking_delay_ms=0, animate=False
    )
    try:
        assert not controller.soundAvailable
        assert not controller.soundEnabled
    finally:
        controller.shutdown()


def test_sound_can_be_toggled_on_a_sounding_controller(qapp):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        sounds=True,
    )
    try:
        assert controller.soundAvailable
        assert controller.soundEnabled

        controller.setSoundEnabled(False)
        assert not controller.soundEnabled
        assert controller.soundAvailable  # still loaded, just muted

        controller.setSoundEnabled(True)
        assert controller.soundEnabled
    finally:
        controller.shutdown()


def test_committing_a_move_does_not_raise_with_sound_enabled(qapp):
    """The sound fires from the commit path; a failing audio backend must not
    take the move with it."""
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        sounds=True,
    )
    try:
        move = controller.session.legal_moves()[0]
        controller.selectPosition(move.source)
        controller.selectPosition(move.destination)
        controller.confirmProposal()
        assert controller.session.history
    finally:
        controller.shutdown()


# -- one sound per hop -------------------------------------------------------


class _CountingPlayer:
    """Stand-in for MovePlayer that only records how often it was asked."""

    def __init__(self):
        self.plays = 0
        self.muted = False
        self.available = True
        self.status = ""
        self.volume = 0.6

    def play(self):
        self.plays += 1
        return True

    def set_muted(self, muted):
        self.muted = bool(muted)

    def set_volume(self, volume):
        self.volume = volume


def _controller_with_counter(board, animate):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=animate,
    )
    controller._sound = _CountingPlayer()
    return controller


def _commit(controller, move):
    controller.selectPosition(move.source)
    controller.selectPosition(move.destination)
    controller.confirmProposal()


def test_a_single_step_plays_once(qapp, board):
    controller = _controller_with_counter(board, animate=True)
    try:
        step = next(m for m in controller.session.legal_moves() if len(m.path) == 2)
        assert len(step.path) == 2  # one hop
        _commit(controller, step)
        assert pump(qapp, lambda: controller.phase != str(Phase.ANIMATING), timeout=5.0)
        assert controller._sound.plays == 1
    finally:
        controller.shutdown()


def _double_jump_position(board):
    """A straight corridor giving exactly one 2-hop chain.

    Holes h0..h4 along one lattice direction: the mover sits on h0, blockers on
    h1 and h3, and h2/h4 are empty, so h0 -> h2 -> h4 is a two-landing chain.
    """
    from diamond.game.coordinates import NUM_DIRECTIONS

    for start in range(len(board)):
        for direction in range(NUM_DIRECTIONS):
            chain = [start]
            hole = start
            for _ in range(4):
                hole = board.neighbour(hole, direction)
                if hole is None:
                    break
                chain.append(hole)
            if len(chain) == 5:
                h0, h1, h2, h3, h4 = chain
                return {h0: 1, h1: 2, h3: 2}, h0, h4
    raise AssertionError("no straight 5-hole corridor on the board")


def test_a_jump_chain_plays_once_per_hop(qapp, board):
    """The sound tracks the piece: a chain of N landings ticks N times."""
    from conftest import make_state

    pieces, source, destination = _double_jump_position(board)
    state = make_state(board, pieces, current_player_id=1)

    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=True,
        initial_state=state,
    )
    controller._sound = _CountingPlayer()
    try:
        move = controller.session.moves_from(source)[destination]
        hops = len(move.path) - 1
        assert hops == 2, move.path

        _commit(controller, move)
        assert pump(qapp, lambda: controller.phase != str(Phase.ANIMATING), timeout=10.0)
        assert controller._sound.plays == hops
    finally:
        controller.shutdown()


def test_without_animation_the_whole_move_gets_one_sound(qapp, board):
    controller = _controller_with_counter(board, animate=False)
    try:
        _commit(controller, controller.session.legal_moves()[0])
        assert controller._sound.plays == 1
    finally:
        controller.shutdown()


# -- volume ------------------------------------------------------------------


def test_volume_round_trips_and_clamps(qapp):
    player = MovePlayer()
    player.set_volume(0.25)
    assert player.volume == pytest.approx(0.25)
    player.set_volume(5)
    assert player.volume == 1.0
    player.set_volume(-1)
    assert player.volume == 0.0


def test_raising_the_volume_unmutes(qapp, board):
    controller = GameController(
        DEFAULT_PLAYERS,
        agents={3: RandomAgent(seed=1)},
        thinking_delay_ms=0,
        animate=False,
        sounds=True,
    )
    try:
        controller.setSoundEnabled(False)
        assert not controller.soundEnabled

        controller.setSoundVolume(0.4)
        assert controller.soundEnabled
        assert controller.soundVolume == pytest.approx(0.4)
    finally:
        controller.shutdown()
