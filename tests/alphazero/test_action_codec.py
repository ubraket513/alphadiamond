from __future__ import annotations

import pytest

from diamond.alphazero.action_codec import ActionCodec, ActionSpaceSpec


def test_diamond_action_space_contract() -> None:
    spec = ActionSpaceSpec(board_size=73, version="diamond73-srcdst-v1")
    codec = ActionCodec(spec)

    assert codec.action_size == 5329
    assert codec.encode(72, 72) == 5328
    assert codec.decode(5328) == (72, 72)


def test_every_source_destination_pair_round_trips() -> None:
    codec = ActionCodec(ActionSpaceSpec.diamond73())

    seen = {
        codec.encode(source, destination)
        for source in range(73)
        for destination in range(73)
    }

    assert seen == set(range(5329))
    assert all(codec.decode(action_id) == divmod(action_id, 73) for action_id in seen)


@pytest.mark.parametrize("source,destination", [(-1, 0), (0, -1), (73, 0), (0, 73)])
def test_encode_rejects_out_of_bounds_positions(source: int, destination: int) -> None:
    codec = ActionCodec(ActionSpaceSpec.diamond73())
    with pytest.raises(ValueError):
        codec.encode(source, destination)


@pytest.mark.parametrize("action_id", [-1, 5329])
def test_decode_rejects_out_of_bounds_action_ids(action_id: int) -> None:
    codec = ActionCodec(ActionSpaceSpec.diamond73())
    with pytest.raises(ValueError):
        codec.decode(action_id)

