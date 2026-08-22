"""Phase 1 step 0: the native tables are the Python tables, element-wise."""

from __future__ import annotations

from diamond.alphazero.native import require_native, topology_tables


def test_native_tables_equal_python_tables() -> None:
    expected = topology_tables()
    actual = require_native().export_tables()

    assert actual["board_size"] == expected["board_size"]
    assert actual["directions"] == expected["directions"]
    for key in (
        "neighbour",
        "camp_positions",
        "pairwise_distance",
        "physical_to_canonical",
        "canonical_to_physical",
    ):
        native_rows = [list(row) for row in actual[key]]
        python_rows = [list(row) for row in expected[key]]
        assert native_rows == python_rows, f"{key} differs"


def test_canonical_mapping_is_an_involution_free_bijection() -> None:
    tables = require_native().export_tables()
    for camp, (forward, inverse) in enumerate(
        zip(tables["physical_to_canonical"], tables["canonical_to_physical"])
    ):
        assert sorted(forward) == list(range(73)), f"camp {camp} forward map is not a bijection"
        for physical, canonical in enumerate(forward):
            assert inverse[canonical] == physical
