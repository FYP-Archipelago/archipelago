from __future__ import annotations

import base64
import struct

import pytest

from archipelago.batch import row_warnings
from archipelago.encoding import genome as genome_mod
from archipelago.encoding.features import DECODE_ERRORS
from archipelago.errors import SchemaError
from archipelago.records import EventLog, apply_event, parse_row, run_meta


def f64_block(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode("ascii")


BASE = {
    "island_id": 0,
    "eval_index": 0,
    "generation": 0,
    "seq": 0,
    "t_wall": 1.0,
    "individual_id": "i0",
    "fitness": 1.0,
    "genome_encoding": "real_vector",
    "genome_hash": "h",
    "parent_ids": [],
    "genome_repr": [1.0],
    "genome_repr_mode": "full",
}


@pytest.mark.parametrize("name", ["seq", "genome_repr"])
def test_required_fields_have_no_fallback(name: str) -> None:
    row = {k: v for k, v in BASE.items() if k != name}
    with pytest.raises(SchemaError, match=name):
        parse_row(row, 0)


def test_empty_genome_repr_raises() -> None:
    with pytest.raises(SchemaError, match="genome_repr"):
        parse_row({**BASE, "genome_repr": ""}, 0)


def test_hashed_mode_is_unsupported() -> None:
    with pytest.raises(genome_mod.GenomeError):
        genome_mod.decode("abc", "hashed", "real_vector")


def test_native_row_parses_to_native_values() -> None:
    row = parse_row({**BASE, "parent_ids": ["a", "b"], "t_wall": 3, "genome_repr": [1.0, 2.0]}, 0)

    assert row.parent_ids == ("a", "b")
    assert row.t_wall == 3.0 and isinstance(row.t_wall, float)
    assert row.genome_repr == [1.0, 2.0]


def test_empty_parent_list_is_a_root() -> None:
    assert parse_row({**BASE, "parent_ids": []}, 0).parent_ids == ()


@pytest.mark.parametrize("parents", ["a;b", "a", ("a",), None])
def test_non_list_parent_ids_raise(parents: object) -> None:
    with pytest.raises(SchemaError, match="parent_ids"):
        parse_row({**BASE, "parent_ids": parents}, 0)


@pytest.mark.parametrize("name", ["seq", "island_id", "eval_index", "generation"])
def test_string_numbers_are_not_coerced(name: str) -> None:
    with pytest.raises(SchemaError, match=name):
        parse_row({**BASE, name: "3"}, 0)


def test_string_float_is_not_coerced() -> None:
    with pytest.raises(SchemaError, match="fitness"):
        parse_row({**BASE, "fitness": "1.0"}, 0)


def test_object_genome_repr_decodes_in_compact_mode() -> None:
    genome = {"x": f64_block([1.5, -2.0]), "v": f64_block([0.25, 0.5])}
    row = parse_row({**BASE, "genome_encoding": "real_vector_velocity", "genome_repr_mode": "compact", "genome_repr": genome}, 0)

    assert row.genome_repr == genome
    decoded = genome_mod.decode(row.genome_repr, row.genome_repr_mode, row.genome_encoding)
    assert decoded == {"x": [1.5, -2.0], "v": [0.25, 0.5]}


def test_full_mode_uses_the_native_value() -> None:
    assert genome_mod.decode([1.0, 2.0], "full", "real_vector") == [1.0, 2.0]
    assert genome_mod.decode({"x": [1.0], "v": [2.0]}, "full", "real_vector_velocity") == {"x": [1.0], "v": [2.0]}
    assert genome_mod.decode("add(x,1)", "full", "expression_tree") == "add(x,1)"


@pytest.mark.parametrize(
    ("value", "mode", "encoding"),
    [
        ([1.0], "compact", "real_vector"),
        ({"x": "AAAAAAAA"}, "compact", "permutation"),
        ([1], "compact", "bitstring"),
        ("AAAAAAAAAAA=", "compact", "real_vector_sigma"),
        ({"x": [1.0]}, "compact", "real_vector_sigma"),
        ([1], "full", "expression_tree"),
        ("x", "other", "real_vector"),
    ],
)
def test_bad_genome_types_raise_decode_errors(value: object, mode: str, encoding: str) -> None:
    with pytest.raises(DECODE_ERRORS):
        genome_mod.decode(value, mode, encoding)


def test_malformed_rows_get_one_warning() -> None:
    assert row_warnings(["a", "b"], 10) == ["2 of 10 evaluations have a malformed genome and are left out of the graph: a, b"]
    assert row_warnings([], 10) == []


@pytest.mark.parametrize(
    ("migration", "expected"),
    [
        ({"topology": "ring", "enabled": True}, "ring"),
        ({"topology": "ring"}, "ring"),
        ({"topology": "ring", "enabled": False}, None),
        (None, None),
    ],
)
def test_migration_topology(migration: dict[str, object] | None, expected: str | None) -> None:
    log = EventLog()
    apply_event(log, {"type": "run_start", "run_id": "r", "algorithm": "de", "benchmark": "sphere", "migration": migration})

    assert run_meta(log).migration_topology == expected
