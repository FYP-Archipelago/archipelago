from __future__ import annotations

import base64
import json
import struct

import pytest

from archipelago.encoding import genome as genome_mod
from archipelago.errors import SchemaError
from archipelago.records import parse_row


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
    "genome_repr": "[1.0]",
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


def test_object_genome_repr_parses_like_its_csv_string() -> None:
    genome = {"x": f64_block([1.5, -2.0]), "v": f64_block([0.25, 0.5])}
    base = {
        "island_id": 0,
        "eval_index": 0,
        "generation": 0,
        "seq": 0,
        "t_wall": 1.0,
        "individual_id": "i0",
        "fitness": 1.0,
        "genome_encoding": "real_vector_velocity",
        "genome_hash": "h",
        "genome_repr_mode": "compact",
    }
    native = parse_row({**base, "genome_repr": genome}, 0)
    csv = parse_row({**base, "genome_repr": json.dumps(genome, separators=(",", ":"))}, 0)

    assert native == csv
    decoded = genome_mod.decode(native.genome_repr, native.genome_repr_mode, native.genome_encoding)
    assert decoded == {"x": [1.5, -2.0], "v": [0.25, 0.5]}
