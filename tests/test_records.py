from __future__ import annotations

import base64
import json
import struct

from archipelago.encoding import genome as genome_mod
from archipelago.records import parse_row


def f64_block(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode("ascii")


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
