from __future__ import annotations

import base64
import hashlib
import json
import struct
from typing import Any

REAL_VECTOR = "real_vector"
REAL_VECTOR_VELOCITY = "real_vector_velocity"
REAL_VECTOR_SIGMA = "real_vector_sigma"
BITSTRING = "bitstring"
PERMUTATION = "permutation"
EXPRESSION_TREE = "expression_tree"

ENCODINGS = frozenset({
    REAL_VECTOR,
    REAL_VECTOR_VELOCITY,
    REAL_VECTOR_SIGMA,
    BITSTRING,
    PERMUTATION,
    EXPRESSION_TREE,
})

# PSO and ES carry search state next to the position. Signature and features use the
# position only: velocity is how a particle intends to move, not where it is. Include
# it and one location splits into as many nodes as there were arrival directions.
COMPOSITE_POSITION_KEY = {REAL_VECTOR_VELOCITY: "x", REAL_VECTOR_SIGMA: "x"}

HASH_DIGEST_SIZE = 16


class GenomeError(ValueError):
    pass


def _decode_f64_block(payload: str) -> list[float]:
    raw = base64.b64decode(payload)
    if len(raw) % 8:
        raise GenomeError(f"float64 block is not a multiple of 8 bytes: {len(raw)}")
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


def _decode_compact(repr_str: str, encoding: str) -> Any:
    if encoding == REAL_VECTOR:
        return _decode_f64_block(repr_str)
    if encoding == PERMUTATION:
        raw = base64.b64decode(repr_str)
        if len(raw) % 4:
            raise GenomeError(f"uint32 block is not a multiple of 4 bytes: {len(raw)}")
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))
    if encoding == BITSTRING:
        length_str, payload = repr_str.split(":", 1)
        n_bits = int(length_str)
        raw = base64.b64decode(payload)
        # LSB-first within each byte. Get this backwards and you still get a
        # well-formed bitstring of the right length - it just isn't the right
        # individual. Fails completely silently, hence the hash check in the tests.
        bits = "".join(f"{byte:08b}"[::-1] for byte in raw)
        return [int(bit) for bit in bits[:n_bits]]
    if encoding in COMPOSITE_POSITION_KEY:
        return {k: _decode_f64_block(v) for k, v in json.loads(repr_str).items()}
    if encoding == EXPRESSION_TREE:
        return repr_str
    raise GenomeError(f"unknown encoding: {encoding}")


def decode(repr_str: str | None, mode: str, encoding: str) -> Any:
    if encoding not in ENCODINGS:
        raise GenomeError(f"unknown encoding: {encoding}")
    if mode == "hashed" or not repr_str:
        return None
    if mode == "full":
        return repr_str if encoding == EXPRESSION_TREE else json.loads(repr_str)
    if mode == "compact":
        return _decode_compact(repr_str, encoding)
    raise GenomeError(f"unknown genome_repr_mode: {mode}")


def position(genome: Any, encoding: str) -> Any:
    key = COMPOSITE_POSITION_KEY.get(encoding)
    return genome[key] if key else genome


def _format_coordinate(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}"
    if text.startswith("-") and float(text) == 0.0:
        return text[1:]
    return text


def signature(genome: Any, encoding: str, precision: int) -> str:
    where = position(genome, encoding)
    if encoding in (REAL_VECTOR, REAL_VECTOR_VELOCITY, REAL_VECTOR_SIGMA):
        body = ",".join(_format_coordinate(v, precision) for v in where)
    elif encoding == BITSTRING:
        body = "".join(str(int(b)) for b in where)
    elif encoding == PERMUTATION:
        body = ",".join(str(int(i)) for i in where)
    elif encoding == EXPRESSION_TREE:
        body = where
    else:
        raise GenomeError(f"unknown encoding: {encoding}")
    return f"{encoding}|{body}"


def location_hash(genome: Any, encoding: str, precision: int) -> str:
    return hashlib.blake2b(signature(genome, encoding, precision).encode(), digest_size=HASH_DIGEST_SIZE).hexdigest()
