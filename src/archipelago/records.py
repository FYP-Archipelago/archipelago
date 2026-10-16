from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .encoding import genome as genome_mod
from .errors import SchemaError, UnsupportedEncoding

SUPPORTED_SCHEMA_MAJOR = "2"


class Evaluation(NamedTuple):
    island_id: int
    eval_index: int
    t_wall: float
    individual_id: str
    parent_ids: tuple[str, ...]
    fitness: float
    genome_encoding: str
    genome_hash: str
    genome_repr: str | None
    genome_repr_mode: str


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    algorithm: str
    benchmark: str
    topology: str | None
    schema_version: str
    maximising: bool
    genome_encoding: str | None
    benchmark_params: dict[str, Any] = field(default_factory=dict)
    clock_offsets: dict[int, float] = field(default_factory=dict)
    # Island ids announced by island_start events.
    started_islands: tuple[int, ...] = ()
    # (origin individual id, arrived individual id), in event order.
    migrations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Run:
    meta: RunMeta
    evaluations: list[Evaluation]
    islands: list[int]

    def corrected_time(self, ev: Evaluation) -> float:
        return ev.t_wall + self.meta.clock_offsets.get(ev.island_id, 0.0)


def _blank(value: Any) -> bool:
    return value is None or value == ""


def _field[T](row: Mapping[str, Any], index: int, name: str, convert: Callable[[Any], T]) -> T:
    value = row.get(name)
    if _blank(value):
        raise SchemaError(f"evaluation row {index}: missing required field {name!r}")
    try:
        return convert(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"evaluation row {index}: cannot parse field {name!r} ({value!r})") from exc


def _parents(value: Any) -> tuple[str, ...]:
    if _blank(value):
        return ()
    if isinstance(value, str):
        return tuple(value.split(";"))
    return tuple(str(v) for v in value)


def parse_evaluations(rows: Sequence[Mapping[str, Any]]) -> list[Evaluation]:
    return [
        Evaluation(
            island_id=_field(row, i, "island_id", int),
            eval_index=_field(row, i, "eval_index", int),
            t_wall=_field(row, i, "t_wall", float),
            individual_id=_field(row, i, "individual_id", str),
            parent_ids=_parents(row.get("parent_ids")),
            fitness=_field(row, i, "fitness", float),
            genome_encoding=_field(row, i, "genome_encoding", str),
            genome_hash=_field(row, i, "genome_hash", str),
            genome_repr=None if _blank(row.get("genome_repr")) else str(row["genome_repr"]),
            genome_repr_mode=_field(row, i, "genome_repr_mode", str),
        )
        for i, row in enumerate(rows)
    ]


def _require(event: Mapping[str, Any], key: str) -> Any:
    value = event.get(key)
    if value is None:
        raise SchemaError(f"{event.get('type')} event: missing required field {key!r}")
    return value


def parse_events(events: Sequence[Mapping[str, Any]]) -> RunMeta:
    start: Mapping[str, Any] | None = None
    first_island: Mapping[str, Any] | None = None
    clock_offsets: dict[int, float] = {}
    started_islands: list[int] = []
    maximising: bool | None = None
    migrations: list[tuple[str, str]] = []

    for event in events:
        kind = event.get("type")
        if kind == "run_start" and start is None:
            start = event
        elif kind == "island_start":
            island_id = int(_require(event, "island_id"))
            started_islands.append(island_id)
            if first_island is None:
                first_island = event
            offset = event.get("clock_offset_ns")
            if offset is not None:
                clock_offsets[island_id] = offset / 1e9
        elif kind == "generation_end" and maximising is None:
            maximising = event.get("maximising")
        elif kind == "migration_arrive":
            origins = event.get("origin_individual_ids") or ()
            arrived = event.get("arrived_individual_ids") or ()
            migrations.extend((str(o), str(a)) for o, a in zip(origins, arrived))

    if start is None:
        raise SchemaError("no run_start event")
    schema_version = str(_require(start, "schema_version"))
    if schema_version.split(".", 1)[0] != SUPPORTED_SCHEMA_MAJOR:
        raise SchemaError(
            f"log schema {schema_version} is not supported (expected {SUPPORTED_SCHEMA_MAJOR}.x)"
        )

    datasets = start.get("datasets") or {}
    return RunMeta(
        run_id=str(_require(start, "run_id")),
        algorithm=str(_require(start, "algorithm")),
        benchmark=str(_require(start, "benchmark")),
        topology=(start.get("migration") or {}).get("topology"),
        schema_version=schema_version,
        maximising=bool(maximising if maximising is not None else datasets.get("maximising", False)),
        genome_encoding=first_island.get("genome_encoding") if first_island else None,
        benchmark_params=dict((first_island or {}).get("benchmark_params") or {}),
        clock_offsets=clock_offsets,
        started_islands=tuple(started_islands),
        migrations=tuple(migrations),
    )


def parse_run(rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> Run:
    meta = parse_events(events)
    evaluations = parse_evaluations(rows)

    # With no island_start events the encoding falls back to the first row's.
    encoding = meta.genome_encoding
    if encoding is None and evaluations:
        encoding = evaluations[0].genome_encoding
        meta = RunMeta(**{**meta.__dict__, "genome_encoding": encoding})
    if encoding is not None and encoding not in genome_mod.ENCODINGS:
        raise UnsupportedEncoding(f"unsupported genome encoding: {encoding}")
    for i, ev in enumerate(evaluations):
        if ev.genome_encoding != encoding:
            raise SchemaError(
                f"evaluation row {i}: genome_encoding {ev.genome_encoding!r} differs from the run's {encoding!r}"
            )

    islands = sorted(set(meta.started_islands) | {ev.island_id for ev in evaluations})
    return Run(meta=meta, evaluations=evaluations, islands=islands)
