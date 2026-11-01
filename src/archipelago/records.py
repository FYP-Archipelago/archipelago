from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .encoding import genome as genome_mod
from .errors import SchemaError, UnsupportedEncoding


class Evaluation(NamedTuple):
    island_id: int
    eval_index: int
    generation: int
    # Per-island record counter shared with the event stream; orders an island's records.
    seq: int
    t_wall: float
    individual_id: str
    parent_ids: tuple[str, ...]
    fitness: float
    genome_encoding: str
    genome_hash: str
    genome_repr: str | list[Any] | dict[str, Any]
    genome_repr_mode: str


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    algorithm: str
    benchmark: str
    # None when the run has migration switched off.
    migration_topology: str | None
    maximising: bool
    genome_encoding: str | None
    budget_per_island: int | None = None
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


def _field[T](row: Mapping[str, Any], index: int, name: str, convert: Callable[[Any], T]) -> T:
    value = row.get(name)
    if value is None:
        raise SchemaError(f"evaluation row {index}: missing required field {name!r}")
    try:
        return convert(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"evaluation row {index}: cannot parse field {name!r} ({value!r})") from exc


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(value)
    return value


# A float field can arrive as a JSON int.
def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(value)
    return float(value)


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(value)
    return value


def _parents(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(value)
    return tuple(_str(v) for v in value)


# A string (compact or expression), a list (full) or an object (composite encodings); never empty.
def _genome_repr(value: Any) -> str | list[Any] | dict[str, Any]:
    if not isinstance(value, (str, list, dict)) or not value:
        raise ValueError(value)
    return value


def parse_row(row: Mapping[str, Any], index: int) -> Evaluation:
    return Evaluation(
        island_id=_field(row, index, "island_id", _int),
        eval_index=_field(row, index, "eval_index", _int),
        generation=_field(row, index, "generation", _int),
        seq=_field(row, index, "seq", _int),
        t_wall=_field(row, index, "t_wall", _float),
        individual_id=_field(row, index, "individual_id", _str),
        parent_ids=_field(row, index, "parent_ids", _parents),
        fitness=_field(row, index, "fitness", _float),
        genome_encoding=_field(row, index, "genome_encoding", _str),
        genome_hash=_field(row, index, "genome_hash", _str),
        genome_repr=_field(row, index, "genome_repr", _genome_repr),
        genome_repr_mode=_field(row, index, "genome_repr_mode", _str),
    )


def parse_evaluations(rows: Sequence[Mapping[str, Any]]) -> list[Evaluation]:
    return [parse_row(row, i) for i, row in enumerate(rows)]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _require(event: Mapping[str, Any], key: str) -> Any:
    value = event.get(key)
    if value is None:
        raise SchemaError(f"{event.get('type')} event: missing required field {key!r}")
    return value


# What the events have said so far. Batch folds every event at once; streaming folds them as
# they arrive (migration_arrive events go through the stream's ordering instead).
@dataclass
class EventLog:
    start: Mapping[str, Any] | None = None
    first_island: Mapping[str, Any] | None = None
    clock_offsets: dict[int, float] = field(default_factory=dict)
    started_islands: list[int] = field(default_factory=list)
    ended_islands: set[int] = field(default_factory=set)
    maximising: bool | None = None
    migrations: list[tuple[str, str]] = field(default_factory=list)


def migration_pairs(event: Mapping[str, Any]) -> list[tuple[str, str]]:
    origins = event.get("origin_individual_ids") or ()
    arrived = event.get("arrived_individual_ids") or ()
    return [(str(o), str(a)) for o, a in zip(origins, arrived)]


def apply_event(log: EventLog, event: Mapping[str, Any]) -> None:
    kind = event.get("type")
    if kind == "run_start" and log.start is None:
        log.start = event
    elif kind == "island_start":
        island_id = int(_require(event, "island_id"))
        log.started_islands.append(island_id)
        if log.first_island is None:
            log.first_island = event
        offset = event.get("clock_offset_ns")
        if offset is not None:
            log.clock_offsets[island_id] = offset / 1e9
    elif kind == "island_end":
        log.ended_islands.add(int(_require(event, "island_id")))
    elif kind == "generation_end" and log.maximising is None:
        log.maximising = event.get("maximising")
    elif kind == "migration_arrive":
        log.migrations.extend(migration_pairs(event))


def run_meta(log: EventLog) -> RunMeta:
    start = log.start
    if start is None:
        raise SchemaError("no run_start event")
    datasets = start.get("datasets") or {}
    migration = start.get("migration") or {}
    maximising = log.maximising if log.maximising is not None else datasets.get("maximising", False)
    return RunMeta(
        run_id=str(_require(start, "run_id")),
        algorithm=str(_require(start, "algorithm")),
        benchmark=str(_require(start, "benchmark")),
        migration_topology=migration.get("topology") if migration.get("enabled", True) else None,
        maximising=bool(maximising),
        budget_per_island=_optional_int(start.get("evaluation_budget_per_island")),
        genome_encoding=log.first_island.get("genome_encoding") if log.first_island else None,
        benchmark_params=dict((log.first_island or {}).get("benchmark_params") or {}),
        clock_offsets=dict(log.clock_offsets),
        started_islands=tuple(log.started_islands),
        migrations=tuple(log.migrations),
    )


def parse_events(events: Sequence[Mapping[str, Any]]) -> RunMeta:
    log = EventLog()
    for event in events:
        apply_event(log, event)
    return run_meta(log)


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
            raise SchemaError(f"evaluation row {i}: genome_encoding {ev.genome_encoding!r} differs from the run's {encoding!r}")

    islands = sorted(set(meta.started_islands) | {ev.island_id for ev in evaluations})
    return Run(meta=meta, evaluations=evaluations, islands=islands)
