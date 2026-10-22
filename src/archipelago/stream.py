from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .api import _clean, malformed_warning, missing_warning
from .config import (
    CALIBRATION_SAMPLE_SIZE,
    CHECKPOINT_FACTOR,
    HALF_LIFE_FRACTION,
    LSH_WIDTH_PERCENTILE,
    NEIGHBOUR_PERCENTILE,
    WARMUP_ROWS,
    Config,
)
from .encoding import genome as genome_mod
from .encoding.features import (
    DECODE_ERRORS,
    Adapter,
    FeatureStore,
    Metric,
    build_adapter,
    encode,
    store_rows,
)
from .errors import ClusteringError, SchemaError, UnsupportedEncoding
from .graph.delta import Sent, diff
from .graph.layout import Projection, fit, transform
from .graph.state import (
    GraphState,
    add_edges,
    add_migrations,
    add_rows,
    columns,
    kill_labels,
    set_coords,
    to_graph,
)
from .models import (
    ClusteringInfo,
    Counts,
    EdgeChanges,
    Edges,
    GraphDelta,
    LayoutInfo,
    MigrationChanges,
    Migrations,
    NodeChanges,
    Nodes,
    PercentileChanges,
    Resolved,
    Result,
    RunInfo,
    Update,
    WeightChanges,
)
from .ordering import Item, Order, enqueue, first_delivery, release
from .records import (
    Evaluation,
    EventLog,
    apply_event,
    migration_pairs,
    parse_row,
    run_meta,
)
from .stages.birch import OnlineBirch
from .stages.calibrate import neighbour_distance, pair_distance
from .stages.denstream import OnlineDenStream
from .stages.lsh import OnlineLSH


# Calibrated values are rounded so last-bit float differences between machines (BLAS builds)
# can't flip a clustering decision when a run is replayed elsewhere.
def _round(value: float) -> float:
    return float(f"{value:.9g}")


# A released usable row waiting for the stages.
@dataclass
class _Row:
    ev: Evaluation
    vector: np.ndarray
    ids: np.ndarray | None
    clock: float
    released: int


def _empty_delta(weighted: bool) -> GraphDelta:
    empty = Edges.model_construct(id=[], source=[], target=[], weight=[])
    fields = {name: [] for name in NodeChanges.model_fields}
    nodes = {name: [] for name in Nodes.model_fields}
    nodes["weight"] = [] if weighted else None
    return GraphDelta.model_construct(
        new_nodes=Nodes.model_construct(**nodes),
        updated_nodes=NodeChanges.model_construct(**fields),
        percentiles=PercentileChanges.model_construct(id=[], fitness_percentile=[]),
        weights=WeightChanges.model_construct(id=[], weight=[]) if weighted else None,
        removed_nodes=[],
        positions=None,
        new_edges=empty,
        edge_updates=EdgeChanges.model_construct(id=[], weight=[]),
        new_migrations=Migrations.model_construct(id=[], source=[], target=[], count=[]),
        migration_updates=MigrationChanges.model_construct(id=[], count=[]),
        island_best=[],
        overall_best=None,
        overlap=0.0,
    )


# Streaming mode: cluster a run while it is going. Feed every new batch of log rows and events
# to cluster(), which returns what changed since the last call; call finish() once the run has
# ended for the final Result. Records are put in one canonical order first, so the same logs
# give the same result however they were chunked, interleaved or redelivered.
class ClusterStream:
    def __init__(self, config: Config | Mapping[str, Any] | None = None):
        self.config = config if isinstance(config, Config) else Config.model_validate(config or {})
        self._order = Order()
        self._log = EventLog()
        self._finished = False
        self._elapsed = 0.0
        self._rows_seen = 0
        self._adapter: Adapter | None = None
        self._encoding: str | None = None
        self._dimension: int | None = None
        self._t0: float | None = None
        self._last_clock = 0.0
        self._maximising = False

        # Every released row, in release order, and where it ended up.
        self._released: list[Evaluation] = []
        self._usable: list[bool] = []
        self._clock: list[float | None] = []
        self._processed_of: list[int | None] = []
        self._span: dict[int, int] = {}
        self._malformed: list[str] = []
        self._unusable_ids: set[str] = set()

        self._buffer: list[_Row] = []
        self._segment: list[_Row] = []
        self._warmed = False
        self._store: FeatureStore | None = None
        self._row_release: list[int] = []
        self._row_clock: list[float] = []

        self._lsh: OnlineLSH | None = None
        self._birch: OnlineBirch | None = None
        self._den: OnlineDenStream | None = None
        # genome_hash -> [lsh, birch, denstream] labels of the genome's first row.
        self._cache: dict[str, list[int | None]] = {}
        self._stage_labels: dict[str, list[int | None]] = {"lsh": [], "birch": [], "den": []}
        self._labels: list[int] = []
        self._how: list[str | None] = []
        self._prunes: list[dict[str, Any]] = []

        self._state: GraphState | None = None
        self._projection: Projection | None = None
        self._sent = Sent()
        self._refit = False
        self._checkpoints: list[dict[str, Any]] = []
        self._next_checkpoint = WARMUP_ROWS * CHECKPOINT_FACTOR
        self._resolved: dict[str, float | None] = dict.fromkeys(Resolved.model_fields)

        # Migrant pairs waiting for a row: individual id -> indices into _pairs.
        self._pairs: list[tuple[str, str]] = []
        self._waiting: dict[str, list[int]] = {}
        self._resolved_pairs: list[tuple[str, str]] = []

    # --- public API ---

    def cluster(self, rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> Update:
        if self._finished:
            raise ClusteringError("this stream is finished; start a new ClusterStream for another run")
        started = time.perf_counter()
        self._ingest(rows, events)
        self._take(release(self._order, self._log, final=False))
        update = self._update(started)
        return update

    def finish(self) -> Result:
        if self._finished:
            raise ClusteringError("this stream is already finished")
        started = time.perf_counter()
        self._take(release(self._order, self._log, final=True))
        meta = run_meta(self._log)
        if not self._warmed and self._buffer:
            self._warm_up(finishing=True)
        self._flush()
        if self._state is None:
            self._state = GraphState(maximising=meta.maximising, weighted=self.config.denstream)
        if self._store is not None and self._store.count:
            self._projection, coords = fit(self._store.rows, self._store.metric, self.config.seed)
            set_coords(self._state, coords)
        diagnostics = self._diagnostics(final=True)
        self._set_weights()
        graph = to_graph(columns(self._state, self._span))
        self._finished = True
        self._elapsed += time.perf_counter() - started
        projection = self._projection
        return Result(
            run=RunInfo(
                run_id=meta.run_id,
                algorithm=meta.algorithm,
                benchmark=meta.benchmark,
                topology=meta.topology,
                islands=sorted(set(meta.started_islands) | set(self._span)),
                maximising=self._maximising if self._warmed else meta.maximising,
            ),
            layout=LayoutInfo(
                method=projection.method if projection else "none",
                variance_kept_2d=projection.variance_kept_2d if projection else 0.0,
                variance_kept_3d=projection.variance_kept_3d if projection else 0.0,
            ),
            graph=graph,
            clustering=self._info(diagnostics, len(graph.nodes.id)),
            rows_without_genome=len(self._released) - sum(self._usable),
            warnings=self._warnings(),
        )

    # --- ingest and release ---

    def _ingest(self, rows: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> None:
        for event in events:
            if not first_delivery(self._order, event):
                continue
            if event.get("type") == "migration_arrive":
                for key in ("seq", "island_id", "t_wall"):
                    if event.get(key) is None:
                        raise SchemaError(f"migration_arrive event: missing required field {key!r}")
                enqueue(
                    self._order,
                    Item(seq=int(event["seq"]), island=int(event["island_id"]), t_wall=float(event["t_wall"]), event=event),
                )
            else:
                apply_event(self._log, event)
        for row in rows:
            ev = parse_row(row, self._rows_seen)
            self._rows_seen += 1
            enqueue(self._order, Item(seq=ev.seq, island=ev.island_id, t_wall=ev.t_wall, row=ev))

    def _take(self, items: list[Item]) -> None:
        for item in items:
            if item.row is not None:
                self._take_row(item.row)
            else:
                # Rows released before this event must be in the graph to resolve its pairs.
                self._flush()
                assert item.event is not None
                for pair in migration_pairs(item.event):
                    self._pairs.append(pair)
                    self._resolve(len(self._pairs) - 1)
        self._flush()

    def _ensure_adapter(self, ev: Evaluation) -> Adapter:
        if self._adapter is None:
            first = self._log.first_island
            encoding = (first or {}).get("genome_encoding") or ev.genome_encoding
            if encoding not in genome_mod.ENCODINGS:
                raise UnsupportedEncoding(f"unsupported genome encoding: {encoding}")
            self._encoding = encoding
            self._adapter = build_adapter(encoding, dict((first or {}).get("benchmark_params") or {}), self.config.seed)
        if ev.genome_encoding != self._encoding:
            raise SchemaError(
                f"evaluation {ev.individual_id}: genome_encoding {ev.genome_encoding!r} differs from the run's {self._encoding!r}"
            )
        return self._adapter

    def _take_row(self, ev: Evaluation) -> None:
        adapter = self._ensure_adapter(ev)
        released = len(self._released)
        self._released.append(ev)
        self._span[ev.island_id] = max(self._span.get(ev.island_id, 0), ev.eval_index + 1)
        try:
            encoded = encode(adapter, ev)
        except DECODE_ERRORS:
            self._malformed.append(ev.individual_id)
            encoded = None
        self._processed_of.append(None)
        if encoded is None:
            self._usable.append(False)
            self._clock.append(None)
            self._unusable_ids.add(ev.individual_id)
            for index in self._waiting.pop(ev.individual_id, []):
                self._resolve(index)
            return

        corrected = ev.t_wall + self._log.clock_offsets.get(ev.island_id, 0.0)
        if self._t0 is None:
            self._t0 = corrected
        self._last_clock = max(corrected - self._t0, self._last_clock)
        self._usable.append(True)
        self._clock.append(self._last_clock)
        vector, ids = encoded
        if self._dimension is not None and vector.size > self._dimension:
            raise SchemaError(f"evaluation {ev.individual_id}: genome length {vector.size} exceeds {self._dimension}")
        row = _Row(ev, vector, ids, self._last_clock, released)
        if self._warmed:
            self._segment.append(row)
        else:
            self._buffer.append(row)
            if len(self._buffer) == WARMUP_ROWS:
                self._warm_up(finishing=False)

    # --- calibration ---

    def _detail(self, value: float) -> float:
        return value / self.config.detail

    def _birch_threshold(self, matrix: np.ndarray) -> float:
        if self.config.birch_threshold is not None:
            return float(self.config.birch_threshold)
        measured = max(
            neighbour_distance(matrix, CALIBRATION_SAMPLE_SIZE, NEIGHBOUR_PERCENTILE, self.config.seed), 1e-12
        )
        return _round(self._detail(measured))

    # DenStream's merge radius is measured on the points it effectively streams: the upstream
    # groups' centroids (rows themselves without an upstream stage).
    def _denstream_epsilon(self, count: int) -> float:
        if self.config.denstream_epsilon is not None:
            return float(self.config.denstream_epsilon)
        assert self._store is not None
        matrix = self._store.rows[:count]
        upstream = self._stage_labels["birch"] if self.config.birch else self._stage_labels["lsh"]
        if self.config.birch or self.config.lsh:
            labels = np.asarray(upstream[:count], dtype=np.int64)
            groups = int(labels.max()) + 1
            sums = np.zeros((groups, matrix.shape[1]))
            np.add.at(sums, labels, matrix)
            points = sums / np.bincount(labels, minlength=groups)[:, None]
        else:
            points = matrix
        measured = neighbour_distance(points, CALIBRATION_SAMPLE_SIZE, NEIGHBOUR_PERCENTILE, self.config.seed)
        return _round(max(self._detail(measured), 1e-12))

    # Auto half-life: a fraction of the run's time span, projected from how much of the
    # evaluation budget has been released so far.
    def _half_life(self, count: int, finishing: bool) -> float:
        if self.config.denstream_half_life is not None:
            return float(self.config.denstream_half_life)
        span = self._row_clock[count - 1]
        budget = self._log.start.get("evaluation_budget") if self._log.start else None
        if budget and not finishing:
            span *= max(1.0, float(budget) / (self._row_release[count - 1] + 1))
        return _round(max(span * HALF_LIFE_FRACTION, 1e-9))

    def _checkpoint_record(self, count: int) -> None:
        self._resolved = {
            "birch_threshold": self._birch.threshold if self._birch else None,
            "denstream_epsilon": self._den.epsilon if self._den else None,
            "denstream_half_life": self._den.half_life if self._den else None,
            "lsh_bucket_width": self._lsh.width if self._lsh else None,
        }
        self._checkpoints.append(
            {
                "rows": count,
                "clock": self._row_clock[count - 1],
                "lsh_width": self._resolved["lsh_bucket_width"],
                "birch_threshold": self._resolved["birch_threshold"],
                "denstream_epsilon": self._resolved["denstream_epsilon"],
                "denstream_half_life": self._resolved["denstream_half_life"],
            }
        )

    def _warm_up(self, finishing: bool) -> None:
        rows = self._buffer
        self._buffer = []
        meta = run_meta(self._log) if self._log.start is not None else None
        if meta is None:
            raise SchemaError("no run_start event")
        self._maximising = meta.maximising
        assert self._adapter is not None
        self._store = FeatureStore(metric=self._adapter.metric)
        store_rows(self._store, [r.vector for r in rows], [r.ids for r in rows])
        self._dimension = self._store.matrix.shape[1]
        matrix = self._store.rows
        config = self.config
        if config.lsh:
            width = None
            if self._store.metric is Metric.L2:
                width = _round(
                    max(pair_distance(matrix, CALIBRATION_SAMPLE_SIZE, LSH_WIDTH_PERCENTILE, config.seed), 1e-12)
                )
            self._lsh = OnlineLSH(config, self._store.metric, self._dimension, width)
        if config.birch:
            self._birch = OnlineBirch(self._birch_threshold(matrix))
        self._state = GraphState(maximising=meta.maximising, weighted=config.denstream)
        self._projection, coords = fit(matrix, self._store.metric, config.seed)
        self._process(rows, coords, stored=True, warm=finishing)
        self._warmed = True
        self._refit = True

    # --- processing ---

    # Push the waiting rows through the stages, split at checkpoints so every recalibration
    # happens at the same row whatever the chunking.
    def _flush(self) -> None:
        while self._segment:
            assert self._store is not None
            room = self._next_checkpoint - self._store.count
            chunk, self._segment = self._segment[:room], self._segment[room:]
            self._process(chunk, None, stored=False, warm=False)
            if self._store.count == self._next_checkpoint:
                self._checkpoint()

    def _checkpoint(self) -> None:
        assert self._store is not None and self._state is not None
        count = self._store.count
        matrix = self._store.rows
        if self._birch is not None and self.config.birch_threshold is None:
            self._birch.set_threshold(self._birch_threshold(matrix))
        if self._den is not None:
            if self.config.denstream_epsilon is None:
                self._den.set_epsilon(self._denstream_epsilon(count))
            if self.config.denstream_half_life is None:
                self._den.set_half_life(self._half_life(count, finishing=False))
        self._projection, coords = fit(matrix, self._store.metric, self.config.seed)
        set_coords(self._state, coords)
        self._refit = True
        self._checkpoint_record(count)
        self._next_checkpoint *= CHECKPOINT_FACTOR

    # Label rows stage by stage. Each stage only looks at earlier rows and earlier stages, so
    # running a stage over the whole chunk before the next equals going row by row.
    # warm: rows are the warm-up buffer (DenStream is calibrated in between); finishing tells
    # the half-life not to project the span forward.
    def _process(self, rows: list[_Row], coords: np.ndarray | None, stored: bool, warm: bool) -> None:
        assert self._store is not None and self._state is not None
        if not rows:
            return
        start = self._store.count - (len(rows) if stored else 0)
        if not stored:
            store_rows(self._store, [r.vector for r in rows], [r.ids for r in rows])
        matrix = self._store.rows[start : start + len(rows)]
        sets = self._store.sets[start : start + len(rows)] if self._store.sets is not None else None
        for r in rows:
            self._row_release.append(r.released)
            self._row_clock.append(r.clock)

        entries: list[list[int | None]] = []
        first: list[bool] = []
        for r in rows:
            entry = self._cache.get(r.ev.genome_hash)
            first.append(entry is None)
            if entry is None:
                entry = self._cache[r.ev.genome_hash] = [None, None, None]
            entries.append(entry)

        lsh_labels: list[int | None] = [None] * len(rows)
        if self._lsh is not None:
            keys = self._lsh.keys(matrix, sets)
            for i, entry in enumerate(entries):
                if first[i]:
                    entry[0] = self._lsh.assign(keys[i])
                else:
                    self._lsh.absorb(int(entry[0]))
                lsh_labels[i] = entry[0]

        birch_labels: list[int | None] = [None] * len(rows)
        if self._birch is not None:
            for i, entry in enumerate(entries):
                if first[i]:
                    entry[1] = self._birch.insert(matrix[i], lsh_labels[i] if lsh_labels[i] is not None else 0)
                else:
                    self._birch.absorb(matrix[i], int(entry[1]))
                birch_labels[i] = entry[1]
        self._stage_labels["lsh"].extend(lsh_labels)
        self._stage_labels["birch"].extend(birch_labels)

        den_labels: list[int | None] = [None] * len(rows)
        how: list[str | None] = [None] * len(rows)
        pruned: list[int] = []
        if self.config.denstream:
            if self._den is None:
                count = self._store.count
                self._den = OnlineDenStream(
                    self.config, self._denstream_epsilon(count), self._half_life(count, finishing=warm), matrix.shape[1], 0.0
                )
            upstream = birch_labels if self._birch is not None else lsh_labels
            for i, entry in enumerate(entries):
                cached = None if first[i] else entry[2]
                prunes_before = len(self._den.prunes)
                label, how[i], gone = self._den.insert(matrix[i], rows[i].clock, cached, upstream[i])
                entry[2] = den_labels[i] = label
                pruned.extend(gone)
                if len(self._den.prunes) > prunes_before:
                    self._prunes.append({"after_usable_row": start + i, "clock": rows[i].clock, "pruned": gone})
        self._stage_labels["den"].extend(den_labels)
        if not self._checkpoints:
            self._checkpoint_record(self._store.count)

        final = den_labels if self.config.denstream else birch_labels if self._birch is not None else lsh_labels
        labels = np.asarray(final, dtype=np.int64)
        self._labels.extend(labels.tolist())
        self._how.extend(how)
        if coords is None:
            assert self._projection is not None
            coords = transform(self._projection, matrix)
        state = self._state
        evs = [r.ev for r in rows]
        add_rows(
            state,
            [ev.individual_id for ev in evs],
            np.array([ev.island_id for ev in evs], dtype=np.int64),
            labels,
            np.array([ev.eval_index for ev in evs], dtype=np.int64),
            np.array([ev.fitness for ev in evs], dtype=np.float64),
            [ev.genome_hash for ev in evs],
            coords,
        )
        lineage = []
        for offset, ev in enumerate(evs):
            child = start + offset
            self._processed_of[rows[offset].released] = child
            for parent_id in ev.parent_ids:
                parent = state.row_of.get(parent_id)
                # Parents on another island are migrated copies; migrations cover those.
                if parent is not None and parent < child and self._released_island(parent) == ev.island_id:
                    lineage.append((parent, child))
        add_edges(state, lineage)
        kill_labels(state, pruned)
        for ev in evs:
            for index in self._waiting.pop(ev.individual_id, []):
                self._resolve(index)

    def _released_island(self, processed: int) -> int:
        return self._released[self._row_release[processed]].island_id

    # Resolve a migrant pair once both rows are in the graph; drop it if either has no genome.
    def _resolve(self, index: int) -> None:
        origin, arrived = self._pairs[index]
        if origin in self._unusable_ids or arrived in self._unusable_ids:
            return
        row_of = self._state.row_of if self._state is not None else {}
        for individual in (origin, arrived):
            if individual not in row_of:
                self._waiting.setdefault(individual, []).append(index)
                return
        assert self._state is not None
        add_migrations(self._state, [(row_of[origin], row_of[arrived])])
        self._resolved_pairs.append((origin, arrived))

    # --- output ---

    def _set_weights(self) -> None:
        if self._den is None or self._state is None:
            return
        weights = self._den.weights()
        state = self._state
        alive = np.flatnonzero(state.alive[: state.nodes])
        state.weight[alive] = [weights[int(label)] for label in state.label[alive].tolist()]

    def _diagnostics(self, final: bool) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if self._lsh is not None:
            out["lsh"] = self._lsh.diagnostics()
        if self._birch is not None:
            out["birch"] = self._birch.diagnostics()
        if self._den is not None:
            out["denstream"] = self._den.diagnostics(final)
        return {stage: {k: _clean(v) for k, v in values.items()} for stage, values in out.items()}

    def _info(self, diagnostics: dict[str, dict[str, Any]], nodes: int) -> ClusteringInfo:
        return ClusteringInfo(
            stages=self.config.stages,
            counts=Counts(
                evaluations=len(self._released),
                rows_clustered=self._store.count if self._store is not None else 0,
                nodes=nodes,
            ),
            resolved=Resolved(**{k: _clean(v) for k, v in self._resolved.items()}),
            diagnostics=diagnostics,
            elapsed_seconds=self._elapsed,
        )

    def _warnings(self) -> list[str]:
        warnings = []
        if self._malformed:
            warnings.append(malformed_warning(self._malformed))
        missing = len(self._released) - sum(self._usable)
        if missing:
            warnings.append(missing_warning(missing, len(self._released)))
        return warnings

    def _update(self, started: float) -> Update:
        if not self._warmed or self._state is None:
            self._elapsed += time.perf_counter() - started
            return Update.model_construct(
                status="warming_up",
                layout=None,
                clustering=self._info({}, 0),
                rows_without_genome=len(self._released) - sum(self._usable),
                warnings=self._warnings(),
                graph=_empty_delta(self.config.denstream),
            )
        self._set_weights()
        state = self._state
        budget = self._log.start.get("evaluation_budget_per_island") if self._log.start else None
        span = {island: int(budget) for island in self._span} if budget else dict(self._span)
        cols = columns(state, span)
        refit = self._refit
        self._refit = False
        delta = diff(self._sent, cols, state.nodes, state.edges, state.migrations, refit)
        projection = self._projection
        layout = (
            LayoutInfo(
                method=projection.method,
                variance_kept_2d=projection.variance_kept_2d,
                variance_kept_3d=projection.variance_kept_3d,
            )
            if refit and projection is not None
            else None
        )
        diagnostics = self._diagnostics(final=False)
        self._elapsed += time.perf_counter() - started
        return Update.model_construct(
            status="live",
            layout=layout,
            clustering=self._info(diagnostics, int(cols["id"].size)),
            rows_without_genome=len(self._released) - sum(self._usable),
            warnings=self._warnings(),
            graph=delta,
        )

    # Private, for the verification scripts: per-row labels and placements, the calibration
    # schedule, DenStream's prunes and lambda changes, and the resolved migrant pairs.
    def _trace(self) -> dict[str, Any]:
        state = self._state
        node_of_row = state.row_node[: state.rows].tolist() if state is not None else []
        rows = []
        for i, ev in enumerate(self._released):
            processed = self._processed_of[i]
            done = processed is not None
            rows.append(
                {
                    "individual_id": ev.individual_id,
                    "island_id": ev.island_id,
                    "seq": ev.seq,
                    "eval_index": ev.eval_index,
                    "fitness": ev.fitness,
                    "genome_hash": ev.genome_hash,
                    "parent_ids": list(ev.parent_ids),
                    "usable": self._usable[i],
                    "clock": self._clock[i],
                    "lsh": self._stage_labels["lsh"][processed] if done else None,
                    "birch": self._stage_labels["birch"][processed] if done else None,
                    "den": self._stage_labels["den"][processed] if done else None,
                    "label": self._labels[processed] if done else None,
                    "node": node_of_row[processed] if done else None,
                    "how": self._how[processed] if done else None,
                }
            )
        store = self._store
        den = self._den
        return {
            "rows": rows,
            "matrix": store.rows.copy() if store is not None else np.zeros((0, 0)),
            "sets": list(store.sets[: store.count]) if store is not None and store.sets is not None else None,
            "coords": state.row_coords[: state.rows].copy() if state is not None else np.zeros((0, 3)),
            "checkpoints": [dict(c) for c in self._checkpoints],
            "prunes": [dict(p) for p in self._prunes],
            "denstream": (
                {"lambda_segments": list(den.lambda_segments), "alive": den.weights(), "final_clock": den.now}
                if den is not None
                else None
            ),
            "migration_pairs": list(self._resolved_pairs),
            "maximising": self._maximising,
            "budget_per_island": self._log.start.get("evaluation_budget_per_island") if self._log.start else None,
            "budget": self._log.start.get("evaluation_budget") if self._log.start else None,
            "metric": store.metric.value if store is not None else None,
            "config": self.config,
        }
