from __future__ import annotations

import pytest

from archipelago.errors import SchemaError
from archipelago.ordering import Item, Order, enqueue, first_delivery, release
from archipelago.records import EventLog, apply_event, parse_row


def item(island: int, seq: int, t_wall: float) -> Item:
    row = parse_row(
        {
            "island_id": island,
            "eval_index": seq,
            "generation": 0,
            "seq": seq,
            "t_wall": t_wall,
            "individual_id": f"{island}-{seq}",
            "fitness": 1.0,
            "genome_encoding": "real_vector",
            "genome_hash": "h",
            "genome_repr_mode": "full",
        },
        0,
    )
    return Item(seq=seq, island=island, t_wall=t_wall, row=row)


def log_for(islands: int, offsets_ns: dict[int, int] | None = None, started: int | None = None, ended: tuple[int, ...] = ()) -> EventLog:
    log = EventLog()
    apply_event(log, {"type": "run_start", "schema_version": "2.0", "run_id": "r", "algorithm": "a", "benchmark": "b", "num_islands": islands})
    for island in range(islands if started is None else started):
        apply_event(log, {"type": "island_start", "island_id": island, "clock_offset_ns": (offsets_ns or {}).get(island, 0)})
    for island in ended:
        apply_event(log, {"type": "island_end", "island_id": island})
    return log


def names(items: list[Item]) -> list[tuple[int, int]]:
    return [(i.island, i.seq) for i in items]


def test_release_is_kway_merge_by_corrected_time_island_seq() -> None:
    # Island 1's clock is 3 s ahead, so its t_wall 4, 6, 20 are corrected to 1, 3, 17.
    log = log_for(2, {1: -3_000_000_000}, ended=(0, 1))
    order = Order()
    for it in (item(0, 0, 1.0), item(0, 1, 5.0), item(0, 2, 10.0), item(1, 0, 4.0), item(1, 1, 6.0), item(1, 2, 20.0)):
        enqueue(order, it)
    # Corrected times: i0 = 1, 5, 10; i1 = 1, 3, 17. The tie at 1.0 goes to the lower island.
    expected = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    assert names(release(order, log, final=False)) == expected


def test_started_island_with_empty_queue_blocks_until_island_end() -> None:
    log = log_for(2)
    order = Order()
    enqueue(order, item(0, 0, 1.0))
    enqueue(order, item(0, 1, 2.0))
    # Island 1 has started, has not ended and has nothing queued: its next record could be earliest.
    assert release(order, log, final=False) == []
    apply_event(log, {"type": "island_end", "island_id": 1})
    assert names(release(order, log, final=False)) == [(0, 0), (0, 1)]
    check_nothing_released_before_run_start_and_all_island_starts()


def check_nothing_released_before_run_start_and_all_island_starts() -> None:
    order = Order()
    enqueue(order, item(0, 0, 1.0))
    assert release(order, EventLog(), final=False) == []
    # run_start announces 2 islands but only one has started.
    log = log_for(2, started=1, ended=(0,))
    assert release(order, log, final=False) == []
    apply_event(log, {"type": "island_start", "island_id": 1})
    apply_event(log, {"type": "island_end", "island_id": 1})
    assert names(release(order, log, final=False)) == [(0, 0)]


def test_redelivered_rows_and_events_are_dropped() -> None:
    order = Order()
    enqueue(order, item(0, 0, 1.0))
    enqueue(order, item(0, 0, 1.0))
    assert len(order.queues[0]) == 1
    event = {"type": "migration_arrive", "emitter": "e", "island_id": 0, "seq": 4}
    assert first_delivery(order, event)
    assert not first_delivery(order, dict(event))
    assert first_delivery(order, {**event, "type": "generation_end"})
    assert first_delivery(order, {**event, "seq": 5})
    check_row_older_than_a_released_seq_raises()


def check_row_older_than_a_released_seq_raises() -> None:
    log = log_for(1, ended=(0,))
    order = Order()
    enqueue(order, item(0, 0, 1.0))
    enqueue(order, item(0, 2, 3.0))
    assert names(release(order, log, final=False)) == [(0, 0), (0, 2)]
    with pytest.raises(SchemaError):
        enqueue(order, item(0, 1, 2.0))
