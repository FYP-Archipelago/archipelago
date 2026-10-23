from __future__ import annotations

import heapq
from bisect import insort
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import SchemaError
from .records import Evaluation, EventLog


# One record waiting to be released: an evaluation row or a migration_arrive event.
@dataclass(frozen=True, order=True)
class Item:
    seq: int
    island: int = field(compare=False)
    t_wall: float = field(compare=False)
    row: Evaluation | None = field(default=None, compare=False)
    event: Mapping[str, Any] | None = field(default=None, compare=False)


# Per-island queues, merged into one order that depends only on the logs: an island's records
# leave in seq order, and across islands the head with the smallest clock-corrected t_wall goes
# first. A live island with nothing queued holds everything back, because its next record
# could still be the earliest.
@dataclass
class Order:
    queues: dict[int, deque[Item]] = field(default_factory=dict)
    released_seq: dict[int, int] = field(default_factory=dict)
    seen_rows: set[tuple[int, int]] = field(default_factory=set)
    seen_events: set[tuple[Any, ...]] = field(default_factory=set)


# False if this record was delivered before (at-least-once delivery is harmless).
def first_delivery(order: Order, event: Mapping[str, Any]) -> bool:
    key = (event.get("emitter"), event.get("island_id"), event.get("seq"), event.get("type"))
    if key in order.seen_events:
        return False
    order.seen_events.add(key)
    return True


def enqueue(order: Order, item: Item) -> None:
    if item.row is not None:
        key = (item.island, item.seq)
        if key in order.seen_rows:
            return
        order.seen_rows.add(key)
    if item.seq <= order.released_seq.get(item.island, -1):
        raise SchemaError(f"island {item.island}: record seq {item.seq} arrived after later records were processed")
    queue = order.queues.setdefault(item.island, deque())
    if not queue or queue[-1].seq < item.seq:
        queue.append(item)
    else:
        items = list(queue)
        insort(items, item)
        order.queues[item.island] = deque(items)


def _ready(log: EventLog) -> bool:
    if log.start is None:
        return False
    expected = log.start.get("num_islands")
    return expected is None or len(set(log.started_islands)) >= int(expected)


def _blocked(order: Order, log: EventLog, island: int) -> bool:
    started = island in log.clock_offsets or island in log.started_islands
    return started and island not in log.ended_islands and not order.queues.get(island)


# Pop every record that is safe to process now, in canonical order. With final=True, nothing
# blocks: the run is over and every queued record is released.
def release(order: Order, log: EventLog, final: bool) -> list[Item]:
    if not final and not _ready(log):
        return []
    offsets = log.clock_offsets
    heads = [
        (queue[0].t_wall + offsets.get(island, 0.0), island, queue[0].seq)
        for island, queue in order.queues.items()
        if queue
    ]
    heapq.heapify(heads)
    started = set(log.started_islands)
    out: list[Item] = []
    while heads:
        if not final and any(_blocked(order, log, island) for island in started):
            break
        _, island, _ = heapq.heappop(heads)
        queue = order.queues[island]
        item = queue.popleft()
        order.released_seq[island] = item.seq
        out.append(item)
        if queue:
            heapq.heappush(heads, (queue[0].t_wall + offsets.get(island, 0.0), island, queue[0].seq))
    return out
