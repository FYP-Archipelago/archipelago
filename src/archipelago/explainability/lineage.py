from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, NamedTuple


# The fields of an evaluation row that lineage needs. Genomes aren't required.
class LineageRow(NamedTuple):
    individual_id: str
    island_id: int
    fitness: float
    parent_ids: tuple[str, ...]
    operator: str | None
    origin_island: int | None
    origin_individual_id: str | None


def lineage_row(row: Mapping[str, Any]) -> LineageRow:
    return LineageRow(
        row["individual_id"],
        row["island_id"],
        float(row["fitness"]),
        tuple(row["parent_ids"]),
        row.get("operator"),
        row.get("origin_island"),
        row.get("origin_individual_id"),
    )


@dataclass
class Node:
    island: int
    fitness: float
    founder: bool
    parents: set[str] = field(default_factory=set)
    # (parent island, this island) for every parent on another island.
    routes: set[tuple[int, int]] = field(default_factory=set)
    # False when a parent, or a migrant's origin, was never logged.
    complete: bool = True


# Links every individual to its logged parents; a migrant's parent is its origin copy on the source island.
# Only an init row without parents is a founder, and any other row without a resolvable parent is incomplete.
def build_graph(rows: Sequence[LineageRow]) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    for row in rows:
        if row.individual_id in nodes:
            raise ValueError(f"Duplicate individual ID: {row.individual_id}")
        if not isfinite(row.fitness):
            raise ValueError(f"Non-finite fitness for {row.individual_id}")
        founder = row.operator == "init" and not row.parent_ids and not row.origin_individual_id
        nodes[row.individual_id] = Node(row.island_id, row.fitness, founder)

    for row in rows:
        node = nodes[row.individual_id]
        parent_ids = set(row.parent_ids)
        # Other operators (PSO) may carry origin fields as provenance; only an arrival depends on its origin.
        if row.operator == "migration_arrival":
            parent_ids.discard(row.origin_individual_id)
            origin = nodes.get(row.origin_individual_id) if row.origin_individual_id else None
            if origin is not None and origin.island == row.origin_island:
                parent_ids.add(row.origin_individual_id)
            else:
                node.complete = False
        for parent_id in parent_ids:
            parent = nodes.get(parent_id)
            if parent is None:
                node.complete = False
                continue
            node.parents.add(parent_id)
            if parent.island != node.island:
                node.routes.add((parent.island, node.island))
        if not node.founder and not node.parents:
            node.complete = False
    return nodes


# Walks parents before children. A node is supported when it and all its ancestors are complete, and gets the
# bitmasks of the islands and routes its ancestry passes through (bit i is islands[i] or routes[i]).
# Also returns the nodes the walk never reached, which only happens at or below a parent cycle.
def ancestry_masks(nodes: dict[str, Node], islands: list[int], routes: list[tuple[int, int]]) -> tuple[dict[str, tuple[int, int]], set[str]]:
    island_bit = {island: 1 << i for i, island in enumerate(islands)}
    route_bit = {route: 1 << i for i, route in enumerate(routes)}
    children: dict[str, list[str]] = defaultdict(list)
    waiting = {}
    for node_id, node in nodes.items():
        waiting[node_id] = len(node.parents)
        for parent_id in node.parents:
            children[parent_id].append(node_id)

    masks: dict[str, tuple[int, int]] = {}
    unreached = set(nodes)
    ready = deque(node_id for node_id, count in waiting.items() if not count)
    while ready:
        node_id = ready.popleft()
        unreached.discard(node_id)
        node = nodes[node_id]
        if node.complete and all(parent_id in masks for parent_id in node.parents):
            island_mask, route_mask = island_bit[node.island], 0
            for parent_id in node.parents:
                island_mask |= masks[parent_id][0]
                route_mask |= masks[parent_id][1]
            for route in node.routes:
                route_mask |= route_bit[route]
            masks[node_id] = (island_mask, route_mask)
        for child_id in children[node_id]:
            waiting[child_id] -= 1
            if not waiting[child_id]:
                ready.append(child_id)
    return masks, unreached
