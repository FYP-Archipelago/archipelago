import random
from collections.abc import Callable, Sequence
from math import comb, sqrt

from ..models import LineageCoverage, Player, RunInfo, ShapleyAttribution, ShapleyCoalition, ShapleyGame, ShapleyPlayer, ShapleyStep
from .lineage import LineageRow, ancestry_masks, build_graph

MAX_ROWS = 300_000  # bounds analysis time
MAX_SAMPLING_WORK = 50_000_000  # candidate checks per sampled game, a few seconds at most
EXACT_MAX_PLAYERS = 10  # larger games are sampled; exact cost grows as 2^n * n
SAMPLES = 1024  # player orders sampled per larger game
EXAMPLES = 6  # example player orders kept for the UI


# Splits the run's improvement over its best founder between islands, and separately between migration routes.
# An individual's gain counts towards a group only if every island (or route) in its ancestry is in the group.
def attribute_improvement(rows: Sequence[LineageRow], run: RunInfo) -> ShapleyAttribution:
    if not rows or len(rows) > MAX_ROWS:
        raise ValueError(f"Expected between 1 and {MAX_ROWS:,} evaluation records.")
    nodes = build_graph(rows)
    best = max if run.maximising else min
    founders = [node.fitness for node in nodes.values() if node.founder]
    if not founders:
        raise ValueError("No explicit initial-population records; cannot define the baseline.")
    baseline = best(founders)

    islands = sorted({node.island for node in nodes.values()})
    routes = sorted({route for node in nodes.values() for route in node.routes})
    masks, unreached = ancestry_masks(nodes, islands, routes)
    sign = 1 if run.maximising else -1
    gains = {node_id: max(0.0, (nodes[node_id].fitness - baseline) * sign) for node_id in masks}

    warnings = []
    excluded = len(nodes) - len(masks)
    if excluded:
        missing = sum(not node.complete for node_id, node in nodes.items() if node_id not in unreached)
        reasons = [
            ("missing or unlogged parent", missing),
            ("in or below a parent cycle", len(unreached)),
            ("descended from those", excluded - missing - len(unreached)),
        ]
        listed = ", ".join(f"{reason}: {count}" for reason, count in reasons if count)
        warnings.append(f"{excluded} evaluations lack fully resolved ancestry and are excluded ({listed}).")

    return ShapleyAttribution(
        baseline_fitness=baseline,
        observed_best_fitness=best(node.fitness for node in nodes.values()),
        supported_best_fitness=best(nodes[node_id].fitness for node_id in masks),
        coverage=LineageCoverage(evaluations=len(nodes), supported=len(masks)),
        warnings=warnings,
        islands=shapley_values(islands, [(masks[node_id][0], gain) for node_id, gain in gains.items()]),
        routes=shapley_values(routes, [(masks[node_id][1], gain) for node_id, gain in gains.items()]),
    )


# Shapley values of the game where a coalition's value is the best gain among candidates whose required players are all in it.
# Exact up to EXACT_MAX_PLAYERS; beyond that averaged over sampled player orders, with a standard error.
def shapley_values(
    players: Sequence[Player], candidates: Sequence[tuple[int, float]], *, seed: int = 20261103, samples: int = SAMPLES
) -> ShapleyGame:
    n = len(players)
    best: dict[int, float] = {}
    for mask, gain in candidates:
        best[mask] = max(best.get(mask, 0.0), gain)
    exact = n <= EXACT_MAX_PLAYERS
    if not exact and len(best) * min(samples * n, 1 << n) > MAX_SAMPLING_WORK:
        raise ValueError("Estimated permutation-sampling work exceeds the Shapley resource limit.")

    value: Callable[[int], float]
    if exact:
        # Every coalition up front: its own best gain, or that of any coalition inside it.
        table = [best.get(mask, 0.0) for mask in range(1 << n)]
        for i in range(n):
            for mask in range(1 << n):
                if mask & (1 << i):
                    table[mask] = max(table[mask], table[mask ^ (1 << i)])
        value = table.__getitem__
    else:
        cache: dict[int, float] = {}

        def value(mask: int) -> float:
            if mask not in cache:
                cache[mask] = max((gain for required, gain in best.items() if required & mask == required), default=0.0)
            return cache[mask]

    empty, full = value(0), value((1 << n) - 1)
    rng = random.Random(seed)

    def steps(order: Sequence[int]) -> list[ShapleyStep]:
        mask, out = 0, []
        for i in order:
            before = value(mask)
            mask |= 1 << i
            out.append(ShapleyStep(player=players[i], before=before, after=value(mask)))
        return out

    credit, squared = [0.0] * n, [0.0] * n
    orders: list[list[ShapleyStep]] = []
    if exact:
        for i in range(n):
            for mask in range(1 << n):
                if not mask & (1 << i):
                    credit[i] += (value(mask | (1 << i)) - value(mask)) / (n * comb(n - 1, mask.bit_count()))
        if n:
            # Forward and reversed first, then shuffles; 12 tries usually yields enough distinct orders.
            tried = [tuple(range(n)), tuple(reversed(range(n)))]
            for _ in range(12):
                order = list(range(n))
                rng.shuffle(order)
                tried.append(tuple(order))
            orders = [steps(order) for order in list(dict.fromkeys(tried))[:EXAMPLES]]
    else:
        for sample in range(samples):
            order = list(range(n))
            rng.shuffle(order)
            mask, before = 0, empty
            for i in order:
                mask |= 1 << i
                after = value(mask)
                marginal = after - before
                credit[i] += marginal
                squared[i] += marginal * marginal
                before = after
            if sample < EXAMPLES:
                orders.append(steps(order))
        credit = [total / samples for total in credit]

    ranked = sorted(range(n), key=lambda i: (-credit[i], i))
    return ShapleyGame(
        empty_value=empty,
        full_value=full,
        sampled_orders=None if exact else samples,
        players=[
            ShapleyPlayer(
                player=players[i],
                value=credit[i],
                standard_error=None if exact else sqrt(max(0, squared[i] - samples * credit[i] ** 2) / (samples - 1) / samples),
            )
            for i in ranked
        ],
        coalitions=[ShapleyCoalition(players=[players[i] for i in range(n) if mask & (1 << i)], value=value(mask)) for mask in range(1 << n)]
        if exact
        else [],
        orders=orders,
    )
