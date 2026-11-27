from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .models import Convergence, ConvergenceLevel, ConvergencePoint, GenerationSummary, IslandEnd

# The shares of the run's improvement whose crossing points are reported.
CONVERGENCE_SHARES = (0.5, 0.9, 0.99, 0.999)

type Axis = Literal["generation", "evaluations"]


@dataclass
class _Series:
    island: int
    generation: list[int]
    evaluations: list[int | None]
    best_so_far: list[float]


def _series(history: Sequence[GenerationSummary]) -> list[_Series]:
    by_island: dict[int, _Series] = {}
    for summary in sorted(history, key=lambda s: (s.island_id, s.generation)):
        if summary.best_so_far_fitness is None:
            continue
        series = by_island.setdefault(summary.island_id, _Series(summary.island_id, [], [], []))
        series.generation.append(summary.generation)
        series.evaluations.append(summary.evaluations_total)
        series.best_so_far.append(summary.best_so_far_fitness)
    return list(by_island.values())


# The share of all islands' generations or evaluations that came after x.
# Each island counts from 0 to its island_end total, or to its last summary when it logged no island_end.
# On the generation axis x is a generation index, so x + 1 generations were done by then.
def _share_after(history: Sequence[GenerationSummary], axis: Axis, x: int, ends: Mapping[int, IslandEnd]) -> float:
    last: dict[int, int] = {}
    for summary in history:
        value = summary.generation + 1 if axis == "generation" else summary.evaluations_total
        if value is not None:
            last[summary.island_id] = max(last.get(summary.island_id, 0), value)
    done = x + 1 if axis == "generation" else x
    after = total = 0
    for island, logged in last.items():
        end = ends.get(island)
        if end is not None:
            logged = max(logged, end.generations_completed if axis == "generation" else end.evaluations_total)
        after += max(0, logged - done)
        total += logged
    return after / total if total else 0.0


# The first island to get its best-so-far past `threshold` (on the scale already applied), along the axis
def _first_past(
    series: list[_Series],
    history: Sequence[GenerationSummary],
    axis: Axis,
    threshold: float,
    scale: Callable[[float], float],
    maximising: bool,
    ends: Mapping[int, IslandEnd],
) -> ConvergencePoint | None:
    first: tuple[int, int, float] | None = None
    for s in series:
        xs = s.generation if axis == "generation" else s.evaluations
        for x, value in zip(xs, s.best_so_far):
            if not (scale(value) >= threshold if maximising else scale(value) <= threshold):
                continue
            if x is not None and (first is None or x < first[1]):
                first = (s.island, x, value)
            break
    if first is None:
        return None
    island, x, value = first
    return ConvergencePoint(island_id=island, x=x, fitness=value, after=_share_after(history, axis, x, ends))


# How far the run's best moved and where it crossed each share of that move.
# The gain runs from the best of the islands' first best-so-far values to the best of all of them.
# ends are the islands' island_end events, which say where each island's search really ended.
def convergence(history: Sequence[GenerationSummary], maximising: bool, ends: Sequence[IslandEnd] = ()) -> Convergence | None:
    series = _series(history)
    by_island = {end.island_id: end for end in ends}
    if not series:
        return None

    def pick(values: list[float]) -> float:
        return max(values) if maximising else min(values)

    values = [v for s in series for v in s.best_so_far]
    start = pick([s.best_so_far[0] for s in series])
    final = pick(values)
    log = not maximising and all(v > 0 for v in values)

    def scale(value: float) -> float:
        return math.log10(value) if log else value

    gain = abs(scale(final) - scale(start))
    levels: list[ConvergenceLevel] = []
    has_evaluations = all(s.evaluations_total is not None for s in history)
    for share in CONVERGENCE_SHARES if gain > 0 else ():
        remaining = (1 - share) * gain
        threshold = scale(final) - remaining if maximising else scale(final) + remaining
        by_generation = _first_past(series, history, "generation", threshold, scale, maximising, by_island)
        assert by_generation is not None, "the island that reached the final best is always past the threshold"
        by_evaluations = _first_past(series, history, "evaluations", threshold, scale, maximising, by_island) if has_evaluations else None
        levels.append(ConvergenceLevel(share=share, generation=by_generation, evaluations=by_evaluations))
    return Convergence(scale="log" if log else "linear", start_fitness=start, final_fitness=final, levels=levels)
