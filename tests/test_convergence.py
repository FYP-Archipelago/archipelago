from __future__ import annotations

from archipelago.convergence import CONVERGENCE_SHARES, convergence
from archipelago.models import GenerationSummary, IslandEnd


# One island's generations from 0, with evaluations_total 10 per generation.
# A None best-so-far is a generation logged without one.
def island(island_id: int, best_so_far: list[float | None], evaluations: bool = True) -> list[GenerationSummary]:
    return [
        GenerationSummary(island_id=island_id, generation=g, evaluations_total=(g + 1) * 10 if evaluations else None, best_so_far_fitness=v)
        for g, v in enumerate(best_so_far)
    ]


# Islands 0 and 1 both end on 60; island 1 gets there first (generation 2, 30 evaluations).
HISTORY = island(0, [100, 80, 80, 60, 60]) + island(1, [100, 90, 60, 60, 60]) + island(2, [100, 95, 90, 85, 70])


def test_dates_each_share_to_the_first_island_to_get_that_close_along_each_axis() -> None:
    result = convergence(HISTORY, maximising=False)
    assert result is not None
    assert (result.scale, result.start_fitness, result.final_fitness) == ("log", 100, 60)
    assert [level.share for level in result.levels] == list(CONVERGENCE_SHARES)
    level = result.levels[2]
    assert (level.generation.island_id, level.generation.x, level.generation.fitness) == (1, 2, 60)
    assert level.evaluations is not None and (level.evaluations.island_id, level.evaluations.x) == (1, 30)
    # Every island ran 5 generations (50 evaluations); 3 were done by generation 2 (30 evaluations), so 2 of 5 came after.
    assert level.generation.after == level.evaluations.after == 0.4


def test_the_share_after_counts_every_island_from_its_start_to_its_logged_end() -> None:
    # Island 1's island_end says it went on to 80 evaluations and 8 generations after its last summary.
    # Its 50 evaluations after 30 (of 80) and 5 generations after the 3rd (of 8) count too.
    ends = [IslandEnd(island_id=1, termination_reason="max_evaluations", generations_completed=8, evaluations_total=80)]
    level = convergence(HISTORY, maximising=False, ends=ends).levels[2]
    assert level.evaluations is not None and level.evaluations.after == (20 + 50 + 20) / (50 + 80 + 50)
    assert level.generation.after == (2 + 5 + 2) / (5 + 8 + 5)


def test_the_scale_is_fixed_by_the_run_not_by_a_chart() -> None:
    # Minimising with positive values counts orders of magnitude, so 10 is two of the four decades from 1000 to 0.1.
    # In fitness it would already be 99% of the way.
    falling = convergence(island(0, [1000, 10, 1, 0.1, 0.1]), maximising=False)
    assert falling is not None and falling.scale == "log"
    assert falling.levels[2].generation.x == 3
    # A zero rules the log scale out: within 1% of the 5 gained is 0.05, first reached at generation 2.
    to_zero = convergence(island(0, [5, 0.5, 0, 0]), maximising=False)
    assert to_zero is not None and to_zero.scale == "linear"
    assert to_zero.levels[2].generation.x == 2
    # Maximising always counts in fitness: half of 10 -> 30 is 20, first passed at generation 2.
    rising = convergence(island(0, [10, 15, 30]), maximising=True)
    assert rising is not None and rising.scale == "linear"
    assert (rising.start_fitness, rising.final_fitness, rising.levels[0].generation.x) == (10, 30, 2)


def test_gaps_missing_evaluations_and_no_movement() -> None:
    # Generations without a best-so-far are skipped, so the start is the first logged one.
    gappy = convergence(island(0, [None, 100, 50]), maximising=False)
    assert gappy is not None and gappy.start_fitness == 100
    without_evaluations = convergence(island(0, [100, 50], evaluations=False), maximising=False)
    assert without_evaluations is not None and all(level.evaluations is None for level in without_evaluations.levels)
    flat = convergence(island(0, [7, 7, 7]), maximising=False)
    assert flat is not None and flat.levels == []
    assert convergence(island(0, [None, None]), maximising=False) is None
    assert convergence([], maximising=False) is None
