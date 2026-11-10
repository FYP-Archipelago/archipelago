from __future__ import annotations

import json
import math
from typing import Any

import pytest

from archipelago import Explainability, ExplainabilityAnalysisUnavailable, RunInfo, ShapleyAttribution, ShapleyGame
from archipelago import explainability as explainability_mod
from archipelago.explainability import Analyses, add, finish
from archipelago.explainability import shapley as shapley_mod
from archipelago.explainability.lineage import lineage_row
from archipelago.explainability.shapley import attribute_improvement, shapley_values

type Row = dict[str, Any]


def row(identity: str, island: int, fitness: float, parents: list[str] | None = None, operator: str = "trial", **extra: Any) -> Row:
    return dict(individual_id=identity, island_id=island, fitness=fitness, parent_ids=parents or [], operator=operator, **extra)


def run_info(maximising: bool = False) -> RunInfo:
    return RunInfo(run_id="run-test", algorithm="ga", benchmark="sphere", migration_topology="ring", islands=[], maximising=maximising)


def analyse(rows: list[Row], maximising: bool = False) -> ShapleyAttribution | ExplainabilityAnalysisUnavailable:
    analyses = Analyses()
    add(analyses, rows)
    return finish(analyses, run_info(maximising)).shapley


def unavailable(rows: list[Row], maximising: bool = False) -> str:
    result = analyse(rows, maximising)
    assert isinstance(result, ExplainabilityAnalysisUnavailable)
    return result.detail


def attribute(rows: list[Row], maximising: bool = False) -> ShapleyAttribution:
    return attribute_improvement([lineage_row(r) for r in rows], run_info(maximising))


def values(game: ShapleyGame) -> list[float]:
    return [player.value for player in game.players]


def test_parent_ids_produce_complementary_credit_without_genomes() -> None:
    rows = [
        row("a", 0, 100, operator="init", genome_repr=[123]),
        row("b", 1, 100, operator="init"),
        row("m", 1, 80, ["a"], "migration_arrival", origin_island=0, origin_individual_id="a"),
        row("w", 1, 10, ["b", "m"]),
    ]
    analyses = Analyses()
    add(analyses, rows)
    result = finish(analyses, run_info()).shapley
    assert isinstance(result, ShapleyAttribution)
    assert values(result.islands) == [45, 45]
    assert [player.player for player in result.routes.players] == [(0, 1)]
    assert "genome" not in repr(analyses.shapley)
    assert "genome" not in result.model_dump_json()


def test_ready_and_unavailable_results_round_trip_through_json() -> None:
    for result in (analyse([row("a", 0, 10, operator="init"), row("b", 0, 5, ["a"])]), analyse([])):
        explainability = Explainability(shapley=result)
        assert Explainability.model_validate_json(explainability.model_dump_json()) == explainability


def test_chunking_and_original_input_mutation_do_not_change_lineage() -> None:
    initial = row("a", 0, 100, operator="init")
    child = row("b", 0, 25, ["a"])
    analyses = Analyses()
    add(analyses, [initial])
    add(analyses, [child])
    child["parent_ids"].append("not-recorded")
    result = finish(analyses, run_info()).shapley
    assert isinstance(result, ShapleyAttribution)
    assert result.islands.full_value == 75


def test_missing_operator_does_not_invent_initial_population() -> None:
    assert "baseline" in unavailable([{"individual_id": "a", "island_id": 0, "fitness": 10, "parent_ids": []}])


def test_row_limit_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapley_mod, "MAX_ROWS", 3)
    assert "3" in unavailable([row(str(i), 0, 10, operator="init") for i in range(4)])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [row("a", 0, math.nan, operator="init")],
        [row("a", 0, 1, operator="init")] * 2,
    ],
    ids=["empty", "nan-fitness", "duplicate-id"],
)
def test_bad_lineage_is_unavailable(rows: list[Row]) -> None:
    assert unavailable(rows).startswith("Shapley analysis unavailable: ")


def test_unexpected_errors_are_unavailable_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: Any) -> ShapleyAttribution:
        raise RuntimeError("bug")

    monkeypatch.setattr(explainability_mod, "attribute_improvement", broken)
    assert "bug" in unavailable([row("a", 0, 10, operator="init")])


def test_sampling_resource_limit_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapley_mod, "MAX_SAMPLING_WORK", 2)
    assert "work" in unavailable([row(str(i), i, 100, operator="init") for i in range(11)])


def test_large_finite_fitness_cannot_emit_nonfinite_json() -> None:
    # The gain over the baseline (2e308) overflows to infinity.
    result = analyse([row("a", 0, -1e308, operator="init"), row("b", 0, 1e308, ["a"])], maximising=True)
    assert isinstance(result, ExplainabilityAnalysisUnavailable)
    json.dumps(result.model_dump(), allow_nan=False)


def test_unresolved_ancestry_keeps_coverage_and_retained_pso_origin_is_only_provenance() -> None:
    result = attribute(
        [
            row("a", 0, 100, operator="init"),
            row("b", 1, 100, operator="init"),
            row("p", 1, 1, [], "pso_move", origin_island=0, origin_individual_id="a"),
            row("w", 1, 5, ["b"], "pso_move", origin_island=0, origin_individual_id="a"),
        ]
    )
    assert (result.coverage.evaluations, result.coverage.supported) == (4, 3)
    assert result.routes.players == []
    assert result.routes.empty_value == 95


def test_maximising_controls_direction() -> None:
    rows = [row("a", 0, 10, operator="init"), row("b", 0, 20, ["a"])]
    assert attribute(rows, maximising=True).islands.full_value == 10
    assert attribute(rows, maximising=False).islands.full_value == 0


def test_full_parent_lineage_not_any_parent_path() -> None:
    rows = [
        row("a", 0, 100, operator="init"),
        row("b", 1, 100, operator="init"),
        row("c", 1, 80, ["a"], "migration_arrival", origin_island=0, origin_individual_id="a"),
        row("d", 1, 10, ["b", "c"]),
    ]
    result = attribute(rows)
    assert values(result.islands) == [45, 45]
    assert result.routes.full_value == 90
    assert result.routes.empty_value == 0
    assert result.coverage.supported == 4


def test_migration_game_only_allocates_gain_beyond_local_only() -> None:
    rows = [
        row("a", 0, 100, operator="init"),
        row("b", 1, 100, operator="init"),
        row("l", 0, 60, ["a"]),
        row("m", 1, 60, ["l"], "migration_arrival", origin_island=0, origin_individual_id="l"),
        row("w", 1, 20, ["b", "m"]),
    ]
    game = attribute(rows).routes
    assert game.empty_value == 40
    assert game.full_value == 80
    assert values(game) == [40]


def test_maximisation_and_input_order_independence() -> None:
    rows = [row("a", 0, -10, operator="init"), row("b", 0, -3, ["a"])]
    forward = attribute(rows, maximising=True)
    assert forward == attribute(list(reversed(rows)), maximising=True)
    assert forward.baseline_fitness == -10
    assert forward.islands.full_value == 7


def test_missing_cycle_and_unlogged_parentage_are_excluded_and_explained() -> None:
    rows = [
        row("a", 0, 100, operator="init"),
        row("b", 0, 1, ["absent"]),
        row("c", 0, 0, ["b"]),
        row("d", 0, -1),
        row("e", 0, -2, ["f"]),
        row("f", 0, -3, ["e"]),
    ]
    result = attribute(rows)
    assert (result.coverage.evaluations, result.coverage.supported) == (6, 1)
    reasons = "missing or unlogged parent: 2, in or below a parent cycle: 2, descended from those: 1"
    assert result.warnings == [f"5 evaluations lack fully resolved ancestry and are excluded ({reasons})."]
    assert result.islands.full_value == 0
    assert result.observed_best_fitness == -3
    assert result.supported_best_fitness == 100


def test_retained_migrant_provenance_cannot_replace_pso_parentage() -> None:
    rows = [
        row("a", 0, 100, operator="init"),
        row("b", 1, 100, operator="init"),
        row("m", 1, 100, ["a"], "migration_arrival", origin_island=0, origin_individual_id="a"),
        row("p", 1, 0, operator="pso_move", origin_island=0, origin_individual_id="a"),
    ]
    result = attribute(rows)
    assert result.coverage.supported == 3
    assert result.islands.full_value == 0


def test_non_arrival_origin_fields_do_not_invent_migration_dependencies() -> None:
    rows = [
        row("a", 0, 100, operator="init"),
        row("b", 1, 100, operator="init"),
        row("p", 1, 0, ["b"], "pso_move", origin_island=0, origin_individual_id="a"),
    ]
    result = attribute(rows)
    assert result.routes.players == []
    assert result.routes.empty_value == 100


@pytest.mark.parametrize(
    "rows",
    [
        [row("a", 0, 1, operator="init")] * 2,
        [row("a", 0, 100, operator="init"), row("a", 1, 100, operator="init"), row("b", 1, 1, ["a"])],
        [row("a", 0, math.nan, operator="init")],
        [row("a", 0, 1)],
    ],
    ids=["duplicate-id", "duplicate-id-across-islands", "nan-fitness", "no-founder"],
)
def test_duplicate_identity_nonfinite_and_missing_initialisation_fail(rows: list[Row]) -> None:
    with pytest.raises(ValueError):
        attribute(rows)


def test_no_routes_and_no_improvement_are_finite() -> None:
    result = attribute([row("a", 0, 1, operator="init")])
    assert result.routes.players == []
    assert values(result.islands) == [0]


def test_complementary_islands_share_joint_gain_and_dummy_is_zero() -> None:
    game = shapley_values([0, 1, 2], [(3, 12)], seed=7)
    assert values(game) == [6, 6, 0]
    assert game.empty_value == 0
    assert game.full_value == 12
    assert sum(values(game)) == pytest.approx(game.full_value - game.empty_value)


def test_exact_game_lists_every_coalition_and_example_orders() -> None:
    game = shapley_values([0, 1], [(3, 12)])
    assert game.sampled_orders is None
    assert [(c.players, c.value) for c in game.coalitions] == [([], 0), ([0], 0), ([1], 0), ([0, 1], 12)]
    assert all(step.after >= step.before for order in game.orders for step in order)
    assert all(player.standard_error is None for player in game.players)


def test_sampling_budget_rejects_large_work_and_preserves_exact_games(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shapley_mod, "MAX_SAMPLING_WORK", 2)
    with pytest.raises(ValueError, match="work"):
        shapley_values(list(range(11)), [(1, 3), (3, 9)])
    assert values(shapley_values([0, 1], [(3, 12)])) == [6, 6]


def test_sampled_game_reproducible_efficient_and_reports_standard_error() -> None:
    game = shapley_values(list(range(11)), [(1, 3), (3, 9)], seed=23)
    assert game == shapley_values(list(range(11)), [(1, 3), (3, 9)], seed=23)
    assert game.sampled_orders == shapley_mod.SAMPLES
    assert game.coalitions == []
    assert sum(values(game)) == pytest.approx(9)
    assert game.players[0].standard_error is not None and game.players[0].standard_error > 0
    assert all(value == 0 for value in values(game)[2:])
