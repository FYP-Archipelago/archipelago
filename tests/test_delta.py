from __future__ import annotations

import numpy as np
from conftest import add_spec, apply_delta, assert_view_matches, new_view

from archipelago.config import DRIFT_TOLERANCE, WEIGHT_TOLERANCE
from archipelago.graph.delta import Sent, diff
from archipelago.graph.state import GraphState, add_edges, add_migrations, columns, kill_labels, set_coords, to_graph

SPAN = {0: 50, 1: 50}


def step(state: GraphState, sent: Sent, refit: bool):
    cols = columns(state, SPAN)
    return cols, diff(sent, cols, state.nodes, state.edges, state.migrations, refit)


def test_viewer_applying_deltas_ends_equal_to_exact_columns() -> None:
    rng = np.random.default_rng(5)
    state = GraphState(maximising=False, weighted=True)
    sent, view = Sent(), new_view()

    def check(refit: bool) -> None:
        cols, delta = step(state, sent, refit)
        apply_delta(view, delta)
        assert_view_matches(view, to_graph(cols))

    # Step 1: two islands, three labels, an edge and a migration; first layout.
    add_spec(state, [(0, 0, 0, 5.0, "a"), (0, 1, 1, 4.0, "b"), (1, 0, 0, 3.0, "c"), (1, 2, 1, 6.0, "d"), (0, 0, 2, 2.0, "e")])
    add_edges(state, [(0, 1), (1, 4), (2, 3)])
    add_migrations(state, [(0, 2)])
    state.weight[: state.nodes] = [1.0, 2.0, 3.0, 4.0]
    check(refit=True)

    # Step 2: more rows, a repeated edge, a new node, weights that move by less than WEIGHT_TOLERANCE.
    add_spec(state, [(0, 1, 3, 1.0, "f"), (1, 3, 2, 0.5, "g"), (0, 0, 4, 7.0, "h")])
    add_edges(state, [(0, 1), (3, 5), (5, 6)])
    add_migrations(state, [(0, 2), (6, 5)])
    state.weight[: state.nodes] = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * (1 - WEIGHT_TOLERANCE / 2)
    cols, delta = step(state, sent, False)
    # Only the node created in this step carries a weight; the older ones moved too little to be re-sent.
    assert delta.weights is not None and delta.weights.id == []
    apply_delta(view, delta)
    assert_view_matches(view, to_graph(cols))

    # Step 3: a weight moves by more than the tolerance, then a label is killed (node 1).
    state.weight[2] *= 1 + 2 * WEIGHT_TOLERANCE
    cols, delta = step(state, sent, False)
    assert delta.weights is not None and 2 in delta.weights.id
    apply_delta(view, delta)
    assert_view_matches(view, to_graph(cols))
    kill_labels(state, [1])
    cols, delta = step(state, sent, False)
    assert delta.removed_nodes == [1]
    apply_delta(view, delta)
    assert_view_matches(view, to_graph(cols))

    # Step 4: nothing changed, so nothing is sent.
    cols, delta = step(state, sent, False)
    assert delta.new_nodes.id == [] and delta.updated_nodes.id == [] and delta.percentiles.id == []
    assert delta.removed_nodes == [] and delta.new_edges.id == [] and delta.edge_updates.id == []
    assert delta.new_migrations.id == [] and delta.migration_updates.id == []

    # Step 5: a layout refit moves every position; a new row in a node sent earlier changes its percentile.
    sent_ids = sorted(view.nodes)
    add_spec(state, [(1, 0, 3, 0.1, "i")], coords=np.zeros((1, 3)))
    set_coords(state, rng.uniform(-1, 1, size=(state.rows, 3)))
    cols, delta = step(state, sent, True)
    assert delta.positions is not None and delta.positions.id == sent_ids
    apply_delta(view, delta)
    assert_view_matches(view, to_graph(cols))

    # Step 6: many fresh rows shift ranks; a percentile moved by more than DRIFT_TOLERANCE is re-sent.
    before = {n: view.nodes[n]["fitness_percentile"] for n in view.nodes}
    add_spec(state, [(0, 0, 5 + k, 10.0 + k, f"z{k}") for k in range(20)])
    cols, delta = step(state, sent, False)
    assert any(abs(p - before[n]) > DRIFT_TOLERANCE for n, p in zip(cols["id"].tolist(), cols["fitness_percentile"].tolist()) if n in before)
    assert delta.percentiles.id
    apply_delta(view, delta)
    assert_view_matches(view, to_graph(cols))
