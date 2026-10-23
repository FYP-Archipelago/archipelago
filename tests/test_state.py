from __future__ import annotations

import numpy as np
import pytest
from conftest import SpecRow, add_spec

from archipelago.graph.state import GraphState, add_edges, add_migrations, add_rows, columns, kill_labels, to_graph

# island, label, eval_index, fitness, genome hash. Nodes by first appearance of (island, label):
# (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,2) -> 3.
ROWS: list[SpecRow] = [
    (0, 0, 0, 4.0, "a"),
    (0, 1, 1, 2.0, "b"),
    (1, 0, 0, 6.0, "c"),
    (0, 0, 2, 2.0, "a"),
    (0, 0, 3, 8.0, "d"),
    (1, 2, 1, 1.0, "e"),
    (0, 1, 4, 2.0, "b"),
]
SPAN = {0: 10, 1: 4}


def build(maximising: bool, rows: list[SpecRow] = ROWS) -> GraphState:
    state = GraphState(maximising=maximising, weighted=False)
    add_spec(state, rows)
    return state


def test_node_columns_by_hand() -> None:
    cols = columns(build(False), SPAN)
    # Rows per node: node 0 = r0, r3, r4 (f 4, 2, 8; eval 0, 2, 3); node 1 = r1, r6 (f 2, 2; eval 1, 4);
    # node 2 = r2 (f 6; eval 0); node 3 = r5 (f 1; eval 1).
    assert cols["id"].tolist() == [0, 1, 2, 3]
    assert cols["island"].tolist() == [0, 0, 1, 1]
    assert cols["visits"].tolist() == [3, 2, 1, 1]
    # Distinct hashes: node 0 has {a, d}, node 1 only {b}.
    assert cols["members"].tolist() == [2, 1, 1, 1]
    assert cols["mean_fitness"].tolist() == pytest.approx([14 / 3, 2.0, 6.0, 1.0], abs=0, rel=1e-15)
    assert cols["best_fitness"].tolist() == [2.0, 2.0, 6.0, 1.0]
    assert cols["first_eval"].tolist() == [0, 1, 0, 1]
    assert cols["last_eval"].tolist() == [3, 4, 0, 1]
    assert cols["genome_hash"] == ["a", "b", "c", "e"]
    # All fitnesses: 4 2 6 2 8 1 2 (7 rows). Rows strictly below each node mean:
    # 14/3 -> {4, 2, 2, 1, 2} = 5; 2 -> {1} = 1; 6 -> 5; 1 -> 0.
    assert cols["fitness_percentile"].tolist() == pytest.approx([5 / 7, 1 / 7, 5 / 7, 0.0], abs=0, rel=1e-15)
    # Mean eval_index / span of the island: (5/3)/10, 2.5/10, 0/4, 1/4.
    assert cols["progress"].tolist() == pytest.approx([1 / 6, 0.25, 0.0, 0.25], abs=0, rel=1e-15)
    # Coordinates of row g are (g, 2g, -g); the node position is the mean over its rows (0, 3, 4 -> 7/3).
    assert cols["x"].tolist() == pytest.approx([7 / 3, 3.5, 2.0, 5.0], abs=0, rel=1e-15)
    assert cols["y"].tolist() == pytest.approx([14 / 3, 7.0, 4.0, 10.0], abs=0, rel=1e-15)
    assert cols["z"].tolist() == pytest.approx([-7 / 3, -3.5, -2.0, -5.0], abs=0, rel=1e-15)
    assert cols["weight"] is None
    # progress is clipped to 1 when the span is smaller than the mean eval_index (2.5 / 2).
    assert columns(build(False), {0: 2, 1: 4})["progress"].tolist()[1] == 1.0
    check_best_flags_minimising_with_ties_to_lowest_id()
    check_mean_is_clipped_into_the_node_range()


def test_maximising_flips_percentile_and_best() -> None:
    cols = columns(build(True), SPAN)
    assert cols["best_fitness"].tolist() == [8.0, 2.0, 6.0, 1.0]
    # A row is better when its fitness is higher. Rows strictly above each node mean:
    # 14/3 -> {6, 8} = 2; 2 -> {4, 6, 8} = 3; 6 -> {8} = 1; 1 -> {4, 2, 6, 2, 8, 2} = 6.
    assert cols["fitness_percentile"].tolist() == pytest.approx([2 / 7, 3 / 7, 1 / 7, 6 / 7], abs=0, rel=1e-15)
    # Overall best is the highest fitness (8, node 0); island 1's best is the 6 on node 2.
    assert cols["overall_best"].tolist() == [True, False, False, False]
    assert cols["island_best"].tolist() == [True, False, True, False]


def check_best_flags_minimising_with_ties_to_lowest_id() -> None:
    cols = columns(build(False), SPAN)
    # Island 0: nodes 0 and 1 both have best 2.0, so node 0 wins; island 1: node 3 (1.0) beats node 2 (6.0).
    assert cols["island_best"].tolist() == [True, False, False, True]
    assert cols["overall_best"].tolist() == [False, False, False, True]
    # Without the island-1 rows, nodes 0 and 1 tie for the overall best too.
    tied = columns(build(False, [r for r in ROWS if r[0] == 0]), SPAN)
    assert tied["overall_best"].tolist() == [True, False]


def check_mean_is_clipped_into_the_node_range() -> None:
    # 0.1 + 0.1 + 0.1 = 0.30000000000000004, and /3 is 0.10000000000000002 > 0.1: unclipped, the node's
    # own rows would count as better than its mean. Likewise 0.3 x 10 sums just below 3.0 (maximising).
    low = columns(build(False, [(0, 0, k, 0.1, f"h{k}") for k in range(3)] + [(0, 1, 3, 5.0, "x"), (0, 1, 4, 6.0, "y")]), {0: 5})
    assert low["mean_fitness"][0] == 0.1
    assert low["fitness_percentile"][0] == 0.0
    high = columns(build(True, [(0, 0, k, 0.3, f"h{k}") for k in range(10)] + [(0, 1, 10, 0.1, "x"), (0, 1, 11, 0.2, "y")]), {0: 12})
    assert high["mean_fitness"][0] == 0.3
    assert high["fitness_percentile"][0] == 0.0


def link_state() -> GraphState:
    state = build(False)
    # Row pairs; rows 0, 3, 4 share node 0, rows 1, 6 share node 1.
    add_edges(state, [(0, 3), (0, 1), (0, 1), (1, 4), (0, 4), (2, 5), (3, 1)])
    add_migrations(state, [(0, 2), (4, 2), (5, 0), (2, 2)])
    return state


def test_edges_drop_self_loops_and_count_repeats() -> None:
    state = link_state()
    cols = columns(state, SPAN)
    # Self loops (0,3), (0,4) are dropped. Edge ids by first occurrence:
    # 0: node0->node1 (3 times: (0,1), (0,1), (3,1)); 1: node1->node0 (once); 2: node2->node3 (once).
    assert cols["edge_id"].tolist() == [0, 1, 2]
    assert cols["edge_pair"].tolist() == [[0, 1], [1, 0], [2, 3]]
    assert cols["edge_weight"].tolist() == [3, 1, 1]
    # Migrations: 0: node0->node2 (twice); 1: node3->node0; (2,2) is a self loop.
    assert cols["migration_id"].tolist() == [0, 1]
    assert cols["migration_pair"].tolist() == [[0, 2], [3, 0]]
    assert cols["migration_count"].tolist() == [2, 1]
    assert cols["in_degree"].tolist() == [1, 1, 0, 1]
    assert cols["out_degree"].tolist() == [1, 1, 1, 0]
    assert cols["migration_endpoint"].tolist() == [True, False, True, True]
    # Labels: 0 on islands {0, 1} (nodes 0, 2); 1 on island 0 (node 1); 2 on island 1 (node 3).
    assert cols["islands_sharing"].tolist() == [2, 1, 2, 1]
    assert cols["overlap"] == pytest.approx((2 + 1 + 1) / 3)


def test_kill_labels_removes_nodes_and_their_links_but_keeps_ids() -> None:
    state = link_state()
    before = columns(state, SPAN)
    kill_labels(state, [1])
    after = columns(state, SPAN)
    # Label 1 is node 1 only. Survivors keep their ids (a gap at 1) and per-node values.
    assert after["id"].tolist() == [0, 2, 3]
    for name in ("visits", "members", "mean_fitness", "best_fitness", "fitness_percentile", "progress", "x", "genome_hash"):
        kept = [before[name][i] for i in (0, 2, 3)]
        assert list(after[name]) == list(kept), name
    # Edges 0 and 1 touched node 1: gone. Edge 2 keeps id 2. Both migrations survive with their ids.
    assert after["edge_id"].tolist() == [2]
    assert after["edge_pair"].tolist() == [[2, 3]]
    assert after["edge_weight"].tolist() == [1]
    assert after["migration_id"].tolist() == [0, 1]
    assert after["in_degree"].tolist() == [0, 0, 1]
    assert after["out_degree"].tolist() == [0, 1, 0]
    # Live labels now: 0 on islands {0, 1}, 2 on island 1.
    assert after["islands_sharing"].tolist() == [2, 2, 1]
    assert after["overlap"] == pytest.approx(1.5)
    assert after["island_best"].tolist() == [True, False, True]
    assert after["overall_best"].tolist() == [False, False, True]
    graph = to_graph(after)
    assert graph.nodes.id == [0, 2, 3]
    assert graph.edges.id == [2]
    # A new node is numbered after every earlier one, dead or alive.
    add_rows(state, ["new"], np.array([0]), np.array([9]), np.array([9]), np.array([3.0]), ["n"], np.zeros((1, 3)))
    assert columns(state, SPAN)["id"].tolist() == [0, 2, 3, 4]
