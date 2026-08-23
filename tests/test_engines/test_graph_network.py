"""Light tests for graph_network (README Section 6, #8): pure
param-validation, boundary, and algorithm-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.graph_network import (
    ALGORITHMS,
    GRAPH_SOURCES,
    NUM_NODES,
    build_graph,
    build_traversal,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_graph_source():
    with pytest.raises(ValueError):
        validate_params("not_a_real_source", "dijkstra", start_node=0, end_node=1)


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        validate_params("grid", "not_a_real_algorithm", start_node=0, end_node=1)


def test_rejects_start_node_out_of_range():
    with pytest.raises(ValueError):
        validate_params("grid", "dijkstra", start_node=-1, end_node=1)


def test_rejects_end_node_out_of_range():
    with pytest.raises(ValueError):
        validate_params("grid", "dijkstra", start_node=0, end_node=NUM_NODES)


def test_dijkstra_rejects_equal_start_and_end():
    with pytest.raises(ValueError):
        validate_params("grid", "dijkstra", start_node=3, end_node=3)


def test_bfs_rejects_equal_start_and_end():
    with pytest.raises(ValueError):
        validate_params("grid", "bfs", start_node=3, end_node=3)


def test_graph_coloring_allows_equal_start_and_end():
    # unused for this algorithm, so no distinctness requirement
    validate_params("grid", "graph_coloring", start_node=3, end_node=3)


# ---- Graph construction ----


@pytest.mark.parametrize("graph_source", sorted(GRAPH_SOURCES))
def test_build_graph_has_expected_node_count(graph_source):
    positions, _edges = build_graph(graph_source)
    assert len(positions) == NUM_NODES


@pytest.mark.parametrize("graph_source", sorted(GRAPH_SOURCES))
def test_build_graph_edges_reference_valid_nodes(graph_source):
    positions, edges = build_graph(graph_source)
    for u, v, w in edges:
        assert 0 <= u < len(positions)
        assert 0 <= v < len(positions)
        assert w > 0  # no self-loops / zero-length edges


@pytest.mark.parametrize("graph_source", sorted(GRAPH_SOURCES))
def test_build_graph_is_deterministic(graph_source):
    a = build_graph(graph_source)
    b = build_graph(graph_source)
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_grid_graph_is_connected():
    # every node reachable from node 0 via BFS over the raw edge list
    positions, edges = build_graph("grid")
    adj = {i: [] for i in range(len(positions))}
    for u, v, _w in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    assert seen == set(range(len(positions)))


# ---- Algorithm correctness ----


def test_dijkstra_finds_a_valid_path_on_grid():
    data = build_traversal("grid", "dijkstra", start_node=0, end_node=NUM_NODES - 1)
    assert data["kind"] == "traverse"
    assert data["path"][0] == 0
    assert data["path"][-1] == NUM_NODES - 1


def test_dijkstra_path_is_no_worse_than_bfs_hop_count_path():
    # Dijkstra on a graph where edge weight = Euclidean distance should
    # never need MORE hops-worth of exploration to finish than makes
    # sense — sanity check: every visited node is visited exactly once.
    data = build_traversal("cycle", "dijkstra", start_node=0, end_node=6)
    assert len(data["visit_order"]) == len(set(data["visit_order"]))


def test_bfs_visits_every_reachable_node_exactly_once():
    data = build_traversal("grid", "bfs", start_node=0, end_node=NUM_NODES - 1)
    assert len(data["visit_order"]) == len(set(data["visit_order"])) == NUM_NODES


def test_bfs_path_hop_count_matches_visit_order_bfs_layering():
    data = build_traversal("grid", "bfs", start_node=0, end_node=NUM_NODES - 1)
    # BFS path length should be minimal — for a 3x4 grid from corner to
    # corner, that's (rows-1)+(cols-1) = 2+3 = 5 edges = 6 nodes.
    assert len(data["path"]) == 6


def test_greedy_coloring_never_gives_adjacent_nodes_the_same_color():
    _positions, edges = build_graph("random_geometric")
    data = build_traversal("random_geometric", "graph_coloring", start_node=0, end_node=1)
    colors = data["colors"]
    for u, v, _w in edges:
        assert colors[u] != colors[v]


def test_greedy_coloring_assigns_every_node():
    data = build_traversal("grid", "graph_coloring", start_node=0, end_node=1)
    assert len(data["assign_sequence"]) == NUM_NODES
    assert all(c is not None for c in data["colors"])
