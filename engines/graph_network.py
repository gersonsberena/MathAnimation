"""graph_network engine (README Section 6, engine 8): Dijkstra
shortest-path race, graph coloring, six-degrees (BFS) visualization.
Mechanic: nodes + edges, algorithm traverses/colors step by step.

Node count is fixed internally (12) — README's params list for this
engine (`graph_source, algorithm, start_node, end_node`) has no
node-count param. `graph_source` picks a deterministic graph layout;
`random_geometric` uses the same independent-irrational Weyl-sequence
trick as grid_cell_coloring's voronoi seeds, since there's no `seed`
param here either.

`build_traversal()` is pure — no Scene dependency.
"""

import heapq
from collections import deque

import numpy as np
from manim import BLUE, GRAY, GREEN, PINK, PURPLE, YELLOW, Dot, Line, ValueTracker, VGroup, linear

from engines.base import ReelScene

GRAPH_SOURCES = {"grid", "cycle", "random_geometric"}
ALGORITHMS = {"dijkstra", "bfs", "graph_coloring"}
NUM_NODES = 12
_PALETTE = [BLUE, PINK, YELLOW, GREEN, PURPLE]


def validate_params(graph_source, algorithm, start_node, end_node) -> None:
    if graph_source not in GRAPH_SOURCES:
        raise ValueError(f"unknown graph_source {graph_source!r}, must be one of {sorted(GRAPH_SOURCES)}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}, must be one of {sorted(ALGORITHMS)}")

    if not isinstance(start_node, int) or not isinstance(end_node, int):
        raise ValueError("start_node and end_node must be integers")
    if not (0 <= start_node < NUM_NODES) or not (0 <= end_node < NUM_NODES):
        raise ValueError(f"start_node/end_node must be in [0, {NUM_NODES - 1}]")

    if algorithm in ("dijkstra", "bfs") and start_node == end_node:
        raise ValueError("start_node and end_node must differ for dijkstra/bfs")
    # graph_coloring: start_node/end_node validated as in-range ints above
    # but unused — coloring has no source/target.


def _grid_positions_and_edges(rows=3, cols=4):
    def idx(r, c):
        return r * cols + c

    positions = [(c, -r) for r in range(rows) for c in range(cols)]
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return positions, edges


def _cycle_positions_and_edges(num_nodes=NUM_NODES):
    positions = [
        (np.cos(2 * np.pi * i / num_nodes), np.sin(2 * np.pi * i / num_nodes)) for i in range(num_nodes)
    ]
    edge_set = set()
    for i in range(num_nodes):
        edge_set.add(tuple(sorted((i, (i + 1) % num_nodes))))
        edge_set.add(tuple(sorted((i, (i + 3) % num_nodes))))  # chords, so shortest paths aren't trivial
    return positions, sorted(edge_set)


def _random_geometric_positions_and_edges(num_nodes=NUM_NODES, radius=0.85):
    # Two independent irrationals (see grid_cell_coloring's voronoi note
    # for why phi/phi**2 would NOT work here) — deterministic without a
    # seed param.
    alpha_x, alpha_y = 5**0.5, 3**0.5
    positions = [(((i * alpha_x) % 1.0) * 2 - 1, ((i * alpha_y) % 1.0) * 2 - 1) for i in range(num_nodes)]
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if (dx * dx + dy * dy) ** 0.5 < radius:
                edges.append((i, j))
    return positions, edges


def build_graph(graph_source):
    """Returns `(positions, weighted_edges)`: `positions` is a list of
    `(x, y)` in an arbitrary world coordinate system; `weighted_edges` is
    a list of `(u, v, weight)` with weight = Euclidean distance.
    """
    if graph_source == "grid":
        positions, edges = _grid_positions_and_edges()
    elif graph_source == "cycle":
        positions, edges = _cycle_positions_and_edges()
    elif graph_source == "random_geometric":
        positions, edges = _random_geometric_positions_and_edges()
    else:
        raise AssertionError(f"unhandled graph_source {graph_source!r}")  # validate_params already checked

    weighted_edges = [
        (u, v, ((positions[u][0] - positions[v][0]) ** 2 + (positions[u][1] - positions[v][1]) ** 2) ** 0.5)
        for u, v in edges
    ]
    return positions, weighted_edges


def _adjacency(num_nodes, weighted_edges):
    adj = [[] for _ in range(num_nodes)]
    for u, v, w in weighted_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _reconstruct_path(prev, start, end):
    if prev[end] is None and end != start:
        return []
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def _dijkstra_steps(num_nodes, weighted_edges, start, end):
    adj = _adjacency(num_nodes, weighted_edges)
    dist = [float("inf")] * num_nodes
    dist[start] = 0
    visited = [False] * num_nodes
    prev = [None] * num_nodes
    pq = [(0, start)]
    visit_order = []

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        visit_order.append(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path = _reconstruct_path(prev, start, end) if dist[end] < float("inf") else []
    return visit_order, path


def _bfs_steps(num_nodes, weighted_edges, start, end):
    adj = _adjacency(num_nodes, weighted_edges)
    visited = [False] * num_nodes
    prev = [None] * num_nodes
    visited[start] = True
    queue = deque([start])
    visit_order = [start]

    while queue:
        u = queue.popleft()
        for v, _w in adj[u]:
            if not visited[v]:
                visited[v] = True
                prev[v] = u
                visit_order.append(v)
                queue.append(v)

    path = _reconstruct_path(prev, start, end) if visited[end] else []
    return visit_order, path


def _greedy_coloring_steps(num_nodes, weighted_edges):
    adj = _adjacency(num_nodes, weighted_edges)
    colors = [None] * num_nodes
    assign_sequence = []

    for u in range(num_nodes):  # fixed deterministic processing order
        used = {colors[v] for v, _w in adj[u] if colors[v] is not None}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
        assign_sequence.append((u, c))

    return assign_sequence, colors


def build_traversal(graph_source, algorithm, start_node, end_node):
    """Returns a dict discriminated by `kind`:
      - dijkstra/bfs: `{"kind": "traverse", "positions", "edges",
        "visit_order", "path"}`
      - graph_coloring: `{"kind": "coloring", "positions", "edges",
        "assign_sequence", "colors"}`
    Pure — no Scene dependency.
    """
    validate_params(graph_source, algorithm, start_node, end_node)
    positions, weighted_edges = build_graph(graph_source)
    num_nodes = len(positions)

    if algorithm == "dijkstra":
        visit_order, path = _dijkstra_steps(num_nodes, weighted_edges, start_node, end_node)
        return {"kind": "traverse", "positions": positions, "edges": weighted_edges, "visit_order": visit_order, "path": path}

    if algorithm == "bfs":
        visit_order, path = _bfs_steps(num_nodes, weighted_edges, start_node, end_node)
        return {"kind": "traverse", "positions": positions, "edges": weighted_edges, "visit_order": visit_order, "path": path}

    if algorithm == "graph_coloring":
        assign_sequence, colors = _greedy_coloring_steps(num_nodes, weighted_edges)
        return {"kind": "coloring", "positions": positions, "edges": weighted_edges, "assign_sequence": assign_sequence, "colors": colors}

    raise AssertionError(f"unhandled algorithm {algorithm!r}")  # validate_params already checked


class GraphNetworkReel(ReelScene):
    """`params`: graph_source, algorithm, start_node, end_node (README
    Section 6, #8).
    """

    graph_source = "grid"
    algorithm = "dijkstra"
    start_node = 0
    end_node = NUM_NODES - 1
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "Shortest Path, Step by Step")

        data = build_traversal(self.graph_source, self.algorithm, self.start_node, self.end_node)
        positions, edges = data["positions"], data["edges"]
        zone = self.zones.content_zone

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        scale = min(zone.width * 0.8 / span_x, zone.height * 0.65 / span_y)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        zone_center = zone.get_center()

        def to_scene(point):
            return np.array([(point[0] - cx) * scale + zone_center[0], (point[1] - cy) * scale + zone_center[1], 0.0])

        edge_lines = VGroup(
            *[
                Line(to_scene(positions[u]), to_scene(positions[v]), stroke_color=GRAY, stroke_width=2, stroke_opacity=0.5)
                for u, v, _w in edges
            ]
        )
        node_dots = [Dot(to_scene(p), radius=0.12, color=GRAY) for p in positions]
        node_group = VGroup(*node_dots)
        self.add(edge_lines, node_group)

        frame_tracker = ValueTracker(0)

        if data["kind"] == "traverse":
            visit_order, path = data["visit_order"], set(data["path"])
            num_steps = len(visit_order)

            def update_nodes(_mob):
                idx = max(0, min(int(round(frame_tracker.get_value())), num_steps))
                visited_now = set(visit_order[:idx])
                for i, dot in enumerate(node_dots):
                    if i in visited_now and i in path:
                        dot.set_color(YELLOW)
                    elif i in visited_now:
                        dot.set_color(BLUE)
                    else:
                        dot.set_color(GRAY)

            node_group.add_updater(update_nodes)
        else:
            assign_sequence = data["assign_sequence"]
            num_steps = len(assign_sequence)

            def update_nodes(_mob):
                idx = max(0, min(int(round(frame_tracker.get_value())), num_steps))
                for dot in node_dots:
                    dot.set_color(GRAY)
                for u, c in assign_sequence[:idx]:
                    node_dots[u].set_color(_PALETTE[c % len(_PALETTE)])

            node_group.add_updater(update_nodes)

        self.play(frame_tracker.animate.set_value(num_steps), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(num_steps)
        self.wait(0.3)
        node_group.clear_updaters()
        update_nodes(node_group)  # apply the final state once more after clearing the updater

        self.set_caption(self.caption_text or "One step at a time")
