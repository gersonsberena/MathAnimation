"""pathfinding_maze engine (README Section 6, engine 12): A* solving a
maze live, ant-colony optimization. Mechanic: search algorithm exploring
a grid, showing frontier expansion.

Maze size is fixed internally (13x13 cells) — README's params for this
engine (`maze_source, algorithm, seed`) have no size field. Mazes are
generated as a set of open "passages" between adjacent cells (everything
not a passage is a wall) via two standard, low-risk-to-get-right
algorithms: randomized-DFS backtracker and randomized Prim's.

`build_maze_solution()` is pure — no Scene dependency.
"""

import heapq

import numpy as np
from manim import BLUE, GRAY, WHITE, YELLOW, Rectangle, Square, ValueTracker, VGroup, linear

from engines.base import ReelScene

MAZE_SOURCES = {"random_dfs_maze", "random_prim_maze"}
ALGORITHMS = {"a_star", "ant_colony"}
SEED_RANGE = (0, 2**31 - 1)

ROWS, COLS = 13, 13
START, GOAL = (0, 0), (ROWS - 1, COLS - 1)
_NUM_ANTS = 30


def validate_params(maze_source, algorithm, seed) -> None:
    if maze_source not in MAZE_SOURCES:
        raise ValueError(f"unknown maze_source {maze_source!r}, must be one of {sorted(MAZE_SOURCES)}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}, must be one of {sorted(ALGORITHMS)}")

    lo, hi = SEED_RANGE
    if not (lo <= seed <= hi):
        raise ValueError(f"seed={seed} out of range [{lo}, {hi}]")


def _grid_neighbors(cell, rows, cols):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols:
            yield nxt


def _generate_maze_dfs(rows, cols, seed):
    rng = np.random.default_rng(seed)
    visited = {(0, 0)}
    passages = set()
    stack = [(0, 0)]

    while stack:
        cell = stack[-1]
        unvisited = [n for n in _grid_neighbors(cell, rows, cols) if n not in visited]
        if not unvisited:
            stack.pop()
            continue
        nxt = unvisited[int(rng.integers(0, len(unvisited)))]
        passages.add(frozenset({cell, nxt}))
        visited.add(nxt)
        stack.append(nxt)

    return passages


def _generate_maze_prim(rows, cols, seed):
    rng = np.random.default_rng(seed)
    visited = {(0, 0)}
    passages = set()

    def frontier_edges(cell):
        return [(cell, n) for n in _grid_neighbors(cell, rows, cols) if n not in visited]

    frontier = frontier_edges((0, 0))
    while frontier:
        idx = int(rng.integers(0, len(frontier)))
        cell_in, cell_out = frontier.pop(idx)
        if cell_out in visited:
            continue
        passages.add(frozenset({cell_in, cell_out}))
        visited.add(cell_out)
        frontier.extend(frontier_edges(cell_out))

    return passages


def build_maze(maze_source, rows, cols, seed):
    if maze_source == "random_dfs_maze":
        return _generate_maze_dfs(rows, cols, seed)
    if maze_source == "random_prim_maze":
        return _generate_maze_prim(rows, cols, seed)
    raise AssertionError(f"unhandled maze_source {maze_source!r}")  # validate_params already checked


def _passable_neighbors(cell, rows, cols, passages):
    for n in _grid_neighbors(cell, rows, cols):
        if frozenset({cell, n}) in passages:
            yield n


def _reconstruct(prev, start, goal):
    if goal != start and goal not in prev:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path


def _a_star_steps(rows, cols, passages, start, goal):
    def heuristic(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    g_score = {start: 0}
    prev = {}
    visited = set()
    visited_order = []
    pq = [(heuristic(start), start)]

    while pq:
        _f, cell = heapq.heappop(pq)
        if cell in visited:
            continue
        visited.add(cell)
        visited_order.append(cell)
        if cell == goal:
            break
        for nxt in _passable_neighbors(cell, rows, cols, passages):
            tentative = g_score[cell] + 1
            if nxt not in g_score or tentative < g_score[nxt]:
                g_score[nxt] = tentative
                prev[nxt] = cell
                heapq.heappush(pq, (tentative + heuristic(nxt), nxt))

    return visited_order, _reconstruct(prev, start, goal)


def _ant_colony_steps(rows, cols, passages, start, goal, seed, num_ants=_NUM_ANTS):
    # Both maze generators produce a perfect-spanning-tree maze (exactly
    # one simple path between any two cells), so a plain no-revisit
    # random walk dead-ends permanently far too often to be a workable
    # search — real ants backtrack out of dead ends and keep exploring
    # rather than dying, so each ant here does a pheromone/heuristic
    # biased DFS-with-backtrack, which is guaranteed to reach the goal
    # (the maze is fully connected — see test_maze_is_fully_connected)
    # while still producing a genuine, seed-dependent search trail.
    # Because the maze is a tree, every ant that finds the goal finds the
    # *same* underlying path (there's only one) — what varies, and what
    # pheromone rewards, is how many steps of wandering it took to find
    # it, i.e. `walk_length`.
    rng = np.random.default_rng(seed)
    pheromone = {edge: 1.0 for edge in passages}

    def heuristic(cell):
        return 1.0 / (1 + abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))

    best_path, best_walk_length = None, float("inf")
    path_history = []  # full walk incl. backtrack steps, for trail rendering

    for _ant in range(num_ants):
        stack = [start]
        seen = {start}
        walk = [start]

        while stack and stack[-1] != goal:
            cell = stack[-1]
            options = [n for n in _passable_neighbors(cell, rows, cols, passages) if n not in seen]
            if not options:
                stack.pop()
                if stack:
                    walk.append(stack[-1])
                continue
            weights = np.array([pheromone[frozenset({cell, n})] * heuristic(n) ** 2 for n in options])
            probs = weights / weights.sum()
            nxt = options[rng.choice(len(options), p=probs)]
            seen.add(nxt)
            stack.append(nxt)
            walk.append(nxt)

        if stack:  # reached goal — stack is the unique tree path start -> goal
            path = list(stack)
            walk_length = len(walk) - 1
            deposit = 1.0 / walk_length
            for i in range(len(path) - 1):
                edge = frozenset({path[i], path[i + 1]})
                pheromone[edge] = pheromone.get(edge, 1.0) + deposit
            if walk_length < best_walk_length:
                best_walk_length, best_path = walk_length, path

        path_history.append(walk)

    return path_history, best_path


def build_maze_solution(maze_source, algorithm, seed):
    """Returns a dict discriminated by `kind`:
      - a_star: `{"kind": "single_path", "passages", "visited_order", "path"}`
      - ant_colony: `{"kind": "multi_path", "passages", "path_history", "best_path"}`
    Both also include `rows`, `cols`, `start`, `goal`. Pure — no Scene
    dependency.
    """
    validate_params(maze_source, algorithm, seed)
    passages = build_maze(maze_source, ROWS, COLS, seed)

    if algorithm == "a_star":
        visited_order, path = _a_star_steps(ROWS, COLS, passages, START, GOAL)
        return {
            "kind": "single_path",
            "passages": passages,
            "visited_order": visited_order,
            "path": path,
            "rows": ROWS,
            "cols": COLS,
            "start": START,
            "goal": GOAL,
        }

    if algorithm == "ant_colony":
        path_history, best_path = _ant_colony_steps(ROWS, COLS, passages, START, GOAL, seed)
        return {
            "kind": "multi_path",
            "passages": passages,
            "path_history": path_history,
            "best_path": best_path,
            "rows": ROWS,
            "cols": COLS,
            "start": START,
            "goal": GOAL,
        }

    raise AssertionError(f"unhandled algorithm {algorithm!r}")  # validate_params already checked


class PathfindingMazeReel(ReelScene):
    """`params`: maze_source, algorithm, seed (README Section 6, #12)."""

    maze_source = "random_dfs_maze"
    algorithm = "a_star"
    seed = 0

    def construct(self):
        self.set_title("Finding The Way Out")

        data = build_maze_solution(self.maze_source, self.algorithm, self.seed)
        rows, cols = data["rows"], data["cols"]
        zone = self.zones.content_zone

        cell_size = min(zone.width * 0.85 / cols, zone.height * 0.65 / rows)
        origin_x = zone.get_center()[0] - cols * cell_size / 2
        origin_y = zone.get_center()[1] + rows * cell_size / 2  # row 0 at top

        def cell_center(cell):
            r, c = cell
            return np.array([origin_x + (c + 0.5) * cell_size, origin_y - (r + 0.5) * cell_size, 0.0])

        # Walls: draw every cell as a small square, connect passage-linked
        # cells with a bridging rectangle; unconnected neighbors keep a
        # visible gap between them, reading as a wall.
        cell_squares = VGroup(
            *[
                Square(side_length=cell_size * 0.8, stroke_color=GRAY, stroke_width=1, fill_opacity=0).move_to(
                    cell_center((r, c))
                )
                for r in range(rows)
                for c in range(cols)
            ]
        )
        bridges = VGroup()
        for edge in data["passages"]:
            a, b = tuple(edge)
            mid = (cell_center(a) + cell_center(b)) / 2
            width = cell_size * 0.8 if a[0] == b[0] else cell_size * 0.3
            height = cell_size * 0.3 if a[0] == b[0] else cell_size * 0.8
            bridges.add(Rectangle(width=width, height=height, stroke_width=0, fill_color=GRAY, fill_opacity=0.3).move_to(mid))

        self.add(bridges, cell_squares)

        frame_tracker = ValueTracker(0)

        if data["kind"] == "single_path":
            visited_order, path = data["visited_order"], set(data["path"])
            num_steps = len(visited_order)

            def update_cells(_mob):
                idx = max(0, min(int(round(frame_tracker.get_value())), num_steps))
                visited_now = set(visited_order[:idx])
                for i, cell in enumerate([(r, c) for r in range(rows) for c in range(cols)]):
                    square = cell_squares[i]
                    if cell in visited_now and cell in path:
                        square.set_fill(YELLOW, opacity=0.9)
                    elif cell in visited_now:
                        square.set_fill(BLUE, opacity=0.5)
                    else:
                        square.set_fill(opacity=0)

            cell_squares.add_updater(update_cells)
            self.play(frame_tracker.animate.set_value(num_steps), run_time=5.0, rate_func=linear)
        else:
            path_history, best_path = data["path_history"], data["best_path"] or []
            num_ants = len(path_history)
            trail_lines = VGroup(*[self._path_line(path, cell_center) for path in path_history])
            for line in trail_lines:
                line.set_stroke(opacity=0)

            def update_trails(_mob):
                idx = max(0, min(int(round(frame_tracker.get_value())), num_ants))
                for i, line in enumerate(trail_lines):
                    line.set_stroke(WHITE, width=1, opacity=0.15 if i < idx else 0)

            trail_lines.add_updater(update_trails)
            self.add(trail_lines)
            self.play(frame_tracker.animate.set_value(num_ants), run_time=4.0, rate_func=linear)
            trail_lines.clear_updaters()

            if best_path:
                best_cells = set(best_path)
                all_cells = [(r, c) for r in range(rows) for c in range(cols)]
                for square, cell in zip(cell_squares, all_cells):
                    if cell in best_cells:
                        square.set_fill(YELLOW, opacity=0.3)
                best_line = self._path_line(best_path, cell_center)
                best_line.set_stroke(YELLOW, width=4, opacity=1.0)
                self.add(best_line)
                self.wait(0.5)

        self.set_caption("The path was always there")

    @staticmethod
    def _path_line(path, cell_center):
        from manim import VMobject

        line = VMobject()
        if len(path) >= 2:
            line.set_points_as_corners([cell_center(cell) for cell in path])
        return line
