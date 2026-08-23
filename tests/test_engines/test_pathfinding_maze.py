"""Light tests for pathfinding_maze (README Section 6, #12): pure
param-validation, boundary, and search-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.pathfinding_maze import (
    COLS,
    GOAL,
    ROWS,
    START,
    _a_star_steps,
    _ant_colony_steps,
    _passable_neighbors,
    build_maze,
    build_maze_solution,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_maze_source():
    with pytest.raises(ValueError):
        validate_params("not_a_real_source", algorithm="a_star", seed=0)


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        validate_params("random_dfs_maze", algorithm="not_a_real_algorithm", seed=0)


def test_rejects_seed_below_range():
    with pytest.raises(ValueError):
        validate_params("random_dfs_maze", algorithm="a_star", seed=-1)


def test_rejects_seed_above_range():
    with pytest.raises(ValueError):
        validate_params("random_dfs_maze", algorithm="a_star", seed=2**31)


# ---- Maze generation correctness ----


@pytest.mark.parametrize("maze_source", ["random_dfs_maze", "random_prim_maze"])
def test_maze_is_fully_connected(maze_source):
    # Both generators build a spanning tree: every one of ROWS*COLS cells
    # must be reachable from START, and there are exactly ROWS*COLS - 1
    # passages (a tree has n-1 edges, no cycles).
    passages = build_maze(maze_source, ROWS, COLS, seed=42)
    assert len(passages) == ROWS * COLS - 1

    seen = {START}
    frontier = [START]
    while frontier:
        cell = frontier.pop()
        for nxt in _passable_neighbors(cell, ROWS, COLS, passages):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert len(seen) == ROWS * COLS


def test_maze_generation_is_deterministic_given_a_seed():
    a = build_maze("random_dfs_maze", ROWS, COLS, seed=7)
    b = build_maze("random_dfs_maze", ROWS, COLS, seed=7)
    assert a == b


def test_different_seeds_produce_different_mazes():
    a = build_maze("random_prim_maze", ROWS, COLS, seed=1)
    b = build_maze("random_prim_maze", ROWS, COLS, seed=2)
    assert a != b


# ---- A* correctness ----


def test_a_star_finds_a_valid_connected_path():
    passages = build_maze("random_dfs_maze", ROWS, COLS, seed=3)
    _visited_order, path = _a_star_steps(ROWS, COLS, passages, START, GOAL)
    assert path[0] == START
    assert path[-1] == GOAL
    for a, b in zip(path, path[1:]):
        assert frozenset({a, b}) in passages


def test_a_star_visits_every_cell_it_needs_before_finding_goal():
    passages = build_maze("random_dfs_maze", ROWS, COLS, seed=3)
    visited_order, path = _a_star_steps(ROWS, COLS, passages, START, GOAL)
    assert GOAL in visited_order
    assert set(path) <= set(visited_order)


def test_a_star_path_length_matches_unique_tree_path():
    # A spanning-tree maze has exactly one simple path between any two
    # cells, so a plain BFS over the same passages must agree with A* on
    # path length.
    from collections import deque

    passages = build_maze("random_prim_maze", ROWS, COLS, seed=11)
    _visited_order, a_star_path = _a_star_steps(ROWS, COLS, passages, START, GOAL)

    dist = {START: 0}
    queue = deque([START])
    while queue:
        cell = queue.popleft()
        for nxt in _passable_neighbors(cell, ROWS, COLS, passages):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)

    assert len(a_star_path) - 1 == dist[GOAL]


# ---- Ant colony correctness ----


def test_ant_colony_best_path_is_valid_and_connected():
    passages = build_maze("random_dfs_maze", ROWS, COLS, seed=5)
    _path_history, best_path = _ant_colony_steps(ROWS, COLS, passages, START, GOAL, seed=5)
    assert best_path is not None, "expected at least one of 30 ants to reach the goal"
    assert best_path[0] == START
    assert best_path[-1] == GOAL
    for a, b in zip(best_path, best_path[1:]):
        assert frozenset({a, b}) in passages


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ant_colony_always_finds_the_goal_across_seeds(seed):
    # DFS-with-backtrack over a fully connected maze is guaranteed to
    # reach the goal eventually — not a probabilistic outcome.
    passages = build_maze("random_dfs_maze", ROWS, COLS, seed=seed)
    _path_history, best_path = _ant_colony_steps(ROWS, COLS, passages, START, GOAL, seed=seed)
    assert best_path is not None


def test_ant_colony_best_path_length_matches_a_star():
    # The maze is a spanning tree, so there's only one simple path
    # start->goal — any ant that finds it must find the same one A* does.
    passages = build_maze("random_dfs_maze", ROWS, COLS, seed=8)
    _visited_order, a_star_path = _a_star_steps(ROWS, COLS, passages, START, GOAL)
    _path_history, ant_best_path = _ant_colony_steps(ROWS, COLS, passages, START, GOAL, seed=8)
    assert ant_best_path == a_star_path


def test_ant_colony_path_history_has_one_entry_per_ant():
    passages = build_maze("random_prim_maze", ROWS, COLS, seed=9)
    path_history, _best_path = _ant_colony_steps(ROWS, COLS, passages, START, GOAL, seed=9, num_ants=10)
    assert len(path_history) == 10


# ---- build_maze_solution() ----


def test_build_maze_solution_a_star_shape():
    data = build_maze_solution("random_dfs_maze", "a_star", seed=1)
    assert data["kind"] == "single_path"
    assert data["path"][0] == data["start"] == START
    assert data["path"][-1] == data["goal"] == GOAL


def test_build_maze_solution_ant_colony_shape():
    data = build_maze_solution("random_prim_maze", "ant_colony", seed=1)
    assert data["kind"] == "multi_path"
    assert len(data["path_history"]) > 0


def test_build_maze_solution_is_deterministic():
    a = build_maze_solution("random_dfs_maze", "a_star", seed=2)
    b = build_maze_solution("random_dfs_maze", "a_star", seed=2)
    assert a == b
