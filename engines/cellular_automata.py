"""cellular_automata engine (README Section 6, engine 7): Conway's Game of
Life, Rule 30, Langton's ant. Mechanic: grid of cells, next state computed
from neighbor rules. A cheap extension of grid_cell_coloring (#6) — same
single-ImageMobject rendering, but stepping through a precomputed sequence
of full `(grid_size, grid_size)` boolean frames via the same
ValueTracker + always_redraw pattern used by particle_physics_sim/
array_bar_race, instead of one static frame.

rule_30 doesn't naturally produce a 2D grid the way game_of_life/
langtons_ant do (an elementary CA evolves one 1D row at a time) — it's
rendered here as the classic cumulative image: each generation's row is
written into the next row of an otherwise-blank canvas, so the frame
sequence is still "the whole canvas at step t," keeping one rendering
path for all three rulesets.

There's no `seed` in this engine's params (`ruleset, grid_size,
initial_state, generations`), so initial states are fixed named patterns,
not randomly generated — same reasoning as grid_cell_coloring's
seed-free voronoi.
"""

import numpy as np
from manim import ImageMobject, ValueTracker, always_redraw, linear

from engines.base import ReelScene

RULESETS = {"game_of_life", "rule_30", "langtons_ant"}
GRID_SIZE_RANGE = (20, 150)
GENERATIONS_RANGE = (10, 200)

GAME_OF_LIFE_PATTERNS = {"glider", "blinker", "r_pentomino", "acorn"}
RULE_30_INITIAL_STATES = {"single_cell"}

_PATTERN_CELLS = {
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "blinker": [(0, 0), (0, 1), (0, 2)],
    "r_pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
}


def validate_params(ruleset, grid_size, initial_state, generations) -> None:
    if ruleset not in RULESETS:
        raise ValueError(f"unknown ruleset {ruleset!r}, must be one of {sorted(RULESETS)}")

    lo, hi = GRID_SIZE_RANGE
    if not (lo <= grid_size <= hi):
        raise ValueError(f"grid_size={grid_size} out of range [{lo}, {hi}]")

    lo, hi = GENERATIONS_RANGE
    if not (lo <= generations <= hi):
        raise ValueError(f"generations={generations} out of range [{lo}, {hi}]")

    if not isinstance(initial_state, str):
        raise ValueError(f"initial_state must be a string, got {initial_state!r}")

    if ruleset == "game_of_life" and initial_state not in GAME_OF_LIFE_PATTERNS:
        raise ValueError(
            f"initial_state {initial_state!r} not one of {sorted(GAME_OF_LIFE_PATTERNS)} for game_of_life"
        )
    if ruleset == "rule_30" and initial_state not in RULE_30_INITIAL_STATES:
        raise ValueError(
            f"initial_state {initial_state!r} not one of {sorted(RULE_30_INITIAL_STATES)} for rule_30"
        )
    # langtons_ant: initial_state validated as a string above but unused —
    # the ant always starts centered on a blank grid.


def _place_pattern(grid, pattern, top_left_y, top_left_x):
    for dy, dx in _PATTERN_CELLS[pattern]:
        y, x = top_left_y + dy, top_left_x + dx
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            grid[y, x] = True
    return grid


def _count_neighbors(grid_int):
    return sum(
        np.roll(np.roll(grid_int, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    )


def _game_of_life_step(grid):
    neighbors = _count_neighbors(grid.astype(int))
    survive = grid & ((neighbors == 2) | (neighbors == 3))
    born = (~grid) & (neighbors == 3)
    return survive | born


def _simulate_game_of_life(grid_size, initial_state, generations):
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    center = grid_size // 2
    grid = _place_pattern(grid, initial_state, center, center)

    frames = [grid.copy()]
    for _ in range(generations):
        grid = _game_of_life_step(grid)
        frames.append(grid.copy())
    return np.stack(frames)


def _apply_elementary_rule(row, rule_number):
    """One step of a 1D elementary cellular automaton (wraparound
    boundary) under Wolfram rule `rule_number` (0-255).
    """
    rule_bits = [(rule_number >> i) & 1 for i in range(8)]
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left.astype(int) << 2) | (row.astype(int) << 1) | right.astype(int)
    return np.array([bool(rule_bits[i]) for i in idx])


def _simulate_rule_30(grid_size, generations):
    # grid_size doubles as row width and canvas height (square canvas);
    # each generation writes one new row into an otherwise-blank canvas.
    num_rows = min(generations + 1, grid_size)
    canvas = np.zeros((grid_size, grid_size), dtype=bool)
    row = np.zeros(grid_size, dtype=bool)
    row[grid_size // 2] = True
    canvas[0] = row

    frames = [canvas.copy()]
    for g in range(1, num_rows):
        row = _apply_elementary_rule(row, rule_number=30)
        canvas[g] = row
        frames.append(canvas.copy())
    return np.stack(frames)


def _simulate_langtons_ant(grid_size, generations):
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    y = x = grid_size // 2
    direction = 0  # 0:up, 1:right, 2:down, 3:left
    dy = [-1, 0, 1, 0]
    dx = [0, 1, 0, -1]

    frames = [grid.copy()]
    for _ in range(generations):
        if grid[y, x]:
            direction = (direction - 1) % 4  # on black: turn left, flip to white
            grid[y, x] = False
        else:
            direction = (direction + 1) % 4  # on white: turn right, flip to black
            grid[y, x] = True
        y = (y + dy[direction]) % grid_size  # torus wraparound — bounded grid_size
        x = (x + dx[direction]) % grid_size
        frames.append(grid.copy())
    return np.stack(frames)


def build_generations(ruleset, grid_size, initial_state, generations):
    """Returns a `(num_frames, grid_size, grid_size)` bool array — one
    full canvas per generation (including the initial state as frame 0).
    Pure — no Scene dependency.
    """
    validate_params(ruleset, grid_size, initial_state, generations)

    if ruleset == "game_of_life":
        return _simulate_game_of_life(grid_size, initial_state, generations)
    if ruleset == "rule_30":
        return _simulate_rule_30(grid_size, generations)
    if ruleset == "langtons_ant":
        return _simulate_langtons_ant(grid_size, generations)

    raise AssertionError(f"unhandled ruleset {ruleset!r}")  # validate_params already checked


class CellularAutomataReel(ReelScene):
    """`params`: ruleset, grid_size, initial_state, generations (README
    Section 6, #7).
    """

    ruleset = "game_of_life"
    grid_size = 60
    initial_state = "glider"
    generations = 80

    def construct(self):
        self.set_title("One Rule, Every Generation")

        frames = build_generations(self.ruleset, self.grid_size, self.initial_state, self.generations)
        num_frames = frames.shape[0]
        zone = self.zones.content_zone

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), num_frames - 1))

        def make_image():
            cells = frames[frame_index()].astype(np.uint8) * 255
            pixels = np.stack([cells] * 3, axis=-1)
            image = ImageMobject(pixels)
            image.height = zone.height * 2
            self.fit_to_zone(image, zone)
            return image

        image = always_redraw(make_image)
        self.add(image)

        self.play(frame_tracker.animate.set_value(num_frames - 1), run_time=5.0, rate_func=linear)

        self.set_caption("The rule never changes")
