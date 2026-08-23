"""Light tests for cellular_automata (README Section 6, #7): pure
param-validation, boundary, and rule-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.cellular_automata import (
    GENERATIONS_RANGE,
    GRID_SIZE_RANGE,
    _apply_elementary_rule,
    _simulate_game_of_life,
    build_generations,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_ruleset():
    with pytest.raises(ValueError):
        validate_params("not_a_real_ruleset", grid_size=40, initial_state="glider", generations=20)


def test_rejects_grid_size_below_min():
    lo, _hi = GRID_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("game_of_life", grid_size=lo - 1, initial_state="glider", generations=20)


def test_rejects_grid_size_above_max():
    _lo, hi = GRID_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("game_of_life", grid_size=hi + 1, initial_state="glider", generations=20)


def test_rejects_generations_below_min():
    lo, _hi = GENERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("game_of_life", grid_size=40, initial_state="glider", generations=lo - 1)


def test_rejects_generations_above_max():
    _lo, hi = GENERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("game_of_life", grid_size=40, initial_state="glider", generations=hi + 1)


def test_game_of_life_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        validate_params("game_of_life", grid_size=40, initial_state="not_a_pattern", generations=20)


def test_rule_30_rejects_unknown_initial_state():
    with pytest.raises(ValueError):
        validate_params("rule_30", grid_size=40, initial_state="random", generations=20)


def test_langtons_ant_accepts_any_string_initial_state():
    # unused for this ruleset, but must still be a string
    validate_params("langtons_ant", grid_size=40, initial_state="whatever", generations=20)


# ---- Elementary rule correctness ----


def test_rule_30_matches_wolfram_truth_table():
    # Rule 30 = 0b00011110: 111->0, 110->0, 101->0, 100->1, 011->1,
    # 010->1, 001->1, 000->0. Build a row where each wraparound-safe
    # triple appears once, isolated by zeros, and check the center cell.
    row = np.array([False, False, False, True, False, False, False], dtype=bool)  # single 1, rest 0
    result = _apply_elementary_rule(row, rule_number=30)
    # Around the lone 1 at index 3: neighborhoods (0,0,1)@2->1, (0,1,0)@3->1, (1,0,0)@4->1
    assert result[2] and result[3] and result[4]
    # Far from the 1, neighborhood is 000 -> 0
    assert not result[0]


def test_game_of_life_blinker_oscillates_with_period_2():
    frames = _simulate_game_of_life(grid_size=15, initial_state="blinker", generations=2)
    assert (frames[0] == frames[2]).all()
    assert not (frames[0] == frames[1]).all()  # actually changes at step 1


# ---- Frame sequence shape/growth, at param bounds ----


@pytest.mark.parametrize("ruleset", sorted(["game_of_life", "rule_30", "langtons_ant"]))
@pytest.mark.parametrize("generations", GENERATIONS_RANGE)
def test_build_generations_frame_count_and_shape(ruleset, generations):
    grid_size = 40
    initial_state = {"game_of_life": "glider", "rule_30": "single_cell", "langtons_ant": "unused"}[ruleset]
    frames = build_generations(ruleset, grid_size, initial_state, generations)
    assert frames.shape[1:] == (grid_size, grid_size)
    assert frames.dtype == bool
    assert frames.shape[0] <= generations + 1


def test_rule_30_activation_front_expands_by_exactly_one_cell_per_side():
    # Structural property of any elementary CA with background rule
    # 000->0 evolving from a single seed: the True-cell envelope can only
    # ever grow by exactly one cell on each side per generation.
    grid_size = 60
    generations = 20
    frames = build_generations("rule_30", grid_size, "single_cell", generations)
    canvas = frames[-1]
    center = grid_size // 2

    for g in range(min(generations, grid_size - 1)):
        row = canvas[g]
        active = np.flatnonzero(row)
        assert active.min() == center - g
        assert active.max() == center + g


def test_build_generations_is_deterministic():
    a = build_generations("game_of_life", 30, "r_pentomino", 15)
    b = build_generations("game_of_life", 30, "r_pentomino", 15)
    assert (a == b).all()
