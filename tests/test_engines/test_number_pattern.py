"""Light tests for number_pattern (README Section 6, #11): pure
param-validation, boundary, and sequence-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.number_pattern import (
    LENGTH_RANGES,
    PI_DIGITS,
    _pascal_mod2_raster,
    _pi_digit_positions,
    _times_table_circle,
    build_number_pattern,
    validate_params,
)


def _binomial_mod2_reference(n, k):
    from math import comb

    return comb(n, k) % 2


# ---- Validation ----


def test_rejects_unknown_sequence_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_sequence", length=50, mapping_rule="binary")


def test_rejects_mapping_rule_invalid_for_sequence_type():
    with pytest.raises(ValueError):
        validate_params("pascals_triangle_mod2", length=50, mapping_rule="color_by_value")


@pytest.mark.parametrize("sequence_type", sorted(LENGTH_RANGES))
def test_rejects_length_below_min(sequence_type):
    lo, _hi = LENGTH_RANGES[sequence_type]
    mapping_rule = {"pascals_triangle_mod2": "binary", "pi_digits": "color_by_value", "times_table_circle": "times_2"}[
        sequence_type
    ]
    with pytest.raises(ValueError):
        validate_params(sequence_type, length=lo - 1, mapping_rule=mapping_rule)


@pytest.mark.parametrize("sequence_type", sorted(LENGTH_RANGES))
def test_rejects_length_above_max(sequence_type):
    _lo, hi = LENGTH_RANGES[sequence_type]
    mapping_rule = {"pascals_triangle_mod2": "binary", "pi_digits": "color_by_value", "times_table_circle": "times_2"}[
        sequence_type
    ]
    with pytest.raises(ValueError):
        validate_params(sequence_type, length=hi + 1, mapping_rule=mapping_rule)


# ---- Pascal's triangle mod 2 correctness ----


def test_pascal_mod2_matches_math_comb_reference_for_small_rows():
    grid = _pascal_mod2_raster(num_rows=20)
    for n in range(20):
        for k in range(n + 1):
            expected = _binomial_mod2_reference(n, k)
            assert grid[n, k] == bool(expected), f"mismatch at n={n}, k={k}"


def test_pascal_mod2_row_edges_are_always_one():
    # C(n, 0) and C(n, n) are always 1, odd, for any n
    grid = _pascal_mod2_raster(num_rows=32)
    for n in range(32):
        assert grid[n, 0]
        assert grid[n, n]


def test_pascal_mod2_outside_the_triangle_is_false():
    grid = _pascal_mod2_raster(num_rows=10)
    for n in range(10):
        for k in range(n + 1, 10):
            assert not grid[n, k]


# ---- Pi digits correctness ----


def test_pi_digits_match_the_well_known_expansion():
    # 3.14159265358979323846... — this specific 20-digit prefix is about
    # as widely verified a fact as exists; if PI_DIGITS is wrong, this
    # catches it immediately.
    assert PI_DIGITS[:20] == "31415926535897932384"


def test_pi_digit_positions_color_by_value_forms_a_grid():
    positions = _pi_digit_positions([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2], "color_by_value")
    assert positions[0] == (0, 0)
    assert positions[9] == (9, 0)
    assert positions[10] == (0, -1)  # wraps to the next row


def test_pi_digit_positions_polar_walk_is_deterministic():
    a = _pi_digit_positions([3, 1, 4, 1, 5], "polar_walk")
    b = _pi_digit_positions([3, 1, 4, 1, 5], "polar_walk")
    assert a == b
    assert len(a) == 5


# ---- Times table circle correctness ----


def test_times_table_circle_points_lie_on_unit_circle():
    points, _lines = _times_table_circle(length=50, k=2)
    for x, y in points:
        assert (x**2 + y**2) ** 0.5 == pytest.approx(1.0)


def test_times_table_circle_lines_use_correct_multiplier():
    length, k = 30, 3
    _points, lines = _times_table_circle(length, k)
    for i, j in lines:
        assert j == (i * k) % length


def test_times_table_circle_produces_one_line_per_point():
    points, lines = _times_table_circle(length=40, k=5)
    assert len(lines) == len(points) == 40


# ---- build_number_pattern() ----


def test_build_number_pattern_raster_shape():
    data = build_number_pattern("pascals_triangle_mod2", length=64, mapping_rule="binary")
    assert data["image"].shape == (64, 64)


def test_build_number_pattern_pi_digits_length_matches():
    data = build_number_pattern("pi_digits", length=30, mapping_rule="polar_walk")
    assert len(data["positions"]) == len(data["colors"]) == 30


def test_build_number_pattern_is_deterministic():
    a = build_number_pattern("times_table_circle", length=60, mapping_rule="times_3")
    b = build_number_pattern("times_table_circle", length=60, mapping_rule="times_3")
    assert a == b
