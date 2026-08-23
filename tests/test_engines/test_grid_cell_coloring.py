"""Light tests for grid_cell_coloring (README Section 6, #6): pure
param-validation, boundary, and grid-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.grid_cell_coloring import (
    GRID_SIZE_RANGE,
    ZOOM_RANGE,
    _mandelbrot_escape_grid,
    _ulam_spiral_numbers,
    _voronoi_seed_points,
    build_grid_image,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_rule_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_rule", grid_size=50, zoom_level=1.0, color_map="grayscale")


def test_rejects_grid_size_below_min():
    lo, _hi = GRID_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("ulam_prime_spiral", grid_size=lo - 1, zoom_level=1.0, color_map="grayscale")


def test_rejects_grid_size_above_max():
    _lo, hi = GRID_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("ulam_prime_spiral", grid_size=hi + 1, zoom_level=1.0, color_map="grayscale")


def test_rejects_unknown_color_map():
    with pytest.raises(ValueError):
        validate_params("ulam_prime_spiral", grid_size=50, zoom_level=1.0, color_map="not_a_real_map")


def test_rejects_non_positive_zoom_level():
    with pytest.raises(ValueError):
        validate_params("mandelbrot", grid_size=50, zoom_level=0, color_map="grayscale")


def test_mandelbrot_rejects_zoom_above_max():
    _lo, hi = ZOOM_RANGE
    with pytest.raises(ValueError):
        validate_params("mandelbrot", grid_size=50, zoom_level=hi + 1, color_map="grayscale")


def test_non_mandelbrot_ignores_out_of_zoom_range_value():
    # zoom_level is validated as "positive" for every rule_type, but only
    # range-checked against ZOOM_RANGE for mandelbrot — a huge value is
    # fine (and unused) for ulam/voronoi.
    validate_params("ulam_prime_spiral", grid_size=50, zoom_level=1e9, color_map="grayscale")


# ---- Ulam spiral correctness ----


def test_ulam_spiral_is_a_bijection_onto_1_through_n_squared():
    n = 21
    grid = _ulam_spiral_numbers(n)
    assert sorted(grid.flatten().tolist()) == list(range(1, n * n + 1))


def test_ulam_spiral_center_cell_is_one():
    n = 15
    grid = _ulam_spiral_numbers(n)
    assert grid[n // 2, n // 2] == 1


# ---- Voronoi correctness ----


def test_voronoi_seed_points_are_deterministic_without_a_seed_param():
    a = _voronoi_seed_points(14, grid_size=100)
    b = _voronoi_seed_points(14, grid_size=100)
    assert a == b


def test_voronoi_seed_points_stay_within_grid_bounds():
    points = _voronoi_seed_points(14, grid_size=100)
    for x, y in points:
        assert 0 <= x < 100
        assert 0 <= y < 100


def test_voronoi_seed_points_are_not_collinear():
    # Regression guard: an earlier version used phi and phi**2 as the two
    # Weyl-sequence steps. Since phi**2 == phi + 1 exactly, that made
    # every "seed" land on the line y=x (see engines/grid_cell_coloring.py
    # for the identity) — degenerate diagonal-stripe output instead of a
    # real 2D Voronoi diagram. Guard against x and y being near-identical
    # sequences again.
    points = np.array(_voronoi_seed_points(14, grid_size=100))
    xs, ys = points[:, 0], points[:, 1]
    correlation = np.corrcoef(xs, ys)[0, 1]
    assert abs(correlation) < 0.9


# ---- Mandelbrot correctness ----


def test_mandelbrot_origin_never_escapes():
    # c=0 is the fixed point z=0 -> stays at 0 forever -> never escapes
    c_grid = np.array([[0.0 + 0.0j]])
    escape = _mandelbrot_escape_grid(c_grid, max_iter=50)
    assert escape[0, 0] == 50


def test_mandelbrot_far_point_escapes_immediately():
    c_grid = np.array([[5.0 + 5.0j]])
    escape = _mandelbrot_escape_grid(c_grid, max_iter=50)
    assert escape[0, 0] == 0


# ---- build_grid_image() shape/dtype, at grid_size param bounds ----


@pytest.mark.parametrize("rule_type", ["ulam_prime_spiral", "voronoi", "mandelbrot"])
@pytest.mark.parametrize("grid_size", GRID_SIZE_RANGE)
def test_build_grid_image_has_expected_shape_and_dtype(rule_type, grid_size):
    image = build_grid_image(rule_type, grid_size, zoom_level=1.0, color_map="grayscale")
    assert image.shape == (grid_size, grid_size, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize("rule_type", ["ulam_prime_spiral", "voronoi", "mandelbrot"])
def test_build_grid_image_is_deterministic(rule_type):
    a = build_grid_image(rule_type, grid_size=40, zoom_level=1.0, color_map="grayscale")
    b = build_grid_image(rule_type, grid_size=40, zoom_level=1.0, color_map="grayscale")
    assert (a == b).all()
