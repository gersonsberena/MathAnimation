"""Light tests for strange_attractor_chaos (README Section 6, #14): pure
param-validation, boundary, and simulation-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.strange_attractor_chaos import (
    LOGISTIC_ITERATIONS_RANGE,
    LORENZ_ITERATIONS_RANGE,
    _rasterize_bifurcation,
    _simulate_logistic_bifurcation_points,
    build_attractor,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_system_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_system", initial_conditions=None, iterations=1000)


def test_rejects_non_dict_initial_conditions():
    with pytest.raises(ValueError):
        validate_params("lorenz", initial_conditions="not a dict", iterations=1000)


def test_lorenz_rejects_iterations_below_min():
    lo, _hi = LORENZ_ITERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("lorenz", initial_conditions=None, iterations=lo - 1)


def test_lorenz_rejects_iterations_above_max():
    _lo, hi = LORENZ_ITERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("lorenz", initial_conditions=None, iterations=hi + 1)


def test_logistic_map_rejects_iterations_below_min():
    lo, _hi = LOGISTIC_ITERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("logistic_map", initial_conditions=None, iterations=lo - 1)


def test_logistic_map_rejects_iterations_above_max():
    _lo, hi = LOGISTIC_ITERATIONS_RANGE
    with pytest.raises(ValueError):
        validate_params("logistic_map", initial_conditions=None, iterations=hi + 1)


# ---- Lorenz correctness ----


@pytest.mark.parametrize("iterations", LORENZ_ITERATIONS_RANGE)
def test_lorenz_trajectory_shape_and_finiteness(iterations):
    data = build_attractor("lorenz", initial_conditions=None, iterations=iterations)
    points = data["points"]
    assert points.shape == (iterations, 2)
    assert np.isfinite(points).all()


def test_lorenz_stays_bounded_within_the_known_attractor_envelope():
    # The classic Lorenz parameters (sigma=10, rho=28, beta=8/3) produce a
    # bounded attractor — x and z should never blow up to large values
    # over a run of this length.
    data = build_attractor("lorenz", initial_conditions=None, iterations=3000)
    points = data["points"]
    assert (np.abs(points) < 100).all()


def test_lorenz_is_deterministic():
    a = build_attractor("lorenz", initial_conditions={"x": 0.1, "y": 0.0, "z": 0.0}, iterations=1000)
    b = build_attractor("lorenz", initial_conditions={"x": 0.1, "y": 0.0, "z": 0.0}, iterations=1000)
    assert (a["points"] == b["points"]).all()


def test_lorenz_sensitive_to_initial_conditions():
    # The defining property of a chaotic system: a tiny perturbation
    # diverges over time (butterfly effect).
    a = build_attractor("lorenz", initial_conditions={"x": 0.1}, iterations=2000)
    b = build_attractor("lorenz", initial_conditions={"x": 0.1 + 1e-6}, iterations=2000)
    early_diff = np.abs(a["points"][10] - b["points"][10]).max()
    late_diff = np.abs(a["points"][-1] - b["points"][-1]).max()
    assert late_diff > early_diff


# ---- Logistic map / bifurcation correctness ----


def test_logistic_map_below_r_3_converges_to_a_single_fixed_point():
    # For r < 3, the logistic map has one stable fixed point — all
    # samples at a given r should converge to (nearly) the same value.
    rs, xs, _r_min, _r_max = _simulate_logistic_bifurcation_points({"x0": 0.5, "r_min": 2.8, "r_max": 2.8}, num_r=1)
    assert np.ptp(xs) < 1e-6


def test_logistic_map_output_stays_in_unit_interval():
    rs, xs, _r_min, _r_max = _simulate_logistic_bifurcation_points(None, num_r=50)
    assert (xs >= 0).all() and (xs <= 1).all()


def test_rasterize_bifurcation_maps_known_point_to_expected_pixel():
    rs = np.array([2.5])  # r_min itself -> column 0
    xs = np.array([1.0])  # x=1 -> row 0 (top, since row = (1-x)*(size-1))
    image = _rasterize_bifurcation(rs, xs, r_min=2.5, r_max=4.0, grid_size=100)
    assert image[0, 0]
    assert image.sum() == 1


@pytest.mark.parametrize("iterations", LOGISTIC_ITERATIONS_RANGE)
def test_logistic_map_raster_has_expected_shape(iterations):
    data = build_attractor("logistic_map", initial_conditions=None, iterations=iterations)
    assert data["image"].shape == (300, 300)
    assert data["image"].dtype == bool
