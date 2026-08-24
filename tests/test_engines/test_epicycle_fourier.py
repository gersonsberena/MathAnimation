"""Light tests for epicycle_fourier (README Section 6, #3): pure
param-validation, boundary, and reconstruction-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.epicycle_fourier import (
    DIFFICULTY_LEVELS,
    NUM_CIRCLES_RANGE,
    PATH_SOURCES,
    build_epicycles,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_path_source():
    with pytest.raises(ValueError):
        validate_params("not_a_real_path", num_circles=5, difficulty_level="medium")


def test_rejects_unknown_difficulty_level():
    with pytest.raises(ValueError):
        validate_params("star", num_circles=5, difficulty_level="not_a_real_level")


def test_rejects_num_circles_below_range():
    lo, _hi = NUM_CIRCLES_RANGE
    with pytest.raises(ValueError):
        validate_params("star", num_circles=lo - 1, difficulty_level="medium")


def test_rejects_num_circles_above_range():
    _lo, hi = NUM_CIRCLES_RANGE
    with pytest.raises(ValueError):
        validate_params("star", num_circles=hi + 1, difficulty_level="medium")


# ---- Shape consistency ----


@pytest.mark.parametrize("path_source", sorted(PATH_SOURCES))
@pytest.mark.parametrize("num_circles", [5, 15, 30])
def test_build_epicycles_centers_shape(path_source, num_circles):
    num_frames = 100
    data = build_epicycles(path_source, num_circles, "medium", num_frames)
    assert data["centers"].shape == (num_frames, num_circles, 2)


@pytest.mark.parametrize("path_source", sorted(PATH_SOURCES))
@pytest.mark.parametrize("num_circles", [5, 15, 30])
def test_build_epicycles_radii_shape(path_source, num_circles):
    num_frames = 100
    data = build_epicycles(path_source, num_circles, "medium", num_frames)
    assert data["radii"].shape == (num_circles,)


@pytest.mark.parametrize("path_source", sorted(PATH_SOURCES))
def test_build_epicycles_traced_shape(path_source):
    num_frames = 100
    num_circles = 10
    data = build_epicycles(path_source, num_circles, "medium", num_frames)
    assert data["traced"].shape == (num_frames, 2)
    assert data["target_points"].shape == (num_frames, 2)


# ---- Difficulty levels & hint opacity ----


@pytest.mark.parametrize("difficulty_level", sorted(DIFFICULTY_LEVELS))
def test_hint_opacity_matches_difficulty(difficulty_level):
    from engines.epicycle_fourier import _HINT_OPACITY

    data = build_epicycles("star", 10, difficulty_level, 50)
    assert data["hint_opacity"] == _HINT_OPACITY[difficulty_level]


# ---- Closed loop property ----


def test_traced_path_is_closed_loop():
    data = build_epicycles("star", 10, "medium", 200)
    traced = data["traced"]
    # First and last frame should be close (full revolution completes)
    assert np.allclose(traced[0], traced[-1], atol=0.05)


# ---- Reconstruction accuracy ----


@pytest.mark.parametrize("path_source", ["star", "heart"])
def test_more_circles_improves_reconstruction(path_source):
    num_frames = 200

    data_small = build_epicycles(path_source, num_circles=3, difficulty_level="medium", num_frames=num_frames)
    traced_small = data_small["traced"]
    target = data_small["target_points"]

    data_large = build_epicycles(path_source, num_circles=30, difficulty_level="medium", num_frames=num_frames)
    traced_large = data_large["traced"]

    error_small = np.mean(np.linalg.norm(traced_small - target, axis=1))
    error_large = np.mean(np.linalg.norm(traced_large - target, axis=1))

    assert error_large < error_small, f"Expected larger num_circles to improve reconstruction: {error_small:.6f} -> {error_large:.6f}"
