"""Light tests for proof_without_words (README Section 6, #16): pure
param-validation and geometry-correctness checks only — no Scene/render
checks, per the current per-engine test cadence (README Section 11.6).
"""

import numpy as np
import pytest

from engines.proof_without_words import (
    _A,
    _B,
    _C,
    _NUM_RINGS,
    _CIRCLE_RADIUS,
    _polygon_area,
    _pythagorean_config1,
    _pythagorean_config2,
    _ring_strip_geometry,
    build_proof,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_proof_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_proof")


# ---- Pythagorean rearrangement correctness ----


def test_3_4_5_triple_satisfies_pythagorean_theorem():
    assert _A**2 + _B**2 == _C**2


def test_config1_triangles_each_have_the_expected_leg_area():
    config1 = _pythagorean_config1(_A, _B)
    expected_area = 0.5 * _A * _B
    for triangle in config1["triangles"]:
        assert _polygon_area(triangle) == pytest.approx(expected_area)


def test_config1_inner_square_area_is_c_squared():
    config1 = _pythagorean_config1(_A, _B)
    assert _polygon_area(config1["inner_square"]) == pytest.approx(_C**2)


def test_config1_pieces_sum_to_the_big_square_area():
    config1 = _pythagorean_config1(_A, _B)
    total = sum(_polygon_area(t) for t in config1["triangles"]) + _polygon_area(config1["inner_square"])
    assert total == pytest.approx((_A + _B) ** 2)


def test_config2_triangles_each_have_the_expected_leg_area():
    config2 = _pythagorean_config2(_A, _B)
    expected_area = 0.5 * _A * _B
    for triangle in config2["triangles"]:
        assert _polygon_area(triangle) == pytest.approx(expected_area)


def test_config2_squares_are_a_squared_and_b_squared():
    config2 = _pythagorean_config2(_A, _B)
    assert _polygon_area(config2["square_a"]) == pytest.approx(_A**2)
    assert _polygon_area(config2["square_b"]) == pytest.approx(_B**2)


def test_config2_pieces_sum_to_the_big_square_area():
    config2 = _pythagorean_config2(_A, _B)
    total = (
        sum(_polygon_area(t) for t in config2["triangles"])
        + _polygon_area(config2["square_a"])
        + _polygon_area(config2["square_b"])
    )
    assert total == pytest.approx((_A + _B) ** 2)


def test_both_configs_cover_identical_total_area():
    config1 = _pythagorean_config1(_A, _B)
    config2 = _pythagorean_config2(_A, _B)
    total1 = sum(_polygon_area(t) for t in config1["triangles"]) + _polygon_area(config1["inner_square"])
    total2 = (
        sum(_polygon_area(t) for t in config2["triangles"])
        + _polygon_area(config2["square_a"])
        + _polygon_area(config2["square_b"])
    )
    assert total1 == pytest.approx(total2)


# ---- Circle-unwrap correctness ----


def test_ring_strip_area_exactly_matches_true_ring_area():
    # Using the ring's midpoint radius for the strip width makes the
    # strip area exactly equal the true annulus area, for any ring
    # count — not just an approximation in the limit (see module
    # docstring derivation).
    strips = _ring_strip_geometry(_CIRCLE_RADIUS, num_rings=10)
    for s in strips:
        true_area = np.pi * (s["outer_radius"] ** 2 - s["inner_radius"] ** 2)
        strip_area = s["width"] * s["height"]
        assert strip_area == pytest.approx(true_area)


@pytest.mark.parametrize("num_rings", [1, 5, 24, 100])
def test_total_strip_area_matches_circle_area_for_any_ring_count(num_rings):
    strips = _ring_strip_geometry(_CIRCLE_RADIUS, num_rings)
    total = sum(s["width"] * s["height"] for s in strips)
    assert total == pytest.approx(np.pi * _CIRCLE_RADIUS**2)


def test_ring_strip_geometry_rings_are_contiguous():
    strips = _ring_strip_geometry(_CIRCLE_RADIUS, _NUM_RINGS)
    assert strips[0]["inner_radius"] == 0.0
    assert strips[-1]["outer_radius"] == pytest.approx(_CIRCLE_RADIUS)
    for prev, nxt in zip(strips, strips[1:]):
        assert prev["outer_radius"] == pytest.approx(nxt["inner_radius"])


def test_strip_widths_increase_monotonically_with_radius():
    strips = _ring_strip_geometry(_CIRCLE_RADIUS, _NUM_RINGS)
    widths = [s["width"] for s in strips]
    assert widths == sorted(widths)


# ---- build_proof() ----


def test_build_proof_pythagorean_shape():
    data = build_proof("pythagorean_rearrangement")
    assert data["kind"] == "pythagorean"
    assert data["a"] == _A and data["b"] == _B and data["c"] == _C


def test_build_proof_circle_unwrap_shape():
    data = build_proof("circle_area_unwrapping")
    assert data["kind"] == "circle_unwrap"
    assert len(data["strips"]) == _NUM_RINGS


def test_build_proof_is_deterministic():
    a = build_proof("circle_area_unwrapping")
    b = build_proof("circle_area_unwrapping")
    assert a == b
