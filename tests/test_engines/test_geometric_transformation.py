"""Light tests for geometric_transformation (README Section 6, #20):
pure param-validation and invariant-tracking correctness checks only —
no Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.geometric_transformation import (
    _apply_reflect,
    _apply_rotate,
    _apply_scale,
    _centroid,
    _polygon_area,
    _regular_polygon_vertices,
    _star_polygon_vertices,
    build_geometric_transformation,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_shape_source():
    with pytest.raises(ValueError):
        validate_params("not_a_real_shape", [{"type": "rotate", "angle": 45}])


def test_rejects_unknown_transform_type():
    with pytest.raises(ValueError):
        validate_params("regular_polygon", [{"type": "not_a_real_transform"}])


def test_rejects_transform_sequence_too_short():
    with pytest.raises(ValueError):
        validate_params("regular_polygon", [{"type": "rotate", "angle": 45}])


def test_rejects_rotate_angle_out_of_range():
    with pytest.raises(ValueError):
        validate_params("regular_polygon", [{"type": "rotate", "angle": 999}, {"type": "rotate", "angle": 10}])


def test_rejects_scale_factor_out_of_range():
    with pytest.raises(ValueError):
        validate_params("regular_polygon", [{"type": "scale", "factor": 100}, {"type": "rotate", "angle": 10}])


# ---- Base shape correctness ----


def test_regular_polygon_vertices_are_equidistant_from_center():
    vertices = _regular_polygon_vertices(n=6, radius=2.0)
    cx, cy = _centroid(vertices)
    for x, y in vertices:
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        assert dist == pytest.approx(2.0, rel=1e-9)


def test_star_polygon_alternates_outer_and_inner_radius():
    vertices = _star_polygon_vertices(n=5, outer_radius=1.0, inner_radius=0.4)
    for i, (x, y) in enumerate(vertices):
        dist = (x**2 + y**2) ** 0.5
        expected = 1.0 if i % 2 == 0 else 0.4
        assert dist == pytest.approx(expected, rel=1e-9)


# ---- Area invariant under each transform type (independent closed-form checks) ----


def test_rotate_preserves_area_exactly():
    vertices = _regular_polygon_vertices(n=6)
    original_area = _polygon_area(vertices)
    for angle in [15, 90, 180, 271, 360]:
        rotated = _apply_rotate(vertices, angle, _centroid(vertices))
        assert _polygon_area(rotated) == pytest.approx(original_area, rel=1e-9)


def test_reflect_preserves_area_exactly():
    vertices = _star_polygon_vertices(n=5)
    original_area = _polygon_area(vertices)
    for axis in [0, 37, 90, 180]:
        reflected = _apply_reflect(vertices, axis, _centroid(vertices))
        assert _polygon_area(reflected) == pytest.approx(original_area, rel=1e-9)


def test_scale_multiplies_area_by_factor_squared():
    vertices = _regular_polygon_vertices(n=6)
    original_area = _polygon_area(vertices)
    for factor in [0.5, 1.5, 2.0, 2.7]:
        scaled = _apply_scale(vertices, factor, _centroid(vertices))
        assert _polygon_area(scaled) == pytest.approx(original_area * factor**2, rel=1e-9)


def test_reflection_is_an_involution():
    # Reflecting across the same axis twice returns to the original shape.
    vertices = _regular_polygon_vertices(n=6)
    center = _centroid(vertices)
    once = _apply_reflect(vertices, 40.0, center)
    twice = _apply_reflect(once, 40.0, center)
    for (x1, y1), (x2, y2) in zip(vertices, twice):
        assert x1 == pytest.approx(x2, abs=1e-9)
        assert y1 == pytest.approx(y2, abs=1e-9)


def test_full_rotation_returns_to_start():
    vertices = _star_polygon_vertices(n=5)
    center = _centroid(vertices)
    rotated = _apply_rotate(vertices, 360.0, center)
    for (x1, y1), (x2, y2) in zip(vertices, rotated):
        assert x1 == pytest.approx(x2, abs=1e-9)
        assert y1 == pytest.approx(y2, abs=1e-9)


# ---- build_geometric_transformation() ----


def test_build_area_sequence_matches_predicted_effect_of_each_step():
    sequence = [
        {"type": "rotate", "angle": 50},
        {"type": "scale", "factor": 2.0},
        {"type": "reflect", "axis": 15},
        {"type": "scale", "factor": 0.5},
    ]
    data = build_geometric_transformation("regular_polygon", sequence)
    areas = data["areas"]
    assert areas[1] == pytest.approx(areas[0], rel=1e-9)  # rotate: unchanged
    assert areas[2] == pytest.approx(areas[1] * 4, rel=1e-9)  # scale x2: *4
    assert areas[3] == pytest.approx(areas[2], rel=1e-9)  # reflect: unchanged
    assert areas[4] == pytest.approx(areas[3] * 0.25, rel=1e-9)  # scale x0.5: *0.25


def test_build_geometric_transformation_frames_length_matches_sequence_plus_one():
    sequence = [{"type": "rotate", "angle": 10}, {"type": "rotate", "angle": 20}, {"type": "rotate", "angle": 30}]
    data = build_geometric_transformation("star_polygon", sequence)
    assert len(data["frames"]) == len(sequence) + 1
    assert len(data["areas"]) == len(sequence) + 1


def test_build_geometric_transformation_is_deterministic():
    sequence = [{"type": "scale", "factor": 1.2}, {"type": "rotate", "angle": 33}]
    a = build_geometric_transformation("regular_polygon", sequence)
    b = build_geometric_transformation("regular_polygon", sequence)
    assert a == b
