"""Light tests for tessellation_growth (README Section 6, #10): pure
param-validation, boundary, and construction-correctness checks only —
no Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.tessellation_growth import (
    FILL_TARGET_RANGE,
    REQUIRED_SYMMETRY,
    _build_hex_tiling,
    _build_square_tiling,
    _penrose_sun,
    _subdivide_penrose_triangles,
    build_tessellation,
    polygon_area,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_tile_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_tile_type", symmetry_group=6, fill_target=0.9)


@pytest.mark.parametrize("tile_type,required", sorted(REQUIRED_SYMMETRY.items()))
def test_rejects_mismatched_symmetry_group(tile_type, required):
    with pytest.raises(ValueError):
        validate_params(tile_type, symmetry_group=required + 1, fill_target=0.9)


@pytest.mark.parametrize("tile_type,required", sorted(REQUIRED_SYMMETRY.items()))
def test_accepts_matching_symmetry_group(tile_type, required):
    validate_params(tile_type, symmetry_group=required, fill_target=0.9)


def test_rejects_fill_target_below_min():
    lo, _hi = FILL_TARGET_RANGE
    with pytest.raises(ValueError):
        validate_params("hex_tiling", symmetry_group=6, fill_target=lo - 0.01)


def test_rejects_fill_target_above_max():
    _lo, hi = FILL_TARGET_RANGE
    with pytest.raises(ValueError):
        validate_params("hex_tiling", symmetry_group=6, fill_target=hi + 0.01)


# ---- Hex / square tiling correctness ----


def test_hex_tiling_produces_regular_hexagons():
    shapes = _build_hex_tiling(num_rings=2)
    for shape in shapes:
        points = shape["points"]
        assert len(points) == 6
        # all 6 vertices equidistant from the tile's own centroid -> regular
        cx = sum(p[0] for p in points) / 6
        cy = sum(p[1] for p in points) / 6
        dists = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in points]
        assert max(dists) - min(dists) < 1e-9


def test_square_tiling_produces_squares():
    shapes = _build_square_tiling(num_rings=2)
    for shape in shapes:
        points = shape["points"]
        assert len(points) == 4
        side_lengths = [
            ((points[i][0] - points[(i + 1) % 4][0]) ** 2 + (points[i][1] - points[(i + 1) % 4][1]) ** 2) ** 0.5
            for i in range(4)
        ]
        assert max(side_lengths) - min(side_lengths) < 1e-9


def test_hex_and_square_tiling_ring_counts_match_known_formula():
    # A hex-grid disk of radius n (cube-coordinate distance) has
    # 1 + 3*n*(n+1) cells — a standard, independently-checkable formula.
    for n in range(1, 4):
        shapes = _build_hex_tiling(num_rings=n)
        assert len(shapes) == 1 + 3 * n * (n + 1)
    # A square grid disk of "radius" n (Chebyshev) has (2n+1)^2 cells.
    for n in range(1, 4):
        shapes = _build_square_tiling(num_rings=n)
        assert len(shapes) == (2 * n + 1) ** 2


# ---- Penrose tiling correctness ----


def test_penrose_sun_has_ten_triangles_of_equal_area():
    triangles = _penrose_sun()
    assert len(triangles) == 10
    areas = [polygon_area([(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]) for _color, a, b, c in triangles]
    assert max(areas) - min(areas) < 1e-9


def test_penrose_subdivision_conserves_total_area():
    # The defining correctness property of a deflation rule: splitting a
    # triangle into smaller ones must exactly tile the same area, no
    # gaps or overlaps. Checked generation by generation, independent of
    # whether the vertex-assignment convention matches any reference
    # implementation's exact labeling.
    triangles = _penrose_sun()

    def total_area(tris):
        return sum(polygon_area([(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]) for _c, a, b, c in tris)

    area_before = total_area(triangles)
    for _generation in range(4):
        triangles = _subdivide_penrose_triangles(triangles)
        area_after = total_area(triangles)
        assert area_after == pytest.approx(area_before, rel=1e-9)


def test_penrose_subdivision_only_produces_two_triangle_types():
    triangles = _penrose_sun()
    for _ in range(3):
        triangles = _subdivide_penrose_triangles(triangles)
    assert {color for color, _a, _b, _c in triangles} <= {0, 1}


# ---- build_tessellation() ----


@pytest.mark.parametrize("tile_type,required", sorted(REQUIRED_SYMMETRY.items()))
def test_build_tessellation_shapes_have_angles_in_valid_range(tile_type, required):
    shapes = build_tessellation(tile_type, symmetry_group=required, fill_target=0.9)
    assert len(shapes) > 0
    for shape in shapes:
        assert 0 <= shape["angle"] < 2 * 3.141592653589793 + 1e-9


def test_build_tessellation_is_deterministic():
    a = build_tessellation("hex_tiling", symmetry_group=6, fill_target=0.9)
    b = build_tessellation("hex_tiling", symmetry_group=6, fill_target=0.9)
    assert a == b
