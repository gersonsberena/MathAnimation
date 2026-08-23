"""tessellation_growth engine (README Section 6, engine 10): hex tiling,
square tiling, Penrose (rhombus/triangle) tiling. Mechanic: shape
repeats/transforms under a fixed rule until it fills the frame.

`symmetry_group` is validated against the tile_type's actual rotational
symmetry order (hex=6, square=4, Penrose=10) rather than being a free
parameter — a mismatch is a real "this pattern can't have that symmetry"
error, not a cosmetic one, so it fails loudly (Section 2, point 4) instead
of silently ignoring the mismatch. The fill-the-frame growth animation is
a single mechanic shared by all three: every shape is built once
up front, each carries the polar angle of its centroid, and the Scene
reveals shapes in angle order as a sweeping ValueTracker passes 0 -> 2*pi
— a radial "growing outward" animation, not per-tile_type logic.

Penrose tiling uses the classic Robinson-triangle deflation method (P3):
start from a 10-triangle "sun," repeatedly split each triangle by its
type. `test_engines/test_tessellation_growth.py` checks total area is
conserved by every subdivision step — a strong, algorithm-independent
correctness check that doesn't depend on getting the vertex-assignment
convention exactly "textbook."

`build_tessellation()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, PINK, WHITE, Polygon, ValueTracker, VGroup, linear

from engines.base import ReelScene

TILE_TYPES = {"hex_tiling", "square_tiling", "penrose_tiling"}
REQUIRED_SYMMETRY = {"hex_tiling": 6, "square_tiling": 4, "penrose_tiling": 10}
FILL_TARGET_RANGE = (0.3, 1.0)

_HEX_NUM_RINGS = 4
_SQUARE_NUM_RINGS = 4
_PENROSE_GENERATIONS = 4
_PHI = (1 + 5**0.5) / 2


def validate_params(tile_type, symmetry_group, fill_target) -> None:
    if tile_type not in TILE_TYPES:
        raise ValueError(f"unknown tile_type {tile_type!r}, must be one of {sorted(TILE_TYPES)}")

    required = REQUIRED_SYMMETRY[tile_type]
    if symmetry_group != required:
        raise ValueError(f"symmetry_group must be {required} for tile_type={tile_type!r}, got {symmetry_group}")

    lo, hi = FILL_TARGET_RANGE
    if not (lo <= fill_target <= hi):
        raise ValueError(f"fill_target={fill_target} out of range [{lo}, {hi}]")


def polygon_area(points):
    """Shoelace formula."""
    n = len(points)
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _shape_angle(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return np.arctan2(cy, cx) % (2 * np.pi)


def _build_hex_tiling(num_rings=_HEX_NUM_RINGS, hex_size=1.0):
    shapes = []
    for q in range(-num_rings, num_rings + 1):
        for r in range(-num_rings, num_rings + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > num_rings:
                continue
            cx = hex_size * (3**0.5) * (q + r / 2)
            cy = hex_size * 1.5 * r
            points = [
                (cx + hex_size * np.cos(np.radians(60 * i + 30)), cy + hex_size * np.sin(np.radians(60 * i + 30)))
                for i in range(6)
            ]
            shapes.append({"points": points, "color_key": (q + r) % 2})
    return shapes


def _build_square_tiling(num_rings=_SQUARE_NUM_RINGS, size=1.0):
    shapes = []
    half = size / 2 * 0.92  # slight gap between tiles for visual clarity
    for i in range(-num_rings, num_rings + 1):
        for j in range(-num_rings, num_rings + 1):
            cx, cy = i * size, j * size
            points = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
            shapes.append({"points": points, "color_key": (i + j) % 2})
    return shapes


def _subdivide_penrose_triangles(triangles):
    new_triangles = []
    for color, a, b, c in triangles:
        if color == 0:  # acute (36 deg apex) "red" triangle
            p = a + (b - a) / _PHI
            new_triangles.append((0, c, p, b))
            new_triangles.append((1, p, c, a))
        else:  # obtuse (108 deg apex) "blue" triangle
            q = b + (a - b) / _PHI
            r = b + (c - b) / _PHI
            new_triangles.append((1, r, c, a))
            new_triangles.append((1, q, r, b))
            new_triangles.append((0, r, q, a))
    return new_triangles


def _penrose_sun(num_wedges=10):
    triangles = []
    for i in range(num_wedges):
        b = complex(np.cos((2 * i - 1) * np.pi / num_wedges), np.sin((2 * i - 1) * np.pi / num_wedges))
        c = complex(np.cos((2 * i + 1) * np.pi / num_wedges), np.sin((2 * i + 1) * np.pi / num_wedges))
        if i % 2 == 0:
            b, c = c, b
        triangles.append((0, 0 + 0j, b, c))
    return triangles


def _build_penrose_tiling(generations=_PENROSE_GENERATIONS):
    triangles = _penrose_sun()
    for _ in range(generations):
        triangles = _subdivide_penrose_triangles(triangles)

    shapes = []
    for color, a, b, c in triangles:
        points = [(a.real, a.imag), (b.real, b.imag), (c.real, c.imag)]
        shapes.append({"points": points, "color_key": color})
    return shapes


def build_tessellation(tile_type, symmetry_group, fill_target):
    """Returns a list of `{"points": [(x, y), ...], "color_key": int,
    "angle": float}` dicts — one per tile/triangle, in an arbitrary world
    coordinate system. Pure — no Scene dependency.
    """
    validate_params(tile_type, symmetry_group, fill_target)

    if tile_type == "hex_tiling":
        shapes = _build_hex_tiling()
    elif tile_type == "square_tiling":
        shapes = _build_square_tiling()
    elif tile_type == "penrose_tiling":
        shapes = _build_penrose_tiling()
    else:
        raise AssertionError(f"unhandled tile_type {tile_type!r}")  # validate_params already checked

    for shape in shapes:
        shape["angle"] = _shape_angle(shape["points"])
    return shapes


class TessellationGrowthReel(ReelScene):
    """`params`: tile_type, symmetry_group, fill_target (README Section
    6, #10).
    """

    tile_type = "hex_tiling"
    symmetry_group = 6
    fill_target = 0.9

    def construct(self):
        self.set_title("One Shape, Endless Pattern")

        shapes = build_tessellation(self.tile_type, self.symmetry_group, self.fill_target)
        zone = self.zones.content_zone

        all_points = [p for s in shapes for p in s["points"]]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        scale = min(zone.width * self.fill_target / span_x, zone.height * self.fill_target / span_y)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        zone_center = zone.get_center()

        def to_scene(point):
            return [(point[0] - cx) * scale + zone_center[0], (point[1] - cy) * scale + zone_center[1], 0.0]

        palette = {0: BLUE, 1: PINK}
        angle_tracker = ValueTracker(0)
        polygons = VGroup()

        for shape in shapes:
            poly = Polygon(
                *[to_scene(p) for p in shape["points"]],
                fill_color=palette[shape["color_key"]],
                fill_opacity=0,
                stroke_color=WHITE,
                stroke_width=1,
                stroke_opacity=0,
            )
            angle = shape["angle"]

            def make_updater(angle=angle):
                def updater(mob):
                    visible = angle_tracker.get_value() >= angle
                    mob.set_fill(opacity=0.85 if visible else 0)
                    mob.set_stroke(opacity=0.6 if visible else 0)

                return updater

            poly.add_updater(make_updater())
            polygons.add(poly)

        self.add(polygons)
        self.play(angle_tracker.animate.set_value(2 * np.pi), run_time=5.0, rate_func=linear)
        for poly in polygons:
            poly.clear_updaters()

        self.set_caption("The rule never changes, the pattern never ends")
