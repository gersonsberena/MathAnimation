"""geometric_transformation engine (README Section 6, engine 20): shape
morphing under rotation/reflection/scaling to reveal symmetry.
Mechanic: continuous transform applied to a shape while tracking an
invariant — here, area.

Rotation and reflection preserve area exactly (rigid motions); scaling
by `factor` multiplies it by `factor**2`. Every transform step's exact
effect on area is independently predictable in closed form, so
correctness is checked against that closed form rather than trusted
from the transform matrices alone (test_rotate_preserves_area_exactly,
test_scale_multiplies_area_by_factor_squared, etc.) — the same
"independent reference" pattern used for population_growth_model's
closed-form checks and number_pattern's pi-digit reference.

`build_geometric_transformation()` is pure — no Scene dependency.
"""

import math

import numpy as np
from manim import BLUE, WHITE, Polygon, Text, Transform

from engines.base import ReelScene

SHAPE_SOURCES = {"regular_polygon", "star_polygon"}
TRANSFORM_TYPES = {"rotate", "reflect", "scale"}
TRANSFORM_SEQUENCE_LENGTH_RANGE = (2, 10)
ANGLE_RANGE = (-360.0, 360.0)
SCALE_FACTOR_RANGE = (0.2, 3.0)

_REGULAR_POLYGON_SIDES = 6
_STAR_POINTS = 5


def validate_params(shape_source, transform_sequence) -> None:
    if shape_source not in SHAPE_SOURCES:
        raise ValueError(f"unknown shape_source {shape_source!r}, must be one of {sorted(SHAPE_SOURCES)}")

    lo, hi = TRANSFORM_SEQUENCE_LENGTH_RANGE
    if not (lo <= len(transform_sequence) <= hi):
        raise ValueError(f"transform_sequence length {len(transform_sequence)} out of range [{lo}, {hi}]")

    for step in transform_sequence:
        t = step.get("type")
        if t not in TRANSFORM_TYPES:
            raise ValueError(f"unknown transform type {t!r}, must be one of {sorted(TRANSFORM_TYPES)}")

        if t == "rotate":
            lo_a, hi_a = ANGLE_RANGE
            if not (lo_a <= step.get("angle", 0) <= hi_a):
                raise ValueError(f"rotate angle={step.get('angle')!r} out of range [{lo_a}, {hi_a}]")
        elif t == "reflect":
            lo_a, hi_a = ANGLE_RANGE
            if not (lo_a <= step.get("axis", 0) <= hi_a):
                raise ValueError(f"reflect axis={step.get('axis')!r} out of range [{lo_a}, {hi_a}]")
        elif t == "scale":
            lo_f, hi_f = SCALE_FACTOR_RANGE
            if not (lo_f <= step.get("factor", 0) <= hi_f):
                raise ValueError(f"scale factor={step.get('factor')!r} out of range [{lo_f}, {hi_f}]")


def _regular_polygon_vertices(n=_REGULAR_POLYGON_SIDES, radius=1.0):
    return [(radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _star_polygon_vertices(n=_STAR_POINTS, outer_radius=1.0, inner_radius=0.45):
    vertices = []
    for k in range(2 * n):
        r = outer_radius if k % 2 == 0 else inner_radius
        angle = math.pi * k / n
        vertices.append((r * math.cos(angle), r * math.sin(angle)))
    return vertices


def _polygon_area(vertices):
    n = len(vertices)
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _centroid(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (sum(xs) / len(vertices), sum(ys) / len(vertices))


def _apply_rotate(vertices, angle_deg, center):
    rad = math.radians(angle_deg)
    cx, cy = center
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [(cx + (x - cx) * cos_a - (y - cy) * sin_a, cy + (x - cx) * sin_a + (y - cy) * cos_a) for x, y in vertices]


def _apply_reflect(vertices, axis_angle_deg, center):
    rad = math.radians(axis_angle_deg)
    cx, cy = center
    cos_2a, sin_2a = math.cos(2 * rad), math.sin(2 * rad)
    result = []
    for x, y in vertices:
        dx, dy = x - cx, y - cy
        rx, ry = dx * cos_2a + dy * sin_2a, dx * sin_2a - dy * cos_2a
        result.append((cx + rx, cy + ry))
    return result


def _apply_scale(vertices, factor, center):
    cx, cy = center
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in vertices]


def _apply_transform_step(vertices, step, center):
    if step["type"] == "rotate":
        return _apply_rotate(vertices, step["angle"], center)
    if step["type"] == "reflect":
        return _apply_reflect(vertices, step["axis"], center)
    if step["type"] == "scale":
        return _apply_scale(vertices, step["factor"], center)
    raise AssertionError(f"unhandled transform type {step['type']!r}")  # validate_params already checked


def build_geometric_transformation(shape_source, transform_sequence):
    """Returns `{"frames", "areas", "transform_sequence"}`. `frames` is a
    list of vertex lists, one per transform step (frames[0] is the
    untransformed starting shape); `areas[i]` is `frames[i]`'s area.
    Pure — no Scene dependency.
    """
    validate_params(shape_source, transform_sequence)

    base = _regular_polygon_vertices() if shape_source == "regular_polygon" else _star_polygon_vertices()
    center = _centroid(base)

    frames = [list(base)]
    current = list(base)
    for step in transform_sequence:
        current = _apply_transform_step(current, step, center)
        frames.append(list(current))

    areas = [_polygon_area(f) for f in frames]
    return {"frames": frames, "areas": areas, "transform_sequence": transform_sequence}


_STEP_LABELS = {
    "rotate": lambda s: f"rotate {s['angle']:.0f}°",
    "reflect": lambda s: f"reflect @ {s['axis']:.0f}°",
    "scale": lambda s: f"scale x{s['factor']:.2f}",
}


class GeometricTransformationReel(ReelScene):
    """`params`: shape_source, transform_sequence (README Section 6,
    #20).
    """

    shape_source = "star_polygon"
    transform_sequence = [
        {"type": "rotate", "angle": 72.0},
        {"type": "reflect", "axis": 30.0},
        {"type": "scale", "factor": 1.4},
        {"type": "rotate", "angle": 144.0},
    ]
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "What Stays The Same?")

        data = build_geometric_transformation(self.shape_source, self.transform_sequence)
        frames, areas, transform_sequence = data["frames"], data["areas"], data["transform_sequence"]
        zone = self.zones.content_zone

        max_extent = max(max(abs(x), abs(y)) for f in frames for x, y in f)
        scale = min(zone.width, zone.height) * 0.35 / max_extent
        center = zone.get_center()

        def to_scene(vertices):
            return [np.array([center[0] + x * scale, center[1] + y * scale, 0.0]) for x, y in vertices]

        shape = Polygon(*to_scene(frames[0]), fill_color=BLUE, fill_opacity=0.75, stroke_color=WHITE, stroke_width=2)
        area_label = Text(f"area = {areas[0]:.2f}", font_size=28, color=WHITE)
        area_label.move_to(center + np.array([0.0, -zone.height * 0.32, 0.0]))
        step_label = Text("start", font_size=24, color=WHITE)
        step_label.move_to(center + np.array([0.0, zone.height * 0.32, 0.0]))

        self.add(shape, area_label, step_label)
        self.wait(0.8)

        for i, step in enumerate(transform_sequence):
            target = Polygon(*to_scene(frames[i + 1]), fill_color=BLUE, fill_opacity=0.75, stroke_color=WHITE, stroke_width=2)
            new_area_label = Text(f"area = {areas[i + 1]:.2f}", font_size=28, color=WHITE).move_to(area_label.get_center())
            new_step_label = Text(_STEP_LABELS[step["type"]](step), font_size=24, color=WHITE).move_to(step_label.get_center())

            self.play(
                Transform(shape, target),
                Transform(area_label, new_area_label),
                Transform(step_label, new_step_label),
                run_time=1.2,
            )
            self.wait(0.4)

        self.wait(0.3)
        self.set_caption(self.caption_text or "Rigid motions never change the area. Only scaling does.")
