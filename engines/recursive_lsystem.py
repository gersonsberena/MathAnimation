"""recursive_lsystem engine (README Section 6, engine 1): fractal tree,
Sierpinski triangle, and other recursive branching/subdivision fractals.

Topics are parameter sets fed into one engine, not separate scripts
(README Section 1) — `rule_type` selects the recursive construction;
`depth` and `branch_angle` shape it. `build_shape()` is a pure function
with no Scene/layout dependency so its geometry is testable without
spinning up a Manim scene.
"""

import numpy as np
from manim import DEGREES, ORIGIN, UP, Create, Line, Polygon, VGroup, rotate_vector

from engines.base import ReelScene

RULE_TYPES = {"binary_branch", "sierpinski_triangle"}

# Recursion blows up exponentially (2^depth line segments for
# binary_branch, 3^depth triangles for sierpinski_triangle) — these
# bounds keep render time sane and are what Section 8's min/max param
# tests exercise.
DEPTH_RANGES = {
    "binary_branch": (1, 10),
    "sierpinski_triangle": (0, 6),
}
BRANCH_ANGLE_RANGE = (5.0, 80.0)


def validate_params(rule_type: str, depth: int, branch_angle: float) -> None:
    if rule_type not in RULE_TYPES:
        raise ValueError(f"unknown rule_type {rule_type!r}, must be one of {sorted(RULE_TYPES)}")

    lo, hi = DEPTH_RANGES[rule_type]
    if not (lo <= depth <= hi):
        raise ValueError(f"depth={depth} out of range [{lo}, {hi}] for rule_type={rule_type!r}")

    lo, hi = BRANCH_ANGLE_RANGE
    if not (lo <= branch_angle <= hi):
        raise ValueError(f"branch_angle={branch_angle} out of range [{lo}, {hi}]")


def _build_binary_branch(start, direction, length, depth, angle_deg, shrink=0.72):
    group = VGroup()
    if depth <= 0 or length < 1e-3:
        return group

    end = start + direction * length
    group.add(Line(start, end))

    angle = angle_deg * DEGREES
    left_dir = rotate_vector(direction, angle)
    right_dir = rotate_vector(direction, -angle)

    group.add(_build_binary_branch(end, left_dir, length * shrink, depth - 1, angle_deg, shrink))
    group.add(_build_binary_branch(end, right_dir, length * shrink, depth - 1, angle_deg, shrink))
    return group


def _build_sierpinski(v0, v1, v2, depth):
    if depth <= 0:
        return VGroup(Polygon(v0, v1, v2))

    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2

    group = VGroup()
    group.add(_build_sierpinski(v0, m01, m20, depth - 1))
    group.add(_build_sierpinski(m01, v1, m12, depth - 1))
    group.add(_build_sierpinski(m20, m12, v2, depth - 1))
    return group


def build_shape(rule_type: str, depth: int, branch_angle: float) -> VGroup:
    """Returns an un-fitted, un-positioned Mobject. Callers (the Scene)
    are responsible for fitting it into content_zone — this function has
    no opinion on layout.
    """
    validate_params(rule_type, depth, branch_angle)

    if rule_type == "binary_branch":
        return _build_binary_branch(ORIGIN, UP, length=2.0, depth=depth, angle_deg=branch_angle)

    if rule_type == "sierpinski_triangle":
        side = 4.0
        height = side * (3**0.5) / 2
        v0 = np.array([-side / 2, -height / 3, 0])
        v1 = np.array([side / 2, -height / 3, 0])
        v2 = np.array([0, 2 * height / 3, 0])
        return _build_sierpinski(v0, v1, v2, depth)

    raise AssertionError(f"unhandled rule_type {rule_type!r}")  # validate_params already checked


class RecursiveLSystemReel(ReelScene):
    """`params`: rule_type, depth, branch_angle (README Section 6, #1).

    Recipe-driven values are set as instance attributes before render
    (mirrors how the orchestrator, Section 9, will parameterize a scene);
    class defaults below are only for a bare manual/CLI render.
    """

    rule_type = "binary_branch"
    depth = 8
    branch_angle = 25.0
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "One Rule. Infinite Branches.")

        shape = build_shape(self.rule_type, self.depth, self.branch_angle)
        self.fit_to_zone(shape, self.zones.content_zone)
        self.play(Create(shape), run_time=4)

        self.set_caption(self.caption_text or "Every branch splits — and leans one way")
