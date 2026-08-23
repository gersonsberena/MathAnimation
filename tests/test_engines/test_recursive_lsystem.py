"""Tests for the recursive_lsystem engine (README Section 6, #1).

Split in two layers per README Section 11.2:
  - pure geometry/validation tests against build_shape() (fast, no Scene)
  - param-boundary tests (item 3) run through the actual Scene, at each
    declared param's min and max, not just one "nice" example
"""

import pytest

from engines.base import ReelScene
from engines.recursive_lsystem import (
    DEPTH_RANGES,
    RecursiveLSystemReel,
    build_shape,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_rule_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_rule", depth=5, branch_angle=25)


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_rejects_depth_below_min(rule_type):
    lo, _hi = DEPTH_RANGES[rule_type]
    with pytest.raises(ValueError):
        validate_params(rule_type, depth=lo - 1, branch_angle=25)


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_rejects_depth_above_max(rule_type):
    _lo, hi = DEPTH_RANGES[rule_type]
    with pytest.raises(ValueError):
        validate_params(rule_type, depth=hi + 1, branch_angle=25)


def test_rejects_branch_angle_below_min():
    with pytest.raises(ValueError):
        validate_params("binary_branch", depth=5, branch_angle=4.9)


def test_rejects_branch_angle_above_max():
    with pytest.raises(ValueError):
        validate_params("binary_branch", depth=5, branch_angle=80.1)


# ---- build_shape() geometry, at declared param min/max bounds ----


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_build_shape_at_min_depth_is_non_empty(rule_type):
    lo, _hi = DEPTH_RANGES[rule_type]
    shape = build_shape(rule_type, depth=lo, branch_angle=25)
    assert len(shape.family_members_with_points()) > 0


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_build_shape_at_max_depth_is_non_empty(rule_type):
    _lo, hi = DEPTH_RANGES[rule_type]
    shape = build_shape(rule_type, depth=hi, branch_angle=25)
    assert len(shape.family_members_with_points()) > 0


def test_binary_branch_segment_count_matches_full_binary_tree():
    # A full binary tree of a given depth has 2^depth - 1 line segments.
    depth = 5
    shape = build_shape("binary_branch", depth=depth, branch_angle=25)
    assert len(shape.family_members_with_points()) == 2**depth - 1


def test_sierpinski_triangle_count_matches_depth():
    # 3^depth triangles at recursion depth `depth` (depth=0 -> 1 triangle).
    depth = 4
    shape = build_shape("sierpinski_triangle", depth=depth, branch_angle=25)
    assert len(shape.family_members_with_points()) == 3**depth


def test_build_shape_is_deterministic():
    a = build_shape("binary_branch", depth=6, branch_angle=30)
    b = build_shape("binary_branch", depth=6, branch_angle=30)
    assert len(a.family_members_with_points()) == len(b.family_members_with_points())


# ---- End-to-end through the Scene, at param bounds ----


def _construct(rule_type, depth, branch_angle):
    scene = RecursiveLSystemReel()
    scene.rule_type = rule_type
    scene.depth = depth
    scene.branch_angle = branch_angle
    scene.setup()
    scene.construct()
    return scene


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_scene_construct_succeeds_at_min_depth(rule_type):
    lo, _hi = DEPTH_RANGES[rule_type]
    # Must not raise, including at depth=1 binary_branch: a single trunk
    # line has zero width, which previously tripped fit_to_zone's
    # zero-size guard (engines/base.py) — see the fix for that.
    _construct(rule_type, depth=lo, branch_angle=25)


@pytest.mark.parametrize("rule_type", sorted(DEPTH_RANGES))
def test_scene_construct_succeeds_at_max_depth(rule_type):
    _lo, hi = DEPTH_RANGES[rule_type]
    _construct(rule_type, depth=hi, branch_angle=25)


def test_scene_propagates_invalid_params_as_error():
    scene = RecursiveLSystemReel()
    scene.rule_type = "binary_branch"
    scene.depth = 999  # out of range
    scene.branch_angle = 25
    scene.setup()
    with pytest.raises(ValueError):
        scene.construct()
