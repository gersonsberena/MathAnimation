"""Shared bounding-box/overlap helpers for layout QA tests (README
Section 11.2). Used by both test_layout.py (the base ReelScene's own
title/caption placement) and test_render_smoke.py (every real engine's
full rendered content, per recipe).
"""

# Adjacent zones are computed via independent formulas (see
# engines/base.py's _build_zones) and share a border at, e.g.,
# title_zone.bottom == content_zone.top. Those two values land a few
# ULPs apart in practice, so a strict "<" comparison flags
# touching-but-not-overlapping zones as overlapping. This tolerance
# treats anything within EPS as "touching."
EPS = 1e-6


def bounds(mobject):
    """(left, right, bottom, top) in scene coordinates."""
    left, bottom, _ = mobject.get_corner([-1, -1, 0])
    right, top, _ = mobject.get_corner([1, 1, 0])
    return left, right, bottom, top


def overlaps(a, b) -> bool:
    a_left, a_right, a_bottom, a_top = bounds(a)
    b_left, b_right, b_bottom, b_top = bounds(b)
    return (
        a_left < b_right - EPS
        and a_right > b_left + EPS
        and a_bottom < b_top - EPS
        and a_top > b_bottom + EPS
    )
