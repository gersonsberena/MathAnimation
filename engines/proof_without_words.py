"""proof_without_words engine (README Section 6, engine 16): Pythagorean
theorem via rearranged squares, circle-area unwrapped into a triangle.
Mechanic: shapes physically rearrange on screen to reveal an identity.

`proof_type` is the only param — side lengths / radius / step counts are
fixed internal constants, not params.

pythagorean_rearrangement uses the classic two-dissection proof: a
(a+b)x(a+b) square split into 4 congruent right triangles (legs a, b,
hypotenuse c) two different ways — once leaving a single tilted c^2
square in the middle, once leaving two axis-aligned a^2/b^2 squares in
opposite corners. Both dissections cover the exact same (a+b)^2 area, so
c^2 = a^2 + b^2. Uses the classic 3-4-5 triple for a clean, exact,
integer-area demo. Vertex coordinates are hand-derived once in this
docstring's sibling module comment and cross-checked by
test_proof_without_words.py via the shoelace formula (same
area-conservation-style verification tessellation_growth's Penrose
tiling uses) rather than trusted from memory alone.

circle_area_unwrapping slices the disk into `_NUM_RINGS` concentric
rings and straightens each ring of *midpoint* radius rho into a
rectangle of width `2*pi*rho` and height `dr` — using the ring's
midpoint radius (not inner or outer) makes each strip's area *exactly*
equal the true ring area (pi*(outer^2 - inner^2)) for any ring count, not
just in the limit (see test_ring_strip_area_exactly_matches_true_ring_area).
Stacking strips narrowest-to-widest forms a triangle of base `2*pi*r`
and height `r` in the limit — area = 1/2 * base * height = pi*r^2,
matching README's literal "unwrapped into a triangle" phrasing (this is
a different classic proof from the sector-rearranged-into-a-parallelogram
one; picked for lower implementation risk, since the ring-to-strip area
match is exact and telescopes cleanly rather than depending on a
sector-zigzag edge-interlock derivation).

`build_proof()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, ORANGE, WHITE, YELLOW, Annulus, FadeIn, FadeOut, Polygon, Rectangle, Text, Transform, VGroup

from engines.base import ReelScene

PROOF_TYPES = {"pythagorean_rearrangement", "circle_area_unwrapping"}

# pythagorean_rearrangement constants — the classic 3-4-5 triple.
_A, _B, _C = 3.0, 4.0, 5.0

# circle_area_unwrapping constants.
_CIRCLE_RADIUS = 3.0
_NUM_RINGS = 24


def validate_params(proof_type) -> None:
    if proof_type not in PROOF_TYPES:
        raise ValueError(f"unknown proof_type {proof_type!r}, must be one of {sorted(PROOF_TYPES)}")


def _polygon_area(vertices):
    """Shoelace formula. `vertices` is a list of (x, y) pairs."""
    n = len(vertices)
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _pythagorean_config1(a, b):
    """Big square dissected into 4 triangles (legs a, b) around a tilted
    inner square of side c — revealing c^2.
    """
    s = a + b
    p1, p2, p3, p4 = (a, 0.0), (s, a), (b, s), (0.0, b)
    corners = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    triangles = [
        [corners[0], p1, p4],
        [corners[1], p2, p1],
        [corners[2], p3, p2],
        [corners[3], p4, p3],
    ]
    return {"triangles": triangles, "inner_square": [p1, p2, p3, p4]}


def _pythagorean_config2(a, b):
    """Same big square, same 4 triangles, rearranged into two opposite
    corners — revealing an a^2 square and a b^2 square. The two
    remaining a x b rectangles (each split along its diagonal into 2
    right triangles with legs a, b) supply the triangles.
    """
    s = a + b
    square_a = [(0.0, 0.0), (a, 0.0), (a, a), (0.0, a)]
    square_b = [(a, a), (s, a), (s, s), (a, s)]
    # Rectangle (0, a)-(a, s), split along its diagonal (0,s)-(a,a).
    rect1_tri1 = [(0.0, a), (a, a), (0.0, s)]
    rect1_tri2 = [(a, a), (a, s), (0.0, s)]
    # Rectangle (a, 0)-(s, a), split along its diagonal (a,0)-(s,a).
    rect2_tri1 = [(a, 0.0), (s, 0.0), (a, a)]
    rect2_tri2 = [(s, 0.0), (s, a), (a, a)]
    return {
        "triangles": [rect1_tri1, rect1_tri2, rect2_tri1, rect2_tri2],
        "square_a": square_a,
        "square_b": square_b,
    }


def _ring_strip_geometry(radius, num_rings):
    dr = radius / num_rings
    strips = []
    for i in range(num_rings):
        inner, outer = i * dr, (i + 1) * dr
        rho = (i + 0.5) * dr  # midpoint radius — makes strip area exact, see module docstring
        strips.append({"inner_radius": inner, "outer_radius": outer, "width": 2 * np.pi * rho, "height": dr})
    return strips


def build_proof(proof_type):
    """Returns a dict discriminated by `kind`:
      - pythagorean_rearrangement: `{"kind": "pythagorean", "a", "b", "c",
        "config1", "config2"}`
      - circle_area_unwrapping: `{"kind": "circle_unwrap", "radius",
        "strips"}`
    Pure — no Scene dependency.
    """
    validate_params(proof_type)

    if proof_type == "pythagorean_rearrangement":
        return {
            "kind": "pythagorean",
            "a": _A,
            "b": _B,
            "c": _C,
            "config1": _pythagorean_config1(_A, _B),
            "config2": _pythagorean_config2(_A, _B),
        }

    if proof_type == "circle_area_unwrapping":
        return {
            "kind": "circle_unwrap",
            "radius": _CIRCLE_RADIUS,
            "strips": _ring_strip_geometry(_CIRCLE_RADIUS, _NUM_RINGS),
        }

    raise AssertionError(f"unhandled proof_type {proof_type!r}")  # validate_params already checked


class ProofWithoutWordsReel(ReelScene):
    """`params`: proof_type (README Section 6, #16)."""

    proof_type = "pythagorean_rearrangement"
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        data = build_proof(self.proof_type)

        if data["kind"] == "pythagorean":
            self._construct_pythagorean(data)
        else:
            self._construct_circle_unwrap(data)

    def _construct_pythagorean(self, data):
        self.set_title(self.title_text or "A Square Proof")
        zone = self.zones.content_zone

        a, b, c = data["a"], data["b"], data["c"]
        s = a + b
        scale = min(zone.width, zone.height) * 0.75 / s
        origin = np.array([zone.get_center()[0] - s * scale / 2, zone.get_center()[1] - s * scale / 2, 0.0])

        def to_scene(pt):
            return origin + np.array([pt[0] * scale, pt[1] * scale, 0.0])

        def make_polygon(vertices, color, opacity=0.85):
            return Polygon(*[to_scene(v) for v in vertices], fill_color=color, fill_opacity=opacity, stroke_color=WHITE, stroke_width=2)

        config1 = data["config1"]
        triangles1 = VGroup(*[make_polygon(t, BLUE) for t in config1["triangles"]])
        inner_square = make_polygon(config1["inner_square"], YELLOW)
        c_label = Text(f"c² = {c ** 2:.0f}", font_size=28, color=WHITE).move_to(inner_square.get_center())

        self.add(triangles1, inner_square, c_label)
        self.wait(1.5)

        config2 = data["config2"]
        triangles2 = VGroup(*[make_polygon(t, BLUE) for t in config2["triangles"]])
        square_a = make_polygon(config2["square_a"], ORANGE)
        square_b = make_polygon(config2["square_b"], YELLOW)
        a_label = Text(f"a² = {a ** 2:.0f}", font_size=24, color=WHITE).move_to(square_a.get_center())
        b_label = Text(f"b² = {b ** 2:.0f}", font_size=24, color=WHITE).move_to(square_b.get_center())

        # 4 triangles map 1:1 between the two configs, so each gets its
        # own Transform; the "revealed" middle shape has a different
        # piece count in each config (1 square vs. 2), so it crossfades
        # instead of trying to Transform between mismatched shapes.
        self.play(
            *[Transform(triangles1[i], triangles2[i]) for i in range(4)],
            FadeOut(inner_square, c_label),
            FadeIn(square_a, square_b, a_label, b_label),
            run_time=2.5,
        )
        self.wait(1.5)

        self.set_caption(self.caption_text or f"{a:.0f}² + {b:.0f}² = {c:.0f}². Same square, rearranged.")
        self.wait(0.5)

    def _construct_circle_unwrap(self, data):
        self.set_title(self.title_text or "Unwrapping a Circle")
        zone = self.zones.content_zone
        radius, strips = data["radius"], data["strips"]

        scale = min(zone.width, zone.height) * 0.7 / (2 * radius)
        circle_center = zone.get_center() + np.array([0.0, zone.height * 0.15, 0.0])

        rings = VGroup(
            *[
                Annulus(
                    inner_radius=max(s["inner_radius"] * scale, 0.001),
                    outer_radius=s["outer_radius"] * scale,
                    fill_color=BLUE if i % 2 == 0 else ORANGE,
                    fill_opacity=0.85,
                    stroke_width=0,
                ).move_to(circle_center)
                for i, s in enumerate(strips)
            ]
        )
        self.add(rings)
        self.wait(1.0)

        triangle_base = 2 * np.pi * radius * scale
        triangle_height = radius * scale
        base_y = circle_center[1] - triangle_height / 2
        strip_rects = VGroup()
        cumulative_y = base_y
        for i, s in enumerate(strips):
            height = s["height"] * scale
            width = s["width"] * scale
            rect = Rectangle(
                width=width, height=height, fill_color=BLUE if i % 2 == 0 else ORANGE, fill_opacity=0.85, stroke_width=0
            )
            rect.move_to([circle_center[0], cumulative_y + height / 2, 0.0])
            strip_rects.add(rect)
            cumulative_y += height

        self.play(Transform(rings, strip_rects), run_time=3.0)
        self.wait(1.0)

        self.set_caption(self.caption_text or f"½ × 2πr × r = πr² ≈ {np.pi * radius ** 2:.1f}")
        self.wait(0.5)
