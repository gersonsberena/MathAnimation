"""number_pattern engine (README Section 6, engine 11): Pascal's triangle
mod 2, digits of pi colored by value, modular "times table" circle
patterns. Mechanic: sequence -> visual mapping.

pascals_triangle_mod2 reuses grid_cell_coloring's rasterized-ImageMobject
pattern (revealed row by row, matching Pascal's triangle's natural
row-by-row generation) — the mod-2 parity test uses the Kummer/Lucas
theorem shortcut `(n & k) == k` rather than computing (potentially huge)
binomial coefficients directly.

pi_digits uses a hardcoded, independently-verifiable 51-digit expansion
of pi rather than a hand-derived arbitrary-precision algorithm — a wrong
custom Machin/Chudnovsky implementation could silently produce
plausible-looking but incorrect digits with no way to self-check; a
literal, well-known reference string carries no such risk (see
PI_DIGITS below and its correctness test).

`build_number_pattern()` is pure — no Scene dependency.
"""

import colorsys

import numpy as np
from manim import BLUE, Dot, ImageMobject, Line, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

SEQUENCE_TYPES = {"pascals_triangle_mod2", "pi_digits", "times_table_circle"}

# The standard 50-decimal-digit expansion of pi (see e.g. Wikipedia's Pi
# article, "Approximate value" section) plus the leading "3" — 51 digits
# total, indexed 0..50.
PI_DIGITS = "314159265358979323846264338327950288419716939937510"

MAPPING_RULES = {
    "pascals_triangle_mod2": {"binary"},
    "pi_digits": {"color_by_value", "polar_walk"},
    "times_table_circle": {"times_2", "times_3", "times_5", "times_7"},
}
LENGTH_RANGES = {
    "pascals_triangle_mod2": (16, 256),
    "pi_digits": (10, len(PI_DIGITS)),
    "times_table_circle": (20, 300),
}
_TIMES_TABLE_MULTIPLIERS = {"times_2": 2, "times_3": 3, "times_5": 5, "times_7": 7}


def validate_params(sequence_type, length, mapping_rule) -> None:
    if sequence_type not in SEQUENCE_TYPES:
        raise ValueError(f"unknown sequence_type {sequence_type!r}, must be one of {sorted(SEQUENCE_TYPES)}")

    valid_rules = MAPPING_RULES[sequence_type]
    if mapping_rule not in valid_rules:
        raise ValueError(
            f"mapping_rule {mapping_rule!r} invalid for sequence_type={sequence_type!r}, "
            f"must be one of {sorted(valid_rules)}"
        )

    lo, hi = LENGTH_RANGES[sequence_type]
    if not (lo <= length <= hi):
        raise ValueError(f"length={length} out of range [{lo}, {hi}] for sequence_type={sequence_type!r}")


def _pascal_mod2_raster(num_rows):
    grid = np.zeros((num_rows, num_rows), dtype=bool)
    for n in range(num_rows):
        for k in range(n + 1):
            grid[n, k] = (n & k) == k  # C(n, k) mod 2 == 1, via Lucas' theorem for p=2
    return grid


def digit_to_rgb(digit):
    hue = digit / 10.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _pi_digit_positions(digits, mapping_rule):
    if mapping_rule == "color_by_value":
        return [(i % 10, -(i // 10)) for i in range(len(digits))]

    if mapping_rule == "polar_walk":
        positions = []
        x, y, angle = 0.0, 0.0, 0.0
        for d in digits:
            angle += d * 36.0  # 360 / 10 possible digit values
            rad = np.radians(angle)
            x += np.cos(rad)
            y += np.sin(rad)
            positions.append((x, y))
        return positions

    raise AssertionError(f"unhandled mapping_rule {mapping_rule!r}")  # validate_params already checked


def _times_table_circle(length, k):
    points = [(np.cos(2 * np.pi * i / length), np.sin(2 * np.pi * i / length)) for i in range(length)]
    lines = [(i, (i * k) % length) for i in range(length)]
    return points, lines


def build_number_pattern(sequence_type, length, mapping_rule):
    """Returns a dict discriminated by `kind`:
      - pascals_triangle_mod2: `{"kind": "raster", "image": (N, N) bool array}`
      - pi_digits: `{"kind": "points", "positions": [...], "colors": [...]}`
      - times_table_circle: `{"kind": "circle_lines", "points": [...], "lines": [(i, j), ...]}`
    Pure — no Scene dependency.
    """
    validate_params(sequence_type, length, mapping_rule)

    if sequence_type == "pascals_triangle_mod2":
        return {"kind": "raster", "image": _pascal_mod2_raster(length)}

    if sequence_type == "pi_digits":
        digits = [int(c) for c in PI_DIGITS[:length]]
        positions = _pi_digit_positions(digits, mapping_rule)
        colors = [digit_to_rgb(d) for d in digits]
        return {"kind": "points", "positions": positions, "colors": colors}

    if sequence_type == "times_table_circle":
        k = _TIMES_TABLE_MULTIPLIERS[mapping_rule]
        points, lines = _times_table_circle(length, k)
        return {"kind": "circle_lines", "points": points, "lines": lines}

    raise AssertionError(f"unhandled sequence_type {sequence_type!r}")  # validate_params already checked


class NumberPatternReel(ReelScene):
    """`params`: sequence_type, length, mapping_rule (README Section 6,
    #11).
    """

    sequence_type = "pascals_triangle_mod2"
    length = 128
    mapping_rule = "binary"

    def construct(self):
        self.set_title("Numbers Hide a Pattern")

        data = build_number_pattern(self.sequence_type, self.length, self.mapping_rule)
        zone = self.zones.content_zone

        if data["kind"] == "raster":
            self._construct_raster(data["image"], zone)
        elif data["kind"] == "points":
            self._construct_points(data["positions"], data["colors"], zone)
        else:
            self._construct_circle_lines(data["points"], data["lines"], zone)

        self.set_caption("Simple numbers. Hidden structure.")

    def _construct_raster(self, full_image, zone):
        num_rows = full_image.shape[0]
        frame_tracker = ValueTracker(0)

        def make_image():
            row = max(1, min(int(round(frame_tracker.get_value())), num_rows))
            partial = np.zeros_like(full_image)
            partial[:row, :] = full_image[:row, :]
            pixels = partial.astype(np.uint8) * 255
            rgb = np.stack([pixels] * 3, axis=-1)
            image = ImageMobject(rgb)
            image.height = zone.height * 2
            self.fit_to_zone(image, zone)
            return image

        image = always_redraw(make_image)
        self.add(image)
        self.play(frame_tracker.animate.set_value(num_rows), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(num_rows)
        self.wait(0.3)

    def _construct_points(self, positions, colors, zone):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        scale = min(zone.width * 0.85 / span_x, zone.height * 0.6 / span_y)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        zone_center = zone.get_center()

        def to_scene(point):
            return [(point[0] - cx) * scale + zone_center[0], (point[1] - cy) * scale + zone_center[1], 0.0]

        frame_tracker = ValueTracker(0)
        dots = VGroup()
        for i, (pos, color) in enumerate(zip(positions, colors)):
            dot = Dot(to_scene(pos), radius=0.05, color=rgb_to_hex(color))

            def make_updater(i=i):
                def updater(mob):
                    mob.set_opacity(1.0 if frame_tracker.get_value() >= i else 0.0)

                return updater

            dot.add_updater(make_updater())
            dots.add(dot)

        self.add(dots)
        self.play(frame_tracker.animate.set_value(len(positions)), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(len(positions))
        self.wait(0.3)
        for dot in dots:
            dot.clear_updaters()

    def _construct_circle_lines(self, points, lines, zone):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span_x = max(max(xs) - min(xs), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        scale = min(zone.width * 0.8 / span_x, zone.height * 0.6 / span_y)
        zone_center = zone.get_center()

        def to_scene(point):
            return [point[0] * scale + zone_center[0], point[1] * scale + zone_center[1], 0.0]

        frame_tracker = ValueTracker(0)
        segments = VGroup()
        for idx, (i, j) in enumerate(lines):
            line = Line(to_scene(points[i]), to_scene(points[j]), stroke_color=BLUE, stroke_width=1, stroke_opacity=0)

            def make_updater(idx=idx):
                def updater(mob):
                    mob.set_stroke(opacity=0.6 if frame_tracker.get_value() >= idx else 0)

                return updater

            line.add_updater(make_updater())
            segments.add(line)

        self.add(segments)
        self.play(frame_tracker.animate.set_value(len(lines)), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(len(lines))
        self.wait(0.3)
        for segment in segments:
            segment.clear_updaters()
