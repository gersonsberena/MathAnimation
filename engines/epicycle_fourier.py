"""epicycle_fourier engine (README Section 6, engine 3): Fourier-epicycle
path reveal, "guess the shape" guessing game. Mechanic: take a closed 2D path,
compute its discrete Fourier series, keep the largest-magnitude terms, and
animate nested epicycles whose combined tip traces the path. Difficulty levels
control hint opacity.

`build_epicycles()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, GRAY, Dot, Line, Polygon, ValueTracker, VGroup, VMobject, always_redraw, linear

from engines.base import ReelScene

PATH_SOURCES = {"star", "heart", "house", "infinity"}
DIFFICULTY_LEVELS = {"easy", "medium", "hard"}
NUM_CIRCLES_RANGE = (2, 50)
_HINT_OPACITY = {"easy": 0.15, "medium": 0.06, "hard": 0.0}
_PATH_LABELS = {"star": "A Five-Pointed Star", "heart": "A Heart", "house": "A House", "infinity": "An Infinity Symbol"}


def validate_params(path_source, num_circles, difficulty_level) -> None:
    if path_source not in PATH_SOURCES:
        raise ValueError(f"unknown path_source {path_source!r}, must be one of {sorted(PATH_SOURCES)}")
    if difficulty_level not in DIFFICULTY_LEVELS:
        raise ValueError(f"unknown difficulty_level {difficulty_level!r}, must be one of {sorted(DIFFICULTY_LEVELS)}")
    lo, hi = NUM_CIRCLES_RANGE
    if not (lo <= num_circles <= hi):
        raise ValueError(f"num_circles={num_circles} out of range [{lo}, {hi}]")


def _arclength_resample(vertices, num_samples):
    """Resample a closed polyline (vertices list) to have `num_samples` points
    distributed evenly by arc length. `vertices` is a list of (x, y) tuples
    or complex numbers; returns a numpy array of complex numbers.
    """
    if len(vertices) < 2:
        raise ValueError("need at least 2 vertices")

    vertices = np.array(vertices, dtype=complex)

    seg_lengths = np.abs(np.diff(vertices))
    total_length = np.sum(seg_lengths)

    if total_length < 1e-9:
        return np.full(num_samples, vertices[0], dtype=complex)

    cumulative = np.concatenate(([0], np.cumsum(seg_lengths)))
    target_distances = np.linspace(0, total_length, num_samples, endpoint=False)

    resampled = []
    for target_dist in target_distances:
        seg_idx = np.searchsorted(cumulative, target_dist) - 1
        seg_idx = np.clip(seg_idx, 0, len(vertices) - 2)

        dist_in_seg = target_dist - cumulative[seg_idx]
        seg_length = seg_lengths[seg_idx]
        if seg_length < 1e-9:
            t = 0
        else:
            t = dist_in_seg / seg_length

        point = vertices[seg_idx] + t * (vertices[seg_idx + 1] - vertices[seg_idx])
        resampled.append(point)

    return np.array(resampled, dtype=complex)


def _path_points(path_source, num_samples):
    """Returns a numpy array of `num_samples` complex numbers (x + 1j*y),
    evenly spaced in the parameter t over [0, 2*pi), tracing the named
    closed curve counterclockwise, roughly centered at the origin.
    Deterministic, no randomness.
    """
    t = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

    if path_source == "star":
        outer_r = 1.0
        inner_r = 0.4
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        vertices = []
        for i, angle in enumerate(angles):
            if i % 2 == 0:
                r = outer_r
            else:
                r = inner_r
            vertices.append(r * np.exp(1j * angle))
        vertices.append(vertices[0])
        return _arclength_resample(vertices, num_samples)

    elif path_source == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        x_norm = np.max(np.abs(x))
        y_norm = np.max(np.abs(y))
        x = x / max(x_norm, y_norm)
        y = y / max(x_norm, y_norm)
        return x + 1j * y

    elif path_source == "house":
        vertices = [
            -0.5 - 0.5j,
            0.5 - 0.5j,
            0.5 + 0.0j,
            0.0 + 0.5j,
            -0.5 + 0.0j,
        ]
        vertices.append(vertices[0])
        return _arclength_resample(vertices, num_samples)

    elif path_source == "infinity":
        x = np.cos(t) / (1 + np.sin(t) ** 2)
        y = np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
        return x + 1j * y

    else:
        raise AssertionError(f"unhandled path_source {path_source!r}")


def _fourier_coefficients(points, num_circles):
    """DFT via np.fft.fft(points)/len(points), paired with integer frequencies.
    Select the `num_circles` largest-magnitude coefficients, sorted by descending
    magnitude. Return a list of (frequency: int, coefficient: complex) tuples.
    """
    n = len(points)
    fft_out = np.fft.fft(points) / n
    freqs = np.fft.fftfreq(n, d=1.0 / n).astype(int)

    mag_freq_coeff = [(np.abs(fft_out[i]), freqs[i], fft_out[i]) for i in range(n)]
    mag_freq_coeff.sort(reverse=True, key=lambda x: x[0])

    selected = mag_freq_coeff[:num_circles]
    selected.sort(reverse=True, key=lambda x: x[0])

    return [(coeff[1], coeff[2]) for coeff in selected]


def build_epicycles(path_source, num_circles, difficulty_level, num_frames):
    """Pure — no Scene dependency. Returns a dict with:
      - "centers": np.ndarray shape (num_frames, num_circles, 2)
      - "radii": np.ndarray shape (num_circles,)
      - "traced": np.ndarray shape (num_frames, 2)
      - "target_points": np.ndarray shape (num_frames, 2)
      - "hint_opacity": float
    """
    validate_params(path_source, num_circles, difficulty_level)

    target_points_complex = _path_points(path_source, num_frames)
    target_points = np.column_stack((target_points_complex.real, target_points_complex.imag))

    coeffs = _fourier_coefficients(target_points_complex, num_circles)
    radii = np.array([np.abs(coeff[1]) for coeff in coeffs])

    t_values = np.linspace(0, 2 * np.pi, num_frames, endpoint=True)

    centers = np.zeros((num_frames, num_circles, 2))
    traced = np.zeros((num_frames, 2))

    for frame_idx, t in enumerate(t_values):
        pos = 0j
        for circle_idx, (freq, coeff) in enumerate(coeffs):
            centers[frame_idx, circle_idx] = np.array([pos.real, pos.imag])
            pos += coeff * np.exp(1j * freq * t)
        traced[frame_idx] = np.array([pos.real, pos.imag])

    hint_opacity = _HINT_OPACITY[difficulty_level]

    return {
        "centers": centers,
        "radii": radii,
        "traced": traced,
        "target_points": target_points,
        "hint_opacity": hint_opacity,
    }


class EpicycleFourierReel(ReelScene):
    """`params`: path_source, num_circles, difficulty_level (README
    Section 6, #3).
    """

    path_source = "star"
    num_circles = 15
    difficulty_level = "medium"
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "What Shape Is This?")

        num_frames = 240
        data = build_epicycles(self.path_source, self.num_circles, self.difficulty_level, num_frames)

        target_points = data["target_points"]
        xs = target_points[:, 0]
        ys = target_points[:, 1]
        span_x = max(np.max(xs) - np.min(xs), 1e-6)
        span_y = max(np.max(ys) - np.min(ys), 1e-6)

        zone = self.zones.content_zone
        scale = min(zone.width * 0.7 / span_x, zone.height * 0.6 / span_y)
        cx, cy = (np.min(xs) + np.max(xs)) / 2, (np.min(ys) + np.max(ys)) / 2
        zone_center = zone.get_center()

        def to_scene(point):
            return np.array([(point[0] - cx) * scale + zone_center[0], (point[1] - cy) * scale + zone_center[1], 0.0])

        if data["hint_opacity"] > 0:
            hint_points = [to_scene(target_points[i]) for i in range(len(target_points))]
            hint_polygon = Polygon(*hint_points, fill_opacity=data["hint_opacity"], stroke_width=0, color=BLUE)
            self.add(hint_polygon)

        frame_tracker = ValueTracker(0)
        centers = data["centers"]
        radii = data["radii"]
        traced = data["traced"]

        def draw_epicycles():
            frame_idx = int(np.clip(frame_tracker.get_value(), 0, num_frames - 1))
            circles_and_lines = VGroup()

            for i in range(len(radii)):
                center = to_scene(centers[frame_idx, i])
                radius = radii[i] * scale
                circle = Dot(center, radius=0.02, color=GRAY)
                circles_and_lines.add(circle)

                if i < len(radii) - 1:
                    next_center = to_scene(centers[frame_idx, i + 1])
                    line = Line(center, next_center, stroke_color=GRAY, stroke_width=1, stroke_opacity=0.5)
                    circles_and_lines.add(line)
                else:
                    traced_point = to_scene(traced[frame_idx])
                    line = Line(center, traced_point, stroke_color=GRAY, stroke_width=1, stroke_opacity=0.5)
                    circles_and_lines.add(line)

            return circles_and_lines

        def draw_trace():
            frame_idx = int(np.clip(frame_tracker.get_value(), 0, num_frames - 1))
            if frame_idx + 1 < 2:
                return VMobject()

            trace_points = [to_scene(traced[i]) for i in range(frame_idx + 1)]
            trace_path = VMobject()
            trace_path.set_points_as_corners(trace_points)
            trace_path.set_color(BLUE)
            trace_path.set_stroke(width=2)
            return trace_path

        epicycles_group = always_redraw(draw_epicycles)
        trace_group = always_redraw(draw_trace)

        self.add(epicycles_group, trace_group)

        self.play(frame_tracker.animate.set_value(num_frames - 1), run_time=6.0, rate_func=linear)
        frame_tracker.set_value(num_frames - 1)
        self.wait(0.3)

        epicycles_group.clear_updaters()
        trace_group.clear_updaters()

        final_epicycles = draw_epicycles()
        final_trace = draw_trace()
        self.remove(epicycles_group, trace_group)
        self.add(final_epicycles, final_trace)

        self.set_caption(self.caption_text or f"{_PATH_LABELS[self.path_source]} -- traced with {self.num_circles} circles")
