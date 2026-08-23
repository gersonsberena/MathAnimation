"""strange_attractor_chaos engine (README Section 6, engine 14): Lorenz
attractor, logistic map bifurcation diagram. A cheap extension of
particle_physics_sim (#2) per README Section 7's priority note — the
Lorenz system reuses that engine's exact dot+trail/ValueTracker rendering
pattern; the logistic map's bifurcation diagram instead reuses
grid_cell_coloring's (#6) rasterized-ImageMobject pattern, since it's a
point cloud (r, x) rather than a single trajectory, and revealing it
column-by-column (as r increases) is the standard way this diagram is
shown.

`build_attractor()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, WHITE, Dot, ImageMobject, TracedPath, ValueTracker, always_redraw, linear

from engines.base import ReelScene

SYSTEM_TYPES = {"lorenz", "logistic_map"}
LORENZ_ITERATIONS_RANGE = (500, 5000)
LOGISTIC_ITERATIONS_RANGE = (100, 1000)  # number of r-samples across the sweep

_LORENZ_SIGMA, _LORENZ_RHO, _LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
_LORENZ_DT = 0.01
_LOGISTIC_TRANSIENT = 200
_LOGISTIC_SAMPLES_PER_R = 40
_LOGISTIC_RASTER_SIZE = 300


def validate_params(system_type, initial_conditions, iterations) -> None:
    if system_type not in SYSTEM_TYPES:
        raise ValueError(f"unknown system_type {system_type!r}, must be one of {sorted(SYSTEM_TYPES)}")

    if initial_conditions is not None and not isinstance(initial_conditions, dict):
        raise ValueError(f"initial_conditions must be a dict or None, got {initial_conditions!r}")

    range_by_type = {"lorenz": LORENZ_ITERATIONS_RANGE, "logistic_map": LOGISTIC_ITERATIONS_RANGE}
    lo, hi = range_by_type[system_type]
    if not (lo <= iterations <= hi):
        raise ValueError(f"iterations={iterations} out of range [{lo}, {hi}] for system_type={system_type!r}")


def _simulate_lorenz(initial_conditions, iterations):
    ic = initial_conditions or {}
    x, y, z = ic.get("x", 0.1), ic.get("y", 0.0), ic.get("z", 0.0)

    trajectory = np.zeros((iterations, 2))  # (x, z) projection — the classic "butterfly" view
    for i in range(iterations):
        trajectory[i] = (x, z)
        dx = _LORENZ_SIGMA * (y - x)
        dy = x * (_LORENZ_RHO - z) - y
        dz = x * y - _LORENZ_BETA * z
        x += dx * _LORENZ_DT
        y += dy * _LORENZ_DT
        z += dz * _LORENZ_DT

    return trajectory


def _simulate_logistic_bifurcation_points(initial_conditions, num_r):
    ic = initial_conditions or {}
    x0 = ic.get("x0", 0.5)
    r_min = ic.get("r_min", 2.5)
    r_max = ic.get("r_max", 4.0)

    r_values = np.linspace(r_min, r_max, num_r)
    rs = np.empty(num_r * _LOGISTIC_SAMPLES_PER_R)
    xs = np.empty(num_r * _LOGISTIC_SAMPLES_PER_R)

    idx = 0
    for r in r_values:
        x = x0
        for _ in range(_LOGISTIC_TRANSIENT):
            x = r * x * (1 - x)
        for _ in range(_LOGISTIC_SAMPLES_PER_R):
            x = r * x * (1 - x)
            rs[idx] = r
            xs[idx] = x
            idx += 1

    return rs, xs, r_min, r_max


def _rasterize_bifurcation(rs, xs, r_min, r_max, grid_size=_LOGISTIC_RASTER_SIZE):
    image = np.zeros((grid_size, grid_size), dtype=bool)
    col = np.clip(((rs - r_min) / (r_max - r_min) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    row = np.clip(((1 - xs) * (grid_size - 1)).astype(int), 0, grid_size - 1)  # flip so x=1 is at top
    image[row, col] = True
    return image


def build_attractor(system_type, initial_conditions, iterations):
    """Returns a dict discriminated by `kind`:
      - lorenz: `{"kind": "trajectory", "points": (iterations, 2) array}`
      - logistic_map: `{"kind": "raster", "image": (N, N) bool array}`
    Pure — no Scene dependency.
    """
    validate_params(system_type, initial_conditions, iterations)

    if system_type == "lorenz":
        return {"kind": "trajectory", "points": _simulate_lorenz(initial_conditions, iterations)}

    if system_type == "logistic_map":
        rs, xs, r_min, r_max = _simulate_logistic_bifurcation_points(initial_conditions, iterations)
        image = _rasterize_bifurcation(rs, xs, r_min, r_max)
        return {"kind": "raster", "image": image}

    raise AssertionError(f"unhandled system_type {system_type!r}")  # validate_params already checked


class StrangeAttractorChaosReel(ReelScene):
    """`params`: system_type, initial_conditions, iterations (README
    Section 6, #14).
    """

    system_type = "lorenz"
    initial_conditions = None
    iterations = 2000
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "Order Hidden in Chaos")

        data = build_attractor(self.system_type, self.initial_conditions, self.iterations)
        zone = self.zones.content_zone

        if data["kind"] == "trajectory":
            self._construct_trajectory(data["points"], zone)
        else:
            self._construct_raster(data["image"], zone)

        self.set_caption(self.caption_text or "Simple rule. Never repeats.")

    def _construct_trajectory(self, points, zone):
        num_frames = points.shape[0]
        mins, maxs = points.min(axis=0), points.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        scale = min(zone.width * 0.9 / span[0], zone.height * 0.9 / span[1])
        center_world = (mins + maxs) / 2
        center_zone = zone.get_center()

        def to_scene(point):
            return np.array(
                [
                    (point[0] - center_world[0]) * scale + center_zone[0],
                    (point[1] - center_world[1]) * scale + center_zone[1],
                    0.0,
                ]
            )

        frame_tracker = ValueTracker(0)
        dot = Dot(color=BLUE, radius=0.05)
        dot.move_to(to_scene(points[0]))

        def updater(mob):
            idx = max(0, min(int(round(frame_tracker.get_value())), num_frames - 1))
            mob.move_to(to_scene(points[idx]))

        dot.add_updater(updater)
        trail = TracedPath(dot.get_center, stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.7)

        self.add(trail, dot)
        self.play(frame_tracker.animate.set_value(num_frames - 1), run_time=6.0, rate_func=linear)
        frame_tracker.set_value(num_frames - 1)
        self.wait(0.3)
        dot.clear_updaters()

    def _construct_raster(self, full_image, zone):
        grid_size = full_image.shape[1]
        frame_tracker = ValueTracker(0)

        def make_image():
            col = max(1, min(int(round(frame_tracker.get_value())), grid_size))
            partial = np.zeros_like(full_image)
            partial[:, :col] = full_image[:, :col]
            pixels = partial.astype(np.uint8) * 255
            rgb = np.stack([pixels] * 3, axis=-1)
            image = ImageMobject(rgb)
            image.height = zone.height * 2
            self.fit_to_zone(image, zone)
            return image

        image = always_redraw(make_image)
        self.add(image)
        self.play(frame_tracker.animate.set_value(grid_size), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(grid_size)
        self.wait(0.3)
