"""wave_signal engine (README Section 6, engine 9): square wave from
stacked sines (Fourier series build-up), interference patterns, standing
waves. Mechanic: superposition of waveforms rendered live.

`wave_components` (a list of `{amplitude, frequency, phase}` dicts) means
slightly different things per `superposition_type`, but the renderer is
the same for all three — reuses monte_carlo_probability's
Axes + ValueTracker + always_redraw growing-line-graph pattern, just
plotting a full waveform curve per frame instead of a partial rate line:

- fourier_square_wave: frames REVEAL one more harmonic at a time (a fixed
  x domain, no time axis) — the classic "watch it become a square wave"
  animation. If `wave_components` is None, the canonical odd-harmonic
  square-wave series is generated internally.
- interference / standing_wave: frames step through TIME with the FULL
  set of components active from frame 0 — wave propagation/oscillation,
  not term-count build-up.

`build_waveform()` is pure — no Scene dependency.
"""

import numpy as np
from manim import Axes, ValueTracker, always_redraw, linear

from engines.base import ReelScene

SUPERPOSITION_TYPES = {"fourier_square_wave", "interference", "standing_wave"}
WAVE_COMPONENTS_COUNT_RANGE = (1, 10)
DURATION_RANGE = (2.0, 10.0)

_X_DOMAIN = (0.0, 4 * np.pi)
_NUM_X_SAMPLES = 400
_NUM_TIME_FRAMES = 90
_DEFAULT_SQUARE_WAVE_HARMONICS = 9  # odd harmonics 1, 3, 5, ..., 17


def _default_square_wave_components(num_harmonics=_DEFAULT_SQUARE_WAVE_HARMONICS):
    return [{"amplitude": 4 / (np.pi * k), "frequency": k, "phase": 0.0} for k in range(1, 2 * num_harmonics, 2)]


def validate_params(superposition_type, wave_components, duration) -> None:
    if superposition_type not in SUPERPOSITION_TYPES:
        raise ValueError(
            f"unknown superposition_type {superposition_type!r}, must be one of {sorted(SUPERPOSITION_TYPES)}"
        )

    if wave_components is None:
        if superposition_type != "fourier_square_wave":
            raise ValueError(f"wave_components is required for superposition_type={superposition_type!r}")
    else:
        lo, hi = WAVE_COMPONENTS_COUNT_RANGE
        if not (lo <= len(wave_components) <= hi):
            raise ValueError(f"wave_components length {len(wave_components)} out of range [{lo}, {hi}]")
        for comp in wave_components:
            if comp.get("amplitude", 0) <= 0:
                raise ValueError(f"amplitude must be positive, got {comp.get('amplitude')!r}")
            if comp.get("frequency", 0) <= 0:
                raise ValueError(f"frequency must be positive, got {comp.get('frequency')!r}")
            if not isinstance(comp.get("phase", 0), (int, float)):
                raise ValueError(f"phase must be numeric, got {comp.get('phase')!r}")

    lo, hi = DURATION_RANGE
    if not (lo <= duration <= hi):
        raise ValueError(f"duration={duration} out of range [{lo}, {hi}]")


def _fourier_progressive_frames(wave_components, x):
    frames = []
    y = np.zeros_like(x)
    for comp in wave_components:
        y = y + comp["amplitude"] * np.sin(comp["frequency"] * x + comp.get("phase", 0.0))
        frames.append(y.copy())
    return np.stack(frames)


def _interference_time_frames(wave_components, x, duration):
    ts = np.linspace(0, duration, _NUM_TIME_FRAMES)
    frames = np.zeros((_NUM_TIME_FRAMES, x.shape[0]))
    for i, t in enumerate(ts):
        y = np.zeros_like(x)
        for comp in wave_components:
            k = comp["frequency"]
            y = y + comp["amplitude"] * np.sin(k * x - k * t + comp.get("phase", 0.0))
        frames[i] = y
    return frames


def _standing_wave_time_frames(wave_components, x, duration):
    ts = np.linspace(0, duration, _NUM_TIME_FRAMES)
    frames = np.zeros((_NUM_TIME_FRAMES, x.shape[0]))
    for i, t in enumerate(ts):
        y = np.zeros_like(x)
        for comp in wave_components:
            k = comp["frequency"]
            # standing wave: two equal, opposite-traveling waves collapse
            # to sin(kx) * cos(kt) — a fixed spatial pattern that only
            # oscillates in place, unlike interference's traveling sum.
            y = y + comp["amplitude"] * np.sin(k * x) * np.cos(k * t)
        frames[i] = y
    return frames


def build_waveform(superposition_type, wave_components, duration):
    """Returns `{"x": (num_x,) array, "y_frames": (num_frames, num_x)
    array}`. Pure — no Scene dependency.
    """
    validate_params(superposition_type, wave_components, duration)

    x = np.linspace(*_X_DOMAIN, _NUM_X_SAMPLES)

    if superposition_type == "fourier_square_wave":
        components = wave_components or _default_square_wave_components()
        y_frames = _fourier_progressive_frames(components, x)
    elif superposition_type == "interference":
        y_frames = _interference_time_frames(wave_components, x, duration)
    elif superposition_type == "standing_wave":
        y_frames = _standing_wave_time_frames(wave_components, x, duration)
    else:
        raise AssertionError(f"unhandled superposition_type {superposition_type!r}")  # validate_params already checked

    return {"x": x, "y_frames": y_frames}


class WaveSignalReel(ReelScene):
    """`params`: wave_components, superposition_type, duration (README
    Section 6, #9) — exposed here as `wave_duration`, NOT `duration`:
    Manim's own `Scene.__init__` sets `self.duration = 0.0` as an
    instance attribute for its internal bookkeeping, which silently
    shadows any subclass's `duration` class-level default the moment the
    Scene is constructed (confirmed by inspecting `Scene.__init__` — this
    isn't a maybe, it happens unconditionally). Any future orchestrator
    mapping this engine's `duration` recipe param onto the scene must set
    `wave_duration`, not `duration`.
    """

    superposition_type = "fourier_square_wave"
    wave_components = None
    wave_duration = 6.0
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "Simple Waves, Complex Shapes")

        data = build_waveform(self.superposition_type, self.wave_components, self.wave_duration)
        x, y_frames = data["x"], data["y_frames"]
        num_frames = y_frames.shape[0]
        y_max = max(np.abs(y_frames).max(), 1e-6)

        zone = self.zones.content_zone
        axes = Axes(
            x_range=[x[0], x[-1], (x[-1] - x[0]) / 4],
            y_range=[-y_max * 1.1, y_max * 1.1, y_max / 2],
            x_length=zone.width * 0.85,
            y_length=zone.height * 0.55,
            axis_config={"stroke_color": "#666666", "include_numbers": False},
        )
        axes.move_to(zone.get_center())

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), num_frames - 1))

        def make_curve():
            y = y_frames[frame_index()]
            return axes.plot_line_graph(x.tolist(), y.tolist(), add_vertex_dots=False, stroke_width=3)

        curve = always_redraw(make_curve)
        self.add(axes, curve)

        self.play(frame_tracker.animate.set_value(num_frames - 1), run_time=self.wave_duration, rate_func=linear)
        frame_tracker.set_value(num_frames - 1)
        self.wait(0.3)

        self.set_caption(self.caption_text or "Add enough waves, get any shape")
        self.wait(0.5)
