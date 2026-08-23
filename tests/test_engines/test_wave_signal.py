"""Light tests for wave_signal (README Section 6, #9): pure
param-validation, boundary, and waveform-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.wave_signal import (
    DURATION_RANGE,
    WAVE_COMPONENTS_COUNT_RANGE,
    WaveSignalReel,
    _default_square_wave_components,
    build_waveform,
    validate_params,
)


def _two_waves():
    return [
        {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0},
        {"amplitude": 0.5, "frequency": 3.0, "phase": 0.0},
    ]


# ---- Validation ----


def test_rejects_unknown_superposition_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_type", wave_components=_two_waves(), duration=5.0)


def test_interference_requires_wave_components():
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=None, duration=5.0)


def test_standing_wave_requires_wave_components():
    with pytest.raises(ValueError):
        validate_params("standing_wave", wave_components=None, duration=5.0)


def test_fourier_square_wave_allows_none_wave_components():
    # falls back to the canonical odd-harmonic series
    validate_params("fourier_square_wave", wave_components=None, duration=5.0)


def test_rejects_wave_components_below_min():
    lo, _hi = WAVE_COMPONENTS_COUNT_RANGE
    assert lo > 0  # otherwise an empty list would be a no-op, not a rejection
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=[], duration=5.0)


def test_rejects_wave_components_above_max():
    _lo, hi = WAVE_COMPONENTS_COUNT_RANGE
    components = [{"amplitude": 1.0, "frequency": float(i + 1), "phase": 0.0} for i in range(hi + 1)]
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=components, duration=5.0)


def test_rejects_non_positive_amplitude():
    components = [{"amplitude": 0, "frequency": 1.0, "phase": 0.0}]
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=components, duration=5.0)


def test_rejects_non_positive_frequency():
    components = [{"amplitude": 1.0, "frequency": 0, "phase": 0.0}]
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=components, duration=5.0)


def test_rejects_duration_below_min():
    lo, _hi = DURATION_RANGE
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=_two_waves(), duration=lo - 0.1)


def test_rejects_duration_above_max():
    _lo, hi = DURATION_RANGE
    with pytest.raises(ValueError):
        validate_params("interference", wave_components=_two_waves(), duration=hi + 0.1)


# ---- Waveform correctness ----


def test_fourier_square_wave_frame_count_matches_harmonic_count():
    data = build_waveform("fourier_square_wave", wave_components=None, duration=5.0)
    assert data["y_frames"].shape[0] == len(_default_square_wave_components())


def test_fourier_square_wave_final_frame_approximates_a_square_wave():
    # At x slightly past 0 (rising edge already passed), the partial sum
    # should be close to +1 (the square wave's plateau value); near the
    # midpoint (x=pi, just after a falling edge) it should be close to -1.
    data = build_waveform("fourier_square_wave", wave_components=None, duration=5.0)
    x, y_frames = data["x"], data["y_frames"]
    final = y_frames[-1]
    near_quarter = np.argmin(np.abs(x - np.pi / 2))
    near_three_quarter = np.argmin(np.abs(x - 3 * np.pi / 2))
    assert final[near_quarter] > 0.8
    assert final[near_three_quarter] < -0.8


def test_fourier_square_wave_converges_as_more_harmonics_are_added():
    # Error against the ideal square wave should shrink (in an L2 sense)
    # as more terms are included — that's the point of a Fourier series.
    data = build_waveform("fourier_square_wave", wave_components=None, duration=5.0)
    x, y_frames = data["x"], data["y_frames"]
    ideal = np.sign(np.sin(x))
    errors = [np.mean((frame - ideal) ** 2) for frame in y_frames]
    assert errors[-1] < errors[0]


def test_interference_frame_count_and_shape():
    data = build_waveform("interference", wave_components=_two_waves(), duration=4.0)
    assert data["y_frames"].shape[1] == data["x"].shape[0]
    assert data["y_frames"].shape[0] == 90


def test_standing_wave_has_fixed_nodes_over_time():
    # A standing wave sin(kx)*cos(kt) has zero displacement at every x
    # where sin(kx)=0, for ALL t — that's what distinguishes it from a
    # traveling wave (interference), whose zero-crossings move over time.
    components = [{"amplitude": 1.0, "frequency": 2.0, "phase": 0.0}]
    data = build_waveform("standing_wave", wave_components=components, duration=4.0)
    x = data["x"]
    node_idx = np.argmin(np.abs(np.sin(2.0 * x)))  # x where sin(2x) ~ 0
    values_at_node_over_time = data["y_frames"][:, node_idx]
    assert np.abs(values_at_node_over_time).max() < 1e-6


def test_build_waveform_is_deterministic():
    a = build_waveform("interference", wave_components=_two_waves(), duration=4.0)
    b = build_waveform("interference", wave_components=_two_waves(), duration=4.0)
    assert (a["y_frames"] == b["y_frames"]).all()


def test_scene_duration_attribute_survives_construction():
    # Regression guard: Manim's Scene.__init__ unconditionally sets
    # self.duration = 0.0 for its own internal bookkeeping, which would
    # silently shadow a class-level `duration` default the moment the
    # Scene is instantiated — the reason this engine's Scene attribute is
    # named `wave_duration`, not `duration`. Confirm it isn't reset.
    scene = WaveSignalReel()
    assert scene.wave_duration == WaveSignalReel.wave_duration
    assert scene.wave_duration != 0.0
