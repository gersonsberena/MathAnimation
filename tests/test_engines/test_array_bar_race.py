"""Light tests for array_bar_race (README Section 6, #4): pure
param-validation, boundary, and frame-generation correctness checks only —
no Scene/render checks, per the current per-engine test cadence
(README Section 11.6).
"""

import pytest

from engines.array_bar_race import (
    ARRAY_LENGTH_RANGE,
    RATE_RANGE,
    SERIES_COUNT_RANGE,
    SPEED_RANGE,
    build_race,
    validate_params,
)


def _array(n):
    return list(range(n, 0, -1))  # reverse-sorted: worst case, exercises every swap


def _series(n):
    return [{"label": f"S{i}", "principal": 1000.0, "rate": 0.1} for i in range(n)]


# ---- Validation ----


def test_rejects_unknown_race_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_race", data_series=_array(5), labels=None, speed=1.0)


@pytest.mark.parametrize("race_type", ["bubble_sort", "quick_sort"])
def test_sort_rejects_array_length_below_min(race_type):
    lo, _hi = ARRAY_LENGTH_RANGE
    with pytest.raises(ValueError):
        validate_params(race_type, data_series=_array(lo - 1), labels=None, speed=1.0)


@pytest.mark.parametrize("race_type", ["bubble_sort", "quick_sort"])
def test_sort_rejects_array_length_above_max(race_type):
    _lo, hi = ARRAY_LENGTH_RANGE
    with pytest.raises(ValueError):
        validate_params(race_type, data_series=_array(hi + 1), labels=None, speed=1.0)


def test_sort_rejects_non_positive_values():
    with pytest.raises(ValueError):
        validate_params("bubble_sort", data_series=[3, -1, 5], labels=None, speed=1.0)


def test_compound_interest_rejects_series_count_below_min():
    lo, _hi = SERIES_COUNT_RANGE
    with pytest.raises(ValueError):
        validate_params("compound_interest", data_series=_series(lo - 1), labels=None, speed=1.0)


def test_compound_interest_rejects_series_count_above_max():
    _lo, hi = SERIES_COUNT_RANGE
    with pytest.raises(ValueError):
        validate_params("compound_interest", data_series=_series(hi + 1), labels=None, speed=1.0)


def test_compound_interest_rejects_non_positive_principal():
    series = [{"label": "A", "principal": 0, "rate": 0.1}, {"label": "B", "principal": 1000, "rate": 0.1}]
    with pytest.raises(ValueError):
        validate_params("compound_interest", data_series=series, labels=None, speed=1.0)


def test_compound_interest_rejects_rate_above_max():
    _lo, hi = RATE_RANGE
    series = [{"label": "A", "principal": 1000, "rate": hi + 0.1}, {"label": "B", "principal": 1000, "rate": 0.1}]
    with pytest.raises(ValueError):
        validate_params("compound_interest", data_series=series, labels=None, speed=1.0)


def test_rejects_mismatched_labels_length():
    with pytest.raises(ValueError):
        validate_params("bubble_sort", data_series=_array(5), labels=["a", "b"], speed=1.0)


def test_rejects_speed_below_min():
    lo, _hi = SPEED_RANGE
    with pytest.raises(ValueError):
        validate_params("bubble_sort", data_series=_array(5), labels=None, speed=lo - 0.01)


def test_rejects_speed_above_max():
    _lo, hi = SPEED_RANGE
    with pytest.raises(ValueError):
        validate_params("bubble_sort", data_series=_array(5), labels=None, speed=hi + 0.01)


# ---- Frame generation, at array-length param bounds ----


@pytest.mark.parametrize("race_type", ["bubble_sort", "quick_sort"])
def test_sort_frames_end_fully_sorted_at_min_length(race_type):
    lo, _hi = ARRAY_LENGTH_RANGE
    frames, _labels = build_race(race_type, _array(lo), labels=None, speed=1.0)
    assert frames[-1] == sorted(_array(lo))


@pytest.mark.parametrize("race_type", ["bubble_sort", "quick_sort"])
def test_sort_frames_end_fully_sorted_at_max_length(race_type):
    _lo, hi = ARRAY_LENGTH_RANGE
    frames, _labels = build_race(race_type, _array(hi), labels=None, speed=1.0)
    assert frames[-1] == sorted(_array(hi))


@pytest.mark.parametrize("race_type", ["bubble_sort", "quick_sort"])
def test_sort_frames_preserve_multiset_every_step(race_type):
    # Sorting only ever permutes values between fixed positions — the
    # multiset of values must be identical across every recorded frame.
    values = _array(7)
    frames, _labels = build_race(race_type, values, labels=None, speed=1.0)
    for frame in frames:
        assert sorted(frame) == sorted(values)


def test_compound_interest_frames_grow_monotonically():
    series = _series(3)
    frames, labels = build_race("compound_interest", series, labels=None, speed=1.0)
    assert labels == ["S0", "S1", "S2"]
    for prev, nxt in zip(frames, frames[1:]):
        assert all(n >= p for p, n in zip(prev, nxt))


def test_sort_default_labels_are_index_based():
    _frames, labels = build_race("bubble_sort", _array(4), labels=None, speed=1.0)
    assert labels == ["0", "1", "2", "3"]


def test_build_race_is_deterministic():
    a, _ = build_race("bubble_sort", _array(6), labels=None, speed=1.0)
    b, _ = build_race("bubble_sort", _array(6), labels=None, speed=1.0)
    assert a == b
