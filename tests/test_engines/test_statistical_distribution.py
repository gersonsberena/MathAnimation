"""Light tests for statistical_distribution (README Section 6, #13): pure
param-validation, boundary, and distribution-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.statistical_distribution import (
    NUM_TRIALS_RANGE,
    build_histogram_trials,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_distribution_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_distribution", num_trials=500, seed=0)


def test_rejects_num_trials_below_min():
    lo, _hi = NUM_TRIALS_RANGE
    with pytest.raises(ValueError):
        validate_params("galton_board", num_trials=lo - 1, seed=0)


def test_rejects_num_trials_above_max():
    _lo, hi = NUM_TRIALS_RANGE
    with pytest.raises(ValueError):
        validate_params("galton_board", num_trials=hi + 1, seed=0)


def test_rejects_negative_seed():
    with pytest.raises(ValueError):
        validate_params("galton_board", num_trials=500, seed=-1)


# ---- Distribution correctness ----


def test_galton_board_bins_are_within_expected_range():
    bins, num_bins = build_histogram_trials("galton_board", num_trials=1000, seed=0)
    assert num_bins == 13  # 12 rows -> 0..12 possible right-moves
    assert (bins >= 0).all() and (bins <= 12).all()


def test_galton_board_histogram_is_roughly_bell_shaped():
    # Binomial(12, 0.5): the middle bin (6) should be the mode, and it
    # should be visited more often than either extreme (0 or 12) over
    # enough trials.
    bins, _num_bins = build_histogram_trials("galton_board", num_trials=3000, seed=0)
    counts = np.bincount(bins, minlength=13)
    assert counts[6] == counts.max()
    assert counts[6] > counts[0]
    assert counts[6] > counts[12]


def test_dice_sum_histogram_bins_are_within_expected_range():
    bins, num_bins = build_histogram_trials("dice_sum_histogram", num_trials=1000, seed=0)
    assert num_bins == 11  # sums 2..12 -> 11 bins
    assert (bins >= 0).all() and (bins <= 10).all()


def test_dice_sum_histogram_peaks_at_seven():
    # sum=7 (bin index 5) is the most likely two-dice sum (6/36).
    bins, _num_bins = build_histogram_trials("dice_sum_histogram", num_trials=3000, seed=0)
    counts = np.bincount(bins, minlength=11)
    assert counts.argmax() == 5


def test_clt_demo_bins_are_within_expected_range():
    bins, num_bins = build_histogram_trials("clt_demo", num_trials=1000, seed=0)
    assert (bins >= 0).all() and (bins < num_bins).all()


def test_clt_demo_concentrates_near_the_theoretical_mean():
    # Sum of 12 U(0,1) vars has mean 6, out of a [0, 12] range split into
    # 25 bins -> the theoretical mean falls in the middle few bins.
    bins, num_bins = build_histogram_trials("clt_demo", num_trials=3000, seed=0)
    counts = np.bincount(bins, minlength=num_bins)
    peak_bin = counts.argmax()
    assert abs(peak_bin - num_bins / 2) < num_bins * 0.2


def test_build_histogram_trials_is_deterministic():
    a, _ = build_histogram_trials("dice_sum_histogram", num_trials=500, seed=42)
    b, _ = build_histogram_trials("dice_sum_histogram", num_trials=500, seed=42)
    assert (a == b).all()


def test_build_histogram_trials_differs_across_seeds():
    a, _ = build_histogram_trials("galton_board", num_trials=500, seed=1)
    b, _ = build_histogram_trials("galton_board", num_trials=500, seed=2)
    assert not (a == b).all()
