"""Light tests for monte_carlo_probability (README Section 6, #5): pure
param-validation, boundary, and convergence-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.monte_carlo_probability import (
    BIRTHDAY_GROUP_SIZE_RANGE,
    DICE_SUM_RANGE,
    NUM_TRIALS_RANGE,
    _birthday_collision_probability,
    _dice_sum_probability,
    build_convergence,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_trial_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_trial", num_trials=200, seed=0, target_value=3)


def test_rejects_num_trials_below_min():
    lo, _hi = NUM_TRIALS_RANGE
    with pytest.raises(ValueError):
        validate_params("monty_hall", num_trials=lo - 1, seed=0, target_value=0)


def test_rejects_num_trials_above_max():
    _lo, hi = NUM_TRIALS_RANGE
    with pytest.raises(ValueError):
        validate_params("monty_hall", num_trials=hi + 1, seed=0, target_value=0)


def test_rejects_negative_seed():
    with pytest.raises(ValueError):
        validate_params("monty_hall", num_trials=200, seed=-1, target_value=0)


def test_birthday_paradox_rejects_group_size_below_min():
    lo, _hi = BIRTHDAY_GROUP_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("birthday_paradox", num_trials=200, seed=0, target_value=lo - 1)


def test_birthday_paradox_rejects_group_size_above_max():
    _lo, hi = BIRTHDAY_GROUP_SIZE_RANGE
    with pytest.raises(ValueError):
        validate_params("birthday_paradox", num_trials=200, seed=0, target_value=hi + 1)


def test_dice_sum_rejects_target_below_min():
    lo, _hi = DICE_SUM_RANGE
    with pytest.raises(ValueError):
        validate_params("dice_sum_convergence", num_trials=200, seed=0, target_value=lo - 1)


def test_dice_sum_rejects_target_above_max():
    _lo, hi = DICE_SUM_RANGE
    with pytest.raises(ValueError):
        validate_params("dice_sum_convergence", num_trials=200, seed=0, target_value=hi + 1)


def test_rejects_non_numeric_target_value():
    with pytest.raises(ValueError):
        validate_params("monty_hall", num_trials=200, seed=0, target_value="not a number")


# ---- Theoretical probability formulas ----


def test_dice_sum_probability_of_seven_is_one_sixth():
    assert _dice_sum_probability(7) == pytest.approx(6 / 36)


def test_dice_sum_probability_of_two_is_one_in_36():
    assert _dice_sum_probability(2) == pytest.approx(1 / 36)


def test_birthday_collision_probability_of_23_exceeds_half():
    # the textbook "23 people" result
    assert _birthday_collision_probability(23) > 0.5


def test_birthday_collision_probability_of_group_size_1_is_zero():
    assert _birthday_collision_probability(1) == pytest.approx(0.0)


# ---- Simulated convergence, at num_trials param bounds ----


@pytest.mark.parametrize("num_trials", NUM_TRIALS_RANGE)
def test_monty_hall_rates_stay_in_unit_interval(num_trials):
    series, theoretical = build_convergence("monty_hall", num_trials, seed=0, target_value=0)
    for name in series:
        assert (series[name] >= 0).all() and (series[name] <= 1).all()
    assert theoretical == {"switch": pytest.approx(2 / 3), "stay": pytest.approx(1 / 3)}


def test_monty_hall_switch_converges_closer_than_stay_over_many_trials():
    series, theoretical = build_convergence("monty_hall", num_trials=2000, seed=0, target_value=0)
    switch_error = abs(series["switch"][-1] - theoretical["switch"])
    stay_error = abs(series["stay"][-1] - theoretical["stay"])
    # both should be reasonably close after 2000 trials; loose bound to
    # avoid a flaky test on an unlucky seed
    assert switch_error < 0.1
    assert stay_error < 0.1


@pytest.mark.parametrize("num_trials", NUM_TRIALS_RANGE)
def test_dice_sum_convergence_has_expected_shape(num_trials):
    series, theoretical = build_convergence(
        "dice_sum_convergence", num_trials, seed=0, target_value=7
    )
    assert series["hit_rate"].shape == (num_trials,)
    assert theoretical["hit_rate"] == pytest.approx(6 / 36)


def test_build_convergence_is_deterministic():
    a, _ = build_convergence("birthday_paradox", num_trials=300, seed=42, target_value=23)
    b, _ = build_convergence("birthday_paradox", num_trials=300, seed=42, target_value=23)
    assert (a["collision_rate"] == b["collision_rate"]).all()


def test_build_convergence_differs_across_seeds():
    a, _ = build_convergence("dice_sum_convergence", num_trials=300, seed=1, target_value=7)
    b, _ = build_convergence("dice_sum_convergence", num_trials=300, seed=2, target_value=7)
    assert not (a["hit_rate"] == b["hit_rate"]).all()
