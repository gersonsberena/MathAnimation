"""Light tests for particle_physics_sim (README Section 6, #2): pure
param-validation and boundary checks only, no Scene/render checks — see
README Section 11.6 on the per-engine test cadence.
"""

import pytest

from engines.particle_physics_sim import (
    NUM_BODIES_RANGES,
    SIM_DURATION_RANGE,
    build_trajectories,
    validate_params,
)

_FAST_FPS = 10  # keep boundary tests quick regardless of sim_duration


def test_rejects_unknown_sim_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_sim", num_bodies=2, sim_duration=4.0, seed=0)


@pytest.mark.parametrize("sim_type", sorted(NUM_BODIES_RANGES))
def test_rejects_num_bodies_below_min(sim_type):
    lo, _hi = NUM_BODIES_RANGES[sim_type]
    if lo == 0:
        pytest.skip("no value below min to test")
    with pytest.raises(ValueError):
        validate_params(sim_type, num_bodies=lo - 1, sim_duration=4.0, seed=0)


@pytest.mark.parametrize("sim_type", sorted(NUM_BODIES_RANGES))
def test_rejects_num_bodies_above_max(sim_type):
    _lo, hi = NUM_BODIES_RANGES[sim_type]
    with pytest.raises(ValueError):
        validate_params(sim_type, num_bodies=hi + 1, sim_duration=4.0, seed=0)


def test_rejects_sim_duration_below_min():
    lo, _hi = SIM_DURATION_RANGE
    with pytest.raises(ValueError):
        validate_params("double_pendulum", num_bodies=2, sim_duration=lo - 0.1, seed=0)


def test_rejects_sim_duration_above_max():
    _lo, hi = SIM_DURATION_RANGE
    with pytest.raises(ValueError):
        validate_params("double_pendulum", num_bodies=2, sim_duration=hi + 0.1, seed=0)


def test_rejects_negative_seed():
    with pytest.raises(ValueError):
        validate_params("double_pendulum", num_bodies=2, sim_duration=4.0, seed=-1)


def test_n_body_gravity_rejects_mismatched_initial_conditions_length():
    with pytest.raises(ValueError):
        build_trajectories(
            "n_body_gravity",
            num_bodies=3,
            initial_conditions=[{"mass": 1, "position": [0, 0], "velocity": [0, 0]}],
            sim_duration=2.0,
            fps=_FAST_FPS,
            seed=0,
        )


@pytest.mark.parametrize("sim_type", sorted(NUM_BODIES_RANGES))
def test_build_trajectories_at_min_bodies_has_expected_shape(sim_type):
    lo, _hi = NUM_BODIES_RANGES[sim_type]
    traj = build_trajectories(
        sim_type, num_bodies=lo, initial_conditions=None, sim_duration=2.0, fps=_FAST_FPS, seed=0
    )
    assert traj.shape[1] == lo
    assert traj.shape[2] == 2
    assert traj.shape[0] == int(2.0 * _FAST_FPS)


@pytest.mark.parametrize("sim_type", sorted(NUM_BODIES_RANGES))
def test_build_trajectories_at_max_bodies_has_expected_shape(sim_type):
    _lo, hi = NUM_BODIES_RANGES[sim_type]
    traj = build_trajectories(
        sim_type, num_bodies=hi, initial_conditions=None, sim_duration=2.0, fps=_FAST_FPS, seed=0
    )
    assert traj.shape[1] == hi
    assert traj.shape[2] == 2


@pytest.mark.parametrize("sim_type", sorted(NUM_BODIES_RANGES))
def test_build_trajectories_is_deterministic_given_seed(sim_type):
    lo, _hi = NUM_BODIES_RANGES[sim_type]
    a = build_trajectories(sim_type, lo, None, 2.0, _FAST_FPS, seed=42)
    b = build_trajectories(sim_type, lo, None, 2.0, _FAST_FPS, seed=42)
    assert (a == b).all()


def test_build_trajectories_produces_finite_positions():
    # A cheap smoke check that the double-pendulum integrator hasn't
    # blown up (NaN/inf) over a full-length run.
    traj = build_trajectories("double_pendulum", 2, None, sim_duration=10.0, fps=_FAST_FPS, seed=0)
    assert bool((traj == traj).all())  # NaN != NaN
    assert bool((abs(traj) < 1e6).all())
