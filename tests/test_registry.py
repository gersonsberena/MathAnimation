import pytest

from engines.registry import scene_class_for_category, validate_params_for_category


def test_scene_class_for_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown category"):
        scene_class_for_category("not_a_real_category")


def test_validate_params_for_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown category"):
        validate_params_for_category("not_a_real_category", {})


def test_validate_params_ignores_keys_validate_params_does_not_declare():
    """particle_physics_sim.validate_params() checks sim_type/num_bodies/
    sim_duration/seed but not initial_conditions -- that key is free-form
    and validated separately, deeper in build_trajectories(). Forwarding
    every params key blindly used to raise a TypeError here; regression
    test for that fix.
    """
    validate_params_for_category(
        "particle_physics_sim",
        {
            "sim_type": "double_pendulum",
            "num_bodies": 2,
            "initial_conditions": {"theta1_deg": 120.0, "theta2_deg": -10.0},
            "sim_duration": 6.0,
            "seed": 0,
        },
    )  # no raise


def test_validate_params_still_enforces_bounds_on_declared_keys():
    with pytest.raises(ValueError, match="out of range"):
        validate_params_for_category(
            "particle_physics_sim",
            {"sim_type": "double_pendulum", "num_bodies": 999, "sim_duration": 6.0, "seed": 0},
        )
