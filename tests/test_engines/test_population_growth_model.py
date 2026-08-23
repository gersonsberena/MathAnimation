"""Light tests for population_growth_model (README Section 6, #18): pure
param-validation and differential-equation correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import numpy as np
import pytest

from engines.population_growth_model import (
    PopulationGrowthModelReel,
    _logistic_vs_exponential_deriv,
    _predator_prey_deriv,
    _rk4_integrate,
    build_population_model,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_model", {"population": 10}, {"growth_rate": 0.1, "carrying_capacity": 100}, 10)


def test_rejects_wrong_initial_population_keys():
    with pytest.raises(ValueError):
        validate_params(
            "predator_prey", {"prey": 10}, {"alpha": 1, "beta": 1, "delta": 1, "gamma": 1}, 10
        )  # missing "predator"


def test_rejects_negative_initial_population():
    with pytest.raises(ValueError):
        validate_params(
            "predator_prey", {"prey": -5, "predator": 10}, {"alpha": 1, "beta": 1, "delta": 1, "gamma": 1}, 10
        )


def test_rejects_wrong_rate_constant_keys():
    with pytest.raises(ValueError):
        validate_params("sir_epidemic", {"S": 99, "I": 1, "R": 0}, {"beta": 0.3}, 10)  # missing "gamma"


def test_rejects_non_positive_rate_constant():
    with pytest.raises(ValueError):
        validate_params("sir_epidemic", {"S": 99, "I": 1, "R": 0}, {"beta": 0.3, "gamma": 0}, 10)


def test_rejects_duration_out_of_range():
    with pytest.raises(ValueError):
        validate_params("logistic_vs_exponential", {"population": 10}, {"growth_rate": 0.1, "carrying_capacity": 100}, 0.5)
    with pytest.raises(ValueError):
        validate_params("logistic_vs_exponential", {"population": 10}, {"growth_rate": 0.1, "carrying_capacity": 100}, 100)


# ---- RK4 integrator sanity ----


def test_rk4_matches_exact_exponential_growth():
    # dP/dt = r*P has the exact closed form P(t) = P0 * exp(r*t) —
    # independent reference, not trusting the RK4 implementation alone.
    r, p0, duration = 0.3, 10.0, 5.0
    deriv = lambda state, _t: r * state  # noqa: E731
    t, states = _rk4_integrate([p0], duration, deriv)
    exact_final = p0 * np.exp(r * duration)
    assert states[-1, 0] == pytest.approx(exact_final, rel=1e-4)


# ---- logistic_vs_exponential correctness ----


def test_logistic_matches_its_closed_form_solution():
    # Logistic equation P(t) = K / (1 + ((K-P0)/P0) * exp(-r*t)).
    r, k, p0, duration = 0.4, 500.0, 20.0, 20.0
    deriv = _logistic_vs_exponential_deriv({"growth_rate": r, "carrying_capacity": k})
    t, states = _rk4_integrate([p0, p0], duration, deriv)
    exact_logistic = k / (1 + ((k - p0) / p0) * np.exp(-r * t))
    assert states[:, 1] == pytest.approx(exact_logistic, rel=1e-3)


def test_exponential_matches_its_closed_form_solution():
    r, k, p0, duration = 0.4, 500.0, 20.0, 20.0
    deriv = _logistic_vs_exponential_deriv({"growth_rate": r, "carrying_capacity": k})
    t, states = _rk4_integrate([p0, p0], duration, deriv)
    exact_exponential = p0 * np.exp(r * t)
    assert states[:, 0] == pytest.approx(exact_exponential, rel=1e-3)


def test_exponential_eventually_exceeds_logistic():
    data = build_population_model(
        "logistic_vs_exponential", {"population": 10.0}, {"growth_rate": 0.5, "carrying_capacity": 200.0}, 20.0
    )
    assert data["exponential"][-1] > data["logistic"][-1]
    assert data["logistic"][-1] <= 200.0 * 1.001  # never meaningfully exceeds carrying capacity


# ---- sir_epidemic correctness ----


def test_sir_total_population_is_conserved():
    data = build_population_model("sir_epidemic", {"S": 990.0, "I": 10.0, "R": 0.0}, {"beta": 0.4, "gamma": 0.1}, 30.0)
    totals = data["S"] + data["I"] + data["R"]
    assert totals == pytest.approx(totals[0], rel=1e-6)


def test_sir_infected_rises_then_falls_for_an_outbreak():
    data = build_population_model("sir_epidemic", {"S": 990.0, "I": 10.0, "R": 0.0}, {"beta": 0.4, "gamma": 0.1}, 50.0)
    infected = data["I"]
    peak_idx = int(np.argmax(infected))
    assert 0 < peak_idx < len(infected) - 1  # peak is interior, not at either endpoint
    assert infected[-1] < infected[peak_idx]


def test_sir_recovered_only_ever_increases():
    data = build_population_model("sir_epidemic", {"S": 990.0, "I": 10.0, "R": 0.0}, {"beta": 0.4, "gamma": 0.1}, 30.0)
    recovered = data["R"]
    assert all(b >= a - 1e-9 for a, b in zip(recovered, recovered[1:]))


# ---- predator_prey correctness ----


def test_predator_prey_equilibrium_point_stays_fixed():
    # The Lotka-Volterra system has a fixed point at
    # (prey, predator) = (gamma/delta, alpha/beta), where both
    # derivatives are exactly zero — integrating from there should stay
    # (approximately) there.
    rates = {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4}
    equilibrium_prey = rates["gamma"] / rates["delta"]
    equilibrium_predator = rates["alpha"] / rates["beta"]
    deriv = _predator_prey_deriv(rates)
    t, states = _rk4_integrate([equilibrium_prey, equilibrium_predator], 20.0, deriv)
    assert states[-1, 0] == pytest.approx(equilibrium_prey, rel=1e-3)
    assert states[-1, 1] == pytest.approx(equilibrium_predator, rel=1e-3)


def test_predator_prey_oscillates_away_from_equilibrium():
    data = build_population_model(
        "predator_prey", {"prey": 40.0, "predator": 9.0}, {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4}, 25.0
    )
    # Away from the fixed point, both populations should meaningfully vary
    # over time (a real cycle), not stay flat.
    assert data["prey"].max() - data["prey"].min() > 5
    assert data["predator"].max() - data["predator"].min() > 2


# ---- build_population_model() / Scene ----


def test_build_population_model_is_deterministic():
    a = build_population_model("sir_epidemic", {"S": 990.0, "I": 10.0, "R": 0.0}, {"beta": 0.4, "gamma": 0.1}, 30.0)
    b = build_population_model("sir_epidemic", {"S": 990.0, "I": 10.0, "R": 0.0}, {"beta": 0.4, "gamma": 0.1}, 30.0)
    for key in ("t", "S", "I", "R"):
        assert np.array_equal(a[key], b[key])


def test_scene_class_does_not_use_duration_as_an_attribute_name():
    # Manim's Scene.__init__ unconditionally sets self.duration = 0.0,
    # which would silently shadow a class-level `duration` default the
    # moment the Scene is constructed (see README Section 8's gotcha,
    # and wave_signal's wave_duration rename for the first instance of
    # this bug). Guard against reintroducing it here.
    assert not hasattr(PopulationGrowthModelReel, "duration")
    assert hasattr(PopulationGrowthModelReel, "growth_duration")
