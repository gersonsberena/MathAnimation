"""population_growth_model engine (README Section 6, engine 18):
predator-prey (Lotka-Volterra), SIR virus-spread sim, logistic vs.
exponential growth race. Mechanic: differential equation stepped
forward, plotted as curves over time.

README's `duration` param names the simulated TIME SPAN, but
`manim.Scene.__init__` unconditionally sets `self.duration = 0.0`,
silently shadowing a Scene-level attribute of that name the moment the
Scene is constructed (documented as a known gotcha in README Section 8,
flagged for this exact engine after wave_signal hit it first). The
Scene class attribute here is `growth_duration` instead; the pure
functions' parameter is still named `duration` since only the Scene
*instance attribute* collides.

Integration is 4th-order Runge-Kutta (RK4), reusing
strange_attractor_chaos's RK4 pattern, over a fixed internal step
count. Rendering reuses monte_carlo_probability/wave_signal's
Axes + ValueTracker + always_redraw growing-line-graph pattern, drawn
progressively as the checkpoint index advances.

Each model_type's correctness is checked against an independent
reference rather than trusting the RK4 implementation alone:
logistic_vs_exponential has an exact closed-form solution (compared
directly); sir_epidemic has a real conserved invariant (S+I+R constant);
predator_prey has a known fixed point (alpha/beta, gamma/delta) whose
derivative is exactly zero, so integrating from it should stay there.

`build_population_model()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, GREEN, ORANGE, UP, Axes, Text, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

MODEL_TYPES = {"predator_prey", "sir_epidemic", "logistic_vs_exponential"}
REQUIRED_INITIAL_KEYS = {
    "predator_prey": {"prey", "predator"},
    "sir_epidemic": {"S", "I", "R"},
    "logistic_vs_exponential": {"population"},
}
REQUIRED_RATE_KEYS = {
    "predator_prey": {"alpha", "beta", "delta", "gamma"},
    "sir_epidemic": {"beta", "gamma"},
    "logistic_vs_exponential": {"growth_rate", "carrying_capacity"},
}
DURATION_RANGE = (1.0, 50.0)
_NUM_STEPS = 400


def validate_params(model_type, initial_populations, rate_constants, duration) -> None:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}, must be one of {sorted(MODEL_TYPES)}")

    required_initial = REQUIRED_INITIAL_KEYS[model_type]
    if set(initial_populations.keys()) != required_initial:
        raise ValueError(f"initial_populations keys must be exactly {sorted(required_initial)} for model_type={model_type!r}")
    for key, value in initial_populations.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"initial_populations[{key!r}]={value!r} must be a non-negative number")

    required_rates = REQUIRED_RATE_KEYS[model_type]
    if set(rate_constants.keys()) != required_rates:
        raise ValueError(f"rate_constants keys must be exactly {sorted(required_rates)} for model_type={model_type!r}")
    for key, value in rate_constants.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"rate_constants[{key!r}]={value!r} must be a positive number")

    lo, hi = DURATION_RANGE
    if not (lo <= duration <= hi):
        raise ValueError(f"duration={duration} out of range [{lo}, {hi}]")


def _rk4_integrate(state0, duration, deriv_fn, num_steps=_NUM_STEPS):
    dt = duration / num_steps
    state = np.array(state0, dtype=float)
    states = [state.copy()]
    t = 0.0
    for _ in range(num_steps):
        k1 = deriv_fn(state, t)
        k2 = deriv_fn(state + dt / 2 * k1, t + dt / 2)
        k3 = deriv_fn(state + dt / 2 * k2, t + dt / 2)
        k4 = deriv_fn(state + dt * k3, t + dt)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        state = np.maximum(state, 0.0)  # populations can't go negative
        t += dt
        states.append(state.copy())
    times = np.linspace(0, duration, num_steps + 1)
    return times, np.array(states)


def _predator_prey_deriv(rates):
    alpha, beta, delta, gamma = rates["alpha"], rates["beta"], rates["delta"], rates["gamma"]

    def deriv(state, _t):
        prey, predator = state
        return np.array([alpha * prey - beta * prey * predator, delta * prey * predator - gamma * predator])

    return deriv


def _sir_deriv(rates):
    beta, gamma = rates["beta"], rates["gamma"]

    def deriv(state, _t):
        s, i, r = state
        n = s + i + r
        new_infections = beta * s * i / n
        return np.array([-new_infections, new_infections - gamma * i, gamma * i])

    return deriv


def _logistic_vs_exponential_deriv(rates):
    r, k = rates["growth_rate"], rates["carrying_capacity"]

    def deriv(state, _t):
        exp_pop, log_pop = state
        return np.array([r * exp_pop, r * log_pop * (1 - log_pop / k)])

    return deriv


def build_population_model(model_type, initial_populations, rate_constants, duration):
    """Returns a dict discriminated by `kind`:
      - predator_prey: `{"kind": "predator_prey", "t", "prey", "predator"}`
      - sir_epidemic: `{"kind": "sir_epidemic", "t", "S", "I", "R"}`
      - logistic_vs_exponential: `{"kind": "logistic_vs_exponential", "t",
        "exponential", "logistic"}`
    Pure — no Scene dependency.
    """
    validate_params(model_type, initial_populations, rate_constants, duration)

    if model_type == "predator_prey":
        state0 = [initial_populations["prey"], initial_populations["predator"]]
        t, states = _rk4_integrate(state0, duration, _predator_prey_deriv(rate_constants))
        return {"kind": "predator_prey", "t": t, "prey": states[:, 0], "predator": states[:, 1]}

    if model_type == "sir_epidemic":
        state0 = [initial_populations["S"], initial_populations["I"], initial_populations["R"]]
        t, states = _rk4_integrate(state0, duration, _sir_deriv(rate_constants))
        return {"kind": "sir_epidemic", "t": t, "S": states[:, 0], "I": states[:, 1], "R": states[:, 2]}

    if model_type == "logistic_vs_exponential":
        pop0 = initial_populations["population"]
        state0 = [pop0, pop0]
        t, states = _rk4_integrate(state0, duration, _logistic_vs_exponential_deriv(rate_constants))
        return {"kind": "logistic_vs_exponential", "t": t, "exponential": states[:, 0], "logistic": states[:, 1]}

    raise AssertionError(f"unhandled model_type {model_type!r}")  # validate_params already checked


class PopulationGrowthModelReel(ReelScene):
    """`params`: model_type, initial_populations, rate_constants,
    duration (README Section 6, #18). NOTE: the Scene class attribute is
    `growth_duration`, not `duration` — see module docstring.
    """

    model_type = "predator_prey"
    initial_populations = {"prey": 40.0, "predator": 9.0}
    rate_constants = {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4}
    growth_duration = 25.0

    def construct(self):
        self.set_title("The Rhythm Of A Population")

        data = build_population_model(self.model_type, self.initial_populations, self.rate_constants, self.growth_duration)

        if data["kind"] == "predator_prey":
            series = {"Prey": (data["prey"], BLUE), "Predator": (data["predator"], ORANGE)}
            caption = "Neither wins. They just keep cycling."
        elif data["kind"] == "sir_epidemic":
            series = {"Susceptible": (data["S"], BLUE), "Infected": (data["I"], ORANGE), "Recovered": (data["R"], GREEN)}
            caption = "The infection always burns out. The question is how fast."
        else:
            series = {"Exponential": (data["exponential"], ORANGE), "Logistic": (data["logistic"], BLUE)}
            caption = "Unlimited growth is a myth. Limits always win."

        t = data["t"]
        zone = self.zones.content_zone
        y_max = max(float(np.max(values)) for values, _color in series.values())

        axes = Axes(
            x_range=[0, t[-1], max(t[-1] / 4, 0.1)],
            y_range=[0, y_max * 1.1, max(y_max / 4, 0.1)],
            x_length=zone.width * 0.85,
            y_length=zone.height * 0.6,
            axis_config={"stroke_color": "#666666", "include_numbers": False},
        )
        axes.move_to(zone.get_center())

        frame_tracker = ValueTracker(0)

        def step_idx():
            return max(0, min(int(round(frame_tracker.get_value())), len(t) - 1))

        def make_line(values, color):
            up_to = step_idx() + 1
            x_vals = t[:up_to].tolist()
            y_vals = values[:up_to].tolist()
            if len(x_vals) < 2:
                x_vals, y_vals = x_vals * 2, y_vals * 2
            return axes.plot_line_graph(x_vals, y_vals, line_color=color, add_vertex_dots=False, stroke_width=4)

        lines = VGroup(*[always_redraw(lambda v=values, c=c: make_line(v, c)) for values, c in series.values()])

        def make_legend():
            parts = [f"{name}: {values[step_idx()]:.0f}" for name, (values, _c) in series.items()]
            return Text("  ".join(parts), font_size=24).next_to(axes, UP, buff=0.3)

        legend = always_redraw(make_legend)

        self.add(axes, lines, legend)
        self.play(frame_tracker.animate.set_value(len(t) - 1), run_time=6.0, rate_func=linear)
        frame_tracker.set_value(len(t) - 1)
        self.wait(0.3)

        self.set_caption(caption)
