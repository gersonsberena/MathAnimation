"""monte_carlo_probability engine (README Section 6, engine 5): Monty
Hall, birthday paradox, dice-sum convergence. Mechanic: run N trials of a
random process, animate a live counter/line converging toward the
theoretical probability.

Every trial_type here reduces to a fraction-of-trials-matching-a-condition
running rate, always in [0, 1] — that uniform shape is what lets one
Scene/Axes render all three. Central-limit-theorem / histogram-style demos
(also mentioned under this engine in README Section 6) are deliberately
NOT included yet: they're a different rendering mechanic (a growing
histogram, not a converging rate line) and README Section 6's own
consolidation note flags this engine as a likely partial merge with #13
(statistical_distribution) — better to fold that in once both exist than
force two mechanics into this engine's params shape now.

All `_simulate_*` functions are pure and seeded via `numpy.random.default_rng`
for reproducibility (Section 2, point 3).
"""

import numpy as np
from manim import BLUE, PINK, UP, YELLOW, Axes, DashedLine, Text, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

TRIAL_TYPES = {"monty_hall", "birthday_paradox", "dice_sum_convergence"}

NUM_TRIALS_RANGE = (50, 2000)
SEED_RANGE = (0, 2**31 - 1)
BIRTHDAY_GROUP_SIZE_RANGE = (2, 100)
DICE_SUM_RANGE = (2, 12)

_MAX_CHECKPOINTS = 120  # decouples animation length from num_trials, like array_bar_race


def validate_params(trial_type, num_trials, seed, target_value) -> None:
    if trial_type not in TRIAL_TYPES:
        raise ValueError(f"unknown trial_type {trial_type!r}, must be one of {sorted(TRIAL_TYPES)}")

    lo, hi = NUM_TRIALS_RANGE
    if not (lo <= num_trials <= hi):
        raise ValueError(f"num_trials={num_trials} out of range [{lo}, {hi}]")

    lo, hi = SEED_RANGE
    if not (lo <= seed <= hi):
        raise ValueError(f"seed={seed} out of range [{lo}, {hi}]")

    if not isinstance(target_value, (int, float)):
        raise ValueError(f"target_value must be numeric, got {target_value!r}")

    if trial_type == "birthday_paradox":
        lo, hi = BIRTHDAY_GROUP_SIZE_RANGE
        if not (lo <= target_value <= hi):
            raise ValueError(f"target_value (group_size)={target_value} out of range [{lo}, {hi}]")
    elif trial_type == "dice_sum_convergence":
        lo, hi = DICE_SUM_RANGE
        if not (lo <= target_value <= hi):
            raise ValueError(f"target_value (target_sum)={target_value} out of range [{lo}, {hi}]")
    # monty_hall: target_value is validated as numeric above but unused —
    # the theoretical targets (2/3 switch, 1/3 stay) are fixed by the game.


def _simulate_monty_hall(num_trials, seed):
    rng = np.random.default_rng(seed)
    switch_wins = 0
    stay_wins = 0
    switch_rate = np.zeros(num_trials)
    stay_rate = np.zeros(num_trials)

    for t in range(num_trials):
        car = int(rng.integers(0, 3))
        pick = int(rng.integers(0, 3))
        remaining = [d for d in range(3) if d != car and d != pick]
        host_opens = remaining[0] if len(remaining) == 1 else remaining[int(rng.integers(0, 2))]
        switch_pick = next(d for d in range(3) if d != pick and d != host_opens)

        if switch_pick == car:
            switch_wins += 1
        if pick == car:
            stay_wins += 1

        switch_rate[t] = switch_wins / (t + 1)
        stay_rate[t] = stay_wins / (t + 1)

    return {"switch": switch_rate, "stay": stay_rate}, {"switch": 2 / 3, "stay": 1 / 3}


def _birthday_collision_probability(group_size):
    prob_no_collision = 1.0
    for i in range(group_size):
        prob_no_collision *= (365 - i) / 365
    return 1.0 - prob_no_collision


def _simulate_birthday_paradox(num_trials, seed, group_size):
    rng = np.random.default_rng(seed)
    collisions = 0
    rate = np.zeros(num_trials)

    for t in range(num_trials):
        birthdays = rng.integers(0, 365, size=group_size)
        if len(set(birthdays.tolist())) < group_size:
            collisions += 1
        rate[t] = collisions / (t + 1)

    theoretical = _birthday_collision_probability(group_size)
    return {"collision_rate": rate}, {"collision_rate": theoretical}


def _dice_sum_probability(target_sum):
    outcomes = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == target_sum)
    return outcomes / 36


def _simulate_dice_sum(num_trials, seed, target_sum):
    rng = np.random.default_rng(seed)
    hits = 0
    rate = np.zeros(num_trials)

    for t in range(num_trials):
        roll = rng.integers(1, 7, size=2)
        if int(roll[0] + roll[1]) == target_sum:
            hits += 1
        rate[t] = hits / (t + 1)

    theoretical = _dice_sum_probability(target_sum)
    return {"hit_rate": rate}, {"hit_rate": theoretical}


def build_convergence(trial_type, num_trials, seed, target_value):
    """Returns `(series, theoretical)`: both dicts keyed by series name,
    `series[name]` a length-`num_trials` running-rate array in [0, 1],
    `theoretical[name]` the probability it should converge toward. Pure —
    no Scene dependency.
    """
    validate_params(trial_type, num_trials, seed, target_value)

    if trial_type == "monty_hall":
        return _simulate_monty_hall(num_trials, seed)
    if trial_type == "birthday_paradox":
        return _simulate_birthday_paradox(num_trials, seed, group_size=int(target_value))
    if trial_type == "dice_sum_convergence":
        return _simulate_dice_sum(num_trials, seed, target_sum=int(target_value))

    raise AssertionError(f"unhandled trial_type {trial_type!r}")  # validate_params already checked


class MonteCarloProbabilityReel(ReelScene):
    """`params`: trial_type, num_trials, seed, target_value (README
    Section 6, #5).
    """

    trial_type = "monty_hall"
    num_trials = 400
    seed = 0
    target_value = 0  # unused for monty_hall

    def construct(self):
        self.set_title("Does It Converge?")

        series, theoretical = build_convergence(self.trial_type, self.num_trials, self.seed, self.target_value)
        names = list(series.keys())
        palette = [BLUE, PINK, YELLOW]

        num_checkpoints = min(_MAX_CHECKPOINTS, self.num_trials)
        checkpoint_indices = np.unique(np.linspace(0, self.num_trials - 1, num_checkpoints).astype(int))
        checkpoint_series = {name: series[name][checkpoint_indices] for name in names}

        zone = self.zones.content_zone
        axes = Axes(
            x_range=[0, self.num_trials, max(self.num_trials // 4, 1)],
            y_range=[0, 1, 0.25],
            x_length=zone.width * 0.85,
            y_length=zone.height * 0.65,
            axis_config={"stroke_color": "#666666", "include_numbers": False},
        )
        axes.move_to(zone.get_center())

        reference_lines = VGroup(
            *[
                DashedLine(
                    axes.c2p(0, theoretical[name]),
                    axes.c2p(self.num_trials, theoretical[name]),
                    color=palette[i % len(palette)],
                    stroke_width=1.5,
                    stroke_opacity=0.6,
                )
                for i, name in enumerate(names)
            ]
        )

        frame_tracker = ValueTracker(0)

        def checkpoint_idx():
            return max(0, min(int(round(frame_tracker.get_value())), len(checkpoint_indices) - 1))

        def make_line(name, color):
            up_to = checkpoint_idx() + 1
            x_vals = (checkpoint_indices[:up_to] + 1).tolist()
            y_vals = checkpoint_series[name][:up_to].tolist()
            if len(x_vals) < 2:
                x_vals, y_vals = x_vals * 2, y_vals * 2
            return axes.plot_line_graph(
                x_vals, y_vals, line_color=color, add_vertex_dots=False, stroke_width=4
            )

        lines = VGroup(
            *[always_redraw(lambda n=name, c=palette[i % len(palette)]: make_line(n, c)) for i, name in enumerate(names)]
        )

        def make_counter():
            idx = checkpoint_idx()
            trial_count = int(checkpoint_indices[idx]) + 1
            parts = [f"{name}: {checkpoint_series[name][idx] * 100:.1f}%" for name in names]
            return Text(f"n={trial_count}  " + "  ".join(parts), font_size=28).next_to(axes, UP, buff=0.3)

        counter = always_redraw(make_counter)

        self.add(axes, reference_lines, lines, counter)
        self.play(
            frame_tracker.animate.set_value(len(checkpoint_indices) - 1),
            run_time=6.0,
            rate_func=linear,
        )
        frame_tracker.set_value(len(checkpoint_indices) - 1)
        self.wait(0.3)

        self.set_caption("More trials, closer to the truth")
