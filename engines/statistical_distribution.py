"""statistical_distribution engine (README Section 6, engine 13): Galton
board/bean machine, dice-sum histogram, central-limit-theorem demo.
README's own consolidation note flags this as sharing a "run N trials"
core with monte_carlo_probability (#5) — and indeed, the CLT/histogram
trial_types deliberately left OUT of that engine (see
engines/monte_carlo_probability.py's module docstring) belong here: a
live-growing histogram is a genuinely different rendering mechanic from
that engine's converging-rate line, which is why it's a separate engine
rather than a forced-in trial_type there.

All three distribution_types reduce to the same shape: bin each trial's
outcome into one of a fixed number of bins, then reveal the growing
per-bin counts as a bar chart — reusing array_bar_race's fixed-position,
growing-bar-height ValueTracker + always_redraw pattern.

`build_histogram_trials()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, Rectangle, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

DISTRIBUTION_TYPES = {"galton_board", "dice_sum_histogram", "clt_demo"}
NUM_TRIALS_RANGE = (100, 3000)
SEED_RANGE = (0, 2**31 - 1)

_GALTON_NUM_ROWS = 12  # -> 13 bins, Binomial(12, 0.5), approximates a bell curve
_CLT_NUM_UNIFORMS = 12  # sum of 12 U(0,1) vars — classic CLT demo, range [0, 12]
_CLT_NUM_BINS = 25
_MAX_CHECKPOINTS = 120  # decouples animation length from num_trials, like array_bar_race


def validate_params(distribution_type, num_trials, seed) -> None:
    if distribution_type not in DISTRIBUTION_TYPES:
        raise ValueError(
            f"unknown distribution_type {distribution_type!r}, must be one of {sorted(DISTRIBUTION_TYPES)}"
        )

    lo, hi = NUM_TRIALS_RANGE
    if not (lo <= num_trials <= hi):
        raise ValueError(f"num_trials={num_trials} out of range [{lo}, {hi}]")

    lo, hi = SEED_RANGE
    if not (lo <= seed <= hi):
        raise ValueError(f"seed={seed} out of range [{lo}, {hi}]")


def _simulate_galton_board(num_trials, seed):
    rng = np.random.default_rng(seed)
    steps = rng.integers(0, 2, size=(num_trials, _GALTON_NUM_ROWS))  # 0:left, 1:right per peg row
    bins = steps.sum(axis=1)  # final bin = number of right-moves, 0..num_rows
    return bins, _GALTON_NUM_ROWS + 1


def _simulate_dice_sum_histogram(num_trials, seed):
    rng = np.random.default_rng(seed)
    rolls = rng.integers(1, 7, size=(num_trials, 2))
    sums = rolls.sum(axis=1)  # 2..12
    return sums - 2, 11  # shift to 0-indexed bins


def _simulate_clt_demo(num_trials, seed):
    rng = np.random.default_rng(seed)
    sums = rng.uniform(0, 1, size=(num_trials, _CLT_NUM_UNIFORMS)).sum(axis=1)  # range [0, k]
    bin_edges = np.linspace(0, _CLT_NUM_UNIFORMS, _CLT_NUM_BINS + 1)
    bins = np.clip(np.digitize(sums, bin_edges) - 1, 0, _CLT_NUM_BINS - 1)
    return bins, _CLT_NUM_BINS


def build_histogram_trials(distribution_type, num_trials, seed):
    """Returns `(bin_indices, num_bins)`: `bin_indices` is a length-
    `num_trials` int array giving each trial's bin, in trial order (so a
    prefix `bin_indices[:k]` is "the first k trials"). Pure — no Scene
    dependency.
    """
    validate_params(distribution_type, num_trials, seed)

    if distribution_type == "galton_board":
        return _simulate_galton_board(num_trials, seed)
    if distribution_type == "dice_sum_histogram":
        return _simulate_dice_sum_histogram(num_trials, seed)
    if distribution_type == "clt_demo":
        return _simulate_clt_demo(num_trials, seed)

    raise AssertionError(f"unhandled distribution_type {distribution_type!r}")  # validate_params already checked


class StatisticalDistributionReel(ReelScene):
    """`params`: distribution_type, num_trials, seed (README Section 6,
    #13).
    """

    distribution_type = "galton_board"
    num_trials = 1000
    seed = 0

    def construct(self):
        self.set_title("Random Becomes Predictable")

        bin_indices, num_bins = build_histogram_trials(self.distribution_type, self.num_trials, self.seed)

        num_checkpoints = min(_MAX_CHECKPOINTS, self.num_trials)
        checkpoint_upto = np.linspace(1, self.num_trials, num_checkpoints).astype(int)
        counts_over_time = np.zeros((num_checkpoints, num_bins), dtype=int)
        running = np.zeros(num_bins, dtype=int)
        prev = 0
        for i, upto in enumerate(checkpoint_upto):
            for b in bin_indices[prev:upto]:
                running[b] += 1
            counts_over_time[i] = running
            prev = upto

        zone = self.zones.content_zone
        max_count = max(int(counts_over_time[-1].max()), 1)
        bar_width = zone.width * 0.85 / num_bins
        max_bar_height = zone.height * 0.75
        bottom_y = zone.get_bottom()[1] + zone.height * 0.05
        left_x = zone.get_left()[0] + zone.width * 0.075

        frame_tracker = ValueTracker(0)

        def checkpoint_idx():
            return max(0, min(int(round(frame_tracker.get_value())), num_checkpoints - 1))

        def make_bar(i):
            count = counts_over_time[checkpoint_idx()][i]
            height = max(count / max_count * max_bar_height, 0.01)
            rect = Rectangle(width=bar_width * 0.85, height=height, fill_color=BLUE, fill_opacity=1, stroke_width=0)
            rect.move_to([left_x + i * bar_width + bar_width / 2, bottom_y + height / 2, 0])
            return rect

        bars = VGroup(*[always_redraw(lambda i=i: make_bar(i)) for i in range(num_bins)])
        self.add(bars)

        self.play(frame_tracker.animate.set_value(num_checkpoints - 1), run_time=5.0, rate_func=linear)

        self.set_caption("Thousands of trials, one shape")
