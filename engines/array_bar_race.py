"""array_bar_race engine (README Section 6, engine 4): sorting-algorithm
race (bubble_sort, quick_sort) and the compound-interest debt-vs-savings
race. Mechanic shared across race_types: an array of (label, value) pairs
changes over discrete time steps; bars are rendered at fixed row
positions and their *length* changes step to step (README's "array of
values changing over time steps, rendered as racing bars").

Sorting and compound growth intentionally do NOT reorder rows by rank:
sorting only ever permutes an existing multiset of values between fixed
array positions (there's nothing to visually "race" by re-ranking), and
fixed rows keep the compound-interest race legible as "which one grows
fastest" rather than a reshuffling scoreboard. `_*_frames()` functions are
pure and testable without a Scene.
"""

import numpy as np
from manim import LEFT, WHITE, Rectangle, Text, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

RACE_TYPES = {"bubble_sort", "quick_sort", "compound_interest"}

ARRAY_LENGTH_RANGE = (3, 10)  # sort types: bar count / worst-case swap count stays sane
SERIES_COUNT_RANGE = (2, 4)  # compound_interest: keep the race to a few competitors
RATE_RANGE = (0.0, 0.5)  # compound_interest: 0-50% growth per step
SPEED_RANGE = (0.25, 4.0)

_NUM_YEARS = 12  # compound_interest step count — not a param, an engine detail
_TARGET_DURATION = {"bubble_sort": 5.0, "quick_sort": 5.0, "compound_interest": 6.0}
_PALETTE = ["#00C2FF", "#FF3EA5", "#FFD23F", "#4EA8FF", "#7CFFB2"]


def validate_params(race_type, data_series, labels, speed) -> None:
    if race_type not in RACE_TYPES:
        raise ValueError(f"unknown race_type {race_type!r}, must be one of {sorted(RACE_TYPES)}")

    if race_type in ("bubble_sort", "quick_sort"):
        lo, hi = ARRAY_LENGTH_RANGE
        if not (lo <= len(data_series) <= hi):
            raise ValueError(f"data_series length {len(data_series)} out of range [{lo}, {hi}]")
        if any((not isinstance(v, (int, float))) or v <= 0 for v in data_series):
            raise ValueError("data_series values must all be positive numbers for a sort race")

    elif race_type == "compound_interest":
        lo, hi = SERIES_COUNT_RANGE
        if not (lo <= len(data_series) <= hi):
            raise ValueError(f"data_series length {len(data_series)} out of range [{lo}, {hi}]")
        for entry in data_series:
            if entry.get("principal", 0) <= 0:
                raise ValueError(f"principal must be positive, got {entry.get('principal')!r}")
            rate = entry.get("rate")
            lo_r, hi_r = RATE_RANGE
            if rate is None or not (lo_r <= rate <= hi_r):
                raise ValueError(f"rate={rate!r} out of range [{lo_r}, {hi_r}]")

    if labels is not None and len(labels) != len(data_series):
        raise ValueError(f"labels length {len(labels)} must match data_series length {len(data_series)}")

    lo, hi = SPEED_RANGE
    if not (lo <= speed <= hi):
        raise ValueError(f"speed={speed} out of range [{lo}, {hi}]")


def _bubble_sort_frames(values):
    arr = list(values)
    frames = [list(arr)]
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                frames.append(list(arr))
                swapped = True
        if not swapped:
            break
    return frames


def _quick_sort_frames(values):
    arr = list(values)
    frames = [list(arr)]

    def partition(lo, hi):
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                frames.append(list(arr))
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        frames.append(list(arr))
        return i + 1

    def sort_range(lo, hi):
        if lo < hi:
            p = partition(lo, hi)
            sort_range(lo, p - 1)
            sort_range(p + 1, hi)

    sort_range(0, len(arr) - 1)
    return frames


def _compound_interest_frames(data_series, num_years=_NUM_YEARS):
    balances = [entry["principal"] for entry in data_series]
    frames = [list(balances)]
    for _year in range(num_years):
        balances = [b * (1 + entry["rate"]) for b, entry in zip(balances, data_series)]
        frames.append(list(balances))
    return frames


def build_race(race_type, data_series, labels, speed):
    """Returns `(frames, resolved_labels)`. `frames` is a list of
    per-step value lists, one entry per row, in fixed row order — pure,
    no Scene dependency.
    """
    validate_params(race_type, data_series, labels, speed)

    if race_type == "bubble_sort":
        frames = _bubble_sort_frames(data_series)
        resolved_labels = labels or [str(i) for i in range(len(data_series))]
    elif race_type == "quick_sort":
        frames = _quick_sort_frames(data_series)
        resolved_labels = labels or [str(i) for i in range(len(data_series))]
    elif race_type == "compound_interest":
        frames = _compound_interest_frames(data_series)
        resolved_labels = labels or [
            entry.get("label", f"Series {i + 1}") for i, entry in enumerate(data_series)
        ]
    else:
        raise AssertionError(f"unhandled race_type {race_type!r}")  # validate_params already checked

    return frames, resolved_labels


class ArrayBarRaceReel(ReelScene):
    """`params`: race_type, data_series, labels, speed (README Section 6,
    #4).
    """

    race_type = "bubble_sort"
    data_series = [5, 3, 8, 1, 9, 2, 7]
    labels = None
    speed = 1.0

    def construct(self):
        self.set_title("Who Sorts Fastest?")

        frames, labels = build_race(self.race_type, self.data_series, self.labels, self.speed)
        num_rows = len(frames[0])
        max_value = max(max(frame) for frame in frames)

        zone = self.zones.content_zone
        row_span = zone.height * 0.9 / num_rows
        bar_height = row_span * 0.6
        max_bar_width = zone.width * 0.6
        left_x = zone.get_left()[0] + zone.width * 0.28
        top_y = zone.get_top()[1] - row_span / 2

        def row_y(i):
            return top_y - i * row_span

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_bar(i):
            value = frames[frame_index()][i]
            width = max(value / max_value * max_bar_width, 0.02)
            rect = Rectangle(
                width=width,
                height=bar_height,
                fill_color=_PALETTE[i % len(_PALETTE)],
                fill_opacity=1,
                stroke_width=0,
            )
            rect.move_to([left_x + width / 2, row_y(i), 0])
            return rect

        def make_value_text(i):
            value = frames[frame_index()][i]
            width = max(value / max_value * max_bar_width, 0.02)
            text = Text(f"{value:,.0f}", font_size=24, color=WHITE)
            text.move_to([left_x + width + 0.3 + text.width / 2, row_y(i), 0])
            return text

        bars = VGroup(*[always_redraw(lambda i=i: make_bar(i)) for i in range(num_rows)])
        value_texts = VGroup(*[always_redraw(lambda i=i: make_value_text(i)) for i in range(num_rows)])
        row_labels = VGroup(
            *[
                Text(labels[i], font_size=24, color=WHITE).next_to(
                    np.array([left_x, row_y(i), 0]), LEFT, buff=0.2
                )
                for i in range(num_rows)
            ]
        )

        self.add(bars, value_texts, row_labels)

        total_duration = _TARGET_DURATION[self.race_type] / self.speed
        self.play(
            frame_tracker.animate.set_value(len(frames) - 1),
            run_time=total_duration,
            rate_func=linear,
        )

        self.set_caption("Every swap, one step closer")
