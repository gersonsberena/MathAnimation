"""game_theory engine (README Section 6, engine 15): iterated prisoner's
dilemma tournament, evolving strategy population. Mechanic: repeated
matchups + score tracking + population shift over generations.

README's params (`strategies, num_rounds, population_size`) have no
`seed` — a strong signal this engine is meant to be fully deterministic,
so the strategy roster is restricted to classic deterministic IPD
strategies (no coin-flip "random" strategy).

Individuals of the same strategy type always behave identically against
a given opponent type, so there's no need to simulate every pairwise
individual match: one deterministic `num_rounds`-round match per
(strategy, strategy) pair yields a payoff matrix, and population shift
across generations is then standard discrete replicator dynamics over
population *fractions* — the textbook way to simulate evolutionary game
theory without O(population_size^2) per-individual simulation.

Rendering reuses array_bar_race's Rectangle + ValueTracker +
always_redraw bar-chart pattern: bars = strategies, height = population
fraction, animated across generations.

`build_tournament()` is pure — no Scene dependency.
"""

import numpy as np
from manim import LEFT, WHITE, Rectangle, Text, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

STRATEGY_TYPES = {"always_cooperate", "always_defect", "tit_for_tat", "grim_trigger", "pavlov"}
STRATEGIES_COUNT_RANGE = (2, len(STRATEGY_TYPES))
NUM_ROUNDS_RANGE = (5, 500)
POPULATION_SIZE_RANGE = (10, 10_000)

# Classic Axelrod prisoner's-dilemma payoffs: R=reward (mutual
# cooperate), S=sucker, T=temptation (mutual defect exploited), P=
# punishment (mutual defect). Requires T > R > P > S and 2R > T + S,
# both satisfied here (5 > 3 > 1 > 0, 6 > 5).
_R, _S, _T, _P = 3, 0, 5, 1
_NUM_GENERATIONS = 20
_PALETTE = ["#00C2FF", "#FF3EA5", "#FFD23F", "#4EA8FF", "#7CFFB2"]


def validate_params(strategies, num_rounds, population_size) -> None:
    if not isinstance(strategies, (list, tuple)):
        raise ValueError("strategies must be a list")

    lo, hi = STRATEGIES_COUNT_RANGE
    if not (lo <= len(strategies) <= hi):
        raise ValueError(f"strategies length {len(strategies)} out of range [{lo}, {hi}]")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    for s in strategies:
        if s not in STRATEGY_TYPES:
            raise ValueError(f"unknown strategy {s!r}, must be one of {sorted(STRATEGY_TYPES)}")

    lo, hi = NUM_ROUNDS_RANGE
    if not (lo <= num_rounds <= hi):
        raise ValueError(f"num_rounds={num_rounds} out of range [{lo}, {hi}]")

    lo, hi = POPULATION_SIZE_RANGE
    if not (lo <= population_size <= hi):
        raise ValueError(f"population_size={population_size} out of range [{lo}, {hi}]")
    if population_size < len(strategies):
        raise ValueError("population_size must be at least len(strategies)")


def _move_always_cooperate(own_history, opp_history):
    return True


def _move_always_defect(own_history, opp_history):
    return False


def _move_tit_for_tat(own_history, opp_history):
    return opp_history[-1] if opp_history else True


def _move_grim_trigger(own_history, opp_history):
    return all(opp_history)  # vacuously True before opponent's first move


def _move_pavlov(own_history, opp_history):
    # Win-stay-lose-shift, equivalently: cooperate iff both players made
    # the same move last round.
    if not own_history:
        return True
    return own_history[-1] == opp_history[-1]


_STRATEGY_FUNCS = {
    "always_cooperate": _move_always_cooperate,
    "always_defect": _move_always_defect,
    "tit_for_tat": _move_tit_for_tat,
    "grim_trigger": _move_grim_trigger,
    "pavlov": _move_pavlov,
}


def _payoff(move_a, move_b):
    if move_a and move_b:
        return _R, _R
    if move_a and not move_b:
        return _S, _T
    if not move_a and move_b:
        return _T, _S
    return _P, _P


def _play_match(strategy_a, strategy_b, num_rounds):
    """Average per-round payoff for each side over one deterministic
    `num_rounds`-round match.
    """
    hist_a, hist_b = [], []
    total_a = total_b = 0
    func_a, func_b = _STRATEGY_FUNCS[strategy_a], _STRATEGY_FUNCS[strategy_b]
    for _round in range(num_rounds):
        move_a, move_b = func_a(hist_a, hist_b), func_b(hist_b, hist_a)
        pay_a, pay_b = _payoff(move_a, move_b)
        total_a += pay_a
        total_b += pay_b
        hist_a.append(move_a)
        hist_b.append(move_b)
    return total_a / num_rounds, total_b / num_rounds


def _build_payoff_matrix(strategies, num_rounds):
    n = len(strategies)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            avg_i, _avg_j = _play_match(strategies[i], strategies[j], num_rounds)
            matrix[i][j] = avg_i
    return matrix


def _initial_population_fractions(n, population_size):
    base, remainder = divmod(population_size, n)
    counts = [base + (1 if i < remainder else 0) for i in range(n)]
    return [c / population_size for c in counts]


def _run_replicator_dynamics(payoff_matrix, initial_fractions, num_generations):
    n = len(initial_fractions)
    fractions = list(initial_fractions)
    history = [list(fractions)]

    for _gen in range(num_generations):
        expected = [sum(fractions[j] * payoff_matrix[i][j] for j in range(n)) for i in range(n)]
        avg_payoff = sum(fractions[i] * expected[i] for i in range(n))
        fractions = [fractions[i] * expected[i] / avg_payoff for i in range(n)]
        total = sum(fractions)  # renormalize against float drift
        fractions = [f / total for f in fractions]
        history.append(list(fractions))

    return history


def build_tournament(strategies, num_rounds, population_size):
    """Returns `{"strategies", "payoff_matrix", "population_history"}`.
    `population_history` is a list of per-generation fraction lists
    (length `_NUM_GENERATIONS + 1`, including the starting distribution),
    one fraction per strategy in `strategies` order. Pure — no Scene
    dependency.
    """
    validate_params(strategies, num_rounds, population_size)

    payoff_matrix = _build_payoff_matrix(strategies, num_rounds)
    initial_fractions = _initial_population_fractions(len(strategies), population_size)
    population_history = _run_replicator_dynamics(payoff_matrix, initial_fractions, _NUM_GENERATIONS)

    return {
        "strategies": list(strategies),
        "payoff_matrix": payoff_matrix,
        "population_history": population_history,
    }


_DISPLAY_NAMES = {
    "always_cooperate": "Always Cooperate",
    "always_defect": "Always Defect",
    "tit_for_tat": "Tit for Tat",
    "grim_trigger": "Grim Trigger",
    "pavlov": "Pavlov",
}


class GameTheoryReel(ReelScene):
    """`params`: strategies, num_rounds, population_size (README Section
    6, #15).
    """

    strategies = ["always_cooperate", "always_defect", "tit_for_tat", "grim_trigger"]
    num_rounds = 50
    population_size = 100

    def construct(self):
        self.set_title("Only The Fittest Strategy Survives")

        data = build_tournament(self.strategies, self.num_rounds, self.population_size)
        history = data["population_history"]
        labels = [_DISPLAY_NAMES[s] for s in data["strategies"]]
        num_rows = len(labels)

        zone = self.zones.content_zone
        row_span = zone.height * 0.9 / num_rows
        bar_height = row_span * 0.6
        max_bar_width = zone.width * 0.55
        left_x = zone.get_left()[0] + zone.width * 0.32
        top_y = zone.get_top()[1] - row_span / 2

        def row_y(i):
            return top_y - i * row_span

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), len(history) - 1))

        def make_bar(i):
            fraction = history[frame_index()][i]
            width = max(fraction * max_bar_width, 0.02)
            rect = Rectangle(
                width=width, height=bar_height, fill_color=_PALETTE[i % len(_PALETTE)], fill_opacity=1, stroke_width=0
            )
            rect.move_to([left_x + width / 2, row_y(i), 0])
            return rect

        def make_value_text(i):
            fraction = history[frame_index()][i]
            width = max(fraction * max_bar_width, 0.02)
            text = Text(f"{fraction * 100:.0f}%", font_size=24, color=WHITE)
            text.move_to([left_x + width + 0.3 + text.width / 2, row_y(i), 0])
            return text

        bars = VGroup(*[always_redraw(lambda i=i: make_bar(i)) for i in range(num_rows)])
        value_texts = VGroup(*[always_redraw(lambda i=i: make_value_text(i)) for i in range(num_rows)])
        row_labels = VGroup(
            *[
                Text(labels[i], font_size=22, color=WHITE).next_to(np.array([left_x, row_y(i), 0]), LEFT, buff=0.2)
                for i in range(num_rows)
            ]
        )

        self.add(bars, value_texts, row_labels)
        self.play(frame_tracker.animate.set_value(len(history) - 1), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(len(history) - 1)
        self.wait(0.3)

        self.set_caption("Cooperation isn't just nice. It's strategy.")
