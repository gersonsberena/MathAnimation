"""Light tests for game_theory (README Section 6, #15): pure
param-validation, boundary, and payoff/replicator-dynamics correctness
checks only — no Scene/render checks, per the current per-engine test
cadence (README Section 11.6).
"""

import pytest

from engines.game_theory import (
    STRATEGY_TYPES,
    _build_payoff_matrix,
    _initial_population_fractions,
    _move_pavlov,
    _play_match,
    _run_replicator_dynamics,
    build_tournament,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        validate_params(["always_cooperate", "not_a_real_strategy"], num_rounds=20, population_size=50)


def test_rejects_duplicate_strategies():
    with pytest.raises(ValueError):
        validate_params(["tit_for_tat", "tit_for_tat"], num_rounds=20, population_size=50)


def test_rejects_too_few_strategies():
    with pytest.raises(ValueError):
        validate_params(["tit_for_tat"], num_rounds=20, population_size=50)


def test_rejects_too_many_strategies():
    with pytest.raises(ValueError):
        validate_params(list(STRATEGY_TYPES) + ["tit_for_tat_dup"], num_rounds=20, population_size=50)


def test_rejects_num_rounds_out_of_range():
    with pytest.raises(ValueError):
        validate_params(["always_cooperate", "always_defect"], num_rounds=1, population_size=50)
    with pytest.raises(ValueError):
        validate_params(["always_cooperate", "always_defect"], num_rounds=1000, population_size=50)


def test_rejects_population_size_out_of_range():
    with pytest.raises(ValueError):
        validate_params(["always_cooperate", "always_defect"], num_rounds=20, population_size=1)
    with pytest.raises(ValueError):
        validate_params(["always_cooperate", "always_defect"], num_rounds=20, population_size=100_000)


# ---- Payoff / match correctness ----


def test_mutual_cooperation_is_the_reward_payoff():
    avg_a, avg_b = _play_match("always_cooperate", "always_cooperate", num_rounds=10)
    assert avg_a == avg_b == 3


def test_mutual_defection_is_the_punishment_payoff():
    avg_a, avg_b = _play_match("always_defect", "always_defect", num_rounds=10)
    assert avg_a == avg_b == 1


def test_defector_exploits_permanent_cooperator():
    avg_cooperator, avg_defector = _play_match("always_cooperate", "always_defect", num_rounds=10)
    assert avg_cooperator == 0
    assert avg_defector == 5


def test_tit_for_tat_matches_always_cooperate_fully():
    avg_a, avg_b = _play_match("tit_for_tat", "always_cooperate", num_rounds=15)
    assert avg_a == avg_b == 3


def test_tit_for_tat_retaliates_against_a_defector():
    # Round 1: tit_for_tat cooperates (gets S=0), defector gets T=5.
    # Every round after: tit_for_tat mirrors the defector's last move,
    # so it's mutual defection (P=1 each) from round 2 onward.
    num_rounds = 20
    avg_tft, avg_defector = _play_match("tit_for_tat", "always_defect", num_rounds=num_rounds)
    expected_tft = (0 + (num_rounds - 1) * 1) / num_rounds
    expected_defector = (5 + (num_rounds - 1) * 1) / num_rounds
    assert avg_tft == pytest.approx(expected_tft)
    assert avg_defector == pytest.approx(expected_defector)


def test_grim_trigger_cooperates_forever_if_never_defected_against():
    avg_a, avg_b = _play_match("grim_trigger", "always_cooperate", num_rounds=15)
    assert avg_a == avg_b == 3


def test_grim_trigger_punishes_a_single_defection_permanently():
    num_rounds = 20
    avg_grim, avg_defector = _play_match("grim_trigger", "always_defect", num_rounds=num_rounds)
    expected_grim = (0 + (num_rounds - 1) * 1) / num_rounds
    expected_defector = (5 + (num_rounds - 1) * 1) / num_rounds
    assert avg_grim == pytest.approx(expected_grim)
    assert avg_defector == pytest.approx(expected_defector)


def test_pavlov_cooperates_forever_against_a_cooperator():
    avg_a, avg_b = _play_match("pavlov", "always_cooperate", num_rounds=15)
    assert avg_a == avg_b == 3


def _reference_pavlov_walk(opponent_moves):
    # Independent reimplementation of the pavlov rule (own move history
    # is what matters, not a hardcoded expected sequence) used to
    # cross-check the engine's own move-by-move behavior.
    own_history, opp_history = [], []
    for opp_move in opponent_moves:
        move = own_history[-1] == opp_history[-1] if own_history else True
        own_history.append(move)
        opp_history.append(opp_move)
    return own_history


def test_pavlov_move_sequence_matches_independent_reference_against_a_defector():
    num_rounds = 8
    own_history, opp_history = [], []
    for _round in range(num_rounds):
        move = _move_pavlov(own_history, opp_history)
        own_history.append(move)
        opp_history.append(False)  # always_defect
    expected = _reference_pavlov_walk([False] * num_rounds)
    assert own_history == expected
    # Sanity: pavlov settles into an alternating C/D pattern against a
    # permanent defector (never converges to steady cooperation or
    # steady defection the way tit_for_tat/grim_trigger do).
    assert own_history[2:] == [True, False] * ((num_rounds - 2) // 2)


# ---- Payoff matrix / population dynamics ----


def test_payoff_matrix_diagonal_for_always_defect_is_mutual_punishment():
    matrix = _build_payoff_matrix(["always_cooperate", "always_defect"], num_rounds=10)
    assert matrix[1][1] == 1  # always_defect vs itself
    assert matrix[0][0] == 3  # always_cooperate vs itself


def test_initial_population_fractions_sum_to_one_and_split_evenly():
    fractions = _initial_population_fractions(4, population_size=100)
    assert fractions == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_initial_population_fractions_handle_uneven_split():
    fractions = _initial_population_fractions(3, population_size=10)
    assert sum(fractions) == pytest.approx(1.0)
    assert len(fractions) == 3


def test_replicator_dynamics_fractions_always_sum_to_one():
    matrix = _build_payoff_matrix(["always_cooperate", "always_defect", "tit_for_tat"], num_rounds=20)
    initial = _initial_population_fractions(3, population_size=90)
    history = _run_replicator_dynamics(matrix, initial, num_generations=15)
    for gen_fractions in history:
        assert sum(gen_fractions) == pytest.approx(1.0)


def test_dominant_strategy_grows_toward_fixation():
    # always_defect strictly out-scores always_cooperate against every
    # opponent (itself and each other) in a two-strategy population, so
    # under replicator dynamics its population share must increase every
    # generation and approach 1.
    matrix = _build_payoff_matrix(["always_cooperate", "always_defect"], num_rounds=20)
    initial = _initial_population_fractions(2, population_size=50)
    history = _run_replicator_dynamics(matrix, initial, num_generations=30)

    defect_shares = [gen[1] for gen in history]
    assert all(b >= a - 1e-12 for a, b in zip(defect_shares, defect_shares[1:]))
    assert defect_shares[-1] > 0.99


# ---- build_tournament() ----


def test_build_tournament_shape():
    data = build_tournament(["always_cooperate", "always_defect", "tit_for_tat"], num_rounds=30, population_size=60)
    assert data["strategies"] == ["always_cooperate", "always_defect", "tit_for_tat"]
    assert len(data["payoff_matrix"]) == 3
    assert all(len(row) == 3 for row in data["payoff_matrix"])
    assert len(data["population_history"]) == 21  # _NUM_GENERATIONS + 1
    for gen_fractions in data["population_history"]:
        assert len(gen_fractions) == 3


def test_build_tournament_is_deterministic():
    a = build_tournament(["tit_for_tat", "grim_trigger", "pavlov"], num_rounds=25, population_size=40)
    b = build_tournament(["tit_for_tat", "grim_trigger", "pavlov"], num_rounds=25, population_size=40)
    assert a == b
