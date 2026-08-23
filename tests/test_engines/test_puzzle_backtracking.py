"""Light tests for puzzle_backtracking (README Section 6, #17): pure
param-validation, boundary, and search-correctness checks only — no
Scene/render checks, per the current per-engine test cadence (README
Section 11.6).
"""

import pytest

from engines.puzzle_backtracking import (
    _FACE_NAMES,
    _MOVES,
    _OPPOSITE_FACE,
    _apply_move,
    _cube_frames,
    _hanoi_frames,
    _n_queens_frames,
    _n_queens_search,
    _solved_cube_state,
    build_puzzle,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_puzzle_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_puzzle", size=5, seed=0)


def test_rejects_size_out_of_range_per_puzzle_type():
    with pytest.raises(ValueError):
        validate_params("tower_of_hanoi", size=1, seed=0)
    with pytest.raises(ValueError):
        validate_params("n_queens", size=2, seed=0)  # unsolvable, excluded by range
    with pytest.raises(ValueError):
        validate_params("rubiks_cube", size=3, seed=0)  # only 2x2 supported


def test_rejects_seed_out_of_range():
    with pytest.raises(ValueError):
        validate_params("tower_of_hanoi", size=5, seed=-1)
    with pytest.raises(ValueError):
        validate_params("tower_of_hanoi", size=5, seed=2**31)


# ---- Tower of Hanoi ----


@pytest.mark.parametrize("num_disks", [2, 3, 5, 8])
def test_hanoi_move_count_matches_2_pow_n_minus_1(num_disks):
    frames, moves = _hanoi_frames(num_disks)
    assert len(moves) == 2**num_disks - 1
    assert len(frames) == len(moves) + 1


def test_hanoi_ends_with_all_disks_on_the_target_peg_in_order():
    frames, _moves = _hanoi_frames(6)
    final_state = frames[-1]
    assert final_state[0] == []
    assert final_state[1] == []
    assert final_state[2] == list(range(6, 0, -1))  # largest at bottom


def test_hanoi_starts_with_all_disks_on_the_source_peg():
    frames, _moves = _hanoi_frames(4)
    assert frames[0][0] == [4, 3, 2, 1]
    assert frames[0][1] == frames[0][2] == []


# ---- N-Queens ----


@pytest.mark.parametrize("n", [4, 5, 6, 8])
def test_n_queens_finds_a_conflict_free_solution(n):
    _steps, solved, board = _n_queens_search(n)
    assert solved
    assert len(board) == n
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            r1, r2 = board[c1], board[c2]
            assert r1 != r2
            assert abs(r1 - r2) != abs(c1 - c2)


def test_n_queens_frames_final_frame_matches_the_solution():
    frames, solved = _n_queens_frames(6)
    assert solved
    final = frames[-1]
    for c1 in range(6):
        for c2 in range(c1 + 1, 6):
            r1, r2 = final[c1], final[c2]
            assert r1 != -1 and r2 != -1
            assert r1 != r2
            assert abs(r1 - r2) != abs(c1 - c2)


def test_n_queens_frames_start_empty():
    frames, _solved = _n_queens_frames(5)
    assert frames[0] == [-1] * 5


# ---- Rubik's cube (2x2x2) ----


def test_solved_cube_state_has_one_color_per_face():
    state = _solved_cube_state()
    for f in range(6):
        face_stickers = state[f * 4 : f * 4 + 4]
        assert len(set(face_stickers)) == 1


def test_every_move_returns_to_start_after_four_applications():
    for move_name in _MOVES:
        state = _solved_cube_state()
        original = list(state)
        for _ in range(4):
            _apply_move(state, move_name, times=1)
        assert state == original


def test_every_move_leaves_its_opposite_face_untouched():
    for move_name, (face_index, _groups) in _MOVES.items():
        opposite_name = _OPPOSITE_FACE[move_name]
        opposite_index = _FACE_NAMES.index(opposite_name)
        state = _solved_cube_state()
        # Scramble first so a same-color face can't hide a real bug.
        _apply_move(state, move_name, times=1)
        before = state[opposite_index * 4 : opposite_index * 4 + 4]
        _apply_move(state, move_name, times=1)
        after = state[opposite_index * 4 : opposite_index * 4 + 4]
        assert before == after


def test_applying_a_move_then_its_inverse_restores_state():
    state = _solved_cube_state()
    _apply_move(state, "F", times=1)
    _apply_move(state, "U", times=1)
    snapshot = list(state)
    _apply_move(state, "U", times=3)  # inverse of U
    _apply_move(state, "F", times=3)  # inverse of F
    assert state == _solved_cube_state()
    # sanity: the snapshot really was scrambled, not accidentally solved
    assert snapshot != _solved_cube_state()


def test_solve_always_returns_to_solved_state():
    for seed in range(10):
        frames, _scramble = _cube_frames(seed)
        assert frames[0] == _solved_cube_state()
        assert frames[-1] == _solved_cube_state()


def test_cube_frames_length_matches_scramble_plus_solve():
    frames, scramble = _cube_frames(seed=3)
    assert len(frames) == 1 + 2 * len(scramble)


# ---- build_puzzle() ----


def test_build_puzzle_hanoi_shape():
    data = build_puzzle("tower_of_hanoi", size=4, seed=0)
    assert data["kind"] == "hanoi"
    assert data["num_disks"] == 4


def test_build_puzzle_n_queens_shape():
    data = build_puzzle("n_queens", size=5, seed=0)
    assert data["kind"] == "n_queens"
    assert data["solved"]


def test_build_puzzle_rubiks_cube_shape():
    data = build_puzzle("rubiks_cube", size=2, seed=7)
    assert data["kind"] == "rubiks_cube"
    assert data["frames"][-1] == _solved_cube_state()


def test_build_puzzle_is_deterministic():
    a = build_puzzle("rubiks_cube", size=2, seed=1)
    b = build_puzzle("rubiks_cube", size=2, seed=1)
    assert a == b
