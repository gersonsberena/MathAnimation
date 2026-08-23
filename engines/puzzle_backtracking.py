"""puzzle_backtracking engine (README Section 6, engine 17): Tower of
Hanoi self-solving, N-Queens backtracking, Rubik's cube solve. Mechanic:
recursive/backtracking algorithm shown step by step.

rubiks_cube is deliberately scoped down to a 2x2x2 pocket cube (`size`
is restricted to exactly 2) rendered as a 6-face unfolded net rather
than a full 3x3 in 3D — a general cube solver (Kociemba's algorithm or
similar) is disproportionate effort for a single variant of one of 20
engines. Its "solve" is a scramble-then-exact-reversal, not a general
solver: each of the 6 face-turn permutations has group order exactly 4
(a 4-cycle on the turning face's own stickers composed with a 4-group
cycle on its neighbors), so applying any move 3 times is guaranteed —
by pure group theory, independent of whether the specific neighbor/
position bookkeeping below is textbook-perfect cube chirality — to be
that move's exact inverse. Solving by replaying the scramble in reverse
with each move inverted this way therefore *always* returns to the
solved state; this is verified directly (test_solve_always_returns_to_
solved_state) rather than assumed. What's NOT independently verified is
whether each move's neighbor-face bookkeeping matches a real physical
cube turn stroke-for-stroke — the one structural property that IS
checked is that a move never touches its opposite face
(test_move_leaves_opposite_face_untouched), which would catch a
face/adjacency transcription bug even though it can't fully certify
"textbook" chirality.

`build_puzzle()` is pure — no Scene dependency.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GRAY,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Line,
    Rectangle,
    Square,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from engines.base import ReelScene

PUZZLE_TYPES = {"tower_of_hanoi", "n_queens", "rubiks_cube"}
SIZE_RANGES = {"tower_of_hanoi": (2, 8), "n_queens": (4, 8), "rubiks_cube": (2, 2)}
SEED_RANGE = (0, 2**31 - 1)

_RUBIK_SCRAMBLE_LENGTH = 12


def validate_params(puzzle_type, size, seed) -> None:
    if puzzle_type not in PUZZLE_TYPES:
        raise ValueError(f"unknown puzzle_type {puzzle_type!r}, must be one of {sorted(PUZZLE_TYPES)}")

    lo, hi = SIZE_RANGES[puzzle_type]
    if not (lo <= size <= hi):
        raise ValueError(f"size={size} out of range [{lo}, {hi}] for puzzle_type={puzzle_type!r}")

    lo, hi = SEED_RANGE
    if not (lo <= seed <= hi):
        raise ValueError(f"seed={seed} out of range [{lo}, {hi}]")


# ---- Tower of Hanoi ----


def _hanoi_moves(n, source=0, aux=1, target=2, moves=None):
    if moves is None:
        moves = []
    if n == 0:
        return moves
    _hanoi_moves(n - 1, source, target, aux, moves)
    moves.append((n, source, target))
    _hanoi_moves(n - 1, aux, source, target, moves)
    return moves


def _hanoi_frames(num_disks):
    pegs = [list(range(num_disks, 0, -1)), [], []]  # peg 0: disk `num_disks` (largest) at index 0, bottom-to-top
    frames = [[list(p) for p in pegs]]
    moves = _hanoi_moves(num_disks)
    for disk, src, tgt in moves:
        assert pegs[src][-1] == disk, "hanoi move disagrees with actual peg state"
        pegs[tgt].append(pegs[src].pop())
        frames.append([list(p) for p in pegs])
    return frames, moves


# ---- N-Queens ----


def _n_queens_search(n):
    board = [-1] * n  # board[col] = row of the queen in that column, -1 if unplaced
    steps = []  # ("place" | "remove", col, row) in attempt order, for step-by-step playback

    def is_safe(col, row):
        for c in range(col):
            r = board[c]
            if r == row or abs(r - row) == abs(c - col):
                return False
        return True

    def backtrack(col):
        if col == n:
            return True
        for row in range(n):
            if is_safe(col, row):
                board[col] = row
                steps.append(("place", col, row))
                if backtrack(col + 1):
                    return True
                steps.append(("remove", col, row))
                board[col] = -1
        return False

    solved = backtrack(0)
    return steps, solved, (list(board) if solved else None)


def _n_queens_frames(n):
    steps, solved, _final_board = _n_queens_search(n)
    board = [-1] * n
    frames = [list(board)]
    for action, col, row in steps:
        board[col] = row if action == "place" else -1
        frames.append(list(board))
    return frames, solved


# ---- Rubik's cube (2x2x2, unfolded net) ----

_FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
_FACE_INDEX = {name: i for i, name in enumerate(_FACE_NAMES)}
_OPPOSITE_FACE = {"U": "D", "D": "U", "F": "B", "B": "F", "L": "R", "R": "L"}
_FACE_COLORS = {"U": "W", "D": "Y", "F": "G", "B": "B", "L": "O", "R": "R"}

# Each move: (face, [4 neighbor (face, [pos_a, pos_b]) groups in transfer
# order — group[k] receives its 2 stickers from group[k-1]]).
_MOVES = {
    "U": (0, [("F", [0, 1]), ("R", [0, 1]), ("B", [0, 1]), ("L", [0, 1])]),
    "D": (1, [("F", [2, 3]), ("L", [2, 3]), ("B", [2, 3]), ("R", [2, 3])]),
    "F": (2, [("U", [2, 3]), ("R", [0, 2]), ("D", [0, 1]), ("L", [1, 3])]),
    "B": (3, [("U", [0, 1]), ("L", [0, 2]), ("D", [2, 3]), ("R", [1, 3])]),
    "L": (4, [("U", [0, 2]), ("F", [0, 2]), ("D", [0, 2]), ("B", [1, 3])]),
    "R": (5, [("U", [1, 3]), ("B", [0, 2]), ("D", [1, 3]), ("F", [1, 3])]),
}


def _solved_cube_state():
    return [_FACE_COLORS[name] for name in _FACE_NAMES for _ in range(4)]


def _rotate_face_stickers(state, face_index):
    i = face_index * 4
    old = state[i : i + 4]
    state[i : i + 4] = [old[2], old[0], old[3], old[1]]  # 90-degree clockwise relabel


def _cycle_neighbor_groups(state, groups):
    saved = [[state[_FACE_INDEX[face] * 4 + p] for p in positions] for face, positions in groups]
    n = len(groups)
    for k in range(n):
        face, positions = groups[k]
        src = saved[(k - 1) % n]
        for idx, p in enumerate(positions):
            state[_FACE_INDEX[face] * 4 + p] = src[idx]


def _apply_move_once(state, move_name):
    face_index, groups = _MOVES[move_name]
    _rotate_face_stickers(state, face_index)
    _cycle_neighbor_groups(state, groups)


def _apply_move(state, move_name, times=1):
    for _ in range(times % 4):
        _apply_move_once(state, move_name)


def _scramble_sequence(seed, length=_RUBIK_SCRAMBLE_LENGTH):
    rng = np.random.default_rng(seed)
    names = list(_MOVES.keys())
    return [names[i] for i in rng.integers(0, len(names), size=length)]


def _cube_frames(seed):
    state = _solved_cube_state()
    frames = [list(state)]
    scramble = _scramble_sequence(seed)

    for move in scramble:
        _apply_move(state, move, times=1)
        frames.append(list(state))
    for move in reversed(scramble):
        _apply_move(state, move, times=3)  # 3x forward == 1x inverse, see module docstring
        frames.append(list(state))

    return frames, scramble


def build_puzzle(puzzle_type, size, seed):
    """Returns a dict discriminated by `kind`:
      - tower_of_hanoi: `{"kind": "hanoi", "frames", "num_disks"}`
      - n_queens: `{"kind": "n_queens", "frames", "n", "solved"}`
      - rubiks_cube: `{"kind": "rubiks_cube", "frames", "scramble"}`
    Pure — no Scene dependency.
    """
    validate_params(puzzle_type, size, seed)

    if puzzle_type == "tower_of_hanoi":
        frames, _moves = _hanoi_frames(size)
        return {"kind": "hanoi", "frames": frames, "num_disks": size}

    if puzzle_type == "n_queens":
        frames, solved = _n_queens_frames(size)
        return {"kind": "n_queens", "frames": frames, "n": size, "solved": solved}

    if puzzle_type == "rubiks_cube":
        frames, scramble = _cube_frames(seed)
        return {"kind": "rubiks_cube", "frames": frames, "scramble": scramble}

    raise AssertionError(f"unhandled puzzle_type {puzzle_type!r}")  # validate_params already checked


_COLOR_HEX = {"W": "#FFFFFF", "Y": "#FFD23F", "G": "#4CAF50", "B": "#2196F3", "O": "#FF9800", "R": "#E53935"}


class PuzzleBacktrackingReel(ReelScene):
    """`params`: puzzle_type, size, seed (README Section 6, #17)."""

    puzzle_type = "tower_of_hanoi"
    size = 5
    seed = 0
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "Solved, One Step At A Time")

        data = build_puzzle(self.puzzle_type, self.size, self.seed)
        if data["kind"] == "hanoi":
            self._construct_hanoi(data)
        elif data["kind"] == "n_queens":
            self._construct_n_queens(data)
        else:
            self._construct_rubiks_cube(data)

    def _construct_hanoi(self, data):
        frames, num_disks = data["frames"], data["num_disks"]
        zone = self.zones.content_zone

        peg_spacing = zone.width * 0.6 / 2
        base_y = zone.get_bottom()[1] + zone.height * 0.1
        peg_height = zone.height * 0.55
        disk_height = min(0.35, peg_height / (num_disks + 1))
        max_disk_width = peg_spacing * 0.85

        def peg_x(i):
            return zone.get_center()[0] + (i - 1) * peg_spacing

        pegs = VGroup(*[Line([peg_x(i), base_y, 0], [peg_x(i), base_y + peg_height, 0], color=GRAY) for i in range(3)])
        self.add(pegs)

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_disks():
            state = frames[frame_index()]
            group = VGroup()
            for peg_i, stack in enumerate(state):
                for level, disk in enumerate(stack):
                    width = max_disk_width * disk / num_disks
                    rect = Rectangle(width=width, height=disk_height * 0.9, fill_color=BLUE, fill_opacity=1, stroke_width=0)
                    rect.move_to([peg_x(peg_i), base_y + (level + 0.5) * disk_height, 0])
                    group.add(rect)
            return group

        disks = always_redraw(make_disks)
        self.add(disks)
        self.play(frame_tracker.animate.set_value(len(frames) - 1), run_time=5.0, rate_func=linear)
        # `.animate` reaches the exact final value, but the video export's
        # last written frame can land a hair short of it (see README
        # Section 8's ValueTracker-reveal gotcha) — pin the value and hold
        # it so the true final state is what actually gets exported.
        frame_tracker.set_value(len(frames) - 1)
        self.wait(0.3)

        self.set_caption(self.caption_text or f"{num_disks} disks, {len(frames) - 1} moves. Every one necessary.")

    def _construct_n_queens(self, data):
        frames, n = data["frames"], data["n"]
        zone = self.zones.content_zone

        cell_size = min(zone.width * 0.85 / n, zone.height * 0.65 / n)
        origin_x = zone.get_center()[0] - n * cell_size / 2
        origin_y = zone.get_center()[1] + n * cell_size / 2

        def cell_center(col, row):
            return np.array([origin_x + (col + 0.5) * cell_size, origin_y - (row + 0.5) * cell_size, 0.0])

        squares = VGroup(
            *[
                Square(
                    side_length=cell_size,
                    fill_color=GRAY,
                    fill_opacity=0.25 if (r + c) % 2 == 0 else 0.1,
                    stroke_color=WHITE,
                    stroke_width=1,
                ).move_to(cell_center(c, r))
                for r in range(n)
                for c in range(n)
            ]
        )
        self.add(squares)

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_queens():
            board = frames[frame_index()]
            group = VGroup()
            for col, row in enumerate(board):
                if row >= 0:
                    text = Text("Q", font_size=int(cell_size * 28), color=YELLOW)
                    text.move_to(cell_center(col, row))
                    group.add(text)
            return group

        queens = always_redraw(make_queens)
        self.add(queens)
        self.play(frame_tracker.animate.set_value(len(frames) - 1), run_time=5.0, rate_func=linear)
        # `.animate` reaches the exact final value, but the video export's
        # last written frame can land a hair short of it (see README
        # Section 8's ValueTracker-reveal gotcha) — pin the value and hold
        # it so the true final state is what actually gets exported.
        frame_tracker.set_value(len(frames) - 1)
        self.wait(0.3)

        self.set_caption(self.caption_text or f"{n} queens, zero conflicts. Backtracking found it.")

    def _construct_rubiks_cube(self, data):
        frames = data["frames"]
        zone = self.zones.content_zone

        cell = min(zone.width, zone.height) * 0.11
        center = zone.get_center()
        # Unfolded cross layout: U above F, D below F, L/F/R/B in a row.
        face_origin = {
            "U": center + UP * (2 * cell),
            "L": center + LEFT * (2 * cell),
            "F": center,
            "R": center + RIGHT * (2 * cell),
            "B": center + RIGHT * (4 * cell),
            "D": center + DOWN * (2 * cell),
        }
        sticker_offsets = [LEFT * 0.5 + UP * 0.5, RIGHT * 0.5 + UP * 0.5, LEFT * 0.5 + DOWN * 0.5, RIGHT * 0.5 + DOWN * 0.5]

        frame_tracker = ValueTracker(0)

        def frame_index():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_net():
            state = frames[frame_index()]
            group = VGroup()
            for f, name in enumerate(_FACE_NAMES):
                for p in range(4):
                    color = _COLOR_HEX[state[f * 4 + p]]
                    sq = Square(side_length=cell * 0.9, fill_color=color, fill_opacity=1, stroke_color=WHITE, stroke_width=1)
                    sq.move_to(face_origin[name] + sticker_offsets[p] * cell)
                    group.add(sq)
            return group

        net = always_redraw(make_net)
        self.add(net)
        self.play(frame_tracker.animate.set_value(len(frames) - 1), run_time=5.0, rate_func=linear)
        # `.animate` reaches the exact final value, but the video export's
        # last written frame can land a hair short of it (see README
        # Section 8's ValueTracker-reveal gotcha) — pin the value and hold
        # it so the true final state is what actually gets exported.
        frame_tracker.set_value(len(frames) - 1)
        self.wait(0.3)

        self.set_caption(self.caption_text or "Every scramble has a way back")
