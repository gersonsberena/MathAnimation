"""Determinism check (README Section 11.2 item 5): for any engine
accepting `seed`, render twice with the same seed and diff frame
pixels — catches accidental use of unseeded randomness.

Compares *decoded pixel content* of an extracted frame, not raw video
file bytes — an mp4 container can embed non-deterministic metadata
(e.g. encode timestamps) even when every rendered pixel is identical,
which would make a whole-file byte/hash comparison flaky.

SLOW — same reasoning as test_render_smoke.py; excluded from the
default `pytest tests/` run, run explicitly with `pytest tests/ -m slow`.
"""

import pathlib
import subprocess

import numpy as np
import pytest
from PIL import Image

from tests.qa_dispatch import load_recipe, RECIPES_DIR
from tests.test_render_smoke import _QA_RENDER_CONFIG, _final_output_file, _render

# Recipes whose category's Scene accepts a `seed` param — the only ones
# where determinism is meaningful to check.
_SEEDED_RECIPE_NAMES = [
    "monte_carlo_probability_monty_hall",
    "particle_physics_sim_double_pendulum",
    "pathfinding_maze_astar",
    "puzzle_backtracking_hanoi",
    "statistical_distribution_galton_board",
]


def _extract_frame_array(video_path, timestamp="0.8"):
    out_png = video_path.with_suffix(".png")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", timestamp, "-i", str(video_path), "-vframes", "1", "-update", "1", str(out_png)],
        check=True,
        capture_output=True,
    )
    return np.array(Image.open(out_png))


@pytest.mark.slow
@pytest.mark.parametrize("recipe_name", _SEEDED_RECIPE_NAMES)
def test_same_seed_produces_identical_output(recipe_name, tmp_path):
    recipe = load_recipe(RECIPES_DIR / f"{recipe_name}.yaml")
    assert "seed" in recipe["params"], f"{recipe_name} has no seed param — remove it from _SEEDED_RECIPE_NAMES"

    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    _render(recipe, dir_a)
    _render(recipe, dir_b)

    file_a, file_b = _final_output_file(dir_a), _final_output_file(dir_b)
    assert file_a is not None and file_b is not None

    frame_a = _extract_frame_array(file_a)
    frame_b = _extract_frame_array(file_b)
    assert frame_a.shape == frame_b.shape
    assert np.array_equal(frame_a, frame_b), f"{recipe_name}: same seed produced different output — unseeded randomness?"
