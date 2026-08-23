"""Shared recipe-dict builder for orchestrator tests -- not a test module
itself (no test_ prefix).
"""

import copy

_BASE_RECIPE = {
    "id": "test-recipe-001",
    "category": "puzzle_backtracking",
    "template_version": 1,
    "title": "Test Title Override",
    "subtitle": None,
    "caption": "Test caption override",
    "fb_post_caption": "test post caption",
    "background": "#000000",
    "color_palette": ["#4EC5E0"],
    "font": None,
    "font_size_title": 44,
    "font_size_caption": 32,
    "accent_color": None,
    "duration": "auto",
    "intro_hold": 0.2,
    "outro_hold": 0.2,
    "fps": 10,
    "music_track": None,
    "music_start_offset": 0.0,
    "music_volume": 0.5,
    "sfx_enabled": False,
    "sfx_volume": 0.8,
    "loop_music": True,
    "resolution": "270x480",
    "safe_zone_top": 0.06,
    "safe_zone_bottom": 0.22,
    "safe_zone_side": 0.16,
    "output_format": "h264_mp4",
    "date_created": "2026-08-23",
    "status": "draft",
    "params": {"puzzle_type": "tower_of_hanoi", "size": 2, "seed": 0},
}


def make_recipe(**overrides) -> dict:
    """Returns a deep copy of a minimal, otherwise-valid recipe dict
    (category=puzzle_backtracking, low-res/fps for fast test renders),
    with `overrides` applied at the top level.
    """
    recipe = copy.deepcopy(_BASE_RECIPE)
    recipe.update(overrides)
    return recipe
