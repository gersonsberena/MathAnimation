"""Render smoke test (README Section 11.2 item 4) + per-engine content
zone-overlap check (item 1, applied to every real engine's actual
rendered output rather than just the base ReelScene's title/caption
placement, which test_layout.py already covers).

SLOW — does a real (if low-res/short) Manim render per recipe. Excluded
from the default `pytest tests/` run (see pyproject.toml's `-m "not
slow"` default and README Section 11.6 step 3's fast-CI / nightly-CI
split); run explicitly with `pytest tests/ -m slow`.
"""

import pathlib

import pytest
from manim import tempconfig

from tests.geometry_helpers import overlaps
from tests.qa_dispatch import all_recipe_paths, build_scene, load_recipe

# Low-res, low-framerate, but the *real* 9:16 aspect ratio — frame_width/
# frame_height must be restated here even though engines/base.py already
# sets them at import time, because tempconfig overrides them for the
# duration of this render and Zones are built from config at Scene.setup()
# time (see engines/base.py's _build_zones), not baked in at import time.
_QA_RENDER_CONFIG = {
    "pixel_width": 270,
    "pixel_height": 480,
    "frame_width": 9,
    "frame_height": 16,
    "frame_rate": 10,
    "disable_caching": True,
    "verbosity": "ERROR",
    "progress_bar": "none",
}


def _render(recipe, media_dir):
    with tempconfig({**_QA_RENDER_CONFIG, "media_dir": str(media_dir)}):
        scene = build_scene(recipe)
        scene.render()
        return scene


def _final_output_file(media_dir):
    candidates = [p for p in pathlib.Path(media_dir).rglob("*.mp4") if "partial_movie_files" not in p.parts]
    return candidates[0] if candidates else None


@pytest.mark.slow
@pytest.mark.parametrize("recipe_path", all_recipe_paths(), ids=lambda p: p.stem)
def test_recipe_renders_without_exception(recipe_path, tmp_path):
    recipe = load_recipe(recipe_path)
    _render(recipe, tmp_path)

    output_file = _final_output_file(tmp_path)
    assert output_file is not None, f"no combined .mp4 output found for {recipe_path.name}"
    assert output_file.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.parametrize("recipe_path", all_recipe_paths(), ids=lambda p: p.stem)
def test_recipe_content_does_not_overlap_title_or_caption_zone(recipe_path, tmp_path):
    recipe = load_recipe(recipe_path)
    scene = _render(recipe, tmp_path)

    non_chrome_mobjects = [m for m in scene.mobjects if m is not scene.title_mobject and m is not scene.caption_mobject]
    for mobject in non_chrome_mobjects:
        assert not overlaps(mobject, scene.zones.title_zone), f"{recipe_path.name}: content overlaps title_zone"
        assert not overlaps(mobject, scene.zones.caption_zone), f"{recipe_path.name}: content overlaps caption_zone"
