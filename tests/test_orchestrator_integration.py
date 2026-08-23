"""End-to-end orchestrator pipeline tests (README Section 9): recipe YAML
in, final muxed video out. SLOW -- does a real (if low-res/short) Manim
render. Excluded from the default `pytest tests/` run, same as
test_render_smoke.py/test_determinism.py; run with `pytest tests/ -m
slow`.

Uses a synthetic ffmpeg-generated tone (tests/orchestrator_test_helpers'
make_recipe already points nowhere by default) rather than a real
licensed music asset -- none exist in assets/music/ yet (README Section
11.4/README Section 9 known-gap note).
"""

import json
import subprocess

import pytest
import yaml

from orchestrator.pipeline import produce_video
from tests.orchestrator_test_helpers import make_recipe


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def _make_tone(path, duration=8):
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-loglevel", "error", str(path)])


def _stream_types(path):
    result = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)])
    return {s["codec_type"] for s in json.loads(result.stdout)["streams"]}


def _write_recipe(recipe, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)
    return path


@pytest.mark.slow
def test_produce_video_with_music_end_to_end(tmp_path):
    music = tmp_path / "music.mp3"
    _make_tone(music)

    recipe = make_recipe(music_track=str(music), music_start_offset=0.5, music_volume=0.5, loop_music=True)
    recipe_path = _write_recipe(recipe, tmp_path / "recipe.yaml")
    output_path = tmp_path / "final.mp4"

    result_path = produce_video(recipe_path, output_path)

    assert result_path == output_path
    assert output_path.exists() and output_path.stat().st_size > 0
    assert _stream_types(output_path) == {"video", "audio"}


@pytest.mark.slow
def test_produce_video_without_music_is_video_only(tmp_path):
    recipe = make_recipe(music_track=None)
    recipe_path = _write_recipe(recipe, tmp_path / "recipe.yaml")
    output_path = tmp_path / "final.mp4"

    produce_video(recipe_path, output_path)

    assert output_path.exists() and output_path.stat().st_size > 0
    assert _stream_types(output_path) == {"video"}


@pytest.mark.slow
def test_render_applies_title_and_caption_overrides(tmp_path):
    from manim import tempconfig

    from engines.registry import scene_class_for_category

    recipe = make_recipe(title="Overridden Title Text", caption="Overridden caption text")
    scene_cls = scene_class_for_category(recipe["category"])

    render_config = {
        "pixel_width": 270,
        "pixel_height": 480,
        "frame_width": 9,
        "frame_height": 16,
        "frame_rate": 10,
        "media_dir": str(tmp_path),
        "disable_caching": True,
        "verbosity": "ERROR",
        "progress_bar": "none",
    }
    with tempconfig(render_config):
        scene = scene_cls()
        scene.title_text = recipe["title"]
        scene.caption_text = recipe["caption"]
        scene.puzzle_type = recipe["params"]["puzzle_type"]
        scene.size = recipe["params"]["size"]
        scene.seed = recipe["params"]["seed"]
        scene.render()

    # Manim's Text.text strips internal spaces (a Manim quirk, not this
    # project's behavior) -- both sides compared stripped.
    assert scene.title_mobject.text == recipe["title"].replace(" ", "")
    assert scene.caption_mobject.text == recipe["caption"].replace(" ", "")


@pytest.mark.slow
def test_missing_recipe_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        produce_video(tmp_path / "does_not_exist.yaml", tmp_path / "out.mp4")
