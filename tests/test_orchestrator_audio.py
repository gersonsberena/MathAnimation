"""orchestrator.audio tests. Uses tiny ffmpeg-lavfi-generated fixtures
(a few frames of solid color, a couple seconds of a sine tone) instead of
a real Manim render or a real music asset -- fast (no rendering), and
doesn't depend on any licensed asset being present.
"""

import json
import subprocess

import pytest

from orchestrator.audio import mix_audio
from tests.orchestrator_test_helpers import make_recipe


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def _make_silent_video(path, duration=2):
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=64x64:d={duration}:r=10", "-pix_fmt", "yuv420p", "-loglevel", "error", str(path)])


def _make_tone(path, duration=5):
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-loglevel", "error", str(path)])


def _stream_types(path):
    result = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)])
    return {s["codec_type"] for s in json.loads(result.stdout)["streams"]}


def test_null_music_track_copies_silent_video_as_final_output(tmp_path):
    silent = tmp_path / "silent.mp4"
    _make_silent_video(silent)
    output = tmp_path / "final.mp4"

    recipe = make_recipe(music_track=None)
    result_path = mix_audio(silent, recipe, output)

    assert result_path == output
    assert output.exists()
    assert _stream_types(output) == {"video"}


def test_missing_music_file_raises(tmp_path):
    silent = tmp_path / "silent.mp4"
    _make_silent_video(silent)
    output = tmp_path / "final.mp4"

    recipe = make_recipe(music_track=str(tmp_path / "does_not_exist.mp3"))
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        mix_audio(silent, recipe, output)


def test_mixes_music_onto_video_producing_both_streams(tmp_path):
    silent = tmp_path / "silent.mp4"
    music = tmp_path / "music.mp3"
    _make_silent_video(silent, duration=2)
    _make_tone(music, duration=5)
    output = tmp_path / "final.mp4"

    recipe = make_recipe(music_track=str(music), music_start_offset=0.5, music_volume=0.4, loop_music=False)
    result_path = mix_audio(silent, recipe, output)

    assert result_path == output
    assert _stream_types(output) == {"video", "audio"}


def test_loops_short_music_to_cover_longer_video(tmp_path):
    silent = tmp_path / "silent.mp4"
    music = tmp_path / "music.mp3"
    _make_silent_video(silent, duration=6)
    _make_tone(music, duration=2)
    output = tmp_path / "final.mp4"

    recipe = make_recipe(music_track=str(music), music_start_offset=0.0, music_volume=1.0, loop_music=True)
    mix_audio(silent, recipe, output)

    result = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=duration", "-of", "json", str(output)])
    audio_duration = float(json.loads(result.stdout)["streams"][0]["duration"])
    assert audio_duration > 5.0  # would be ~2s if it hadn't looped
