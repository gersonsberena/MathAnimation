"""Mixes a recipe's music track onto a rendered silent video and produces
the final encoded output (README Section 9, steps 5-6).

`music_track: null` is a valid recipe -- the silent video is used as the
final output as-is, no ffmpeg pass. A non-null `music_track` that doesn't
exist on disk raises `FileNotFoundError` rather than silently shipping a
video without the music the recipe asked for (README Section 8's "fail
loudly" pattern -- the same reasoning `fit_to_zone` uses for layout).
`orchestrator.recipe.validate_recipe` already checks this before the
(expensive) render step runs; the check here is a defensive re-check for
any caller that reaches this module without going through validation.

Mixing command, verified manually against synthetic fixtures before
writing this module: seek the music input to `music_start_offset`,
optionally `-stream_loop -1` it to cover videos longer than the track,
apply `music_volume` as a `volume` filter, and `-shortest` to trim to the
(always shorter, non-looped) video's length. Video is stream-copied
untouched; only the audio is (re-)encoded.
"""

import shutil
import subprocess
from pathlib import Path

from orchestrator.recipe import REPO_ROOT


def _resolve_music_path(music_track: str) -> Path:
    path = Path(music_track)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def mix_audio(silent_video_path, recipe: dict, output_path) -> Path:
    """Produces the final video at `output_path`. Returns `output_path`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    music_track = recipe["music_track"]
    if not music_track:
        shutil.copyfile(silent_video_path, output_path)
        return output_path

    music_path = _resolve_music_path(music_track)
    if not music_path.exists():
        raise FileNotFoundError(f"recipe's music_track {music_track!r} does not exist on disk (resolved: {music_path})")

    cmd = ["ffmpeg", "-y", "-i", str(silent_video_path)]
    if recipe["loop_music"]:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-ss", str(recipe["music_start_offset"]), "-i", str(music_path)]
    cmd += [
        "-filter_complex",
        f"[1:a]volume={recipe['music_volume']}[aout]",
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-loglevel",
        "error",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed (exit {result.returncode}):\n{result.stderr}")

    return output_path
