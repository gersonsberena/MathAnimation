"""Recipe loading and validation (README Section 9, steps 1-2).

Validates the recipe's top-level schema fields (README Section 5) plus,
via `engines.registry.validate_params_for_category`, the category-
specific `params` block against that engine's own `validate_params()`.

V1 scope (see README Section 9 for the full rationale):
  - `duration` must be `"auto"` (the engine's natural length) -- a fixed-
    seconds override would require generically rescaling every engine's
    internal animation timing, a real architecture change, so it's
    rejected loudly rather than silently ignored.
  - `output_format` must be `"h264_mp4"`, the only encoder path V1
    implements.
  - `sfx_enabled: true` is rejected -- no engine defines sfx trigger
    points or an asset field for it, so honoring it would silently do
    nothing.
  - `font`, `font_size_title`, `font_size_caption`, `accent_color`,
    `color_palette` are recognized but not applied to the render. Unlike
    the fields above, an unapplied style choice is immediately visible on
    watching the output and not a functional correctness problem, so
    this only warns (via the `warnings` module) rather than raising.
  - `subtitle`, `fb_post_caption`, `sfx_volume` are metadata no engine
    renders; accepted and otherwise unused.
  - A non-null `music_track` that doesn't exist on disk raises here (not
    only in `orchestrator.audio`, which re-checks defensively) so a
    missing asset fails before the expensive render step runs, not after.
  - `music_mood` (see `MUSIC_MOODS`) must be one of a fixed vocabulary,
    and -- when `music_track` is set -- must match the name of the
    `assets/music/<mood>/` folder the track actually lives in. The mood
    folder is the source of truth for a track's mood; this just catches
    the two fields drifting apart (e.g. a track moved to a different
    mood folder without updating `music_mood` to match).
"""

import warnings
from pathlib import Path

import yaml

from engines.registry import validate_params_for_category

REPO_ROOT = Path(__file__).resolve().parent.parent

# assets/music/<mood>/ -- see README Section 5. Kept small and by feel
# (not by math category) so a handful of tracks per mood covers all 19
# engines without every recipe needing its own track.
MUSIC_MOODS = {
    "tense": "suspenseful/building -- reveal-driven engines (search, backtracking, probability puzzles)",
    "ambient": "calm/minimal -- contemplative engines and elegant-proof reveals",
    "rhythmic": "driving/repetitive -- generative growth or chaotic-motion engines",
    "playful": "upbeat/energetic -- competitive or lighthearted engines",
}

REQUIRED_KEYS = {
    "id",
    "category",
    "template_version",
    "title",
    "subtitle",
    "caption",
    "fb_post_caption",
    "background",
    "color_palette",
    "font",
    "font_size_title",
    "font_size_caption",
    "accent_color",
    "duration",
    "intro_hold",
    "outro_hold",
    "fps",
    "music_track",
    "music_mood",
    "music_start_offset",
    "music_volume",
    "sfx_enabled",
    "sfx_volume",
    "loop_music",
    "resolution",
    "safe_zone_top",
    "safe_zone_bottom",
    "safe_zone_side",
    "output_format",
    "date_created",
    "status",
    "params",
}

_UNAPPLIED_STYLE_FIELDS = ("font", "font_size_title", "font_size_caption", "accent_color", "color_palette")


def load_recipe(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_recipe(recipe: dict) -> None:
    """Raises ValueError/NotImplementedError on anything the orchestrator
    can't honor. Warns (doesn't raise) for cosmetic fields it silently
    doesn't apply. Does not touch disk or render anything.
    """
    missing = REQUIRED_KEYS - recipe.keys()
    if missing:
        raise ValueError(f"recipe missing required key(s): {sorted(missing)}")

    if recipe["duration"] != "auto":
        raise NotImplementedError(
            f"duration={recipe['duration']!r} is not supported in V1 -- only "
            "'auto' (the engine's natural length) is implemented; a fixed-"
            "seconds override would require rescaling per-engine animation "
            "timing"
        )

    if recipe["output_format"] != "h264_mp4":
        raise NotImplementedError(f"output_format={recipe['output_format']!r} is not supported in V1 -- only 'h264_mp4' is implemented")

    if recipe["sfx_enabled"]:
        raise NotImplementedError("sfx_enabled=true is not supported in V1 -- no engine defines sfx trigger points or an asset path for it")

    music_mood = recipe["music_mood"]
    if music_mood not in MUSIC_MOODS:
        raise ValueError(f"music_mood={music_mood!r} is not one of {sorted(MUSIC_MOODS)}")

    music_track = recipe["music_track"]
    if music_track:
        music_path = Path(music_track)
        if not music_path.is_absolute():
            music_path = REPO_ROOT / music_path
        if not music_path.exists():
            raise FileNotFoundError(f"recipe's music_track {music_track!r} does not exist on disk (resolved: {music_path})")
        if music_path.parent.name != music_mood:
            raise ValueError(
                f"music_track {music_track!r} lives under a {music_path.parent.name!r} folder but "
                f"music_mood={music_mood!r} -- the mood folder is the source of truth, update one to match the other"
            )

    applied_style_fields = [field for field in _UNAPPLIED_STYLE_FIELDS if recipe.get(field) is not None]
    if applied_style_fields:
        warnings.warn(
            f"recipe sets {applied_style_fields}, but V1 does not apply these to the render (README Section 9 known limitation) -- output will use each engine's built-in styling",
            stacklevel=2,
        )

    validate_params_for_category(recipe["category"], recipe["params"])
