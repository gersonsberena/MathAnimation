"""Renders a recipe's target engine to a silent video (README Section 9,
steps 3-4: dispatch + render at real resolution/fps/duration).

Config (resolution/fps/background/safe zones) is applied via `tempconfig`
for the duration of the render, the same pattern `tests/test_render_
smoke.py` uses. `intro_hold`/`outro_hold` are applied generically, with
no per-engine changes, by wrapping the target Scene's `construct()` in a
dynamically-created subclass that waits before/after calling the real
`construct()`.
"""

import pathlib

from manim import tempconfig

from engines.registry import PARAM_KEY_OVERRIDES, scene_class_for_category


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width_str, height_str = resolution.lower().split("x")
        width, height = int(width_str), int(height_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"resolution must look like '1080x1920', got {resolution!r}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"resolution dimensions must be positive, got {resolution!r}")

    # Engines' internal geometry is written in scene units relative to a
    # fixed 9:16 frame_width/frame_height (engines/base.py), not to pixel
    # dimensions -- a different pixel resolution only changes output
    # crispness and is safe to support generically, but a different
    # ASPECT ratio would silently distort every engine's zone math, so
    # it's rejected rather than silently stretched.
    if abs(width / height - 9 / 16) > 1e-3:
        raise ValueError(f"resolution {resolution!r} is not a 9:16 aspect ratio (engines assume 9:16 scene geometry)")

    return width, height


def _make_scene_class(base_cls, intro_hold: float, outro_hold: float):
    class _WithHolds(base_cls):
        def construct(self):
            if intro_hold > 0:
                self.wait(intro_hold)
            super().construct()
            if outro_hold > 0:
                self.wait(outro_hold)

    _WithHolds.__name__ = f"{base_cls.__name__}WithHolds"
    return _WithHolds


def _final_output_file(media_dir) -> pathlib.Path:
    candidates = [p for p in pathlib.Path(media_dir).rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if not candidates:
        raise RuntimeError(f"render produced no combined .mp4 output under {media_dir}")
    return candidates[0]


def render_silent_video(recipe: dict, media_dir) -> pathlib.Path:
    """Renders the recipe's target engine into `media_dir`, applying
    resolution/fps/background/safe-zone/title/caption/intro-outro-hold
    overrides plus the recipe's `params`. Returns the path to the final
    combined (silent) .mp4.
    """
    category = recipe["category"]
    width, height = _parse_resolution(recipe["resolution"])

    scene_cls = scene_class_for_category(category)
    wrapped_cls = _make_scene_class(scene_cls, recipe["intro_hold"], recipe["outro_hold"])

    render_config = {
        "pixel_width": width,
        "pixel_height": height,
        "frame_width": 9,
        "frame_height": 16,
        "frame_rate": recipe["fps"],
        "background_color": recipe["background"],
        "media_dir": str(media_dir),
        "disable_caching": True,
        "verbosity": "ERROR",
        "progress_bar": "none",
    }

    with tempconfig(render_config):
        scene = wrapped_cls()
        scene.safe_zone_top = recipe["safe_zone_top"]
        scene.safe_zone_bottom = recipe["safe_zone_bottom"]
        scene.safe_zone_side = recipe["safe_zone_side"]
        scene.title_text = recipe["title"]
        if recipe.get("caption"):
            scene.caption_text = recipe["caption"]

        overrides = PARAM_KEY_OVERRIDES.get(category, {})
        for key, value in recipe["params"].items():
            attr = overrides.get(key, key)
            setattr(scene, attr, value)

        scene.render()

    return _final_output_file(media_dir)
