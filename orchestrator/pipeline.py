"""Top-level recipe -> final video entry point (README Section 9)."""

import tempfile
from pathlib import Path

from orchestrator.audio import mix_audio
from orchestrator.recipe import load_recipe, validate_recipe
from orchestrator.render import render_silent_video


def produce_video(recipe_path, output_path) -> Path:
    """Loads, validates, renders, and mixes the recipe at `recipe_path`
    into a final video at `output_path`. Returns `output_path`.
    """
    recipe = load_recipe(recipe_path)
    validate_recipe(recipe)

    with tempfile.TemporaryDirectory(prefix="mathanimation_render_") as tmp_dir:
        silent_video_path = render_silent_video(recipe, tmp_dir)
        return mix_audio(silent_video_path, recipe, output_path)
