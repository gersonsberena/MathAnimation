import pytest

from orchestrator.recipe import validate_recipe
from tests.orchestrator_test_helpers import make_recipe


def test_valid_recipe_passes_with_style_field_warning():
    recipe = make_recipe(font="Helvetica", accent_color="#FFFFFF")
    with pytest.warns(UserWarning, match="font.*accent_color|accent_color.*font"):
        validate_recipe(recipe)


def test_valid_recipe_with_no_style_fields_set_does_not_warn(recwarn):
    recipe = make_recipe(font=None, font_size_title=None, font_size_caption=None, accent_color=None, color_palette=None)
    validate_recipe(recipe)
    assert len(recwarn) == 0


def test_missing_required_key_raises():
    recipe = make_recipe()
    del recipe["fps"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_recipe(recipe)


def test_fixed_duration_not_supported():
    recipe = make_recipe(duration=15)
    with pytest.raises(NotImplementedError, match="duration"):
        validate_recipe(recipe)


def test_non_h264_output_format_not_supported():
    recipe = make_recipe(output_format="webm")
    with pytest.raises(NotImplementedError, match="output_format"):
        validate_recipe(recipe)


def test_sfx_enabled_not_supported():
    recipe = make_recipe(sfx_enabled=True)
    with pytest.raises(NotImplementedError, match="sfx_enabled"):
        validate_recipe(recipe)


def test_missing_music_file_raises_before_render():
    recipe = make_recipe(music_track="assets/music/does_not_exist_xyz.mp3")
    with pytest.raises(FileNotFoundError, match="does_not_exist_xyz"):
        validate_recipe(recipe)


def test_unknown_category_raises():
    recipe = make_recipe(category="not_a_real_category", params={})
    with pytest.raises(ValueError, match="unknown category"):
        validate_recipe(recipe)


def test_invalid_params_for_category_raises():
    recipe = make_recipe(params={"puzzle_type": "tower_of_hanoi", "size": 999, "seed": 0})
    with pytest.raises(ValueError, match="out of range"):
        validate_recipe(recipe)
