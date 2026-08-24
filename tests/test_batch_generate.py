import datetime
import warnings

import pytest

from batch.generate import generate_batch, generate_recipe, load_variations
from engines.registry import CATEGORY_TO_SCENE_CLASS
from orchestrator.recipe import validate_recipe

_A_DATE = datetime.date(2026, 8, 24)


def test_generate_recipe_uses_variant_text():
    recipe0 = generate_recipe("puzzle_backtracking", variant_index=0, date=_A_DATE)
    recipe1 = generate_recipe("puzzle_backtracking", variant_index=1, date=_A_DATE)
    assert recipe0["title"] != recipe1["title"]
    assert recipe0["caption"] != recipe1["caption"]


def test_generate_recipe_wraps_around_variant_index():
    num_variants = len(load_variations("puzzle_backtracking"))
    recipe = generate_recipe("puzzle_backtracking", variant_index=0, date=_A_DATE)
    wrapped = generate_recipe("puzzle_backtracking", variant_index=num_variants, date=_A_DATE)
    assert wrapped["title"] == recipe["title"]


def test_generate_recipe_sets_fresh_id_and_date():
    recipe = generate_recipe("puzzle_backtracking", variant_index=0, date=_A_DATE)
    assert recipe["date_created"] == "2026-08-24"
    assert "2026-08-24" in recipe["id"]
    assert recipe["status"] == "draft"


def test_generate_recipe_keeps_template_params_unchanged():
    recipe = generate_recipe("puzzle_backtracking", variant_index=0, date=_A_DATE)
    assert recipe["params"] == {"puzzle_type": "tower_of_hanoi", "size": 5, "seed": 0}


def test_generate_recipe_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown category"):
        generate_recipe("not_a_real_category", variant_index=0, date=_A_DATE)


def test_generate_batch_returns_one_per_category():
    categories = ["puzzle_backtracking", "wave_signal"]
    batch = generate_batch(categories, date=_A_DATE)
    assert set(batch) == set(categories)


@pytest.mark.parametrize("category", sorted(CATEGORY_TO_SCENE_CLASS))
def test_every_registered_category_has_variations_and_a_template(category):
    variations = load_variations(category)
    assert len(variations) >= 1
    for variant in variations:
        assert variant["title"] and variant["caption"] and variant["fb_post_caption"]
    generate_recipe(category, variant_index=0, date=_A_DATE)  # raises if no template recipe file exists


@pytest.mark.parametrize("category", sorted(CATEGORY_TO_SCENE_CLASS))
def test_generated_recipe_passes_orchestrator_validation(category):
    recipe = generate_recipe(category, variant_index=0, date=_A_DATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # style fields (font/color) are a known V1 orchestrator limitation, not this test's concern
        validate_recipe(recipe)  # no raise
