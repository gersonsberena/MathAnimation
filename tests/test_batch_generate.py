import datetime
import subprocess
import sys
import warnings

import pytest
import yaml

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


def test_generate_recipe_keeps_template_params_unchanged_when_variant_has_none():
    recipe = generate_recipe("puzzle_backtracking", variant_index=0, date=_A_DATE)
    assert recipe["params"] == {"puzzle_type": "tower_of_hanoi", "size": 5, "seed": 0}


def test_generate_recipe_overrides_params_when_variant_declares_them():
    variations = load_variations("puzzle_backtracking")
    n_queens_index = next(i for i, v in enumerate(variations) if v.get("params", {}).get("puzzle_type") == "n_queens")
    recipe = generate_recipe("puzzle_backtracking", variant_index=n_queens_index, date=_A_DATE)
    assert recipe["params"]["puzzle_type"] == "n_queens"
    assert recipe["params"] != {"puzzle_type": "tower_of_hanoi", "size": 5, "seed": 0}


def test_generate_recipe_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown category"):
        generate_recipe("not_a_real_category", variant_index=0, date=_A_DATE)


def test_generate_batch_returns_one_per_category():
    categories = ["puzzle_backtracking", "wave_signal"]
    batch = generate_batch(categories, date=_A_DATE)
    assert set(batch) == set(categories)


def test_cli_variant_flag_targets_a_specific_topic(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "batch", "--categories", "puzzle_backtracking", "--variant", "2", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    generated = list(tmp_path.glob("*.yaml"))
    assert len(generated) == 1
    with open(generated[0], encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    assert recipe["params"]["puzzle_type"] == "n_queens"


@pytest.mark.parametrize("category", sorted(CATEGORY_TO_SCENE_CLASS))
def test_every_registered_category_has_variations_and_a_template(category):
    variations = load_variations(category)
    assert len(variations) >= 1
    for variant in variations:
        assert variant["title"] and variant["caption"] and variant["fb_post_caption"]
    generate_recipe(category, variant_index=0, date=_A_DATE)  # raises if no template recipe file exists


def _all_category_variant_pairs():
    """(category, variant_index) for every variant of every registered
    category -- not just index 0, since a variant's own `params` override
    (a genuinely different topic, not just different text) is exactly
    what could be wrong and needs its own validation pass.
    """
    return [(category, i) for category in sorted(CATEGORY_TO_SCENE_CLASS) for i in range(len(load_variations(category)))]


@pytest.mark.parametrize(("category", "variant_index"), _all_category_variant_pairs(), ids=lambda v: str(v))
def test_generated_recipe_passes_orchestrator_validation(category, variant_index):
    recipe = generate_recipe(category, variant_index=variant_index, date=_A_DATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # style fields (font/color) are a known V1 orchestrator limitation, not this test's concern
        validate_recipe(recipe)  # no raise
