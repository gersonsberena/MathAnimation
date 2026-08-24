"""Generates ready-to-render recipe dicts by combining a category's
curated example recipe (common-block styling: background, resolution,
safe zones, etc.) with one entry from `recipes/variations.yaml` --
title/caption/fb_post_caption always come from there (README Section
10.2's "hook phrasing varies" checklist item), and `params` comes from
there too whenever a variant declares its own (a genuinely different
topic within the category, e.g. n_queens vs. tower_of_hanoi for
puzzle_backtracking), falling back to the example recipe's `params`
otherwise (pure text-hook rotation on the same topic).

Each variant's `params`, when present, is hand-written directly against
that category's own `validate_params()` -- not sampled or generated --
so adding a new topic variant means writing one `params` dict by hand
and letting `tests/test_batch_generate.py`'s sweep catch any mistake
(it validates every variant of every category, not just index 0).
"""

import datetime
from pathlib import Path

import yaml

from engines.registry import CATEGORY_TO_SCENE_CLASS
from orchestrator.recipe import REPO_ROOT

VARIATIONS_PATH = REPO_ROOT / "recipes" / "variations.yaml"
EXAMPLES_DIR = REPO_ROOT / "recipes" / "examples"


def load_variations(category: str) -> list:
    all_variations = yaml.safe_load(VARIATIONS_PATH.read_text(encoding="utf-8"))
    if category not in all_variations or not all_variations[category]:
        raise ValueError(f"no title/caption variations defined for category {category!r} in {VARIATIONS_PATH}")
    return all_variations[category]


def _load_template_recipe(category: str) -> dict:
    matches = sorted(EXAMPLES_DIR.glob(f"{category}_*.yaml"))
    if not matches:
        raise ValueError(f"no template recipe found under {EXAMPLES_DIR} for category {category!r}")
    with open(matches[0], encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_recipe(category: str, variant_index: int, date: datetime.date) -> dict:
    """Returns a new recipe dict for `category`: common-block styling
    copied from its curated example recipe, title/caption/fb_post_caption
    always swapped to variant `variant_index` from recipes/variations.yaml
    (wrapping around if there are fewer variants than `variant_index`),
    and `params` swapped too if that variant declares its own (otherwise
    the example recipe's `params` are kept unchanged), plus a fresh `id`
    and `date_created`.
    """
    if category not in CATEGORY_TO_SCENE_CLASS:
        raise ValueError(f"unknown category {category!r}, must be one of {sorted(CATEGORY_TO_SCENE_CLASS)}")

    recipe = _load_template_recipe(category)
    variations = load_variations(category)
    picked_index = variant_index % len(variations)
    variant = variations[picked_index]

    recipe["title"] = variant["title"]
    recipe["caption"] = variant["caption"]
    recipe["fb_post_caption"] = variant["fb_post_caption"]
    if "params" in variant:
        recipe["params"] = variant["params"]
    recipe["id"] = f"{category}-{date.isoformat()}-v{picked_index}"
    recipe["date_created"] = date.isoformat()
    recipe["status"] = "draft"
    return recipe


def generate_batch(categories, date: datetime.date) -> dict:
    """Returns `{category: recipe_dict}`, one per entry in `categories`.
    The text-variant index for every category is derived from `date`'s
    day-of-year, so re-running for the same date is idempotent but
    different dates rotate through each category's available variants.
    """
    day_index = date.timetuple().tm_yday
    return {category: generate_recipe(category, day_index, date) for category in categories}
