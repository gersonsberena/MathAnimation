"""Recipe -> Scene dispatch for the QA render/determinism tests
(README Section 11.2 items 4-5).

Thin test-only wrapper around engines/registry.py (the shared category
-> Scene class / param-key-override source of truth, also used by the
production orchestrator). This module adds only what's test-specific:
loading recipe YAML files from disk and instantiating+configuring a
Scene from one.
"""

from pathlib import Path

import yaml

from engines.registry import PARAM_KEY_OVERRIDES, scene_class_for_category

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes" / "examples"


def load_recipe(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_recipe_paths():
    return sorted(RECIPES_DIR.glob("*.yaml"))


def build_scene(recipe):
    """Returns a configured (but not yet rendered) Scene *instance* for a
    loaded recipe dict — params applied as instance attributes, so the
    class-level defaults used elsewhere (e.g. manual renders) are
    untouched.
    """
    scene_cls = scene_class_for_category(recipe["category"])
    overrides = PARAM_KEY_OVERRIDES.get(recipe["category"], {})
    scene = scene_cls()
    for key, value in recipe["params"].items():
        attr = overrides.get(key, key)
        setattr(scene, attr, value)
    return scene
