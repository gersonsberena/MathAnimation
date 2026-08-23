"""Recipe -> Scene dispatch for the QA render/determinism tests
(README Section 11.2 items 4-5).

This is deliberately NOT the "orchestrator" README Section 9 marks out
of scope (recipe loading, calling the right engine, final ffmpeg audio
mix/encode as a production pipeline). It's just enough glue for tests to
turn one `recipes/examples/*.yaml` file into a configured Scene instance:
map `category` to a Scene class, and set each `params` entry as a class
attribute. Nothing here mixes audio, resolves asset paths, or is meant
to be reused by a real render pipeline.

One category's recipe param key doesn't match its Scene attribute name:
population_growth_model's `duration` (README's param name) collides with
`manim.Scene`'s own `self.duration` instance attribute (see README
Section 8's gotcha), so the Scene class attribute is `growth_duration`
instead — PARAM_KEY_OVERRIDES documents that one exception; every other
category's recipe params map straight onto identically-named Scene
attributes.
"""

from pathlib import Path

import yaml

from engines.array_bar_race import ArrayBarRaceReel
from engines.cellular_automata import CellularAutomataReel
from engines.game_theory import GameTheoryReel
from engines.geometric_transformation import GeometricTransformationReel
from engines.graph_network import GraphNetworkReel
from engines.grid_cell_coloring import GridCellColoringReel
from engines.monte_carlo_probability import MonteCarloProbabilityReel
from engines.number_pattern import NumberPatternReel
from engines.particle_physics_sim import ParticlePhysicsSimReel
from engines.pathfinding_maze import PathfindingMazeReel
from engines.population_growth_model import PopulationGrowthModelReel
from engines.proof_without_words import ProofWithoutWordsReel
from engines.puzzle_backtracking import PuzzleBacktrackingReel
from engines.recursive_lsystem import RecursiveLSystemReel
from engines.statistical_distribution import StatisticalDistributionReel
from engines.strange_attractor_chaos import StrangeAttractorChaosReel
from engines.tessellation_growth import TessellationGrowthReel
from engines.tree_data_structure import TreeDataStructureReel
from engines.wave_signal import WaveSignalReel

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes" / "examples"

CATEGORY_TO_SCENE_CLASS = {
    "array_bar_race": ArrayBarRaceReel,
    "cellular_automata": CellularAutomataReel,
    "game_theory": GameTheoryReel,
    "geometric_transformation": GeometricTransformationReel,
    "graph_network": GraphNetworkReel,
    "grid_cell_coloring": GridCellColoringReel,
    "monte_carlo_probability": MonteCarloProbabilityReel,
    "number_pattern": NumberPatternReel,
    "particle_physics_sim": ParticlePhysicsSimReel,
    "pathfinding_maze": PathfindingMazeReel,
    "population_growth_model": PopulationGrowthModelReel,
    "proof_without_words": ProofWithoutWordsReel,
    "puzzle_backtracking": PuzzleBacktrackingReel,
    "recursive_lsystem": RecursiveLSystemReel,
    "statistical_distribution": StatisticalDistributionReel,
    "strange_attractor_chaos": StrangeAttractorChaosReel,
    "tessellation_growth": TessellationGrowthReel,
    "tree_data_structure": TreeDataStructureReel,
    "wave_signal": WaveSignalReel,
}

PARAM_KEY_OVERRIDES = {
    "population_growth_model": {"duration": "growth_duration"},
}


def load_recipe(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_recipe_paths():
    return sorted(RECIPES_DIR.glob("*.yaml"))


def scene_class_for_recipe(recipe):
    category = recipe["category"]
    if category not in CATEGORY_TO_SCENE_CLASS:
        raise ValueError(f"no Scene class registered for category {category!r} — add it to CATEGORY_TO_SCENE_CLASS")
    return CATEGORY_TO_SCENE_CLASS[category]


def build_scene(recipe):
    """Returns a configured (but not yet rendered) Scene *instance* for a
    loaded recipe dict — params applied as instance attributes, so the
    class-level defaults used elsewhere (e.g. manual renders) are
    untouched.
    """
    scene_cls = scene_class_for_recipe(recipe)
    overrides = PARAM_KEY_OVERRIDES.get(recipe["category"], {})
    scene = scene_cls()
    for key, value in recipe["params"].items():
        attr = overrides.get(key, key)
        setattr(scene, attr, value)
    return scene
