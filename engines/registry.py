"""Recipe `category` -> engine module/Scene-class registry.

Single source of truth for turning a recipe's `category` string into the
right `ReelScene` subclass and its `validate_params` function. Used by
both the production orchestrator (`orchestrator/`) and the QA test
harness (`tests/qa_dispatch.py`) so the two can never drift apart.

Recipe `params` dict keys are validated using the ORIGINAL README param
names (matches every engine's `validate_params(...)` signature exactly).
Applying params as Scene *attributes* is a separate step where one
category needs a rename — see PARAM_KEY_OVERRIDES — because
`population_growth_model`'s README param is literally named `duration`,
which collides with `manim.Scene`'s own `self.duration` instance
attribute (README Section 8's documented gotcha); its Scene class
attribute is `growth_duration` instead. `validate_params()` itself is
untouched by that collision (it's a plain function, not a Scene
attribute), so validation always uses the raw recipe key.

`epicycle_fourier` (README Section 6, #3) is a known, documented gap —
marked "(built)" in an earlier README draft but never implemented (see
the note in README Section 6). It's deliberately absent from this
registry; a recipe with that category fails lookup with a clear error
rather than silently doing nothing.
"""

import engines.array_bar_race as array_bar_race
import engines.cellular_automata as cellular_automata
import engines.game_theory as game_theory
import engines.geometric_transformation as geometric_transformation
import engines.graph_network as graph_network
import engines.grid_cell_coloring as grid_cell_coloring
import engines.monte_carlo_probability as monte_carlo_probability
import engines.number_pattern as number_pattern
import engines.particle_physics_sim as particle_physics_sim
import engines.pathfinding_maze as pathfinding_maze
import engines.population_growth_model as population_growth_model
import engines.proof_without_words as proof_without_words
import engines.puzzle_backtracking as puzzle_backtracking
import engines.recursive_lsystem as recursive_lsystem
import engines.statistical_distribution as statistical_distribution
import engines.strange_attractor_chaos as strange_attractor_chaos
import engines.tessellation_growth as tessellation_growth
import engines.tree_data_structure as tree_data_structure
import engines.wave_signal as wave_signal

CATEGORY_TO_SCENE_CLASS = {
    "array_bar_race": array_bar_race.ArrayBarRaceReel,
    "cellular_automata": cellular_automata.CellularAutomataReel,
    "game_theory": game_theory.GameTheoryReel,
    "geometric_transformation": geometric_transformation.GeometricTransformationReel,
    "graph_network": graph_network.GraphNetworkReel,
    "grid_cell_coloring": grid_cell_coloring.GridCellColoringReel,
    "monte_carlo_probability": monte_carlo_probability.MonteCarloProbabilityReel,
    "number_pattern": number_pattern.NumberPatternReel,
    "particle_physics_sim": particle_physics_sim.ParticlePhysicsSimReel,
    "pathfinding_maze": pathfinding_maze.PathfindingMazeReel,
    "population_growth_model": population_growth_model.PopulationGrowthModelReel,
    "proof_without_words": proof_without_words.ProofWithoutWordsReel,
    "puzzle_backtracking": puzzle_backtracking.PuzzleBacktrackingReel,
    "recursive_lsystem": recursive_lsystem.RecursiveLSystemReel,
    "statistical_distribution": statistical_distribution.StatisticalDistributionReel,
    "strange_attractor_chaos": strange_attractor_chaos.StrangeAttractorChaosReel,
    "tessellation_growth": tessellation_growth.TessellationGrowthReel,
    "tree_data_structure": tree_data_structure.TreeDataStructureReel,
    "wave_signal": wave_signal.WaveSignalReel,
}

CATEGORY_TO_MODULE = {
    "array_bar_race": array_bar_race,
    "cellular_automata": cellular_automata,
    "game_theory": game_theory,
    "geometric_transformation": geometric_transformation,
    "graph_network": graph_network,
    "grid_cell_coloring": grid_cell_coloring,
    "monte_carlo_probability": monte_carlo_probability,
    "number_pattern": number_pattern,
    "particle_physics_sim": particle_physics_sim,
    "pathfinding_maze": pathfinding_maze,
    "population_growth_model": population_growth_model,
    "proof_without_words": proof_without_words,
    "puzzle_backtracking": puzzle_backtracking,
    "recursive_lsystem": recursive_lsystem,
    "statistical_distribution": statistical_distribution,
    "strange_attractor_chaos": strange_attractor_chaos,
    "tessellation_growth": tessellation_growth,
    "tree_data_structure": tree_data_structure,
    "wave_signal": wave_signal,
}

PARAM_KEY_OVERRIDES = {
    "population_growth_model": {"duration": "growth_duration"},
}


def scene_class_for_category(category):
    if category not in CATEGORY_TO_SCENE_CLASS:
        raise ValueError(f"unknown category {category!r}, must be one of {sorted(CATEGORY_TO_SCENE_CLASS)}")
    return CATEGORY_TO_SCENE_CLASS[category]


def validate_params_for_category(category, params):
    if category not in CATEGORY_TO_MODULE:
        raise ValueError(f"unknown category {category!r}, must be one of {sorted(CATEGORY_TO_MODULE)}")
    CATEGORY_TO_MODULE[category].validate_params(**params)
