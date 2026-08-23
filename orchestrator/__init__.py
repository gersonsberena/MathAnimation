"""Recipe -> final video orchestrator (README Section 9).

Loads a recipe YAML, validates it, renders the target engine at the
recipe's real resolution/fps/duration, mixes in the specified music
track, and encodes the final output. See `orchestrator/recipe.py` for
exactly which recipe fields V1 honors.

Public entry point: `orchestrator.pipeline.produce_video`.
"""
