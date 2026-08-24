"""Batch recipe generation (not the orchestrator's render pipeline).

Turns the curated `recipes/examples/*.yaml` (one per engine category,
already-valid params) into ready-to-render recipes with rotated title/
caption/fb_post_caption text, so producing a batch of reels doesn't mean
hand-writing a new YAML file per video. See `batch/generate.py`.
"""
