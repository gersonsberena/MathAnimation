# Environment setup

Pinned versions (see `requirements.txt`): Python 3.11, Manim 0.21.0,
pytest 9.1.1. Manim's rendered output has changed across versions in the
past — don't float these without re-checking every engine's debug render.

## Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) on `PATH` (Manim shells out to it for encoding)
- A LaTeX distribution only if an engine renders `MathTex`/`Tex` (not
  required for the base layout system)

## Setup

```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Onboarding check

This is the actual "did setup work" test — not prose:

```bash
pytest tests/test_layout.py -v
```

If that passes, the base `ReelScene` layout system is importable and
correct on your machine. Engine-specific tests live under
`tests/test_engines/` and are added as each engine in README Section 7 is
built.

## Rendering a single recipe (manual smoke check)

```bash
python -m manim render -ql engines/recursive_lsystem.py <SceneClassName>
```

`-ql` = low quality/fast preview. Drop it for a full-res render.

## Running the full QA suite (render smoke test + determinism check)

`pytest tests/` only runs the fast (no-render) checks by default — see
`pyproject.toml`'s `addopts`. The slow checks (README Section 11.2 items
4-5: an actual low-res render of every `recipes/examples/*.yaml` recipe,
a content-vs-zone overlap check on that render, and a same-seed-twice
determinism check) are marked `@pytest.mark.slow` and run explicitly:

```bash
pytest tests/ -v -m slow
```

## Producing a final video from a recipe (orchestrator)

README Section 9. Renders the recipe's target engine at its real
resolution/fps, mixes in the specified music track, and writes a final
`h264_mp4`:

```bash
python -m orchestrator recipes/examples/puzzle_backtracking_hanoi.yaml -o out.mp4
```

Requires the recipe's `music_track` file to actually exist on disk if
it's non-null — `assets/music/` only has a `.gitkeep` today (no real
tracks committed yet, see Section 11.4), so every committed example
recipe will fail validation with a clear `FileNotFoundError` until real
music assets are added. Set `music_track: null` in a copy of a recipe to
try the pipeline without one.

Takes about 2 minutes; needs ffmpeg on `PATH` (same requirement as any
render). This is what `.github/workflows/nightly-render-qa.yml` runs.
