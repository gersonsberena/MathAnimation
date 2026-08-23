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
