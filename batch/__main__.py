"""CLI: `python -m batch [--categories cat1,cat2,...] [--out-dir DIR] [--date YYYY-MM-DD]`

Writes one ready-to-render recipe YAML per category into `--out-dir`
(default `recipes/generated/`, gitignored -- these are day-to-day output,
not curated examples). Each still needs `music_track` filled in (or left
`null`) before rendering, same as any recipe -- see `orchestrator/recipe.py`.
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml

from batch.generate import generate_batch
from engines.registry import CATEGORY_TO_SCENE_CLASS


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a batch of recipe YAMLs with rotated title/caption text.")
    parser.add_argument("--categories", default=None, help="comma-separated category names (default: all)")
    parser.add_argument("--out-dir", default="recipes/generated", help="directory to write generated recipe YAMLs into")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today (drives which text variant rotates in)")
    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else sorted(CATEGORY_TO_SCENE_CLASS)
    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipes = generate_batch(categories, date)
    for recipe in recipes.values():
        out_path = out_dir / f"{recipe['id']}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(recipe, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
