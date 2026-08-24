"""CLI: `python -m batch [--categories cat1,cat2,...] [--out-dir DIR] [--date YYYY-MM-DD] [--variant N]`

Writes one ready-to-render recipe YAML per category into `--out-dir`
(default `recipes/generated/`, gitignored -- these are day-to-day output,
not curated examples). Each still needs `music_track` filled in (or left
`null`) before rendering, same as any recipe -- see `orchestrator/recipe.py`.

Without `--variant`, which variant each category gets is derived from
`--date`'s day-of-year (or today's, by default) -- convenient for a
"today's batch" run, but not a practical way to target one specific
variant (e.g. `puzzle_backtracking`'s Rubik's-cube topic) since that
means hunting for a date that happens to land on it. `--variant N`
overrides that: every requested category uses variant index N directly
(wrapping around per-category, same as the date-derived index does) --
pair it with `--categories` naming just the one category you want a
specific variant of. List a category's variants (index + title) with
`python -c "from batch.generate import load_variations; [print(i, v['title']) for i, v in enumerate(load_variations('puzzle_backtracking'))]"`.
"""

import argparse
import datetime
import sys
from pathlib import Path

import yaml

from batch.generate import generate_batch, generate_recipe
from engines.registry import CATEGORY_TO_SCENE_CLASS


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a batch of recipe YAMLs with rotated title/caption/topic variants.")
    parser.add_argument("--categories", default=None, help="comma-separated category names (default: all)")
    parser.add_argument("--out-dir", default="recipes/generated", help="directory to write generated recipe YAMLs into")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today (drives which variant rotates in, unless --variant is set)")
    parser.add_argument("--variant", type=int, default=None, help="explicit variant index, applied to every requested category (overrides date-based rotation)")
    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else sorted(CATEGORY_TO_SCENE_CLASS)
    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant is not None:
        recipes = {category: generate_recipe(category, variant_index=args.variant, date=date) for category in categories}
    else:
        recipes = generate_batch(categories, date)

    for recipe in recipes.values():
        out_path = out_dir / f"{recipe['id']}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(recipe, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
