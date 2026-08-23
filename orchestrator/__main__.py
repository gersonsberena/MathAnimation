"""CLI entry point: `python -m orchestrator <recipe.yaml> -o <output.mp4>`."""

import argparse
import sys

from orchestrator.pipeline import produce_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a MathAnimation recipe YAML into a final video.")
    parser.add_argument("recipe", help="Path to a recipe YAML file")
    parser.add_argument("-o", "--output", required=True, help="Path to write the final .mp4 to")
    args = parser.parse_args()

    output_path = produce_video(args.recipe, args.output)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
