"""Summarise completed experiment output; does not run inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import question_disagreement, variant_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Path to generations.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results = pd.read_json(args.results, lines=True)
    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_summary(results).to_csv(output_dir / "variant_summary.csv", index=False)
    disagreement = question_disagreement(results)
    disagreement.to_csv(output_dir / "question_disagreement.csv", index=False)
    print(variant_summary(results).to_string(index=False))
    print(f"Answer disagreement: {(disagreement.n_unique_predictions > 1).mean():.1%} of questions")


if __name__ == "__main__":
    main()
