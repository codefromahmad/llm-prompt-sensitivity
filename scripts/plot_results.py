"""Plot accuracy by prompt variant from completed generation records."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# This script is designed for non-interactive environments (for example Colab
# and CI), where the macOS GUI backend can fail or hang during font setup.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation import variant_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Path to generations.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = variant_summary(pd.read_json(args.results, lines=True))
    sns.set_theme(style="whitegrid")
    axis = sns.barplot(data=summary, x="prompt_variant", y="accuracy", color="C0")
    axis.set(xlabel="Prompt variant", ylabel="Accuracy", ylim=(0, 1), title="CommonsenseQA accuracy by prompt wording")
    for bar, accuracy in zip(axis.patches, summary["accuracy"]):
        axis.annotate(
            f"{accuracy:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )
    figure = axis.get_figure()
    figure.tight_layout()
    output = args.output or args.results.parent / "accuracy_by_variant.png"
    figure.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
