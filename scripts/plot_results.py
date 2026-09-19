"""Plot accuracy by prompt variant from completed generation records."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    axis = sns.barplot(data=summary, x="prompt_variant", y="accuracy", hue="prompt_variant", legend=False)
    axis.set(xlabel="Prompt variant", ylabel="Accuracy", ylim=(0, 1), title="CommonsenseQA accuracy by prompt wording")
    for container in axis.containers:
        axis.bar_label(container, labels=[f"{value:.1%}" for value in summary.accuracy], padding=3)
    figure = axis.get_figure()
    figure.tight_layout()
    output = args.output or args.results.parent / "accuracy_by_variant.png"
    figure.savefig(output, dpi=200)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
