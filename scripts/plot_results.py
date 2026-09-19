"""Create publication-quality figures from completed generation records."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Supports headless environments such as Colab and CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import prediction_stability_summary


PROMPT_ORDER = ["P1", "P2", "P3", "P4", "P5"]
FIGURE_DPI = 400


def _poster_style() -> dict:
    """Return consistent, readable styling for poster-scale static figures."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 23,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def accuracy_plot_values(analysis_dir: Path) -> pd.DataFrame:
    """Load the precomputed bootstrap accuracy intervals in prompt order."""
    path = analysis_dir / "bootstrap_accuracy_ci.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/analyze.py first.")
    values = pd.read_csv(path)
    required = {"prompt_variant", "accuracy", "ci_95_lower", "ci_95_upper"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"Bootstrap file is missing columns: {sorted(missing)}")
    return values.set_index("prompt_variant").loc[PROMPT_ORDER].reset_index()


def save_figure(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    """Save a high-resolution PNG and a vector PDF for one figure."""
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def plot_accuracy(values: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot accuracy percentages with question-level bootstrap 95% CIs."""
    with plt.rc_context(_poster_style()):
        figure, axis = plt.subplots(figsize=(11, 7))
        percentages = 100 * values["accuracy"].to_numpy()
        lower_errors = percentages - 100 * values["ci_95_lower"].to_numpy()
        upper_errors = 100 * values["ci_95_upper"].to_numpy() - percentages
        positions = np.arange(len(values))
        bars = axis.bar(positions, percentages, color="#0072B2", width=0.66)
        axis.errorbar(
            positions, percentages, yerr=np.vstack([lower_errors, upper_errors]),
            fmt="none", color="#1A1A1A", capsize=6, linewidth=1.8, zorder=3,
        )
        axis.set_xticks(positions, values["prompt_variant"])
        axis.set_xlabel("Prompt variant")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_title("Accuracy by prompt wording")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        for bar, percentage, upper in zip(bars, percentages, 100 * values["ci_95_upper"]):
            axis.annotate(
                f"{percentage:.1f}%",
                (bar.get_x() + bar.get_width() / 2, upper),
                ha="center", va="bottom", xytext=(0, 9), textcoords="offset points", fontsize=16,
            )
        figure.tight_layout()
        return save_figure(figure, output_dir / "figure1_accuracy_by_prompt")


def plot_prediction_stability(values: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot mutually exclusive stability categories for all sampled questions."""
    values = values[values["question_count"] > 0].reset_index(drop=True)
    colors = ["#0072B2", "#D55E00", "#CC79A7", "#7A7A7A", "#4D4D4D"]
    with plt.rc_context(_poster_style()):
        figure, axis = plt.subplots(figsize=(12, 5.5))
        left = 0.0
        for row, color in zip(values.itertuples(index=False), colors):
            axis.barh("All questions", row.percentage, left=left, height=0.5, color=color, label=row.category)
            center = left + row.percentage / 2
            axis.text(
                center, 0, f"{row.question_count}\n{row.percentage:.1f}%", ha="center", va="center",
                color="white", fontsize=15, fontweight="semibold",
            )
            left += row.percentage
        axis.set_xlim(0, 100)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Questions (%)")
        axis.set_yticks([])
        axis.set_title("Prediction stability across equivalent prompt variants")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
                    handlelength=1.2, columnspacing=1.6)
        axis.text(0, -0.57, "Valid prediction = extracted A–E answer; n = 300 questions.", fontsize=14)
        figure.subplots_adjust(bottom=0.35, left=0.08, right=0.98, top=0.86)
        return save_figure(figure, output_dir / "figure2_prediction_stability")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to generations.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figures and plotted-value CSVs")
    args = parser.parse_args()

    results = pd.read_json(args.results, lines=True)
    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_values = accuracy_plot_values(args.results.parent)
    stability_values = prediction_stability_summary(results)
    stability_plot_values = stability_values[stability_values["question_count"] > 0].reset_index(drop=True)
    accuracy_values.to_csv(output_dir / "figure1_accuracy_values.csv", index=False)
    stability_plot_values.to_csv(output_dir / "figure2_prediction_stability_values.csv", index=False)
    accuracy_paths = plot_accuracy(accuracy_values, output_dir)
    stability_paths = plot_prediction_stability(stability_plot_values, output_dir)
    for path in (*accuracy_paths, *stability_paths):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
