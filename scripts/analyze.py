"""Summarise completed experiment output; does not run inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation import p1_correctness_flips, question_disagreement, variant_summary
from src.statistics import cochran_q_test, paired_bootstrap, pairwise_exact_mcnemar


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
    flips = p1_correctness_flips(results)
    flips.to_csv(output_dir / "p1_correctness_flips.csv", index=False)
    cochran_q = cochran_q_test(results)
    (output_dir / "cochran_q.json").write_text(json.dumps(cochran_q, indent=2) + "\n")
    mcnemar = pairwise_exact_mcnemar(results)
    mcnemar.to_csv(output_dir / "pairwise_exact_mcnemar_holm.csv", index=False)
    bootstrap_accuracy, bootstrap_differences = paired_bootstrap(results)
    bootstrap_accuracy.to_csv(output_dir / "bootstrap_accuracy_ci.csv", index=False)
    bootstrap_differences.to_csv(output_dir / "bootstrap_difference_vs_p1_ci.csv", index=False)
    consistency = {
        "questions": int(results["question_id"].nunique()),
        "all_five_identical_predictions": int(
            results.groupby("question_id")["predicted_answer"].agg(lambda x: x.nunique(dropna=False) == 1).sum()
        ),
        "non_null_answer_disagreement_questions": int((disagreement["n_unique_predictions"] > 1).sum()),
    }
    (output_dir / "consistency_summary.json").write_text(json.dumps(consistency, indent=2) + "\n")
    print(variant_summary(results).to_string(index=False))
    print(json.dumps(consistency, indent=2))
    print(flips.to_string(index=False))
    print("Cochran's Q")
    print(json.dumps(cochran_q, indent=2))
    print("Pairwise exact McNemar tests with Holm adjustment")
    print(mcnemar.to_string(index=False))
    print("Paired bootstrap accuracy 95% CIs")
    print(bootstrap_accuracy.to_string(index=False))
    print("Paired bootstrap differences vs P1 (percentage points)")
    print(bootstrap_differences.to_string(index=False))


if __name__ == "__main__":
    main()
