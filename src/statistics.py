"""Paired statistical analyses for prompt-variant correctness outcomes."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar

if TYPE_CHECKING:
    import pandas as pd


PROMPT_VARIANTS = ("P1", "P2", "P3", "P4", "P5")


def correctness_matrix(results: pd.DataFrame, prompts: tuple[str, ...] = PROMPT_VARIANTS) -> pd.DataFrame:
    """Return a question-by-prompt boolean correctness matrix in fixed order."""
    import pandas as pd

    matrix = results.pivot(index="question_id", columns="prompt_variant", values="correct")
    missing = set(prompts) - set(matrix.columns)
    if missing:
        raise ValueError(f"Missing prompt variants: {sorted(missing)}")
    matrix = matrix.loc[:, list(prompts)]
    if matrix.isna().any().any():
        raise ValueError("Each question must have one correctness value for every prompt variant.")
    return matrix.astype(bool)


def cochran_q_test(results: pd.DataFrame) -> dict:
    """Run Cochran's Q across the five paired prompt conditions."""
    matrix = correctness_matrix(results)
    result = cochrans_q(matrix.to_numpy(dtype=int))
    return {
        "q_statistic": float(result.statistic),
        "degrees_of_freedom": len(PROMPT_VARIANTS) - 1,
        "p_value": float(result.pvalue),
        "questions": int(len(matrix)),
    }


def pairwise_exact_mcnemar(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run all exact paired McNemar tests and Holm-adjust their p-values."""
    import pandas as pd

    matrix = correctness_matrix(results)
    rows = []
    for prompt_a, prompt_b in combinations(PROMPT_VARIANTS, 2):
        correct_a = matrix[prompt_a]
        correct_b = matrix[prompt_b]
        a_correct_b_incorrect = int((correct_a & ~correct_b).sum())
        a_incorrect_b_correct = int((~correct_a & correct_b).sum())
        table = [
            [int((correct_a & correct_b).sum()), a_correct_b_incorrect],
            [a_incorrect_b_correct, int((~correct_a & ~correct_b).sum())],
        ]
        exact_result = mcnemar(table, exact=True, correction=False)
        rows.append({
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "a_correct_b_incorrect": a_correct_b_incorrect,
            "a_incorrect_b_correct": a_incorrect_b_correct,
            "raw_exact_p_value": float(exact_result.pvalue),
        })
    output = pd.DataFrame(rows)
    rejected, adjusted, _, _ = multipletests(output["raw_exact_p_value"], alpha=alpha, method="holm")
    output["holm_adjusted_p_value"] = adjusted
    output["significant_after_holm"] = rejected
    output["alpha"] = alpha
    return output


def paired_bootstrap(results: pd.DataFrame, seed: int = 42, samples: int = 10_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap whole questions to preserve prompt-pairing in CIs."""
    import pandas as pd

    matrix = correctness_matrix(results).to_numpy(dtype=float)
    question_count = matrix.shape[0]
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, question_count, size=(samples, question_count))
    bootstrap_accuracies = matrix[resample_indices].mean(axis=1)
    point_accuracies = matrix.mean(axis=0)
    lower, upper = np.percentile(bootstrap_accuracies, [2.5, 97.5], axis=0)
    accuracy_rows = []
    difference_rows = []
    for index, prompt in enumerate(PROMPT_VARIANTS):
        accuracy_rows.append({
            "prompt_variant": prompt,
            "accuracy": point_accuracies[index],
            "ci_95_lower": lower[index],
            "ci_95_upper": upper[index],
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        })
        if prompt != "P1":
            differences_pp = 100 * (bootstrap_accuracies[:, index] - bootstrap_accuracies[:, 0])
            difference_rows.append({
                "prompt_variant": prompt,
                "difference_vs_p1_percentage_points": 100 * (point_accuracies[index] - point_accuracies[0]),
                "ci_95_lower_percentage_points": np.percentile(differences_pp, 2.5),
                "ci_95_upper_percentage_points": np.percentile(differences_pp, 97.5),
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            })
    return pd.DataFrame(accuracy_rows), pd.DataFrame(difference_rows)
