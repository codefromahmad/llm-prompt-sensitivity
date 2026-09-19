import numpy as np
import pandas as pd

from src.statistics import cochran_q_test, paired_bootstrap, pairwise_exact_mcnemar


def synthetic_results() -> pd.DataFrame:
    correctness = {
        "q1": [True, False, True, False, True],
        "q2": [True, True, False, False, True],
        "q3": [False, True, False, False, False],
        "q4": [False, False, True, True, False],
        "q5": [True, False, True, True, False],
        "q6": [False, True, False, True, True],
    }
    return pd.DataFrame([
        {"question_id": question_id, "prompt_variant": f"P{index + 1}", "correct": value}
        for question_id, values in correctness.items()
        for index, value in enumerate(values)
    ])


def test_pairwise_mcnemar_counts_and_holm_output():
    tests = pairwise_exact_mcnemar(synthetic_results())
    assert len(tests) == 10
    p1_p2 = tests[(tests.prompt_a == "P1") & (tests.prompt_b == "P2")].iloc[0]
    assert p1_p2.a_correct_b_incorrect == 2
    assert p1_p2.a_incorrect_b_correct == 2
    assert p1_p2.raw_exact_p_value == 1.0
    assert (tests.holm_adjusted_p_value >= tests.raw_exact_p_value).all()


def test_cochran_q_and_paired_bootstrap_are_reproducible():
    results = synthetic_results()
    cochran = cochran_q_test(results)
    assert cochran["degrees_of_freedom"] == 4
    assert cochran["questions"] == 6
    assert cochran["q_statistic"] >= 0
    first_accuracy, first_differences = paired_bootstrap(results, seed=42, samples=500)
    second_accuracy, second_differences = paired_bootstrap(results, seed=42, samples=500)
    assert len(first_accuracy) == 5
    assert len(first_differences) == 4
    np.testing.assert_allclose(
        first_accuracy.select_dtypes(include="number").to_numpy(),
        second_accuracy.select_dtypes(include="number").to_numpy(),
    )
    np.testing.assert_allclose(
        first_differences.select_dtypes(include="number").to_numpy(),
        second_differences.select_dtypes(include="number").to_numpy(),
    )
