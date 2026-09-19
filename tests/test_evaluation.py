import pandas as pd

from src.evaluation import extract_answer, p1_correctness_flips, prediction_stability_summary


def test_extracts_exact_answer():
    assert extract_answer("C") == "C"


def test_extracts_answer_with_punctuation():
    assert extract_answer("(C)") == "C"
    assert extract_answer("C.") == "C"
    assert extract_answer("C)") == "C"


def test_extracts_labelled_answer():
    assert extract_answer("Answer: C") == "C"


def test_extracts_sentence_answer():
    assert extract_answer("The answer is C") == "C"


def test_extracts_leading_answer_before_explanation():
    assert extract_answer("C. talented\n\nA person who is good at sports is typically considered talented.") == "C"
    assert extract_answer("C) Explanation: a person may be talented.") == "C"
    assert extract_answer("(C) Explanation: a person who is good at sports is talented.") == "C"
    assert extract_answer("A. Explanation: a person may be talented.") == "A"
    assert extract_answer("C\nExplanation: A person may be talented.") == "C"


def test_extracts_final_answer_cue():
    assert extract_answer("Final answer: C") == "C"


def test_extracts_markdown_formatted_explicit_answer_cue():
    assert extract_answer("The correct answer is **D. city or town**.") == "D"
    assert extract_answer("The most logical choice is **A") == "A"


def test_extracts_one_standalone_letter():
    assert extract_answer("I select B") == "B"


def test_rejects_ambiguous_answer():
    assert extract_answer("Answer: A ... final answer: C") is None


def test_rejects_no_answer():
    assert extract_answer("I cannot determine the answer.") is None


def test_p1_correctness_flips():
    results = pd.DataFrame([
        {"question_id": "q1", "prompt_variant": "P1", "correct": True},
        {"question_id": "q2", "prompt_variant": "P1", "correct": False},
        {"question_id": "q1", "prompt_variant": "P2", "correct": False},
        {"question_id": "q2", "prompt_variant": "P2", "correct": True},
    ])
    flips = p1_correctness_flips(results).set_index("prompt_variant")
    assert flips.loc["P2", "correct_to_incorrect"] == 1
    assert flips.loc["P2", "incorrect_to_correct"] == 1


def test_prediction_stability_summary_keeps_invalid_answers_distinct():
    predictions = {
        "stable": ["A", "A", "A", "A", "A"],
        "different": ["A", "B", "A", "A", "A"],
        "invalid_and_different": ["A", None, "B", "A", "A"],
        "invalid_and_identical": ["A", None, "A", "A", "A"],
        "all_invalid": [None, None, None, None, None],
    }
    results = pd.DataFrame([
        {"question_id": question_id, "prompt_variant": f"P{index + 1}", "predicted_answer": prediction}
        for question_id, answers in predictions.items()
        for index, prediction in enumerate(answers)
    ])
    summary = prediction_stability_summary(results).set_index("category")
    assert summary["question_count"].to_dict() == {
        "All five valid, same answer": 1,
        "All five valid, ≥2 answers": 1,
        "≥1 invalid, ≥2 valid answers": 1,
        "≥1 invalid, one valid answer": 1,
        "All predictions invalid": 1,
    }
    assert summary["percentage"].sum() == 100.0
