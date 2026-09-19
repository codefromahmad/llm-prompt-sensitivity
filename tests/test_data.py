import pytest

from src.data import normalise_example


def test_normalise_example_preserves_question_and_choices():
    example = {
        "id": "example-1",
        "question": "Question?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["a", "b", "c", "d", "e"]},
        "answerKey": "C",
    }
    assert normalise_example(example) == {
        "question_id": "example-1",
        "question": "Question?",
        "choices": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"},
        "ground_truth": "C",
    }


def test_normalise_example_rejects_mismatched_choice_lists():
    example = {
        "id": "bad-example",
        "question": "Question?",
        "choices": {"label": ["A", "B"], "text": ["a"]},
        "answerKey": "A",
    }
    with pytest.raises(ValueError, match="length mismatch"):
        normalise_example(example)
