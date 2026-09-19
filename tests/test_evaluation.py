from src.evaluation import extract_answer


def test_extracts_exact_answer():
    assert extract_answer("C") == "C"


def test_extracts_answer_with_punctuation():
    assert extract_answer("(C)") == "C"
    assert extract_answer("C.") == "C"


def test_extracts_labelled_answer():
    assert extract_answer("Answer: C") == "C"


def test_extracts_sentence_answer():
    assert extract_answer("The answer is C") == "C"


def test_extracts_one_standalone_letter():
    assert extract_answer("I select B") == "B"


def test_rejects_ambiguous_answer():
    assert extract_answer("Answer: A ... final answer: C") is None


def test_rejects_no_answer():
    assert extract_answer("I cannot determine the answer.") is None
