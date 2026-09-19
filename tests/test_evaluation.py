from src.evaluation import extract_answer


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
