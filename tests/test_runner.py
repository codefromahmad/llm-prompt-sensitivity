from dataclasses import replace
import json

from src import runner
from src.config import ExperimentConfig


def test_runner_flushes_each_record_and_checkpoints_metadata(tmp_path, monkeypatch):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path))
    example = {
        "id": "question-1",
        "question": "Which option is correct?",
        "choices": {"label": list("ABCDE"), "text": list("abcde")},
        "answerKey": "A",
    }
    output_dir = config.output_path(pilot=True)
    generations_path = output_dir / "generations.jsonl"
    calls = 0

    def fake_generate(*_args):
        nonlocal calls
        assert generations_path.exists()
        assert len(generations_path.read_text().splitlines()) == calls
        metadata = json.loads((output_dir / "run_metadata.json").read_text())
        assert metadata["actual_generations"] == calls
        calls += 1
        return "A", 1, 0.01

    monkeypatch.setattr(runner, "ExperimentConfig", lambda: config)
    monkeypatch.setattr(runner, "load_questions", lambda *_args: [example])
    monkeypatch.setattr(runner, "dataset_provenance", lambda *_args: {"fingerprint": "test"})
    monkeypatch.setattr(runner, "load_model_and_tokenizer", lambda *_args: (object(), object()))
    monkeypatch.setattr(runner, "generate_response", fake_generate)
    monkeypatch.setattr(runner, "git_commit_hash", lambda: "test-commit")

    results = runner.run(pilot=True)

    assert len(results) == 5
    assert calls == 5
    assert len(generations_path.read_text().splitlines()) == 5
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["actual_generations"] == 5
    assert metadata["expected_generations"] == 5
    assert json.loads((output_dir / "sampled_question_ids.json").read_text()) == ["question-1"]
