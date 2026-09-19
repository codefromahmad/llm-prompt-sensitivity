"""CLI runner for the pilot or full experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .config import ExperimentConfig
from .data import dataset_provenance, load_questions, normalise_example
from .evaluation import extract_answer
from .model import generate_response, load_model_and_tokenizer
from .prompts import PROMPT_VARIANTS, format_prompt


def run(pilot: bool) -> pd.DataFrame:
    config = ExperimentConfig()
    output_dir = config.output_path(pilot)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_dataset = load_questions(config, pilot)
    rows = [normalise_example(row) for row in sampled_dataset]
    model, tokenizer = load_model_and_tokenizer(config)
    records: list[dict] = []
    for row_index, row in enumerate(rows, start=1):
        for variant in PROMPT_VARIANTS:
            prompt = format_prompt(row["question"], row["choices"], variant)
            raw_response, generated_tokens, inference_time = generate_response(model, tokenizer, prompt, config)
            predicted = extract_answer(raw_response)
            records.append({
                **row,
                "prompt_variant": variant.identifier,
                "raw_response": raw_response,
                "predicted_answer": predicted,
                "correct": predicted == row["ground_truth"],
                "generated_tokens": generated_tokens,
                "inference_time": inference_time,
            })
        print(f"Completed question {row_index}/{len(rows)}", flush=True)
    results = pd.DataFrame(records)
    results.to_json(output_dir / "generations.jsonl", orient="records", lines=True, force_ascii=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": pilot,
        "expected_generations": len(rows) * len(PROMPT_VARIANTS),
        "config": config.as_dict(),
        "dataset_provenance": dataset_provenance(sampled_dataset),
        "sampled_question_ids": [row["question_id"] for row in rows],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "sampled_question_ids.json").write_text(
        json.dumps(metadata["sampled_question_ids"], indent=2) + "\n"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 10 questions × 5 variants (50 generations).")
    parser.add_argument("--full", action="store_true", help="Run 300 questions × 5 variants (1500 generations).")
    args = parser.parse_args()
    if args.pilot == args.full:
        parser.error("Specify exactly one of --pilot or --full.")
    run(pilot=args.pilot)


if __name__ == "__main__":
    main()
