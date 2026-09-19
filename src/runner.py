"""CLI runner for the pilot or full experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from .config import ExperimentConfig
from .data import dataset_provenance, load_questions, normalise_example
from .evaluation import extract_answer
from .model import generate_response, load_model_and_tokenizer
from .prompts import PROMPT_VARIANTS, format_prompt


def git_commit_hash() -> str | None:
    """Return the checked-out commit when the runner is launched from a Git clone."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def write_run_metadata(output_dir, metadata: dict) -> None:
    """Persist run progress so a runtime reset leaves an auditable checkpoint."""
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def run(pilot: bool) -> pd.DataFrame:
    config = ExperimentConfig()
    output_dir = config.output_path(pilot)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_dataset = load_questions(config, pilot)
    rows = [normalise_example(row) for row in sampled_dataset]
    generations_path = output_dir / "generations.jsonl"

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot": pilot,
        "expected_generations": len(rows) * len(PROMPT_VARIANTS),
        "actual_sample_size": len(rows),
        "actual_generations": 0,
        "config": config.as_dict(),
        "generation_configuration": {
            "do_sample": config.do_sample,
            "max_new_tokens": config.max_new_tokens,
            "enable_thinking": config.enable_thinking,
        },
        "prompt_variants": [
            {
                "identifier": variant.identifier,
                "label": variant.label,
                "instruction": variant.instruction,
            }
            for variant in PROMPT_VARIANTS
        ],
        "dataset_provenance": dataset_provenance(sampled_dataset),
        "sampled_question_ids": [row["question_id"] for row in rows],
        "git_commit": git_commit_hash(),
    }

    # Establish all persisted artifacts before model loading or generation.
    generations_path.write_text("", encoding="utf-8")
    (output_dir / "sampled_question_ids.json").write_text(
        json.dumps(metadata["sampled_question_ids"], indent=2) + "\n"
    )
    write_run_metadata(output_dir, metadata)

    model, tokenizer = load_model_and_tokenizer(config)

    records: list[dict] = []

    with generations_path.open("a", encoding="utf-8") as generations_file:
        for row_index, row in enumerate(rows, start=1):
            for variant in PROMPT_VARIANTS:
                prompt = format_prompt(
                    row["question"],
                    row["choices"],
                    variant,
                )

                raw_response, generated_tokens, inference_time = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    config,
                )

                predicted = extract_answer(raw_response)

                record = {
                    **row,
                    "prompt_variant": variant.identifier,
                    "raw_response": raw_response,
                    "predicted_answer": predicted,
                    "correct": predicted == row["ground_truth"],
                    "generated_tokens": generated_tokens,
                    "inference_time": inference_time,
                }

                records.append(record)

                # Save every complete generation immediately before continuing.
                generations_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                generations_file.flush()
                metadata["actual_generations"] = len(records)
                write_run_metadata(output_dir, metadata)

            print(
                f"Completed question {row_index}/{len(rows)} "
                f"({len(records)} generations saved)",
                flush=True,
            )

    results = pd.DataFrame(records)
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
