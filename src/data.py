"""CommonsenseQA loading and deterministic nested sampling."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from .config import ExperimentConfig


def load_questions(config: ExperimentConfig, pilot: bool) -> Dataset:
    """Return the first N rows of one fixed seeded permutation.

    Thus the 10-question pilot is always a subset of the 300-question sample.
    """
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    required = config.final_sample_size
    if len(dataset) < required:
        raise ValueError(f"Dataset has {len(dataset)} rows, fewer than requested {required}.")
    return dataset.shuffle(seed=config.seed).select(range(config.sample_size(pilot)))


def normalise_example(example: dict) -> dict:
    """Map a CommonsenseQA row to the experiment's stable representation."""
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    if len(labels) != len(texts):
        raise ValueError(f"Choice label/text length mismatch for {example['id']}")
    choices = dict(zip(labels, texts))
    if set(choices) != set("ABCDE"):
        raise ValueError(f"Expected choices A-E, found {sorted(choices)} for {example['id']}")
    return {
        "question_id": example["id"],
        "question": example["question"],
        "choices": choices,
        "ground_truth": example["answerKey"],
    }


def dataset_provenance(dataset: Dataset) -> dict:
    """Capture dataset identity information exposed by the local HF dataset."""
    info = dataset.info
    version = getattr(info, "version", None)
    revision = getattr(info, "revision", None)
    split = getattr(info, "split", None)
    return {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "hf_dataset_info_version": str(version) if version is not None else None,
        "hf_dataset_revision": str(revision) if revision is not None else None,
        "dataset_builder": getattr(info, "builder_name", None),
        "dataset_config": getattr(info, "config_name", None),
        "dataset_split": str(split) if split is not None else None,
    }
