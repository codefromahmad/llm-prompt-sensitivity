"""Answer extraction and aggregation helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

VALID_ANSWERS = frozenset("ABCDE")
PREDICTION_STABILITY_CATEGORIES = (
    "All five valid, same answer",
    "All five valid, ≥2 answers",
    "≥1 invalid, ≥2 valid answers",
    "≥1 invalid, one valid answer",
    "All predictions invalid",
)
_STRICT = re.compile(r"^\s*\(?\s*([A-E])\s*\)?(?:\s*[\.!,:;]?\s*)$", re.IGNORECASE)
_ANSWER_CUE = re.compile(
    r"(?:final\s+)?(?:answer|option|choice|letter)\s*(?:is|:)?\s*(?:[*_`]+\s*)*\(?\s*([A-E])\b",
    re.IGNORECASE,
)
_LEADING_ANSWER = re.compile(
    r"^\s*(?:\(\s*([A-E])\s*\)|([A-E])\s*[\.)]|([A-E])(?=\s*[\r\n]+))(?=\s|$)",
    re.IGNORECASE,
)
_STANDALONE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def extract_answer(raw_response: str) -> str | None:
    """Extract a defensible A-E answer, returning None if it is ambiguous."""
    strict = _STRICT.match(raw_response)
    if strict:
        return strict.group(1).upper()
    cue_matches = _ANSWER_CUE.findall(raw_response)
    if cue_matches:
        unique_cues = {item.upper() for item in cue_matches}
        return next(iter(unique_cues)) if len(unique_cues) == 1 else None
    leading = _LEADING_ANSWER.match(raw_response)
    if leading:
        return next(item.upper() for item in leading.groups() if item is not None)
    standalone = _STANDALONE.findall(raw_response)
    unique = {item.upper() for item in standalone}
    return next(iter(unique)) if len(unique) == 1 else None


def variant_summary(results: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    return (results.groupby("prompt_variant", as_index=False)
            .agg(n=("question_id", "size"), accuracy=("correct", "mean"),
                 valid_answer_rate=("predicted_answer", lambda x: x.notna().mean()),
                 mean_inference_time_s=("inference_time", "mean")))


def question_disagreement(results: pd.DataFrame) -> pd.DataFrame:
    """Quantify whether variants produced different extracted answers per question."""
    return (results.groupby("question_id", as_index=False)
            .agg(n_unique_predictions=("predicted_answer", lambda x: x.dropna().nunique()),
                 any_invalid=("predicted_answer", lambda x: x.isna().any())))


def prediction_stability_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-question answer stability while keeping nulls distinct.

    A question is stable only when all five extracted predictions are valid and
    identical. Questions containing one or more null extractions are reported
    separately, according to whether their remaining valid predictions agree.
    """
    import pandas as pd

    rows = []
    for _, group in results.groupby("question_id", sort=False):
        predictions = group["predicted_answer"]
        valid_predictions = predictions.dropna()
        valid_count = int(valid_predictions.size)
        unique_valid_predictions = int(valid_predictions.nunique())
        if valid_count == len(predictions) and unique_valid_predictions == 1:
            category = PREDICTION_STABILITY_CATEGORIES[0]
        elif valid_count == len(predictions):
            category = PREDICTION_STABILITY_CATEGORIES[1]
        elif unique_valid_predictions > 1:
            category = PREDICTION_STABILITY_CATEGORIES[2]
        elif valid_count == 0:
            category = PREDICTION_STABILITY_CATEGORIES[4]
        else:
            category = PREDICTION_STABILITY_CATEGORIES[3]
        rows.append({"category": category})

    counts = pd.DataFrame(rows).value_counts("category").reindex(
        PREDICTION_STABILITY_CATEGORIES, fill_value=0
    )
    total_questions = int(counts.sum())
    return pd.DataFrame({
        "category": counts.index,
        "question_count": counts.values,
        "percentage": 100 * counts.values / total_questions,
    })


def p1_correctness_flips(results: pd.DataFrame) -> pd.DataFrame:
    """Count correctness changes for each variant against P1 on shared questions."""
    import pandas as pd

    baseline = results[results["prompt_variant"] == "P1"].set_index("question_id")["correct"]
    rows = []
    for variant in sorted(set(results["prompt_variant"]) - {"P1"}):
        comparison = results[results["prompt_variant"] == variant].set_index("question_id")["correct"]
        aligned_baseline, aligned_comparison = baseline.align(comparison, join="inner")
        rows.append({
            "prompt_variant": variant,
            "correct_to_incorrect": int((aligned_baseline & ~aligned_comparison).sum()),
            "incorrect_to_correct": int((~aligned_baseline & aligned_comparison).sum()),
        })
    return pd.DataFrame(rows)
