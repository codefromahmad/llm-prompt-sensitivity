"""Answer extraction and aggregation helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

VALID_ANSWERS = frozenset("ABCDE")
_STRICT = re.compile(r"^\s*\(?\s*([A-E])\s*\)?(?:\s*[\.!,:;]?\s*)$", re.IGNORECASE)
_ANSWER_CUE = re.compile(r"(?:answer|option|choice|letter)\s*(?:is|:)?\s*\(?\s*([A-E])\b", re.IGNORECASE)
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
