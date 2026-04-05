from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable, Mapping
import math
import re


_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_EVALUATOR_KEYS = (
    "evaluator_name",
    "asr_system",
    "workflow",
    "condition_name",
    "model_tag",
    "model_name",
    "system",
)
_WER_KEYS = ("wer", "WER")
_CER_KEYS = ("cer", "CER")
_RUNTIME_KEYS = ("runtime_seconds", "elapsed_seconds", "duration_seconds")


def normalize_text_for_metrics(text: Any) -> str:
    normalized = str(text or "").lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def compute_wer_cer(reference: Any, hypothesis: Any) -> dict[str, Any]:
    from jiwer import cer, wer

    normalized_reference = normalize_text_for_metrics(reference)
    normalized_hypothesis = normalize_text_for_metrics(hypothesis)

    if not normalized_reference:
        error_rate = 0.0 if not normalized_hypothesis else 1.0
        return {
            "reference_normalized": normalized_reference,
            "hypothesis_normalized": normalized_hypothesis,
            "wer": error_rate,
            "cer": error_rate,
        }

    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "wer": float(wer(normalized_reference, normalized_hypothesis)),
        "cer": float(cer(normalized_reference, normalized_hypothesis)),
    }


def _levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)

    previous_row = list(range(len(text_b) + 1))
    for index_a, char_a in enumerate(text_a, start=1):
        current_row = [index_a]
        for index_b, char_b in enumerate(text_b, start=1):
            insertion_cost = current_row[index_b - 1] + 1
            deletion_cost = previous_row[index_b] + 1
            substitution_cost = previous_row[index_b - 1] + (char_a != char_b)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row
    return previous_row[-1]


def compute_text_similarity(text_a: Any, text_b: Any) -> float:
    normalized_a = normalize_text_for_metrics(text_a)
    normalized_b = normalize_text_for_metrics(text_b)
    max_length = max(len(normalized_a), len(normalized_b), 1)
    distance = _levenshtein_distance(normalized_a, normalized_b)
    return max(0.0, 1.0 - (distance / max_length))


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isnan(float(value))
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(converted)


def summarize_metric_column(values: Iterable[Any]) -> dict[str, Any]:
    numeric_values = [float(value) for value in values if _is_number(value)]
    if not numeric_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(numeric_values),
        "mean": float(mean(numeric_values)),
        "median": float(median(numeric_values)),
        "min": float(min(numeric_values)),
        "max": float(max(numeric_values)),
    }


def _to_records(results_df: Any) -> list[dict[str, Any]]:
    if results_df is None:
        return []
    if hasattr(results_df, "to_dict"):
        try:
            records = results_df.to_dict(orient="records")
            return [dict(record) for record in records]
        except TypeError:
            pass
    if isinstance(results_df, Mapping):
        return [dict(results_df)]
    return [dict(record) for record in results_df]


def _find_first_present_key(records: list[dict[str, Any]], candidate_keys: tuple[str, ...]) -> str | None:
    for key in candidate_keys:
        if any(key in record for record in records):
            return key
    return None


def _extract_numeric_values(records: list[dict[str, Any]], key: str | None) -> list[float]:
    if key is None:
        return []
    return [float(record[key]) for record in records if _is_number(record.get(key))]


def pick_best_asr_evaluator(results_df: Any) -> dict[str, Any]:
    records = _to_records(results_df)
    if not records:
        raise ValueError("No ASR evaluation records were provided.")

    evaluator_key = _find_first_present_key(records, _EVALUATOR_KEYS)
    wer_key = _find_first_present_key(records, _WER_KEYS)
    cer_key = _find_first_present_key(records, _CER_KEYS)
    runtime_key = _find_first_present_key(records, _RUNTIME_KEYS)

    if evaluator_key is None:
        raise ValueError("Could not determine the evaluator-identifying column.")
    if wer_key is None:
        raise ValueError("Could not determine the WER column for evaluator selection.")

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[str(record.get(evaluator_key, "unknown"))].append(record)

    summaries = []
    for evaluator_name, evaluator_records in grouped_records.items():
        wer_values = _extract_numeric_values(evaluator_records, wer_key)
        cer_values = _extract_numeric_values(evaluator_records, cer_key)
        runtime_values = _extract_numeric_values(evaluator_records, runtime_key)

        summary = {
            "evaluator": evaluator_name,
            "sample_count": len(evaluator_records),
            "mean_wer": float(mean(wer_values)) if wer_values else float("inf"),
            "mean_cer": float(mean(cer_values)) if cer_values else float("inf"),
            "mean_runtime_seconds": float(mean(runtime_values)) if runtime_values else float("inf"),
        }
        summaries.append(summary)

    best_summary = min(
        summaries,
        key=lambda item: (
            item["mean_wer"],
            item["mean_cer"],
            item["mean_runtime_seconds"],
            item["evaluator"],
        ),
    )
    best_summary["selection_rule"] = "minimum mean WER, then mean CER, then mean runtime"
    return best_summary


__all__ = [
    "normalize_text_for_metrics",
    "compute_wer_cer",
    "compute_text_similarity",
    "summarize_metric_column",
    "pick_best_asr_evaluator",
]