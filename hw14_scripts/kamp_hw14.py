from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Any
import csv
import json

from hw14_analysis_utils import pick_best_asr_evaluator
from hw14_data_utils import CHECKPOINT_PATH, EXPERIMENTS_DIR, PROJECT_ROOT, load_checkpoint, now_iso, save_audio, save_checkpoint, save_text_artifact
import hw14_experiment_runner as runner

ASR_MODELS = {
    "ga20a_whisper_pipeline": "openai/whisper-small",
    "ga20b_wav2vec2_ctc": "facebook/wav2vec2-base-960h",
    "ga20c_whisper_direct": "openai/whisper-small",
    "ga20e_speecht5_asr": "microsoft/speecht5_asr",
}

TTS_MODELS = {
    "ga20f_speecht5_tts": "microsoft/speecht5_tts",
    "ga20f_speecht5_vocoder": "microsoft/speecht5_hifigan",
    "ga20g_mms_tts": "facebook/mms-tts-eng",
    "ga20h_bark": "suno/bark-small",
    "ga20k_bark": "suno/bark-small",
}

TEXT_BANK = {
    "hello_dog": "Hello, my dog is cute.",
    "llamas": "There are llamas all around.",
    "banking_request": "I need to transfer money from savings to checking today.",
    "account_followup": "Please review the recent activity on my account and let me know whether any charges require follow-up.",
}

DEFAULT_SEEDS = [42, 123, 555]

SHARED_ASSETS_DIR = EXPERIMENTS_DIR / "shared_assets"
MANIFESTS_DIR = SHARED_ASSETS_DIR / "manifests"
TEXT_BANK_PATH = SHARED_ASSETS_DIR / "text_bank" / "text_bank.json"
LONG_AUDIO_PATH = SHARED_ASSETS_DIR / "long_audio_cache" / "long_audio.wav"
FIXED_SUBSET_PATH = MANIFESTS_DIR / "minds14_fixed_subset.csv"
RESAMPLING_SUBSET_PATH = MANIFESTS_DIR / "minds14_resampling_subset.csv"

GROUP_DIRS = {
    "group_a_asr": EXPERIMENTS_DIR / "group_a_asr",
    "group_b_structured_tts": EXPERIMENTS_DIR / "group_b_structured_tts",
    "group_c_bark_cross": EXPERIMENTS_DIR / "group_c_bark_cross",
}

GROUP_EXECUTION_MAP = {
    "group_a_asr": {
        "experiments": ["exp1_baselines", "exp2_shortform_asr", "exp3_asr_sensitivity", "exp4_longform_whisper"],
        "expected_counts": {
            "baseline_rows": 5,
            "exp2_rows": 96,
            "exp3_rows": 114,
            "exp4_rows": 6,
        },
        "verification_script": "hw14_scripts/verification_scripts/verify_group_a.py",
    },
    "group_b_structured_tts": {
        "experiments": ["exp1_baselines", "exp5_speecht5_tts", "exp6_mms_tts"],
        "expected_counts": {
            "baseline_rows": 2,
            "exp5_rows": 6,
            "exp6_rows": 12,
        },
        "verification_script": "hw14_scripts/verification_scripts/verify_group_b.py",
    },
    "group_c_bark_cross": {
        "experiments": ["exp1_baselines", "exp7_bark", "exp8_cross_tts"],
        "expected_counts": {
            "baseline_rows": 2,
            "exp7_rows": 20,
            "exp8_rows": 9,
        },
        "verification_script": "hw14_scripts/verification_scripts/verify_group_c.py",
    },
}

GROUP_A_BASELINE_HEADER = [
    "experiment",
    "script",
    "task_type",
    "baseline_mode",
    "dataset_sample_index",
    "reference_text",
    "predicted_text",
    "model_name",
    "input_text",
    "voice_condition",
    "input_sampling_rate",
    "saved_sampling_rate",
    "output_audio_path",
    "output_text_path",
    "output_figure_path",
    "runtime_sec",
    "timestamp",
    "notes",
]

GROUP_A_EXP2_HEADER = [
    "experiment",
    "script",
    "workflow_name",
    "sample_index",
    "model_name",
    "input_sampling_rate",
    "audio_duration_sec",
    "predicted_text_raw",
    "predicted_text_normalized",
    "reference_text_raw",
    "reference_text_normalized",
    "wer",
    "cer",
    "inference_time_sec",
    "timestamp",
]

GROUP_A_EXP3_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "workflow_name",
    "sample_index",
    "resampling_rate",
    "decode_policy",
    "predicted_text_raw",
    "predicted_text_clean",
    "predicted_text_normalized",
    "reference_text_normalized",
    "raw_wer",
    "raw_cer",
    "clean_wer",
    "clean_cer",
    "inference_time_sec",
    "timestamp",
]

GROUP_A_EXP4_HEADER = [
    "experiment",
    "script",
    "chunk_length_s",
    "batch_size",
    "timestamp_mode",
    "input_duration_sec",
    "num_segments",
    "total_words",
    "total_characters",
    "mean_segment_duration_sec",
    "transcript_text",
    "transcript_similarity_to_baseline",
    "inference_time_sec",
    "timestamp",
]

GROUP_A_SHORTFORM_WORKFLOWS = [
    ("GA20A", "ga20a_whisper_pipeline"),
    ("GA20B", "ga20b_wav2vec2_ctc"),
    ("GA20C", "ga20c_whisper_direct"),
    ("GA20E", "ga20e_speecht5_asr"),
]

GROUP_A_RESAMPLING_WORKFLOWS = [
    ("GA20A", "ga20a_whisper_pipeline"),
    ("GA20B", "ga20b_wav2vec2_ctc"),
    ("GA20E", "ga20e_speecht5_asr"),
]

GROUP_A_RESAMPLING_RATES = [8000, 16000, 24000]
MAX_RESAMPLING_DURATION_SECONDS = 20.0

GROUP_A_LONGFORM_CONDITIONS = [
    {"chunk_length_s": 5, "batch_size": 8},
    {"chunk_length_s": 5, "batch_size": 4},
    {"chunk_length_s": 10, "batch_size": 4},
    {"chunk_length_s": 10, "batch_size": 8},
    {"chunk_length_s": 15, "batch_size": 4},
    {"chunk_length_s": 15, "batch_size": 8},
]

GROUP_TTS_BASELINE_HEADER = list(GROUP_A_BASELINE_HEADER)

GROUP_B_EXP5_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "text_tag",
    "input_text",
    "speaker_condition",
    "speaker_embedding_source",
    "spectrogram_path",
    "waveform_path",
    "spectrogram_frames",
    "spectrogram_bins",
    "waveform_duration_sec",
    "sample_rate_saved",
    "waveform_rms",
    "generation_time_sec",
    "timestamp",
]

GROUP_B_EXP6_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "model_name",
    "seed",
    "text_tag",
    "input_text",
    "speaking_rate",
    "output_path",
    "sample_rate_saved",
    "waveform_num_samples",
    "waveform_duration_sec",
    "waveform_rms",
    "generation_time_sec",
    "timestamp",
]

GROUP_B_GA20F_BASELINE_TEXT = TEXT_BANK["llamas"]
GROUP_B_GA20G_BASELINE_TEXT = "Hello - my dog is cute"
GROUP_B_EXP5_TEXT_TAGS = ["llamas", "hello_dog", "banking_request"]
GROUP_B_EXP5_FIXED_SEED = 42
GROUP_B_EXP5_SPEAKER_CONDITIONS = [
    {
        "speaker_condition": "helper_embedding",
        "speaker_embedding_source": "cmu_arctic_reference_embedding",
    },
    {
        "speaker_condition": "zero_vector",
        "speaker_embedding_source": "documented_zero_vector_control",
    },
]
GROUP_B_EXP6_TEXT_TAGS = ["hello_dog", "banking_request"]
GROUP_B_EXP6_SEEDS = [42, 123, 555]
GROUP_B_EXP6_SPEAKING_RATES = [0.8, 1.0, 1.2]

GROUP_C_EXP7_HEADER = [
    "experiment",
    "sub_experiment",
    "script",
    "prompt_mode",
    "voice_condition",
    "voice_preset",
    "seed",
    "input_text",
    "output_path",
    "model_sample_rate",
    "saved_sample_rate",
    "save_rate_mode",
    "waveform_duration_sec",
    "generation_time_sec",
    "round_trip_transcription",
    "round_trip_wer",
    "timestamp",
]

GROUP_C_EXP8_HEADER = [
    "experiment",
    "system_id",
    "source_script",
    "text_tag",
    "input_text",
    "output_path",
    "saved_sample_rate",
    "waveform_duration_sec",
    "generation_time_sec",
    "asr_evaluator",
    "round_trip_transcription",
    "round_trip_wer",
    "round_trip_cer",
    "controllability_mode",
    "reused_from_experiment",
    "timestamp",
]

GROUP_C_GA20H_BASELINE_PROMPT = (
    "Hello, my name is Suno. And, uh \u2014 and I like pizza. [laughs]\n"
    "But I also have other interests such as playing tic tac toe."
)

GROUP_C_GA20K_BASELINE_TEXT = "Hello, my dog is cute"
GROUP_C_BARK_PRESET_DEFAULT = "v2/en_speaker_5"

GROUP_C_PROMPT_STYLES = [
    {
        "prompt_mode": "plain",
        "text": "Hello, my name is Suno and I like pizza. But I also have other interests such as playing tic tac toe.",
    },
    {
        "prompt_mode": "expr",
        "text": GROUP_C_GA20H_BASELINE_PROMPT,
    },
    {
        "prompt_mode": "strong_expr",
        "text": "Hello, my name is Suno! [laughs] I absolutely love pizza. [gasps] But I also have other interests, such as playing tic tac toe!",
    },
]

GROUP_C_VOICE_CONDITIONS = [
    {"voice_condition": "no_preset", "voice_preset": None},
    {"voice_condition": "preset5", "voice_preset": "v2/en_speaker_5"},
    {"voice_condition": "preset6", "voice_preset": "v2/en_speaker_6"},
]

GROUP_C_EXP7_SEEDS = [42, 123]
GROUP_C_EXP8_TEXT_TAGS = ["hello_dog", "llamas", "banking_request"]
GROUP_C_EXP8_MMS_SEED = 555
GROUP_C_EXP8_MMS_SPEAKING_RATE = 1.0
GROUP_C_EXP8_BARK_SEED = 42

GROUP_C_EXP7_AUDIT_TARGETS = [
    {"sub_experiment": "7A", "prompt_mode": "plain", "seed": 42, "audit_tag": "plain"},
    {"sub_experiment": "7A", "prompt_mode": "expr", "seed": 42, "audit_tag": "expr"},
    {"sub_experiment": "7B", "voice_condition": "no_preset", "seed": 42, "audit_tag": "nopreset"},
    {"sub_experiment": "7B", "voice_condition": "preset5", "seed": 42, "audit_tag": "preset5"},
]


def _write_csv_rows(filepath: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def _read_manifest_rows(filepath: Path) -> list[dict[str, Any]]:
    if not filepath.exists():
        return []
    with filepath.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_raw_minds_dataset() -> Any:
    from datasets import load_dataset

    return load_dataset(runner.MINDS_DATASET_NAME, name=runner.MINDS_DATASET_CONFIG, split=runner.MINDS_SPLIT)


def _project_relative_string(path_like: str | Path | None) -> str:
    if not path_like:
        return ""
    path = Path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _audio_duration_seconds(audio_array: Any, sampling_rate: int) -> float:
    return round(len(audio_array) / float(sampling_rate or 1), 6)


def _sample_tag(sample_index: int | None) -> str:
    if sample_index is None:
        return "sample_unknown"
    return f"sample{int(sample_index):03d}"


@lru_cache(maxsize=1)
def _load_native_rate_minds_dataset() -> Any:
    return _load_raw_minds_dataset()


def _load_minds_example_for_rate(sample_index: int, sampling_rate: int) -> dict[str, Any]:
    if int(sampling_rate) == 8000:
        dataset = _load_native_rate_minds_dataset()
    else:
        dataset = runner._load_minds_dataset_cached(int(sampling_rate))
    return dict(dataset[int(sample_index)])


def _group_a_root_for_mode(mode: str, output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    elif mode == "dry_run":
        root = GROUP_DIRS["group_a_asr"] / "dry_run"
    else:
        root = GROUP_DIRS["group_a_asr"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _group_a_dirs(group_root: Path) -> dict[str, Path]:
    directories = {
        "root": group_root,
        "exp1": group_root / "exp1_baselines",
        "exp2": group_root / "exp2_shortform_asr",
        "exp3": group_root / "exp3_asr_sensitivity",
        "exp4": group_root / "exp4_longform_whisper",
        "full": group_root / "full",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _update_group_checkpoint(
    group_name: str,
    mode: str,
    group_root: Path,
    block_name: str,
    details: dict[str, Any],
) -> Path:
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    if not isinstance(checkpoint, dict):
        checkpoint = {}

    groups = checkpoint.setdefault("groups", {})
    group_state = groups.setdefault(group_name, {})
    mode_state = group_state.setdefault(mode, {})
    completed_blocks = list(mode_state.get("completed_blocks", []))
    if block_name not in completed_blocks:
        completed_blocks.append(block_name)

    block_summaries = dict(mode_state.get("block_summaries", {}))
    block_summaries[block_name] = dict(details)

    mode_state.update(
        {
            "group_root": str(group_root),
            "updated_at": now_iso(),
            "completed_blocks": completed_blocks,
            "block_summaries": block_summaries,
        }
    )
    group_state[mode] = mode_state
    groups[group_name] = group_state
    checkpoint.update(
        {
            "groups": groups,
            "last_group": group_name,
            "last_mode": mode,
            "last_block": block_name,
        }
    )
    return save_checkpoint(checkpoint, CHECKPOINT_PATH)


def _group_a_baseline_row(result: dict[str, Any], script: str) -> dict[str, Any]:
    return {
        "experiment": "exp1_baselines",
        "script": script,
        "task_type": "asr",
        "baseline_mode": True,
        "dataset_sample_index": result.get("sample_index", ""),
        "reference_text": result.get("reference_text_raw", ""),
        "predicted_text": result.get("predicted_text_raw", ""),
        "model_name": result.get("model_name", ""),
        "input_text": "",
        "voice_condition": "",
        "input_sampling_rate": result.get("input_sampling_rate", runner.DEFAULT_RESAMPLED_RATE),
        "saved_sampling_rate": "",
        "output_audio_path": "",
        "output_text_path": _project_relative_string(result.get("output_text_path") or result.get("output_json_path")),
        "output_figure_path": _project_relative_string(result.get("output_figure_path")),
        "runtime_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
        "notes": result.get("notes", ""),
    }


def _group_a_exp2_row(result: dict[str, Any], script: str, audio_duration_sec: float) -> dict[str, Any]:
    return {
        "experiment": "exp2_shortform_asr",
        "script": script,
        "workflow_name": result.get("workflow_name", ""),
        "sample_index": result.get("sample_index", ""),
        "model_name": result.get("model_name", ""),
        "input_sampling_rate": result.get("input_sampling_rate", ""),
        "audio_duration_sec": audio_duration_sec,
        "predicted_text_raw": result.get("predicted_text_raw", ""),
        "predicted_text_normalized": result.get("predicted_text_normalized", ""),
        "reference_text_raw": result.get("reference_text_raw", ""),
        "reference_text_normalized": result.get("reference_text_normalized", ""),
        "wer": result.get("wer", ""),
        "cer": result.get("cer", ""),
        "inference_time_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_a_exp3a_row(result: dict[str, Any], script: str) -> dict[str, Any]:
    return {
        "experiment": "exp3_asr_sensitivity",
        "sub_experiment": "3A",
        "script": script,
        "workflow_name": result.get("workflow_name", ""),
        "sample_index": result.get("sample_index", ""),
        "resampling_rate": result.get("resampling_rate", result.get("input_sampling_rate", "")),
        "decode_policy": result.get("decode_policy", "clean"),
        "predicted_text_raw": result.get("predicted_text_raw", ""),
        "predicted_text_clean": result.get("predicted_text_raw", ""),
        "predicted_text_normalized": result.get("predicted_text_normalized", ""),
        "reference_text_normalized": result.get("reference_text_normalized", ""),
        "raw_wer": result.get("wer", ""),
        "raw_cer": result.get("cer", ""),
        "clean_wer": result.get("wer", ""),
        "clean_cer": result.get("cer", ""),
        "inference_time_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_a_exp3b_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "exp3_asr_sensitivity",
        "sub_experiment": "3B",
        "script": "GA20C",
        "workflow_name": result.get("workflow_name", ""),
        "sample_index": result.get("sample_index", ""),
        "resampling_rate": result.get("input_sampling_rate", runner.DEFAULT_RESAMPLED_RATE),
        "decode_policy": "raw_vs_clean",
        "predicted_text_raw": result.get("predicted_text_raw", ""),
        "predicted_text_clean": result.get("predicted_text_clean", ""),
        "predicted_text_normalized": result.get("predicted_text_normalized", ""),
        "reference_text_normalized": result.get("reference_text_normalized", ""),
        "raw_wer": result.get("raw_wer", ""),
        "raw_cer": result.get("raw_cer", ""),
        "clean_wer": result.get("clean_wer", ""),
        "clean_cer": result.get("clean_cer", ""),
        "inference_time_sec": "",
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_a_exp4_row(result: dict[str, Any], input_duration_sec: float) -> dict[str, Any]:
    transcript_text = str(result.get("predicted_text_raw", ""))
    normalized_words = str(result.get("predicted_text_normalized", "")).split()
    return {
        "experiment": "exp4_longform_whisper",
        "script": "GA20D",
        "chunk_length_s": result.get("chunk_length_s", ""),
        "batch_size": result.get("batch_size", ""),
        "timestamp_mode": "return_timestamps=True" if result.get("return_timestamps") else "return_timestamps=False",
        "input_duration_sec": input_duration_sec,
        "num_segments": result.get("segment_count", ""),
        "total_words": len(normalized_words),
        "total_characters": result.get("transcript_length_chars", len(transcript_text)),
        "mean_segment_duration_sec": result.get("mean_segment_duration_sec", ""),
        "transcript_text": transcript_text,
        "transcript_similarity_to_baseline": result.get("similarity_to_baseline", ""),
        "inference_time_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
    }


def _select_best_asr_evaluator(exp2_rows: list[dict[str, Any]], output_path: str | Path | None = None) -> dict[str, Any]:
    selection_rows = [
        {
            "workflow": row["workflow_name"],
            "wer": row["wer"],
            "cer": row["cer"],
            "runtime_seconds": row["inference_time_sec"],
        }
        for row in exp2_rows
    ]
    summary = pick_best_asr_evaluator(selection_rows)
    workflow_name = str(summary.get("evaluator", ""))
    best_summary = {
        "workflow_name": workflow_name,
        "evaluator": workflow_name,
        "sample_count": summary.get("sample_count"),
        "mean_wer": summary.get("mean_wer"),
        "mean_cer": summary.get("mean_cer"),
        "mean_runtime_seconds": summary.get("mean_runtime_seconds"),
        "selection_rule": summary.get("selection_rule"),
        "source_experiment": "exp2_shortform_asr",
        "selected_at": now_iso(),
    }
    if output_path is not None:
        save_text_artifact(output_path, best_summary, as_json=True)
    return best_summary


def _group_c_root_for_mode(mode: str, output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    elif mode == "dry_run":
        root = GROUP_DIRS["group_c_bark_cross"] / "dry_run"
    else:
        root = GROUP_DIRS["group_c_bark_cross"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _group_b_root_for_mode(mode: str, output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    elif mode == "dry_run":
        root = GROUP_DIRS["group_b_structured_tts"] / "dry_run"
    else:
        root = GROUP_DIRS["group_b_structured_tts"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _group_b_dirs(group_root: Path) -> dict[str, Path]:
    directories = {
        "root": group_root,
        "exp1": group_root / "exp1_baselines",
        "exp5": group_root / "exp5_speecht5_tts",
        "exp6": group_root / "exp6_mms_tts",
        "full": group_root / "full",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _group_c_dirs(group_root: Path) -> dict[str, Path]:
    directories = {
        "root": group_root,
        "exp1": group_root / "exp1_baselines",
        "exp7": group_root / "exp7_bark",
        "exp8": group_root / "exp8_cross_tts",
        "full": group_root / "full",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _resolve_artifact_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_csv_rows_if_exists(filepath: str | Path) -> list[dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _first_existing_audio_path(row: dict[str, Any], candidate_keys: tuple[str, ...]) -> Path | None:
    for key in candidate_keys:
        path = _resolve_artifact_path(row.get(key))
        if path is not None and path.exists():
            return path
    return None


def _audio_metadata_from_path(audio_path: str | Path | None) -> dict[str, Any]:
    path = _resolve_artifact_path(audio_path)
    if path is None or not path.exists():
        return {"saved_sampling_rate": 0, "waveform_duration_sec": 0.0}
    audio_array, sampling_rate = runner._load_wav(path)
    return {
        "saved_sampling_rate": int(sampling_rate),
        "waveform_duration_sec": _audio_duration_seconds(audio_array, int(sampling_rate)),
    }


def _load_group_a_best_asr_evaluator() -> dict[str, Any]:
    best_eval_path = GROUP_DIRS["group_a_asr"] / "full" / "best_asr_evaluator.json"
    if not best_eval_path.exists():
        raise RuntimeError(
            f"Group A best ASR evaluator is missing: {best_eval_path}. Complete IMPL-3 before running Group C."
        )
    with best_eval_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload.get("workflow_name"):
        raise RuntimeError(f"workflow_name missing in {best_eval_path}")
    return payload


def _group_tts_baseline_row(result: dict[str, Any], script: str, voice_condition: str, notes: str) -> dict[str, Any]:
    return {
        "experiment": "exp1_baselines",
        "script": script,
        "task_type": "tts",
        "baseline_mode": True,
        "dataset_sample_index": "",
        "reference_text": "",
        "predicted_text": "",
        "model_name": result.get("model_name", ""),
        "input_text": result.get("input_text", ""),
        "voice_condition": voice_condition,
        "input_sampling_rate": "",
        "saved_sampling_rate": result.get("saved_sampling_rate", ""),
        "output_audio_path": _project_relative_string(result.get("output_audio_path")),
        "output_text_path": _project_relative_string(result.get("output_text_path") or result.get("output_json_path")),
        "output_figure_path": _project_relative_string(result.get("output_figure_path")),
        "runtime_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
        "notes": notes,
    }


def _group_b_exp5_row(
    result: dict[str, Any],
    text_tag: str,
    speaker_condition: str,
    speaker_embedding_source: str,
) -> dict[str, Any]:
    return {
        "experiment": "exp5_speecht5_tts",
        "sub_experiment": "5A",
        "script": "GA20F",
        "text_tag": text_tag,
        "input_text": result.get("input_text", ""),
        "speaker_condition": speaker_condition,
        "speaker_embedding_source": speaker_embedding_source,
        "spectrogram_path": _project_relative_string(result.get("output_figure_path")),
        "waveform_path": _project_relative_string(result.get("output_audio_path")),
        "spectrogram_frames": result.get("spectrogram_frames", ""),
        "spectrogram_bins": result.get("spectrogram_bins", ""),
        "waveform_duration_sec": result.get("duration_seconds", ""),
        "sample_rate_saved": result.get("saved_sampling_rate", ""),
        "waveform_rms": result.get("rms", ""),
        "generation_time_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_b_exp6_row(result: dict[str, Any], text_tag: str, sub_experiment: str) -> dict[str, Any]:
    return {
        "experiment": "exp6_mms_tts",
        "sub_experiment": sub_experiment,
        "script": "GA20G",
        "model_name": result.get("model_name", ""),
        "seed": result.get("seed", ""),
        "text_tag": text_tag,
        "input_text": result.get("input_text", ""),
        "speaking_rate": result.get("speaking_rate", ""),
        "output_path": _project_relative_string(result.get("output_audio_path")),
        "sample_rate_saved": result.get("saved_sampling_rate", ""),
        "waveform_num_samples": result.get("sample_count", ""),
        "waveform_duration_sec": result.get("duration_seconds", ""),
        "waveform_rms": result.get("rms", ""),
        "generation_time_sec": result.get("inference_time_sec", ""),
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_c_exp7_row(
    result: dict[str, Any],
    sub_experiment: str,
    script: str,
    prompt_mode: str = "",
    voice_condition: str = "",
    roundtrip_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "exp7_bark",
        "sub_experiment": sub_experiment,
        "script": script,
        "prompt_mode": prompt_mode,
        "voice_condition": voice_condition,
        "voice_preset": result.get("voice_preset", ""),
        "seed": result.get("seed", ""),
        "input_text": result.get("input_text", ""),
        "output_path": _project_relative_string(result.get("output_audio_path")),
        "model_sample_rate": result.get("native_sampling_rate", ""),
        "saved_sample_rate": result.get("saved_sampling_rate", ""),
        "save_rate_mode": result.get("save_rate_mode", ""),
        "waveform_duration_sec": result.get("duration_seconds", ""),
        "generation_time_sec": result.get("inference_time_sec", ""),
        "round_trip_transcription": roundtrip_result.get("predicted_text_raw", "") if roundtrip_result else "",
        "round_trip_wer": roundtrip_result.get("wer", "") if roundtrip_result else "",
        "timestamp": result.get("timestamp", now_iso()),
    }


def _group_c_source_meta_from_result(
    result: dict[str, Any],
    source_script: str,
    controllability_mode: str,
    reused_from_experiment: str = "",
) -> dict[str, Any]:
    metadata = _audio_metadata_from_path(result.get("output_audio_path"))
    return {
        "source_script": source_script,
        "input_text": result.get("input_text", ""),
        "audio_path": _resolve_artifact_path(result.get("output_audio_path")),
        "saved_sampling_rate": _coerce_int(result.get("saved_sampling_rate"), metadata["saved_sampling_rate"]),
        "waveform_duration_sec": _coerce_float(result.get("duration_seconds"), metadata["waveform_duration_sec"]),
        "generation_time_sec": _coerce_float(result.get("inference_time_sec"), 0.0),
        "controllability_mode": controllability_mode,
        "reused_from_experiment": reused_from_experiment,
    }


def _group_c_exp8_row(
    source_meta: dict[str, Any],
    roundtrip_result: dict[str, Any],
    system_id: str,
    text_tag: str,
    asr_evaluator: str,
) -> dict[str, Any]:
    return {
        "experiment": "exp8_cross_tts",
        "system_id": system_id,
        "source_script": source_meta.get("source_script", ""),
        "text_tag": text_tag,
        "input_text": source_meta.get("input_text", ""),
        "output_path": _project_relative_string(source_meta.get("audio_path")),
        "saved_sample_rate": source_meta.get("saved_sampling_rate", ""),
        "waveform_duration_sec": source_meta.get("waveform_duration_sec", ""),
        "generation_time_sec": source_meta.get("generation_time_sec", ""),
        "asr_evaluator": asr_evaluator,
        "round_trip_transcription": roundtrip_result.get("predicted_text_raw", ""),
        "round_trip_wer": roundtrip_result.get("wer", ""),
        "round_trip_cer": roundtrip_result.get("cer", ""),
        "controllability_mode": source_meta.get("controllability_mode", ""),
        "reused_from_experiment": source_meta.get("reused_from_experiment", ""),
        "timestamp": roundtrip_result.get("timestamp", now_iso()),
    }


def _load_exp5_speecht5_reuse(text_tag: str) -> dict[str, Any] | None:
    exp5_csv = GROUP_DIRS["group_b_structured_tts"] / "exp5_speecht5_tts" / "exp5_speecht5_tts.csv"
    for row in _read_csv_rows_if_exists(exp5_csv):
        if row.get("text_tag") != text_tag or row.get("speaker_condition") != "helper_embedding":
            continue
        audio_path = _first_existing_audio_path(row, ("waveform_path", "output_audio_path", "output_path"))
        if audio_path is None:
            continue
        metadata = _audio_metadata_from_path(audio_path)
        return {
            "source_script": "GA20F",
            "input_text": row.get("input_text", TEXT_BANK[text_tag]),
            "audio_path": audio_path,
            "saved_sampling_rate": _coerce_int(row.get("sample_rate_saved") or row.get("saved_sampling_rate"), metadata["saved_sampling_rate"]),
            "waveform_duration_sec": _coerce_float(row.get("waveform_duration_sec"), metadata["waveform_duration_sec"]),
            "generation_time_sec": _coerce_float(row.get("generation_time_sec"), 0.0),
            "controllability_mode": "helper_embedding",
            "reused_from_experiment": "exp5_speecht5_tts",
        }
    return None


def _load_exp6_mms_reuse(text_tag: str) -> dict[str, Any] | None:
    exp6_csv = GROUP_DIRS["group_b_structured_tts"] / "exp6_mms_tts" / "exp6_mms_tts.csv"
    rows = _read_csv_rows_if_exists(exp6_csv)
    candidates: list[tuple[int, dict[str, Any], Path]] = []
    for row in rows:
        if row.get("text_tag") != text_tag:
            continue
        if _coerce_int(row.get("seed"), -1) != GROUP_C_EXP8_MMS_SEED:
            continue
        audio_path = _first_existing_audio_path(row, ("output_path", "output_audio_path", "waveform_path"))
        if audio_path is None:
            continue
        sub_experiment = row.get("sub_experiment", "")
        speaking_rate = str(row.get("speaking_rate", "")).strip()
        if sub_experiment == "6B" and speaking_rate in {"1", "1.0", "1.00"}:
            priority = 0
        elif sub_experiment == "6A":
            priority = 1
        else:
            continue
        candidates.append((priority, row, audio_path))

    if not candidates:
        return None

    _, row, audio_path = sorted(candidates, key=lambda item: item[0])[0]
    metadata = _audio_metadata_from_path(audio_path)
    return {
        "source_script": "GA20G",
        "input_text": row.get("input_text", TEXT_BANK[text_tag]),
        "audio_path": audio_path,
        "saved_sampling_rate": _coerce_int(row.get("sample_rate_saved") or row.get("saved_sampling_rate"), metadata["saved_sampling_rate"]),
        "waveform_duration_sec": _coerce_float(row.get("waveform_duration_sec"), metadata["waveform_duration_sec"]),
        "generation_time_sec": _coerce_float(row.get("generation_time_sec"), 0.0),
        "controllability_mode": "seed_555_default_rate",
        "reused_from_experiment": "exp6_mms_tts",
    }


def _select_group_c_audit_targets(primary_records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    target_specs = GROUP_C_EXP7_AUDIT_TARGETS[:1] if mode == "dry_run" else GROUP_C_EXP7_AUDIT_TARGETS
    for target in target_specs:
        for record in primary_records:
            if record.get("sub_experiment") != target["sub_experiment"]:
                continue
            if record.get("seed") != target["seed"]:
                continue
            if target["sub_experiment"] == "7A" and record.get("prompt_mode") != target["prompt_mode"]:
                continue
            if target["sub_experiment"] == "7B" and record.get("voice_condition") != target["voice_condition"]:
                continue
            selected.append({"audit_tag": target["audit_tag"], **record})
            break
    return selected


def build_fixed_minds_subset(
    manifest_path: str | Path = FIXED_SUBSET_PATH,
    resampling_manifest_path: str | Path = RESAMPLING_SUBSET_PATH,
    seed: int = DEFAULT_SEEDS[0],
    fixed_count: int = 24,
    resampling_count: int = 10,
    target_sampling_rate: int = runner.DEFAULT_RESAMPLED_RATE,
    force: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    resampling_manifest_path = Path(resampling_manifest_path)

    if manifest_path.exists() and resampling_manifest_path.exists() and not force:
        return {
            "fixed_manifest": str(manifest_path),
            "resampling_manifest": str(resampling_manifest_path),
            "seed": int(seed),
            "fixed_rows": _read_manifest_rows(manifest_path),
            "resampling_rows": _read_manifest_rows(resampling_manifest_path),
        }

    dataset = _load_raw_minds_dataset()
    dataset_size = len(dataset)
    if fixed_count > dataset_size:
        raise ValueError(f"Requested {fixed_count} fixed samples from a dataset of size {dataset_size}.")
    if resampling_count > fixed_count:
        raise ValueError("resampling_count cannot exceed fixed_count.")

    rng = Random(int(seed))
    selected_indices = rng.sample(range(dataset_size), int(fixed_count))
    fieldnames = [
        "selection_seed",
        "sample_order",
        "sample_index",
        "reference_text",
        "native_sampling_rate",
        "resampled_sampling_rate",
        "duration_seconds",
        "dataset_name",
        "dataset_config",
        "split",
    ]

    fixed_rows: list[dict[str, Any]] = []
    for order, sample_index in enumerate(selected_indices):
        example = dataset[int(sample_index)]
        audio_info = example["audio"]
        duration_seconds = len(audio_info["array"]) / float(audio_info["sampling_rate"])
        fixed_rows.append(
            {
                "selection_seed": int(seed),
                "sample_order": order,
                "sample_index": int(sample_index),
                "reference_text": example["english_transcription"],
                "native_sampling_rate": int(audio_info["sampling_rate"]),
                "resampled_sampling_rate": int(target_sampling_rate),
                "duration_seconds": round(float(duration_seconds), 6),
                "dataset_name": runner.MINDS_DATASET_NAME,
                "dataset_config": runner.MINDS_DATASET_CONFIG,
                "split": runner.MINDS_SPLIT,
            }
        )

    eligible_resampling_rows = [
        row for row in fixed_rows if float(row["duration_seconds"]) <= float(MAX_RESAMPLING_DURATION_SECONDS)
    ]
    if len(eligible_resampling_rows) < int(resampling_count):
        raise ValueError(
            "The fixed MINDS-14 subset did not contain enough <=20 second clips for the resampling study. "
            "Rerun build_fixed_minds_subset with a different seed or fixed_count."
        )

    resampling_rows = eligible_resampling_rows[: int(resampling_count)]
    for order, row in enumerate(resampling_rows):
        row["sample_order"] = order

    _write_csv_rows(manifest_path, fieldnames, fixed_rows)
    _write_csv_rows(resampling_manifest_path, fieldnames, resampling_rows)

    return {
        "fixed_manifest": str(manifest_path),
        "resampling_manifest": str(resampling_manifest_path),
        "seed": int(seed),
        "fixed_rows": fixed_rows,
        "resampling_rows": resampling_rows,
    }


def persist_text_bank(text_bank_path: str | Path = TEXT_BANK_PATH, force: bool = False) -> str:
    text_bank_path = Path(text_bank_path)
    if text_bank_path.exists() and not force:
        return str(text_bank_path)
    save_text_artifact(text_bank_path, TEXT_BANK, as_json=True)
    return str(text_bank_path)


def cache_long_audio_asset(
    output_path: str | Path = LONG_AUDIO_PATH,
    target_seconds: int = 60,
    target_sample_rate: int = runner.DEFAULT_RESAMPLED_RATE,
    force: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        from scipy.io import wavfile

        sampling_rate, array = wavfile.read(output_path)
        return {
            "path": str(output_path),
            "sample_rate": int(sampling_rate),
            "sample_count": int(len(array)),
            "cached": True,
        }

    long_audio = runner._generate_long_audio_array(
        target_seconds=int(target_seconds),
        target_sample_rate=int(target_sample_rate),
    )
    audio_stats = save_audio(long_audio, int(target_sample_rate), output_path, label="long_audio_cache")
    audio_stats["cached"] = False
    return audio_stats


def ensure_shared_assets(force: bool = False) -> dict[str, Any]:
    manifests = build_fixed_minds_subset(force=force)
    text_bank_path = persist_text_bank(force=force)
    long_audio_stats = cache_long_audio_asset(force=force)
    return {
        "fixed_manifest": manifests["fixed_manifest"],
        "resampling_manifest": manifests["resampling_manifest"],
        "text_bank": text_bank_path,
        "long_audio": long_audio_stats["path"],
        "seed": manifests["seed"],
        "fixed_rows": manifests["fixed_rows"],
        "resampling_rows": manifests["resampling_rows"],
    }


def _resolve_group_root(group_name: str, output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        path = Path(output_root)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if group_name not in GROUP_DIRS:
        raise ValueError(f"Unsupported group: {group_name}")
    GROUP_DIRS[group_name].mkdir(parents=True, exist_ok=True)
    return GROUP_DIRS[group_name]


def _prepare_group_plan(group_name: str, mode: str, output_root: str | Path | None = None) -> dict[str, Any]:
    shared_assets = ensure_shared_assets(force=False)
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    group_root = _resolve_group_root(group_name, output_root)
    group_config = GROUP_EXECUTION_MAP[group_name]
    return {
        "group_name": group_name,
        "mode": mode,
        "prepared_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "group_root": str(group_root),
        "shared_assets": {
            "fixed_manifest": shared_assets["fixed_manifest"],
            "resampling_manifest": shared_assets["resampling_manifest"],
            "text_bank": shared_assets["text_bank"],
            "long_audio": shared_assets["long_audio"],
        },
        "planned_blocks": list(group_config["experiments"]),
        "expected_counts": dict(group_config["expected_counts"]),
        "verification_script": group_config["verification_script"],
        "checkpoint": checkpoint,
        "notes": "Step 2 infrastructure prepared. Step-specific notebooks will execute these plans in later implementation steps.",
    }


def run_group_a_asr(mode: str = "dry_run", output_root: str | Path | None = None) -> dict[str, Any]:
    if mode not in {"dry_run", "full"}:
        raise ValueError(f"Unsupported Group A mode: {mode}")

    shared_assets = ensure_shared_assets(force=False)
    group_root = _group_a_root_for_mode(mode, output_root)
    directories = _group_a_dirs(group_root)

    fixed_rows = _read_manifest_rows(FIXED_SUBSET_PATH)
    resampling_rows = _read_manifest_rows(RESAMPLING_SUBSET_PATH)
    if not fixed_rows:
        raise RuntimeError(f"Fixed Group A manifest is missing or empty: {FIXED_SUBSET_PATH}")
    if not resampling_rows:
        raise RuntimeError(f"Resampling Group A manifest is missing or empty: {RESAMPLING_SUBSET_PATH}")

    long_audio_array, long_audio_rate = runner._load_wav(LONG_AUDIO_PATH)
    long_audio_duration_sec = _audio_duration_seconds(long_audio_array, long_audio_rate)

    baseline_rows: list[dict[str, Any]] = []
    exp2_rows: list[dict[str, Any]] = []
    exp3_rows: list[dict[str, Any]] = []
    exp4_rows: list[dict[str, Any]] = []

    best_evaluator: dict[str, Any] | None = None

    try:
        baseline_specs = [
            ("GA20A", lambda: runner.run_ga20a_baseline(output_dir=directories["exp1"])),
            ("GA20B", lambda: runner.run_ga20b_baseline(output_dir=directories["exp1"])),
            ("GA20C", lambda: runner.run_ga20c_baseline(output_dir=directories["exp1"])),
            (
                "GA20D",
                lambda: runner.run_ga20d_corrected_baseline(
                    output_dir=directories["exp1"],
                    audio_array=long_audio_array,
                ),
            ),
            ("GA20E", lambda: runner.run_ga20e_baseline(output_dir=directories["exp1"])),
        ]
        for script, baseline_fn in baseline_specs:
            baseline_rows.append(_group_a_baseline_row(baseline_fn(), script))
            runner.clear_model_caches()
        _write_csv_rows(directories["exp1"] / "exp1_baselines.csv", GROUP_A_BASELINE_HEADER, baseline_rows)
        _update_group_checkpoint(
            "group_a_asr",
            mode,
            group_root,
            "exp1_baselines",
            {"rows": len(baseline_rows), "csv": str(directories["exp1"] / "exp1_baselines.csv")},
        )

        exp2_manifest_rows = fixed_rows[:1] if mode == "dry_run" else fixed_rows
        exp2_dataset = runner._load_minds_dataset_cached(runner.DEFAULT_RESAMPLED_RATE)
        for script, workflow_name in GROUP_A_SHORTFORM_WORKFLOWS:
            for manifest_row in exp2_manifest_rows:
                sample_index = int(manifest_row["sample_index"])
                example = dict(exp2_dataset[sample_index])
                result = runner.run_shortform_asr_condition(
                    workflow_name=workflow_name,
                    audio_array=example["audio"]["array"],
                    sampling_rate=int(example["audio"]["sampling_rate"]),
                    reference_text=manifest_row["reference_text"],
                    sample_index=sample_index,
                    decode_policy="clean",
                    label=f"{workflow_name}_exp2_{_sample_tag(sample_index)}",
                )
                exp2_rows.append(
                    _group_a_exp2_row(
                        result,
                        script,
                        _audio_duration_seconds(example["audio"]["array"], int(example["audio"]["sampling_rate"])),
                    )
                )
            runner.clear_model_caches()

        _write_csv_rows(directories["exp2"] / "exp2_shortform_asr.csv", GROUP_A_EXP2_HEADER, exp2_rows)
        _update_group_checkpoint(
            "group_a_asr",
            mode,
            group_root,
            "exp2_shortform_asr",
            {"rows": len(exp2_rows), "csv": str(directories["exp2"] / "exp2_shortform_asr.csv")},
        )

        exp3a_manifest_rows = resampling_rows[:1] if mode == "dry_run" else resampling_rows
        exp3a_workflows = GROUP_A_RESAMPLING_WORKFLOWS[:1] if mode == "dry_run" else GROUP_A_RESAMPLING_WORKFLOWS
        for script, workflow_name in exp3a_workflows:
            for rate in GROUP_A_RESAMPLING_RATES:
                for manifest_row in exp3a_manifest_rows:
                    sample_index = int(manifest_row["sample_index"])
                    example = _load_minds_example_for_rate(sample_index, rate)
                    result = runner.run_asr_resampling_condition(
                        workflow_name=workflow_name,
                        audio_array=example["audio"]["array"],
                        original_sampling_rate=int(example["audio"]["sampling_rate"]),
                        target_sampling_rate=int(rate),
                        reference_text=manifest_row["reference_text"],
                        sample_index=sample_index,
                        label=f"{workflow_name}_exp3a_{int(rate)}hz_{_sample_tag(sample_index)}",
                    )
                    exp3_rows.append(_group_a_exp3a_row(result, script))
            runner.clear_model_caches()

        _write_csv_rows(directories["exp3"] / "exp3_asr_sensitivity.csv", GROUP_A_EXP3_HEADER, exp3_rows)
        _update_group_checkpoint(
            "group_a_asr",
            mode,
            group_root,
            "exp3_asr_sensitivity_3a",
            {"rows": len(exp3_rows), "csv": str(directories["exp3"] / "exp3_asr_sensitivity.csv")},
        )

        exp3b_manifest_rows = fixed_rows[:1] if mode == "dry_run" else fixed_rows
        exp3b_dataset = runner._load_minds_dataset_cached(runner.DEFAULT_RESAMPLED_RATE)
        for manifest_row in exp3b_manifest_rows:
            sample_index = int(manifest_row["sample_index"])
            example = dict(exp3b_dataset[sample_index])
            result = runner.run_ga20c_decode_comparison(
                audio_array=example["audio"]["array"],
                sampling_rate=int(example["audio"]["sampling_rate"]),
                reference_text=manifest_row["reference_text"],
                sample_index=sample_index,
                label=f"ga20c_exp3b_raw_cleanup_{_sample_tag(sample_index)}",
            )
            exp3_rows.append(_group_a_exp3b_row(result))
        runner.clear_model_caches()

        _write_csv_rows(directories["exp3"] / "exp3_asr_sensitivity.csv", GROUP_A_EXP3_HEADER, exp3_rows)
        _update_group_checkpoint(
            "group_a_asr",
            mode,
            group_root,
            "exp3_asr_sensitivity_3b",
            {"rows": len(exp3_rows), "csv": str(directories["exp3"] / "exp3_asr_sensitivity.csv")},
        )

        longform_conditions = GROUP_A_LONGFORM_CONDITIONS[:1] if mode == "dry_run" else GROUP_A_LONGFORM_CONDITIONS
        baseline_transcript: str | None = None
        for condition in longform_conditions:
            chunk_length_s = int(condition["chunk_length_s"])
            batch_size = int(condition["batch_size"])
            label = f"ga20d_exp4_chunk{chunk_length_s:02d}_batch{batch_size:02d}_longaudio"
            result = runner.run_longform_whisper_condition(
                audio_array=long_audio_array,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                output_dir=directories["exp4"],
                label=label,
                baseline_text=baseline_transcript,
                return_timestamps=True,
            )
            if baseline_transcript is None:
                baseline_transcript = str(result.get("predicted_text_raw", ""))
                result["similarity_to_baseline"] = 1.0
            exp4_rows.append(_group_a_exp4_row(result, long_audio_duration_sec))

        _write_csv_rows(directories["exp4"] / "exp4_longform_whisper.csv", GROUP_A_EXP4_HEADER, exp4_rows)
        _update_group_checkpoint(
            "group_a_asr",
            mode,
            group_root,
            "exp4_longform_whisper",
            {"rows": len(exp4_rows), "csv": str(directories["exp4"] / "exp4_longform_whisper.csv")},
        )
        runner.clear_model_caches()

        if mode == "full":
            best_evaluator = _select_best_asr_evaluator(
                exp2_rows,
                output_path=directories["full"] / "best_asr_evaluator.json",
            )
            _update_group_checkpoint(
                "group_a_asr",
                mode,
                group_root,
                "best_asr_evaluator_selected",
                {"workflow_name": best_evaluator["workflow_name"], "path": str(directories["full"] / "best_asr_evaluator.json")},
            )
        else:
            best_evaluator = _select_best_asr_evaluator(exp2_rows)
    finally:
        runner.clear_model_caches()

    return {
        "group_name": "group_a_asr",
        "mode": mode,
        "executed_at": now_iso(),
        "group_root": str(group_root),
        "shared_assets": {
            "fixed_manifest": shared_assets["fixed_manifest"],
            "resampling_manifest": shared_assets["resampling_manifest"],
            "text_bank": shared_assets["text_bank"],
            "long_audio": shared_assets["long_audio"],
        },
        "counts": {
            "baseline_rows": len(baseline_rows),
            "exp2_rows": len(exp2_rows),
            "exp3_rows": len(exp3_rows),
            "exp4_rows": len(exp4_rows),
        },
        "paths": {
            "exp1_csv": str(directories["exp1"] / "exp1_baselines.csv"),
            "exp2_csv": str(directories["exp2"] / "exp2_shortform_asr.csv"),
            "exp3_csv": str(directories["exp3"] / "exp3_asr_sensitivity.csv"),
            "exp4_csv": str(directories["exp4"] / "exp4_longform_whisper.csv"),
            "best_asr_evaluator": str(directories["full"] / "best_asr_evaluator.json") if mode == "full" else "",
            "checkpoint": str(CHECKPOINT_PATH),
        },
        "best_asr_evaluator": best_evaluator,
    }


def run_group_b_structured_tts(mode: str = "dry_run", output_root: str | Path | None = None) -> dict[str, Any]:
    if mode not in {"dry_run", "full"}:
        raise ValueError(f"Unsupported Group B mode: {mode}")

    shared_assets = ensure_shared_assets(force=False)
    group_root = _group_b_root_for_mode(mode, output_root)
    directories = _group_b_dirs(group_root)

    baseline_rows: list[dict[str, Any]] = []
    exp5_rows: list[dict[str, Any]] = []
    exp6_rows: list[dict[str, Any]] = []

    if mode == "dry_run":
        exp5_plan = [("hello_dog", GROUP_B_EXP5_SPEAKER_CONDITIONS[1])]
        exp6a_plan = [("hello_dog", GROUP_B_EXP6_SEEDS[0])]
        exp6b_plan = [("banking_request", GROUP_B_EXP6_SPEAKING_RATES[0])]
    else:
        exp5_plan = [
            (text_tag, speaker_condition)
            for text_tag in GROUP_B_EXP5_TEXT_TAGS
            for speaker_condition in GROUP_B_EXP5_SPEAKER_CONDITIONS
        ]
        exp6a_plan = [
            (text_tag, seed)
            for text_tag in GROUP_B_EXP6_TEXT_TAGS
            for seed in GROUP_B_EXP6_SEEDS
        ]
        exp6b_plan = [
            (text_tag, speaking_rate)
            for text_tag in GROUP_B_EXP6_TEXT_TAGS
            for speaking_rate in GROUP_B_EXP6_SPEAKING_RATES
        ]

    try:
        ga20f_baseline_result = runner.run_speecht5_tts_condition(
            text=GROUP_B_GA20F_BASELINE_TEXT,
            output_dir=directories["exp1"] / "ga20f_baseline",
            label="ga20f_baseline",
            speaker_label="helper_embedding",
        )
        baseline_rows.append(
            _group_tts_baseline_row(
                ga20f_baseline_result,
                script="GA20F",
                voice_condition="helper_embedding",
                notes="Baseline-equivalent SpeechT5 helper-embedding path with spectrogram and 16 kHz waveform export.",
            )
        )

        ga20g_baseline_result = runner.run_mms_tts_condition(
            text=GROUP_B_GA20G_BASELINE_TEXT,
            output_dir=directories["exp1"] / "ga20g_baseline",
            label="ga20g_baseline",
            save_rate_override=runner.DEFAULT_RESAMPLED_RATE,
        )
        baseline_rows.append(
            _group_tts_baseline_row(
                ga20g_baseline_result,
                script="GA20G",
                voice_condition="default",
                notes="Baseline-equivalent MMS-TTS path with the original hard-coded 16 kHz save behavior preserved.",
            )
        )

        exp1_csv_path = _write_csv_rows(directories["exp1"] / "exp1_baselines.csv", GROUP_TTS_BASELINE_HEADER, baseline_rows)
        _update_group_checkpoint(
            "group_b_structured_tts",
            mode,
            group_root,
            "exp1_baselines_structured_tts",
            {"rows": len(baseline_rows), "csv": str(exp1_csv_path)},
        )

        reference_embedding = runner.load_reference_speaker_embedding()
        zero_vector_embedding = reference_embedding * 0.0
        for text_tag, speaker_condition in exp5_plan:
            condition_name = speaker_condition["speaker_condition"]
            result = runner.run_speecht5_tts_condition(
                text=TEXT_BANK[text_tag],
                output_dir=directories["exp5"],
                label=f"ga20f_exp5_{text_tag}_{condition_name}",
                speaker_embedding=None if condition_name == "helper_embedding" else zero_vector_embedding,
                speaker_label=condition_name,
                seed=GROUP_B_EXP5_FIXED_SEED,
            )
            exp5_rows.append(
                _group_b_exp5_row(
                    result,
                    text_tag=text_tag,
                    speaker_condition=condition_name,
                    speaker_embedding_source=speaker_condition["speaker_embedding_source"],
                )
            )

        exp5_csv_path = _write_csv_rows(directories["exp5"] / "exp5_speecht5_tts.csv", GROUP_B_EXP5_HEADER, exp5_rows)
        _update_group_checkpoint(
            "group_b_structured_tts",
            mode,
            group_root,
            "exp5_speecht5_tts",
            {"rows": len(exp5_rows), "csv": str(exp5_csv_path)},
        )

        for text_tag, seed in exp6a_plan:
            result = runner.run_mms_tts_condition(
                text=TEXT_BANK[text_tag],
                output_dir=directories["exp6"],
                label=f"ga20g_exp6a_{text_tag}_seed{seed}",
                seed=seed,
            )
            exp6_rows.append(_group_b_exp6_row(result, text_tag=text_tag, sub_experiment="6A"))

        exp6_csv_path = _write_csv_rows(directories["exp6"] / "exp6_mms_tts.csv", GROUP_B_EXP6_HEADER, exp6_rows)
        _update_group_checkpoint(
            "group_b_structured_tts",
            mode,
            group_root,
            "exp6a_mms_seed",
            {"rows": len(exp6_rows), "csv": str(exp6_csv_path)},
        )

        for text_tag, speaking_rate in exp6b_plan:
            rate_tag = str(speaking_rate).replace(".", "p")
            result = runner.run_mms_tts_condition(
                text=TEXT_BANK[text_tag],
                output_dir=directories["exp6"],
                label=f"ga20g_exp6b_{text_tag}_rate{rate_tag}_seed555",
                seed=555,
                speaking_rate=speaking_rate,
            )
            exp6_rows.append(_group_b_exp6_row(result, text_tag=text_tag, sub_experiment="6B"))

        exp6_csv_path = _write_csv_rows(directories["exp6"] / "exp6_mms_tts.csv", GROUP_B_EXP6_HEADER, exp6_rows)
        _update_group_checkpoint(
            "group_b_structured_tts",
            mode,
            group_root,
            "exp6b_mms_speaking_rate",
            {"rows": len(exp6_rows), "csv": str(exp6_csv_path)},
        )
    finally:
        runner.clear_model_caches()

    summary_payload = {
        "group_name": "group_b_structured_tts",
        "mode": mode,
        "executed_at": now_iso(),
        "group_root": str(group_root),
        "shared_assets": {
            "fixed_manifest": shared_assets["fixed_manifest"],
            "resampling_manifest": shared_assets["resampling_manifest"],
            "text_bank": shared_assets["text_bank"],
            "long_audio": shared_assets["long_audio"],
        },
        "counts": {
            "baseline_rows": len(baseline_rows),
            "exp5_rows": len(exp5_rows),
            "exp6_rows": len(exp6_rows),
        },
        "paths": {
            "exp1_csv": str(directories["exp1"] / "exp1_baselines.csv"),
            "exp5_csv": str(directories["exp5"] / "exp5_speecht5_tts.csv"),
            "exp6_csv": str(directories["exp6"] / "exp6_mms_tts.csv"),
            "checkpoint": str(CHECKPOINT_PATH),
        },
    }
    summary_path = directories["root"] / f"group_b_{mode}_summary.json"
    save_text_artifact(summary_path, summary_payload, as_json=True)
    summary_payload["paths"]["summary"] = str(summary_path)
    return summary_payload


def run_group_c_bark_cross(mode: str = "dry_run", output_root: str | Path | None = None) -> dict[str, Any]:
    if mode not in {"dry_run", "full"}:
        raise ValueError(f"Unsupported Group C mode: {mode}")

    shared_assets = ensure_shared_assets(force=False)
    best_asr_evaluator = _load_group_a_best_asr_evaluator()
    evaluator_name = best_asr_evaluator["workflow_name"]

    group_root = _group_c_root_for_mode(mode, output_root)
    directories = _group_c_dirs(group_root)
    best_evaluator_reference_path = directories["root"] / "best_asr_evaluator_reference.json"
    save_text_artifact(best_evaluator_reference_path, best_asr_evaluator, as_json=True)

    baseline_rows: list[dict[str, Any]] = []
    exp7_rows: list[dict[str, Any]] = []
    exp8_rows: list[dict[str, Any]] = []
    exp7_primary_records: list[dict[str, Any]] = []

    prompt_conditions = GROUP_C_PROMPT_STYLES[:1] if mode == "dry_run" else GROUP_C_PROMPT_STYLES
    prompt_seeds = GROUP_C_EXP7_SEEDS[:1] if mode == "dry_run" else GROUP_C_EXP7_SEEDS
    voice_conditions = [GROUP_C_VOICE_CONDITIONS[1]] if mode == "dry_run" else GROUP_C_VOICE_CONDITIONS
    voice_seeds = GROUP_C_EXP7_SEEDS[:1] if mode == "dry_run" else GROUP_C_EXP7_SEEDS
    exp8_plan = [("speecht5", "hello_dog")] if mode == "dry_run" else [
        (system_id, text_tag)
        for system_id in ("speecht5", "mms_tts", "bark_preset")
        for text_tag in GROUP_C_EXP8_TEXT_TAGS
    ]

    try:
        ga20h_baseline_result = runner.run_bark_prompt_condition(
            text=GROUP_C_GA20H_BASELINE_PROMPT,
            output_dir=directories["exp1"] / "ga20h_baseline",
            label="ga20h_baseline",
            save_rate_mode="baseline_hardcoded_16k",
        )
        baseline_rows.append(
            _group_tts_baseline_row(
                ga20h_baseline_result,
                script="GA20H",
                voice_condition="expressive_prompt",
                notes="Baseline-equivalent expressive Bark export preserved at hard-coded 16 kHz.",
            )
        )

        ga20k_baseline_result = runner.run_bark_preset_condition(
            text=GROUP_C_GA20K_BASELINE_TEXT,
            voice_preset=GROUP_C_BARK_PRESET_DEFAULT,
            output_dir=directories["exp1"] / "ga20k_baseline",
            label="ga20k_baseline",
            save_rate_mode="baseline_hardcoded_16k",
        )
        baseline_rows.append(
            _group_tts_baseline_row(
                ga20k_baseline_result,
                script="GA20K",
                voice_condition=GROUP_C_BARK_PRESET_DEFAULT,
                notes="Baseline-equivalent Bark preset export preserved at hard-coded 16 kHz.",
            )
        )

        exp1_csv_path = _write_csv_rows(directories["exp1"] / "exp1_baselines.csv", GROUP_TTS_BASELINE_HEADER, baseline_rows)
        _update_group_checkpoint(
            "group_c_bark_cross",
            mode,
            group_root,
            "exp1_baselines",
            {"rows": len(baseline_rows), "csv": str(exp1_csv_path)},
        )

        exp7_roundtrip_dir = directories["exp7"] / "roundtrip"
        exp7_prompt_dir = directories["exp7"] / "7a_prompt_styles"
        exp7_voice_dir = directories["exp7"] / "7b_voice_conditions"
        exp7_audit_dir = directories["exp7"] / "7c_export_audit"

        for prompt_condition in prompt_conditions:
            for seed in prompt_seeds:
                label = f"ga20h_exp7a_{prompt_condition['prompt_mode']}_seed{seed}"
                result = runner.run_bark_prompt_condition(
                    text=prompt_condition["text"],
                    output_dir=exp7_prompt_dir,
                    label=label,
                    seed=seed,
                    save_rate_mode="native",
                )
                roundtrip = runner.run_roundtrip_asr_eval(
                    evaluator_name=evaluator_name,
                    reference_text=prompt_condition["text"],
                    audio_path=result["output_audio_path"],
                    sampling_rate=int(result["saved_sampling_rate"]),
                    output_dir=exp7_roundtrip_dir,
                    label=f"{label}_{evaluator_name}_roundtrip",
                )
                exp7_rows.append(
                    _group_c_exp7_row(
                        result,
                        sub_experiment="7A",
                        script="GA20H",
                        prompt_mode=prompt_condition["prompt_mode"],
                        roundtrip_result=roundtrip,
                    )
                )
                exp7_primary_records.append(
                    {
                        "sub_experiment": "7A",
                        "script": "GA20H",
                        "prompt_mode": prompt_condition["prompt_mode"],
                        "voice_condition": "",
                        **result,
                    }
                )

        exp7b_text = TEXT_BANK["hello_dog"]
        for voice_condition in voice_conditions:
            for seed in voice_seeds:
                label = f"ga20k_exp7b_{voice_condition['voice_condition']}_seed{seed}"
                result = runner.run_bark_preset_condition(
                    text=exp7b_text,
                    voice_preset=voice_condition["voice_preset"],
                    output_dir=exp7_voice_dir,
                    label=label,
                    seed=seed,
                    save_rate_mode="native",
                )
                roundtrip = runner.run_roundtrip_asr_eval(
                    evaluator_name=evaluator_name,
                    reference_text=exp7b_text,
                    audio_path=result["output_audio_path"],
                    sampling_rate=int(result["saved_sampling_rate"]),
                    output_dir=exp7_roundtrip_dir,
                    label=f"{label}_{evaluator_name}_roundtrip",
                )
                exp7_rows.append(
                    _group_c_exp7_row(
                        result,
                        sub_experiment="7B",
                        script="GA20K",
                        voice_condition=voice_condition["voice_condition"],
                        roundtrip_result=roundtrip,
                    )
                )
                exp7_primary_records.append(
                    {
                        "sub_experiment": "7B",
                        "script": "GA20K",
                        "prompt_mode": "",
                        "voice_condition": voice_condition["voice_condition"],
                        **result,
                    }
                )

        audit_targets = _select_group_c_audit_targets(exp7_primary_records, mode)
        for target in audit_targets:
            audit_label = f"{target['script'].lower()}_exp7c_{target['audit_tag']}_seed{target['seed']}"
            audit_results = runner.run_bark_export_audit(
                text=target["input_text"],
                output_dir=exp7_audit_dir,
                label=audit_label,
                voice_preset=target.get("voice_preset"),
                seed=target.get("seed"),
                do_sample=bool(target.get("do_sample", True)),
                audio_path=target.get("output_audio_path"),
                native_sample_rate=_coerce_int(target.get("native_sampling_rate"), _coerce_int(target.get("saved_sampling_rate"), 24000)),
                generation_time_sec=_coerce_float(target.get("inference_time_sec"), 0.0),
            )
            for audit_result in audit_results:
                roundtrip = runner.run_roundtrip_asr_eval(
                    evaluator_name=evaluator_name,
                    reference_text=target["input_text"],
                    audio_path=audit_result["output_audio_path"],
                    sampling_rate=int(audit_result["saved_sampling_rate"]),
                    output_dir=exp7_roundtrip_dir,
                    label=f"{audit_label}_{audit_result['save_rate_mode']}_{evaluator_name}_roundtrip",
                )
                exp7_rows.append(
                    _group_c_exp7_row(
                        audit_result,
                        sub_experiment="7C",
                        script=target["script"],
                        prompt_mode=target.get("prompt_mode", ""),
                        voice_condition=target.get("voice_condition", ""),
                        roundtrip_result=roundtrip,
                    )
                )

        exp7_csv_path = _write_csv_rows(directories["exp7"] / "exp7_bark.csv", GROUP_C_EXP7_HEADER, exp7_rows)
        _update_group_checkpoint(
            "group_c_bark_cross",
            mode,
            group_root,
            "exp7_bark",
            {"rows": len(exp7_rows), "csv": str(exp7_csv_path)},
        )

        bark_exp8_reuse: dict[str, dict[str, Any]] = {}
        for record in exp7_primary_records:
            if record.get("sub_experiment") != "7B":
                continue
            if record.get("voice_condition") != "preset5":
                continue
            if _coerce_int(record.get("seed"), -1) != GROUP_C_EXP8_BARK_SEED:
                continue
            if record.get("input_text") != TEXT_BANK["hello_dog"]:
                continue
            bark_exp8_reuse["hello_dog"] = _group_c_source_meta_from_result(
                record,
                source_script="GA20K",
                controllability_mode="preset5_seed42",
                reused_from_experiment="exp7_bark",
            )

        exp8_roundtrip_dir = directories["exp8"] / "roundtrip"
        for system_id, text_tag in exp8_plan:
            input_text = TEXT_BANK[text_tag]
            if system_id == "speecht5":
                source_meta = _load_exp5_speecht5_reuse(text_tag)
                if source_meta is None:
                    result = runner.run_speecht5_tts_condition(
                        text=input_text,
                        output_dir=directories["exp8"] / system_id / text_tag,
                        label=f"ga20f_exp8_{text_tag}",
                        speaker_label="helper_embedding",
                    )
                    source_meta = _group_c_source_meta_from_result(
                        result,
                        source_script="GA20F",
                        controllability_mode="helper_embedding",
                    )
            elif system_id == "mms_tts":
                source_meta = _load_exp6_mms_reuse(text_tag)
                if source_meta is None:
                    result = runner.run_mms_tts_condition(
                        text=input_text,
                        output_dir=directories["exp8"] / system_id / text_tag,
                        label=f"ga20g_exp8_{text_tag}_seed{GROUP_C_EXP8_MMS_SEED}",
                        seed=GROUP_C_EXP8_MMS_SEED,
                        speaking_rate=GROUP_C_EXP8_MMS_SPEAKING_RATE,
                    )
                    source_meta = _group_c_source_meta_from_result(
                        result,
                        source_script="GA20G",
                        controllability_mode="seed_555_default_rate",
                    )
            elif system_id == "bark_preset":
                source_meta = bark_exp8_reuse.get(text_tag)
                if source_meta is None:
                    result = runner.run_bark_preset_condition(
                        text=input_text,
                        voice_preset=GROUP_C_BARK_PRESET_DEFAULT,
                        output_dir=directories["exp8"] / system_id / text_tag,
                        label=f"ga20k_exp8_preset5_{text_tag}_seed{GROUP_C_EXP8_BARK_SEED}",
                        seed=GROUP_C_EXP8_BARK_SEED,
                        save_rate_mode="native",
                    )
                    source_meta = _group_c_source_meta_from_result(
                        result,
                        source_script="GA20K",
                        controllability_mode="preset5_seed42",
                    )
                else:
                    source_meta = dict(source_meta)
            else:
                raise ValueError(f"Unsupported Group C Exp 8 system: {system_id}")

            roundtrip = runner.run_roundtrip_asr_eval(
                evaluator_name=evaluator_name,
                reference_text=input_text,
                audio_path=source_meta["audio_path"],
                sampling_rate=int(source_meta["saved_sampling_rate"]),
                output_dir=exp8_roundtrip_dir,
                label=f"{system_id}_exp8_{text_tag}_{evaluator_name}_roundtrip",
            )
            exp8_rows.append(_group_c_exp8_row(source_meta, roundtrip, system_id, text_tag, evaluator_name))

        exp8_csv_path = _write_csv_rows(directories["exp8"] / "exp8_cross_tts.csv", GROUP_C_EXP8_HEADER, exp8_rows)
        _update_group_checkpoint(
            "group_c_bark_cross",
            mode,
            group_root,
            "exp8_cross_tts",
            {"rows": len(exp8_rows), "csv": str(exp8_csv_path), "asr_evaluator": evaluator_name},
        )
    finally:
        runner.clear_model_caches()

    summary_payload = {
        "group_name": "group_c_bark_cross",
        "mode": mode,
        "executed_at": now_iso(),
        "group_root": str(group_root),
        "shared_assets": {
            "fixed_manifest": shared_assets["fixed_manifest"],
            "resampling_manifest": shared_assets["resampling_manifest"],
            "text_bank": shared_assets["text_bank"],
            "long_audio": shared_assets["long_audio"],
        },
        "counts": {
            "baseline_rows": len(baseline_rows),
            "exp7_rows": len(exp7_rows),
            "exp8_rows": len(exp8_rows),
        },
        "paths": {
            "exp1_csv": str(directories["exp1"] / "exp1_baselines.csv"),
            "exp7_csv": str(directories["exp7"] / "exp7_bark.csv"),
            "exp8_csv": str(directories["exp8"] / "exp8_cross_tts.csv"),
            "best_asr_evaluator_reference": str(best_evaluator_reference_path),
            "checkpoint": str(CHECKPOINT_PATH),
        },
        "best_asr_evaluator": best_asr_evaluator,
    }
    summary_path = directories["root"] / f"group_c_{mode}_summary.json"
    save_text_artifact(summary_path, summary_payload, as_json=True)
    summary_payload["paths"]["summary"] = str(summary_path)
    return summary_payload


def main(group: str | None = None, mode: str = "dry_run") -> dict[str, Any]:
    if group is None:
        return {
            "group_a_asr": run_group_a_asr(mode=mode),
            "group_b_structured_tts": run_group_b_structured_tts(mode=mode),
            "group_c_bark_cross": run_group_c_bark_cross(mode=mode),
        }
    if group == "group_a_asr":
        return run_group_a_asr(mode=mode)
    if group == "group_b_structured_tts":
        return run_group_b_structured_tts(mode=mode)
    if group == "group_c_bark_cross":
        return run_group_c_bark_cross(mode=mode)
    raise ValueError(f"Unsupported group: {group}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare or inspect HW14 experiment group execution plans.")
    parser.add_argument("--group", choices=["group_a_asr", "group_b_structured_tts", "group_c_bark_cross"], default=None)
    parser.add_argument("--mode", choices=["dry_run", "full"], default="dry_run")
    args = parser.parse_args()
    print(json.dumps(main(group=args.group, mode=args.mode), indent=2))


__all__ = [
    "ASR_MODELS",
    "TTS_MODELS",
    "TEXT_BANK",
    "DEFAULT_SEEDS",
    "build_fixed_minds_subset",
    "cache_long_audio_asset",
    "ensure_shared_assets",
    "run_group_a_asr",
    "run_group_b_structured_tts",
    "run_group_c_bark_cross",
    "main",
]