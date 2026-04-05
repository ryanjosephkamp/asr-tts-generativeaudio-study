from __future__ import annotations

from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from hw14_analysis_utils import compute_text_similarity, compute_wer_cer, normalize_text_for_metrics
from hw14_data_utils import cleanup_model, get_audio_stats, now_iso, save_audio, save_spectrogram_figure, save_text_artifact, timer

MINDS_DATASET_NAME = "PolyAI/minds14"
MINDS_DATASET_CONFIG = "en-AU"
MINDS_SPLIT = "train"
DEFAULT_RESAMPLED_RATE = 16_000

ASR_MODEL_IDS = {
    "ga20a_whisper_pipeline": "openai/whisper-small",
    "ga20b_wav2vec2_ctc": "facebook/wav2vec2-base-960h",
    "ga20c_whisper_direct": "openai/whisper-small",
    "ga20d_whisper_longform": "openai/whisper-small",
    "ga20e_speecht5_asr": "microsoft/speecht5_asr",
}

TTS_MODEL_IDS = {
    "ga20f_speecht5_tts": "microsoft/speecht5_tts",
    "ga20f_speecht5_vocoder": "microsoft/speecht5_hifigan",
    "ga20g_mms_tts": "facebook/mms-tts-eng",
    "ga20h_bark": "suno/bark-small",
    "ga20k_bark": "suno/bark-small",
}


@lru_cache(maxsize=1)
def _get_ga20a_pipeline() -> Any:
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_IDS["ga20a_whisper_pipeline"],
        max_new_tokens=200,
    )


@lru_cache(maxsize=1)
def _get_ga20b_components() -> tuple[Any, Any]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_IDS["ga20b_wav2vec2_ctc"])
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_IDS["ga20b_wav2vec2_ctc"])
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _get_ga20c_components() -> tuple[Any, Any]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ASR_MODEL_IDS["ga20c_whisper_direct"])
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_IDS["ga20c_whisper_direct"])
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _get_ga20d_pipeline() -> Any:
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_IDS["ga20d_whisper_longform"],
    )


@lru_cache(maxsize=1)
def _get_ga20e_components() -> tuple[Any, Any]:
    from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

    processor = SpeechT5Processor.from_pretrained(ASR_MODEL_IDS["ga20e_speecht5_asr"])
    model = SpeechT5ForSpeechToText.from_pretrained(ASR_MODEL_IDS["ga20e_speecht5_asr"])
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _get_ga20f_components() -> tuple[Any, Any, Any]:
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    processor = SpeechT5Processor.from_pretrained(TTS_MODEL_IDS["ga20f_speecht5_tts"])
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL_IDS["ga20f_speecht5_tts"])
    vocoder = SpeechT5HifiGan.from_pretrained(TTS_MODEL_IDS["ga20f_speecht5_vocoder"])
    model.eval()
    vocoder.eval()
    return processor, model, vocoder


@lru_cache(maxsize=1)
def _get_ga20g_components() -> tuple[Any, Any]:
    from transformers import VitsModel, VitsTokenizer

    tokenizer = VitsTokenizer.from_pretrained(TTS_MODEL_IDS["ga20g_mms_tts"])
    model = VitsModel.from_pretrained(TTS_MODEL_IDS["ga20g_mms_tts"])
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=2)
def _get_bark_components(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return processor, model


def clear_model_caches() -> dict[str, Any]:
    _get_ga20a_pipeline.cache_clear()
    _get_ga20b_components.cache_clear()
    _get_ga20c_components.cache_clear()
    _get_ga20d_pipeline.cache_clear()
    _get_ga20e_components.cache_clear()
    _get_ga20f_components.cache_clear()
    _get_ga20g_components.cache_clear()
    _get_bark_components.cache_clear()
    cleanup = cleanup_model()
    cleanup["cleared_at"] = now_iso()
    return cleanup


def _as_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "artifact"


def _to_model_audio(audio_array: Any) -> np.ndarray:
    array = np.asarray(audio_array)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = np.asarray([array.item()])
    if array.dtype.kind in {"i", "u"}:
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / float(scale or 1)
    else:
        array = array.astype(np.float32)
    return array


def _save_json_record(output_dir: Path | None, label: str, record: Mapping[str, Any]) -> str | None:
    if output_dir is None:
        return None
    artifact_path = output_dir / f"{_slugify(label)}.json"
    save_text_artifact(artifact_path, dict(record), as_json=True)
    return str(artifact_path)


def _get_model_sample_rate(model: Any, fallback: int = DEFAULT_RESAMPLED_RATE) -> int:
    for holder_name in ("generation_config", "config"):
        holder = getattr(model, holder_name, None)
        if holder is not None and getattr(holder, "sample_rate", None):
            return int(holder.sample_rate)
        if holder is not None and getattr(holder, "sampling_rate", None):
            return int(holder.sampling_rate)
    return int(fallback)


def _extract_pipeline_text(output: Any) -> str:
    if isinstance(output, Mapping):
        return str(output.get("text", ""))
    if isinstance(output, Sequence) and output and not isinstance(output, (str, bytes, bytearray)):
        return str(output[0])
    return str(output)


def _resample_audio(audio_array: Any, original_rate: int, target_rate: int) -> np.ndarray:
    array = _to_model_audio(audio_array)
    if int(original_rate) == int(target_rate):
        return array
    from scipy.signal import resample_poly

    divisor = gcd(int(original_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(original_rate) // divisor
    return resample_poly(array, up, down).astype(np.float32)


def _load_wav(audio_path: str | Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sampling_rate, array = wavfile.read(audio_path)
    return _to_model_audio(array), int(sampling_rate)


@lru_cache(maxsize=4)
def _load_minds_dataset_cached(sampling_rate: int | None) -> Any:
    from datasets import Audio, load_dataset

    dataset = load_dataset(MINDS_DATASET_NAME, name=MINDS_DATASET_CONFIG, split=MINDS_SPLIT)
    dataset = dataset.add_column("dataset_index", list(range(len(dataset))))
    if sampling_rate is not None:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=int(sampling_rate)))
    return dataset


def _load_random_minds_example(sampling_rate: int | None = DEFAULT_RESAMPLED_RATE) -> Mapping[str, Any]:
    dataset = _load_minds_dataset_cached(None if sampling_rate is None else int(sampling_rate))
    return dataset.shuffle()[0]


def _load_minds_example(sample_index: int, sampling_rate: int | None = DEFAULT_RESAMPLED_RATE) -> Mapping[str, Any]:
    dataset = _load_minds_dataset_cached(None if sampling_rate is None else int(sampling_rate))
    return dataset[int(sample_index)]


def _generate_long_audio_array(target_seconds: int = 60, target_sample_rate: int = DEFAULT_RESAMPLED_RATE) -> np.ndarray:
    from datasets import load_dataset

    try:
        stream = load_dataset(
            "librispeech_asr",
            split="train.clean.360",
            streaming=True,
            trust_remote_code=True,
        )
    except TypeError:
        stream = load_dataset("librispeech_asr", split="train.clean.360", streaming=True)

    target_length = int(target_seconds) * int(target_sample_rate)
    chunks: list[np.ndarray] = []
    current_length = 0
    for sample in stream:
        chunk = _to_model_audio(sample["audio"]["array"])
        chunks.append(chunk)
        current_length += len(chunk)
        if current_length >= target_length:
            break

    if not chunks:
        raise RuntimeError("Could not build long audio cache from LibriSpeech streaming samples.")

    return np.concatenate(chunks)[:target_length].astype(np.float32)


@lru_cache(maxsize=1)
def _get_cmu_arctic_xvector_manifest() -> tuple[Path, tuple[str, ...]]:
    from huggingface_hub import hf_hub_download
    import zipfile

    zip_path = Path(
        hf_hub_download(
            repo_id="Matthijs/cmu-arctic-xvectors",
            filename="spkrec-xvect.zip",
            repo_type="dataset",
        )
    )
    with zipfile.ZipFile(zip_path) as archive:
        entries = tuple(sorted(name for name in archive.namelist() if name.endswith(".npy")))
    if not entries:
        raise RuntimeError("No CMU Arctic xvector files were found in the downloaded archive.")
    return zip_path, entries


def load_reference_speaker_embedding(embedding_id: int = 7440) -> np.ndarray:
    try:
        from io import BytesIO
        import zipfile

        zip_path, entries = _get_cmu_arctic_xvector_manifest()
        index = int(embedding_id)
        if index < 0 or index >= len(entries):
            raise IndexError(f"embedding_id {index} is out of range for {len(entries)} CMU Arctic xvectors")
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(entries[index]) as handle:
                return np.asarray(np.load(BytesIO(handle.read())), dtype=np.float32)
    except Exception:
        try:
            from datasets import load_dataset

            try:
                dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors",
                    split="validation",
                    trust_remote_code=True,
                )
            except TypeError:
                dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            return np.asarray(dataset[int(embedding_id)]["xvector"], dtype=np.float32)
        except Exception:
            try:
                from genaibook.core import get_speaker_embeddings

                return np.asarray(get_speaker_embeddings(int(embedding_id)), dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the CMU Arctic speaker embedding required for SpeechT5 experiments."
                ) from exc


def _base_record(label: str, model_name: str, sampling_rate: int | None = None) -> dict[str, Any]:
    record = {
        "label": label,
        "model_name": model_name,
        "timestamp": now_iso(),
    }
    if sampling_rate is not None:
        record["input_sampling_rate"] = int(sampling_rate)
    return record


def run_shortform_asr_condition(
    workflow_name: str,
    audio_array: Any,
    sampling_rate: int,
    declared_sampling_rate: int | None = None,
    reference_text: str | None = None,
    sample_index: int | None = None,
    decode_policy: str = "clean",
    output_dir: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    output_root = _as_path(output_dir)
    audio = _to_model_audio(audio_array)
    workflow_key = workflow_name.lower()
    artifact_label = label or f"{workflow_key}_{sample_index if sample_index is not None else 'sample'}"
    model_input_sampling_rate = int(declared_sampling_rate or sampling_rate)

    if workflow_key in {"ga20a", "ga20a_whisper_pipeline", "ga20a_whisper_pipeline_asr"}:
        pipeline_runner = _get_ga20a_pipeline()
        with timer(artifact_label) as timing:
            pipeline_output = pipeline_runner(audio)
        predicted_text = _extract_pipeline_text(pipeline_output)
        model_name = ASR_MODEL_IDS["ga20a_whisper_pipeline"]
    elif workflow_key in {"ga20b", "ga20b_wav2vec2_ctc"}:
        import torch

        processor, model = _get_ga20b_components()
        inputs = processor(audio, sampling_rate=model_input_sampling_rate, return_tensors="pt")
        with timer(artifact_label) as timing:
            with torch.inference_mode():
                outputs = model(**inputs)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            decoded = processor.batch_decode(predicted_ids)
        predicted_text = str(decoded[0])
        model_name = ASR_MODEL_IDS["ga20b_wav2vec2_ctc"]
    elif workflow_key in {"ga20c", "ga20c_whisper_direct"}:
        import torch

        processor, model = _get_ga20c_components()
        inputs = processor(audio, sampling_rate=model_input_sampling_rate, return_tensors="pt")
        skip_special_tokens = decode_policy.lower() != "raw"
        with timer(artifact_label) as timing:
            with torch.inference_mode():
                generated_ids = model.generate(**inputs)
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
        predicted_text = str(decoded[0])
        model_name = ASR_MODEL_IDS["ga20c_whisper_direct"]
    elif workflow_key in {"ga20e", "ga20e_speecht5_asr"}:
        import torch

        processor, model = _get_ga20e_components()
        inputs = processor(audio=audio, sampling_rate=model_input_sampling_rate, return_tensors="pt")
        with timer(artifact_label) as timing:
            with torch.inference_mode():
                predicted_ids = model.generate(**inputs, max_new_tokens=70)
            decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predicted_text = str(decoded[0])
        model_name = ASR_MODEL_IDS["ga20e_speecht5_asr"]
    else:
        raise ValueError(f"Unsupported short-form ASR workflow: {workflow_name}")

    record = _base_record(artifact_label, model_name, sampling_rate)
    record.update(
        {
            "workflow_name": workflow_name,
            "sample_index": sample_index,
            "decode_policy": decode_policy,
            "predicted_text_raw": predicted_text,
            "predicted_text_normalized": normalize_text_for_metrics(predicted_text),
            "reference_text_raw": reference_text,
            "reference_text_normalized": normalize_text_for_metrics(reference_text) if reference_text is not None else None,
            "inference_time_sec": float(timing["elapsed_seconds"]),
        }
    )
    if model_input_sampling_rate != int(sampling_rate):
        record["model_input_sampling_rate"] = model_input_sampling_rate
    if reference_text is not None:
        record.update(compute_wer_cer(reference_text, predicted_text))

    artifact_path = _save_json_record(output_root, artifact_label, record)
    if artifact_path:
        record["output_text_path"] = artifact_path
    return record


def run_ga20a_baseline(sample_index: int | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    example = _load_minds_example(sample_index) if sample_index is not None else _load_random_minds_example()
    result = run_shortform_asr_condition(
        workflow_name="ga20a_whisper_pipeline",
        audio_array=example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        reference_text=example["english_transcription"],
        sample_index=int(example.get("dataset_index", sample_index)) if example.get("dataset_index") is not None else sample_index,
        decode_policy="clean",
        output_dir=output_dir,
        label="ga20a_baseline",
    )
    result.update({"script": "GA20A", "baseline_mode": True})
    return result


def run_ga20b_baseline(sample_index: int | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    example = _load_minds_example(sample_index) if sample_index is not None else _load_random_minds_example()
    result = run_shortform_asr_condition(
        workflow_name="ga20b_wav2vec2_ctc",
        audio_array=example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        reference_text=example["english_transcription"],
        sample_index=int(example.get("dataset_index", sample_index)) if example.get("dataset_index") is not None else sample_index,
        decode_policy="clean",
        output_dir=output_dir,
        label="ga20b_baseline",
    )
    result.update({"script": "GA20B", "baseline_mode": True})
    return result


def run_ga20c_baseline(sample_index: int | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    example = _load_minds_example(sample_index) if sample_index is not None else _load_random_minds_example()
    result = run_shortform_asr_condition(
        workflow_name="ga20c_whisper_direct",
        audio_array=example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        reference_text=example["english_transcription"],
        sample_index=int(example.get("dataset_index", sample_index)) if example.get("dataset_index") is not None else sample_index,
        decode_policy="raw",
        output_dir=output_dir,
        label="ga20c_baseline",
    )
    result.update({"script": "GA20C", "baseline_mode": True})
    return result


def run_ga20e_baseline(sample_index: int | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    example = _load_minds_example(sample_index) if sample_index is not None else _load_random_minds_example()
    result = run_shortform_asr_condition(
        workflow_name="ga20e_speecht5_asr",
        audio_array=example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
        reference_text=example["english_transcription"],
        sample_index=int(example.get("dataset_index", sample_index)) if example.get("dataset_index") is not None else sample_index,
        decode_policy="clean",
        output_dir=output_dir,
        label="ga20e_baseline",
    )
    result.update({"script": "GA20E", "baseline_mode": True})
    return result


def run_asr_resampling_condition(
    workflow_name: str,
    audio_array: Any,
    original_sampling_rate: int,
    target_sampling_rate: int,
    reference_text: str | None = None,
    sample_index: int | None = None,
    output_dir: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    resampled_audio = _resample_audio(audio_array, int(original_sampling_rate), int(target_sampling_rate))
    workflow_key = workflow_name.lower()
    model_input_sampling_rate = int(target_sampling_rate)
    if workflow_key in {"ga20b", "ga20b_wav2vec2_ctc", "ga20c", "ga20c_whisper_direct", "ga20e", "ga20e_speecht5_asr"}:
        # Exp 3A intentionally varies the waveform rate while processor-based models still
        # require a declared 16 kHz rate. This preserves the mismatch condition instead of
        # silently resampling back to the model's preferred rate.
        model_input_sampling_rate = DEFAULT_RESAMPLED_RATE
    record = run_shortform_asr_condition(
        workflow_name=workflow_name,
        audio_array=resampled_audio,
        sampling_rate=int(target_sampling_rate),
        declared_sampling_rate=model_input_sampling_rate,
        reference_text=reference_text,
        sample_index=sample_index,
        decode_policy="clean",
        output_dir=output_dir,
        label=label or f"{workflow_name}_{target_sampling_rate}hz",
    )
    record.update(
        {
            "original_sampling_rate": int(original_sampling_rate),
            "resampling_rate": int(target_sampling_rate),
            "model_input_sampling_rate": int(model_input_sampling_rate),
            "sampling_rate_mode": "mismatched" if int(model_input_sampling_rate) != int(target_sampling_rate) else "matched",
            "sub_experiment": "3A",
        }
    )
    return record


def run_ga20c_decode_comparison(
    audio_array: Any,
    sampling_rate: int,
    reference_text: str | None = None,
    sample_index: int | None = None,
    output_dir: str | Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    raw_result = run_shortform_asr_condition(
        workflow_name="ga20c_whisper_direct",
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        reference_text=reference_text,
        sample_index=sample_index,
        decode_policy="raw",
        output_dir=None,
        label=f"{label or 'ga20c'}_raw",
    )
    clean_result = run_shortform_asr_condition(
        workflow_name="ga20c_whisper_direct",
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        reference_text=reference_text,
        sample_index=sample_index,
        decode_policy="clean",
        output_dir=None,
        label=f"{label or 'ga20c'}_clean",
    )
    comparison = {
        "label": label or "ga20c_decode_comparison",
        "workflow_name": "ga20c_whisper_direct",
        "sample_index": sample_index,
        "input_sampling_rate": int(sampling_rate),
        "predicted_text_raw": raw_result["predicted_text_raw"],
        "predicted_text_clean": clean_result["predicted_text_raw"],
        "predicted_text_normalized": clean_result["predicted_text_normalized"],
        "reference_text_normalized": clean_result.get("reference_text_normalized"),
        "raw_wer": raw_result.get("wer"),
        "raw_cer": raw_result.get("cer"),
        "clean_wer": clean_result.get("wer"),
        "clean_cer": clean_result.get("cer"),
        "sub_experiment": "3B",
        "timestamp": now_iso(),
    }
    artifact_path = _save_json_record(_as_path(output_dir), comparison["label"], comparison)
    if artifact_path:
        comparison["output_text_path"] = artifact_path
    return comparison


def run_longform_whisper_condition(
    audio_array: Any,
    chunk_length_s: int,
    batch_size: int,
    output_dir: str | Path | None = None,
    label: str = "ga20d_longform",
    baseline_text: str | None = None,
    task: str = "transcribe",
    return_timestamps: bool = True,
) -> dict[str, Any]:
    output_root = _as_path(output_dir)
    pipeline_runner = _get_ga20d_pipeline()
    with timer(label) as timing:
        output = pipeline_runner(
            _to_model_audio(audio_array),
            generate_kwargs={"task": task},
            chunk_length_s=int(chunk_length_s),
            batch_size=int(batch_size),
            return_timestamps=bool(return_timestamps),
        )

    transcript_text = _extract_pipeline_text(output)
    chunks = output.get("chunks", []) if isinstance(output, Mapping) else []
    durations = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp")
        if isinstance(timestamp, Sequence) and len(timestamp) == 2 and None not in timestamp:
            durations.append(float(timestamp[1]) - float(timestamp[0]))

    record = _base_record(label, ASR_MODEL_IDS["ga20d_whisper_longform"], DEFAULT_RESAMPLED_RATE)
    record.update(
        {
            "chunk_length_s": int(chunk_length_s),
            "batch_size": int(batch_size),
            "return_timestamps": bool(return_timestamps),
            "predicted_text_raw": transcript_text,
            "predicted_text_normalized": normalize_text_for_metrics(transcript_text),
            "segment_count": len(chunks),
            "mean_segment_duration_sec": float(np.mean(durations)) if durations else None,
            "transcript_length_chars": len(transcript_text),
            "inference_time_sec": float(timing["elapsed_seconds"]),
        }
    )
    if baseline_text is not None:
        record["similarity_to_baseline"] = compute_text_similarity(baseline_text, transcript_text)

    if output_root is not None:
        json_path = output_root / f"{_slugify(label)}.json"
        txt_path = output_root / f"{_slugify(label)}.txt"
        save_text_artifact(json_path, output, as_json=True)
        save_text_artifact(txt_path, transcript_text, as_json=False)
        record["output_json_path"] = str(json_path)
        record["output_text_path"] = str(txt_path)
    return record


def run_ga20d_corrected_baseline(
    output_dir: str | Path | None = None,
    audio_array: Any | None = None,
) -> dict[str, Any]:
    long_audio = _to_model_audio(audio_array) if audio_array is not None else _generate_long_audio_array()
    result = run_longform_whisper_condition(
        audio_array=long_audio,
        chunk_length_s=5,
        batch_size=8,
        output_dir=output_dir,
        label="ga20d_corrected_baseline",
    )
    result.update(
        {
            "script": "GA20D",
            "baseline_mode": True,
            "notes": "Corrected baseline-equivalent wrapper used because GA20D.py omits the pipeline import.",
        }
    )
    return result


def run_speecht5_tts_condition(
    text: str,
    output_dir: str | Path | None = None,
    label: str = "speecht5_tts",
    speaker_embedding: Any | None = None,
    speaker_label: str = "reference_embedding",
    seed: int | None = None,
) -> dict[str, Any]:
    import torch

    output_root = _as_path(output_dir)
    processor, model, vocoder = _get_ga20f_components()

    if seed is not None:
        torch.manual_seed(int(seed))

    if speaker_embedding is None:
        speaker_embedding = load_reference_speaker_embedding()
    speaker_tensor = torch.tensor(np.asarray(speaker_embedding), dtype=torch.float32).unsqueeze(0)
    inputs = processor(text=text, return_tensors="pt")

    with timer(label) as timing:
        with torch.inference_mode():
            spectrogram = model.generate_speech(inputs["input_ids"], speaker_tensor)
            speech = vocoder(spectrogram)

    sample_rate = _get_model_sample_rate(vocoder, fallback=DEFAULT_RESAMPLED_RATE)
    record = _base_record(label, TTS_MODEL_IDS["ga20f_speecht5_tts"], sample_rate)
    record.update(
        {
            "input_text": text,
            "speaker_condition": speaker_label,
            "seed": seed,
            "inference_time_sec": float(timing["elapsed_seconds"]),
            "saved_sampling_rate": sample_rate,
        }
    )

    if output_root is not None:
        spectrogram_array = np.asarray(spectrogram)
        spectrogram_array = np.squeeze(spectrogram_array)
        if spectrogram_array.ndim == 1:
            spectrogram_array = np.expand_dims(spectrogram_array, axis=0)
        spectrogram_path = output_root / f"{_slugify(label)}_spectrogram.png"
        waveform_path = output_root / f"{_slugify(label)}.wav"
        save_spectrogram_figure(spectrogram_array, spectrogram_path, title=label)
        audio_stats = save_audio(speech.numpy(), sample_rate, waveform_path, label=label)
        record.update(
            {
                "output_figure_path": str(spectrogram_path),
                "output_audio_path": str(waveform_path),
                "spectrogram_frames": int(spectrogram_array.shape[-1]),
                "spectrogram_bins": int(spectrogram_array.shape[0]),
            }
        )
        record.update(audio_stats)
        _save_json_record(output_root, label, record)
    return record


def run_mms_tts_condition(
    text: str,
    output_dir: str | Path | None = None,
    label: str = "mms_tts",
    seed: int | None = None,
    speaking_rate: float | None = None,
    save_rate_override: int | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    output_root = _as_path(output_dir)
    tokenizer, model = _get_ga20g_components()
    if seed is not None:
        set_seed(int(seed))

    original_speaking_rate = model.speaking_rate
    if speaking_rate is not None:
        model.speaking_rate = float(speaking_rate)
    inputs = tokenizer(text=text, return_tensors="pt")

    with timer(label) as timing:
        with torch.inference_mode():
            outputs = model(inputs["input_ids"])

    sample_rate = int(save_rate_override or _get_model_sample_rate(model, fallback=DEFAULT_RESAMPLED_RATE))
    record = _base_record(label, TTS_MODEL_IDS["ga20g_mms_tts"], sample_rate)
    record.update(
        {
            "input_text": text,
            "seed": seed,
            "speaking_rate": float(model.speaking_rate),
            "baseline_speaking_rate": float(original_speaking_rate),
            "saved_sampling_rate": sample_rate,
            "inference_time_sec": float(timing["elapsed_seconds"]),
        }
    )

    if output_root is not None:
        waveform_path = output_root / f"{_slugify(label)}.wav"
        audio_stats = save_audio(outputs.waveform[0].numpy(), sample_rate, waveform_path, label=label)
        record["output_audio_path"] = str(waveform_path)
        record.update(audio_stats)
        _save_json_record(output_root, label, record)
    model.speaking_rate = original_speaking_rate
    return record


def run_bark_prompt_condition(
    text: str,
    output_dir: str | Path | None = None,
    label: str = "bark_prompt",
    seed: int | None = None,
    do_sample: bool = True,
    save_rate_mode: str = "native",
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    output_root = _as_path(output_dir)
    if seed is not None:
        set_seed(int(seed))

    processor, model = _get_bark_components(TTS_MODEL_IDS["ga20h_bark"])
    inputs = processor(text=[text], return_tensors="pt")

    with timer(label) as timing:
        with torch.inference_mode():
            speech_values = model.generate(**inputs, do_sample=do_sample)

    native_sample_rate = _get_model_sample_rate(model, fallback=24_000)
    saved_sample_rate = 16_000 if save_rate_mode == "baseline_hardcoded_16k" else native_sample_rate
    record = _base_record(label, TTS_MODEL_IDS["ga20h_bark"], saved_sample_rate)
    record.update(
        {
            "input_text": text,
            "seed": seed,
            "do_sample": bool(do_sample),
            "native_sampling_rate": native_sample_rate,
            "saved_sampling_rate": saved_sample_rate,
            "save_rate_mode": save_rate_mode,
            "inference_time_sec": float(timing["elapsed_seconds"]),
        }
    )

    if output_root is not None:
        waveform_path = output_root / f"{_slugify(label)}.wav"
        audio_stats = save_audio(speech_values.numpy().squeeze(), saved_sample_rate, waveform_path, label=label)
        record["output_audio_path"] = str(waveform_path)
        record.update(audio_stats)
        _save_json_record(output_root, label, record)
    return record


def run_bark_preset_condition(
    text: str,
    voice_preset: str | None,
    output_dir: str | Path | None = None,
    label: str = "bark_preset",
    seed: int | None = None,
    do_sample: bool = True,
    save_rate_mode: str = "native",
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    output_root = _as_path(output_dir)
    if seed is not None:
        set_seed(int(seed))

    processor, model = _get_bark_components(TTS_MODEL_IDS["ga20k_bark"])
    if voice_preset is None:
        inputs = processor(text=[text], return_tensors="pt")
    else:
        inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")

    with timer(label) as timing:
        with torch.inference_mode():
            speech_values = model.generate(**inputs, do_sample=do_sample)

    native_sample_rate = _get_model_sample_rate(model, fallback=24_000)
    saved_sample_rate = 16_000 if save_rate_mode == "baseline_hardcoded_16k" else native_sample_rate
    record = _base_record(label, TTS_MODEL_IDS["ga20k_bark"], saved_sample_rate)
    record.update(
        {
            "input_text": text,
            "voice_preset": voice_preset,
            "seed": seed,
            "do_sample": bool(do_sample),
            "native_sampling_rate": native_sample_rate,
            "saved_sampling_rate": saved_sample_rate,
            "save_rate_mode": save_rate_mode,
            "inference_time_sec": float(timing["elapsed_seconds"]),
        }
    )

    if output_root is not None:
        waveform_path = output_root / f"{_slugify(label)}.wav"
        audio_stats = save_audio(speech_values.numpy().squeeze(), saved_sample_rate, waveform_path, label=label)
        record["output_audio_path"] = str(waveform_path)
        record.update(audio_stats)
        _save_json_record(output_root, label, record)
    return record


def run_bark_export_audit(
    text: str,
    output_dir: str | Path | None = None,
    label: str = "bark_export_audit",
    voice_preset: str | None = None,
    seed: int | None = None,
    do_sample: bool = True,
    save_rates: Sequence[int] | None = None,
    audio_path: str | Path | None = None,
    audio_array: Any | None = None,
    native_sample_rate: int | None = None,
    generation_time_sec: float | None = None,
) -> list[dict[str, Any]]:
    import torch
    from transformers import set_seed

    output_root = _as_path(output_dir)
    waveform: np.ndarray
    native_rate: int
    elapsed_seconds = float(generation_time_sec or 0.0)

    if audio_array is not None:
        waveform = _to_model_audio(audio_array)
        native_rate = int(native_sample_rate or DEFAULT_RESAMPLED_RATE)
    elif audio_path is not None:
        waveform, inferred_rate = _load_wav(audio_path)
        native_rate = int(native_sample_rate or inferred_rate)
    else:
        if seed is not None:
            set_seed(int(seed))

        processor, model = _get_bark_components(TTS_MODEL_IDS["ga20k_bark"])
        if voice_preset is None:
            inputs = processor(text=[text], return_tensors="pt")
        else:
            inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")

        with timer(label) as timing:
            with torch.inference_mode():
                speech_values = model.generate(**inputs, do_sample=do_sample)

        waveform = speech_values.numpy().squeeze()
        native_rate = _get_model_sample_rate(model, fallback=24_000)
        elapsed_seconds = float(timing["elapsed_seconds"])

    audit_rates = list(save_rates) if save_rates is not None else [16_000, native_rate]
    records: list[dict[str, Any]] = []
    for rate in audit_rates:
        record = _base_record(f"{label}_{rate}hz", TTS_MODEL_IDS["ga20k_bark"], int(rate))
        record.update(
            {
                "input_text": text,
                "voice_preset": voice_preset,
                "seed": seed,
                "do_sample": bool(do_sample),
                "native_sampling_rate": native_rate,
                "saved_sampling_rate": int(rate),
                "save_rate_mode": "hardcoded_16k" if int(rate) == 16_000 else "model_native",
                "inference_time_sec": elapsed_seconds,
                "audio_source_path": str(audio_path) if audio_path is not None else None,
            }
        )
        if output_root is not None:
            waveform_path = output_root / f"{_slugify(label)}_{int(rate)}hz.wav"
            audio_stats = save_audio(waveform, int(rate), waveform_path, label=f"{label}_{rate}hz")
            record["output_audio_path"] = str(waveform_path)
            record.update(audio_stats)
            _save_json_record(output_root, f"{label}_{rate}hz", record)
        records.append(record)
    return records


def run_roundtrip_asr_eval(
    evaluator_name: str,
    reference_text: str,
    audio_path: str | Path | None = None,
    audio_array: Any | None = None,
    sampling_rate: int | None = None,
    output_dir: str | Path | None = None,
    label: str = "roundtrip_eval",
) -> dict[str, Any]:
    source_sampling_rate: int | None = None
    if audio_array is None:
        if audio_path is None:
            raise ValueError("Either audio_path or audio_array must be provided for round-trip ASR evaluation.")
        audio_array, inferred_rate = _load_wav(audio_path)
        sampling_rate = inferred_rate if sampling_rate is None else sampling_rate
    elif sampling_rate is None:
        raise ValueError("sampling_rate is required when audio_array is provided directly.")

    source_sampling_rate = int(sampling_rate)
    target_sampling_rate = DEFAULT_RESAMPLED_RATE
    if source_sampling_rate != target_sampling_rate:
        audio_array = _resample_audio(audio_array, source_sampling_rate, target_sampling_rate)
        sampling_rate = target_sampling_rate

    record = run_shortform_asr_condition(
        workflow_name=evaluator_name,
        audio_array=audio_array,
        sampling_rate=int(sampling_rate),
        reference_text=reference_text,
        output_dir=output_dir,
        label=label,
    )
    record.update(
        {
            "roundtrip_evaluator": evaluator_name,
            "audio_source_path": str(audio_path) if audio_path is not None else None,
            "audio_source_sampling_rate": source_sampling_rate,
        }
    )
    return record


__all__ = [
    "clear_model_caches",
    "run_ga20a_baseline",
    "run_ga20b_baseline",
    "run_ga20c_baseline",
    "run_ga20d_corrected_baseline",
    "run_ga20e_baseline",
    "run_shortform_asr_condition",
    "run_asr_resampling_condition",
    "run_ga20c_decode_comparison",
    "run_longform_whisper_condition",
    "run_speecht5_tts_condition",
    "run_mms_tts_condition",
    "run_bark_prompt_condition",
    "run_bark_preset_condition",
    "run_bark_export_audit",
    "run_roundtrip_asr_eval",
    "load_reference_speaker_embedding",
]