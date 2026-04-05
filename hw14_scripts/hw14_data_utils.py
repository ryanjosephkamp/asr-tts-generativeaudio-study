from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import csv
import gc
import json
import sys
import time
import zipfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hw14_experiments"
PRINTOUTS_DIR = PROJECT_ROOT / "hw14_printouts"
CHECKPOINT_PATH = EXPERIMENTS_DIR / "checkpoint.json"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def timer(label: str = "") -> Iterator[dict[str, Any]]:
    started_at = now_iso()
    start_time = time.perf_counter()
    timing = {"label": label, "started_at": started_at}
    try:
        yield timing
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        timing["finished_at"] = now_iso()
        timing["elapsed_seconds"] = elapsed_seconds
        if label:
            print(f"[timer] {label}: {elapsed_seconds:.3f}s")


def _as_path(filepath: str | Path) -> Path:
    return Path(filepath)


def _ensure_parent(filepath: str | Path) -> Path:
    path = _as_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_header(filepath: str | Path) -> list[str]:
    path = _as_path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def init_csv(filepath: str | Path, header_row: Sequence[Any]) -> Path:
    path = _ensure_parent(filepath)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(header_row))
    return path


def append_csv_row(filepath: str | Path, row_data: Sequence[Any] | Mapping[str, Any]) -> Path:
    path = _ensure_parent(filepath)
    if isinstance(row_data, Mapping):
        header = _read_csv_header(path)
        if not header:
            header = list(row_data.keys())
            init_csv(path, header)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writerow({column: row_data.get(column, "") for column in header})
        return path

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist yet: {path}. Call init_csv() before appending sequence rows."
        )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(row_data))
    return path


def count_csv_rows(filepath: str | Path) -> int:
    path = _as_path(filepath)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.reader(handle))
    return max(row_count - 1, 0)


def _to_audio_array(audio_array: Any) -> np.ndarray:
    array = np.asarray(audio_array)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = np.asarray([array.item()])
    return array


def _prepare_wav_array(audio_array: Any) -> np.ndarray:
    array = _to_audio_array(audio_array)
    if array.dtype.kind == "f":
        clipped = np.clip(array.astype(np.float32), -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
    if array.dtype.kind in {"i", "u"}:
        clipped = np.clip(array.astype(np.int64), -32768, 32767)
        return clipped.astype(np.int16)
    raise TypeError(f"Unsupported audio dtype for WAV export: {array.dtype}")


def get_audio_stats(audio_array: Any, sampling_rate: int) -> dict[str, float | int]:
    array = _to_audio_array(audio_array)
    if array.size == 0:
        return {
            "sample_rate": int(sampling_rate),
            "sample_count": 0,
            "duration_seconds": 0.0,
            "rms": 0.0,
            "max_amplitude": 0.0,
            "min_amplitude": 0.0,
        }

    if array.dtype.kind in {"i", "u"}:
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        normalized = array.astype(np.float32) / float(scale or 1)
    else:
        normalized = array.astype(np.float32)

    return {
        "sample_rate": int(sampling_rate),
        "sample_count": int(array.size),
        "duration_seconds": float(array.size / float(sampling_rate or 1)),
        "rms": float(np.sqrt(np.mean(np.square(normalized)))),
        "max_amplitude": float(np.max(normalized)),
        "min_amplitude": float(np.min(normalized)),
    }


def save_audio(audio_array: Any, sampling_rate: int, filepath: str | Path, label: str = "") -> dict[str, Any]:
    from scipy.io import wavfile

    path = _ensure_parent(filepath)
    wav_array = _prepare_wav_array(audio_array)
    wavfile.write(path, int(sampling_rate), wav_array)
    stats = get_audio_stats(wav_array, sampling_rate)
    stats.update({"path": str(path), "label": label})
    if label:
        print(f"[audio] Saved {label} -> {path} @ {sampling_rate} Hz")
    return stats


def save_text_artifact(filepath: str | Path, payload: Any, as_json: bool = True) -> Path:
    path = _ensure_parent(filepath)
    if as_json:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        return path

    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        text = "\n".join(str(item) for item in payload)
    else:
        text = str(payload)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def save_spectrogram_figure(spectrogram: Any, filepath: str | Path, title: str = "") -> Path:
    import matplotlib.pyplot as plt

    path = _ensure_parent(filepath)
    spec = np.asarray(spectrogram)
    spec = np.squeeze(spec)
    if spec.ndim == 1:
        spec = np.expand_dims(spec, axis=0)

    figure = plt.figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    image = axis.imshow(spec, aspect="auto", origin="lower")
    axis.set_xlabel("Frame")
    axis.set_ylabel("Bin")
    if title:
        axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


class TeeLogger:
    def __init__(self, filepath: str | Path, stream: Any | None = None, mode: str = "a") -> None:
        self.filepath = _ensure_parent(filepath)
        self.stream = stream if stream is not None else sys.stdout
        self.mode = mode
        self._handle = None

    def _open_if_needed(self) -> None:
        if self._handle is None:
            self._handle = self.filepath.open(self.mode, encoding="utf-8")

    def write(self, text: str) -> int:
        self._open_if_needed()
        written = self.stream.write(text)
        self._handle.write(text)
        return written

    def flush(self) -> None:
        if hasattr(self.stream, "flush"):
            self.stream.flush()
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TeeLogger":
        self._open_if_needed()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def save_checkpoint(data: Mapping[str, Any], checkpoint_path: str | Path = CHECKPOINT_PATH) -> Path:
    path = _ensure_parent(checkpoint_path)
    payload = dict(data)
    payload.setdefault("saved_at", now_iso())
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def load_checkpoint(checkpoint_path: str | Path = CHECKPOINT_PATH, default: Any | None = None) -> Any:
    path = _as_path(checkpoint_path)
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _project_relative(filepath: Path) -> Path:
    try:
        return filepath.relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(filepath.name)


def bundle_results(
    source_dir: str | Path,
    zip_path: str | Path,
    extra_paths: Sequence[str | Path] | None = None,
) -> Path:
    source_path = _as_path(source_dir)
    archive_path = _ensure_parent(zip_path)
    extras = list(extra_paths) if extra_paths is not None else ([PRINTOUTS_DIR] if PRINTOUTS_DIR.exists() else [])

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if source_path.exists():
            for child in sorted(source_path.rglob("*")):
                if child.is_file():
                    archive.write(child, arcname=str(_project_relative(child)))

        wrote_logs = False
        for extra in extras:
            extra_path = _as_path(extra)
            if not extra_path.exists():
                continue
            if extra_path.is_file():
                archive.write(extra_path, arcname=str(Path("logs") / extra_path.name))
                wrote_logs = True
                continue
            for child in sorted(extra_path.rglob("*")):
                if child.is_file():
                    archive.write(child, arcname=str(Path("logs") / child.relative_to(extra_path)))
                    wrote_logs = True

        if not wrote_logs:
            archive.writestr("logs/.keep", "")

    return archive_path


def cleanup_model(*objects: Any) -> dict[str, Any]:
    released_objects = len(objects)
    del objects
    gc.collect()

    cuda_cleared = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            cuda_cleared = True
    except ImportError:
        cuda_cleared = False

    return {"released_objects": released_objects, "cuda_cleared": cuda_cleared}


__all__ = [
    "PROJECT_ROOT",
    "EXPERIMENTS_DIR",
    "PRINTOUTS_DIR",
    "CHECKPOINT_PATH",
    "timer",
    "now_iso",
    "init_csv",
    "append_csv_row",
    "count_csv_rows",
    "save_audio",
    "get_audio_stats",
    "save_text_artifact",
    "save_spectrogram_figure",
    "TeeLogger",
    "save_checkpoint",
    "load_checkpoint",
    "bundle_results",
    "cleanup_model",
]