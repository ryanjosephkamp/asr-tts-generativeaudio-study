from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

from hw14_analysis_utils import compute_text_similarity
from hw14_data_utils import PROJECT_ROOT


plt.style.use("seaborn-v0_8-whitegrid")

EXPERIMENTS_DIR = PROJECT_ROOT / "hw14_experiments"
REPORT_FIGS_DIR = PROJECT_ROOT / "hw14_reports" / "figures"
LATEX_FIGS_DIR = PROJECT_ROOT / "hw14_reports" / "latex" / "figures"

GROUP_A_DIR = EXPERIMENTS_DIR / "group_a_asr"
GROUP_B_DIR = EXPERIMENTS_DIR / "group_b_structured_tts"
GROUP_C_DIR = EXPERIMENTS_DIR / "group_c_bark_cross"

FIGURE_OUTPUTS = {
    "exp1_baseline_summary.png": EXPERIMENTS_DIR / "exp1_baseline_summary.png",
    "asr_family_wer_bar.png": GROUP_A_DIR / "exp2_shortform_asr" / "asr_family_wer_bar.png",
    "asr_family_cer_bar.png": GROUP_A_DIR / "exp2_shortform_asr" / "asr_family_cer_bar.png",
    "asr_family_runtime.png": GROUP_A_DIR / "exp2_shortform_asr" / "asr_family_runtime.png",
    "whisper_interface_delta.png": GROUP_A_DIR / "exp2_shortform_asr" / "whisper_interface_delta.png",
    "asr_sampling_rate_curve.png": GROUP_A_DIR / "exp3_asr_sensitivity" / "asr_sampling_rate_curve.png",
    "asr_sampling_rate_runtime.png": GROUP_A_DIR / "exp3_asr_sensitivity" / "asr_sampling_rate_runtime.png",
    "ga20c_cleanup_delta.png": GROUP_A_DIR / "exp3_asr_sensitivity" / "ga20c_cleanup_delta.png",
    "longform_segment_counts.png": GROUP_A_DIR / "exp4_longform_whisper" / "longform_segment_counts.png",
    "longform_runtime.png": GROUP_A_DIR / "exp4_longform_whisper" / "longform_runtime.png",
    "longform_similarity_heatmap.png": GROUP_A_DIR / "exp4_longform_whisper" / "longform_similarity_heatmap.png",
    "speecht5_spectrogram_panel.png": GROUP_B_DIR / "exp5_speecht5_tts" / "speecht5_spectrogram_panel.png",
    "speecht5_duration_summary.png": GROUP_B_DIR / "exp5_speecht5_tts" / "speecht5_duration_summary.png",
    "mms_seed_duration.png": GROUP_B_DIR / "exp6_mms_tts" / "mms_seed_duration.png",
    "mms_speaking_rate_duration.png": GROUP_B_DIR / "exp6_mms_tts" / "mms_speaking_rate_duration.png",
    "bark_prompt_style_grid.png": GROUP_C_DIR / "exp7_bark" / "bark_prompt_style_grid.png",
    "bark_voice_preset_summary.png": GROUP_C_DIR / "exp7_bark" / "bark_voice_preset_summary.png",
    "bark_sample_rate_audit.png": GROUP_C_DIR / "exp7_bark" / "bark_sample_rate_audit.png",
    "tts_roundtrip_wer.png": GROUP_C_DIR / "exp8_cross_tts" / "tts_roundtrip_wer.png",
    "tts_roundtrip_cer.png": GROUP_C_DIR / "exp8_cross_tts" / "tts_roundtrip_cer.png",
    "tts_runtime_comparison.png": GROUP_C_DIR / "exp8_cross_tts" / "tts_runtime_comparison.png",
    "tts_control_tradeoff_table.png": GROUP_C_DIR / "exp8_cross_tts" / "tts_control_tradeoff_table.png",
}

SCRIPT_ORDER = ["GA20A", "GA20B", "GA20C", "GA20D", "GA20E", "GA20F", "GA20G", "GA20H", "GA20K"]
TEXT_TAG_ORDER = ["hello_dog", "llamas", "banking_request"]
SPEAKER_ORDER = ["helper_embedding", "zero_vector"]
SEED_ORDER = [42, 123, 555]
SPEAKING_RATE_ORDER = [0.8, 1.0, 1.2]
PROMPT_ORDER = ["plain", "expr", "strong_expr"]
VOICE_ORDER = ["no_preset", "preset5", "preset6"]
SAVE_RATE_ORDER = ["model_native", "hardcoded_16k"]
SYSTEM_ORDER = ["speecht5", "mms_tts", "bark_preset"]

WORKFLOW_LABELS = {
    "ga20a_whisper_pipeline": "GA20A Whisper pipeline",
    "ga20b_wav2vec2_ctc": "GA20B Wav2Vec2 CTC",
    "ga20c_whisper_direct": "GA20C direct Whisper",
    "ga20e_speecht5_asr": "GA20E SpeechT5 ASR",
}

MODEL_LABELS = {
    "openai/whisper-small": "Whisper small",
    "facebook/wav2vec2-base-960h": "Wav2Vec2 base",
    "microsoft/speecht5_asr": "SpeechT5 ASR",
    "microsoft/speecht5_tts": "SpeechT5 TTS",
    "facebook/mms-tts-eng": "MMS-TTS",
    "suno/bark-small": "Bark small",
}

TEXT_LABELS = {
    "hello_dog": "Hello, my dog is cute.",
    "llamas": "There are llamas all around.",
    "banking_request": "Banking request",
}

SPEAKER_LABELS = {
    "helper_embedding": "Helper embedding",
    "zero_vector": "Zero vector",
}

PROMPT_LABELS = {
    "plain": "Plain prompt",
    "expr": "Expressive prompt",
    "strong_expr": "Strong expressive prompt",
}

VOICE_LABELS = {
    "no_preset": "No preset",
    "preset5": "Preset 5",
    "preset6": "Preset 6",
}

SAVE_RATE_LABELS = {
    "model_native": "Model-native save rate",
    "hardcoded_16k": "Hard-coded 16 kHz",
    "native": "Native save rate",
}

SYSTEM_LABELS = {
    "speecht5": "SpeechT5",
    "mms_tts": "MMS-TTS",
    "bark_preset": "Bark preset",
}

CONTROL_LABELS = {
    "speecht5": "Speaker embedding",
    "mms_tts": "Seed + speaking rate",
    "bark_preset": "Voice preset + seed",
}

GROUP_COLORS = {
    "speech": "#2F6FED",
    "speech_alt": "#6C8EF5",
    "mms": "#11A579",
    "bark": "#E17C05",
    "warning": "#B85C38",
    "neutral": "#5F6B7A",
}


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve_path(raw_path: object) -> Path:
    if pd.isna(raw_path):
        raise ValueError("Encountered an empty artifact path.")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _save_figure(fig: plt.Figure, save_path: Path) -> Path:
    output_path = _ensure_parent(save_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_rate(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return str(int(round(float(numeric))))


def _format_condition_label(chunk_length_s: object, batch_size: object) -> str:
    return f"{int(chunk_length_s)}s / bs{int(batch_size)}"


def _artifact_summary(row: pd.Series) -> str:
    outputs = []
    if isinstance(row.get("output_text_path"), str) and row["output_text_path"]:
        outputs.append("text")
    if isinstance(row.get("output_audio_path"), str) and row["output_audio_path"]:
        outputs.append("audio")
    if isinstance(row.get("output_figure_path"), str) and row["output_figure_path"]:
        outputs.append("figure")
    if not outputs:
        return str(row.get("task_type", "artifact")).upper()
    return "+".join(outputs)


def _annotate_vertical_bars(ax: plt.Axes, bars, decimals: int = 3) -> None:
    lower, upper = ax.get_ylim()
    offset = max((upper - lower) * 0.02, 0.01)
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset if value >= 0 else value - offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )


def _make_table_figure(table_df: pd.DataFrame, title: str, save_path: Path, font_size: int = 8) -> Path:
    fig_height = 2.5 + 0.42 * len(table_df)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.12, 1.35)
    for cell_key, cell in table.get_celld().items():
        if cell_key[0] == 0:
            cell.set_facecolor("#D9E6FB")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    return _save_figure(fig, save_path)


def _make_bar_chart(summary: pd.Series, title: str, ylabel: str, save_path: Path, color: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(summary.index, summary.values, color=color, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=15)
    _annotate_vertical_bars(ax, bars)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def _make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    x_order: list,
    group_order: list,
    group_labels: dict[str, str],
    title: str,
    ylabel: str,
    save_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x_lookup = {value: index for index, value in enumerate(x_order)}
    x_positions = np.arange(len(x_order))
    for group_name in group_order:
        subset = df[df[group_col] == group_name].copy()
        if subset.empty:
            continue
        subset["_x"] = subset[x_col].map(x_lookup)
        subset = subset.sort_values("_x")
        ax.plot(
            subset["_x"],
            subset[y_col],
            marker="o",
            linewidth=2,
            label=group_labels.get(group_name, group_name),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(item) for item in x_order])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def _make_grouped_bar_chart(
    pivot_df: pd.DataFrame,
    title: str,
    ylabel: str,
    save_path: Path,
    legend_title: str,
    label_map: dict[str, str] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = np.arange(len(pivot_df.index))
    width = 0.8 / max(len(pivot_df.columns), 1)
    colors = [GROUP_COLORS["speech"], GROUP_COLORS["mms"], GROUP_COLORS["bark"], GROUP_COLORS["warning"]]

    for index, column in enumerate(pivot_df.columns):
        values = pivot_df[column].to_numpy(dtype=float)
        bars = ax.bar(
            x_positions + (index - (len(pivot_df.columns) - 1) / 2) * width,
            values,
            width=width,
            label=label_map.get(column, column) if label_map else column,
            color=colors[index % len(colors)],
            edgecolor="black",
        )
        _annotate_vertical_bars(ax, bars, decimals=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index, rotation=12, ha="right")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=legend_title, frameon=True, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, save_path)


def _make_image_grid(image_paths: list[Path], labels: list[str], title: str, save_path: Path, ncols: int = 2) -> Path:
    nrows = math.ceil(len(image_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, image_path, label in zip(axes.flat, image_paths, labels):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, save_path)


def _make_audio_spectrogram_grid(audio_paths: list[Path], labels: list[str], title: str, save_path: Path, ncols: int = 2) -> Path:
    nrows = math.ceil(len(audio_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, audio_path, label in zip(axes.flat, audio_paths, labels):
        sample_rate, audio = wavfile.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        window = min(1024, max(128, len(audio)))
        ax.specgram(audio, Fs=sample_rate, NFFT=window, noverlap=window // 2, cmap="magma")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency")
    for ax in axes.flat[len(audio_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(fig, save_path)


def generate_exp1_baseline_summary() -> Path:
    baseline_sources = [
        ("Group A", GROUP_A_DIR / "exp1_baselines" / "exp1_baselines.csv"),
        ("Group B", GROUP_B_DIR / "exp1_baselines" / "exp1_baselines.csv"),
        ("Group C", GROUP_C_DIR / "exp1_baselines" / "exp1_baselines.csv"),
    ]
    frames = []
    for group_name, csv_path in baseline_sources:
        frame = _read_csv(csv_path).copy()
        frame["group"] = group_name
        frames.append(frame)
    baseline_df = pd.concat(frames, ignore_index=True)
    baseline_df["runtime_sec"] = pd.to_numeric(baseline_df["runtime_sec"], errors="coerce")
    baseline_df["script_order"] = baseline_df["script"].map({name: index for index, name in enumerate(SCRIPT_ORDER)})
    baseline_df = baseline_df.sort_values("script_order")
    baseline_df["model_label"] = baseline_df["model_name"].map(MODEL_LABELS).fillna(baseline_df["model_name"])
    baseline_df["artifacts"] = baseline_df.apply(_artifact_summary, axis=1)

    table_df = pd.DataFrame(
        {
            "Script": baseline_df["script"],
            "Group": baseline_df["group"],
            "Task": baseline_df["task_type"].str.upper(),
            "Model": baseline_df["model_label"],
            "Outputs": baseline_df["artifacts"],
            "Runtime (s)": baseline_df["runtime_sec"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-"),
            "Save Hz": baseline_df["saved_sampling_rate"].map(_format_rate),
        }
    )
    return _make_table_figure(
        table_df,
        "Experiment 1 Baseline Artifact Summary Across All Nine Scripts",
        FIGURE_OUTPUTS["exp1_baseline_summary.png"],
    )


def generate_exp2_figures() -> list[Path]:
    exp2_df = _read_csv(GROUP_A_DIR / "exp2_shortform_asr" / "exp2_shortform_asr.csv").copy()
    exp2_df["wer"] = pd.to_numeric(exp2_df["wer"], errors="coerce")
    exp2_df["cer"] = pd.to_numeric(exp2_df["cer"], errors="coerce")
    exp2_df["inference_time_sec"] = pd.to_numeric(exp2_df["inference_time_sec"], errors="coerce")
    order = [name for name in WORKFLOW_LABELS if name in set(exp2_df["workflow_name"])]

    results = []
    for metric, ylabel, filename in [
        ("wer", "Mean WER", "asr_family_wer_bar.png"),
        ("cer", "Mean CER", "asr_family_cer_bar.png"),
        ("inference_time_sec", "Mean runtime (s)", "asr_family_runtime.png"),
    ]:
        summary = exp2_df.groupby("workflow_name")[metric].mean().reindex(order)
        summary.index = [WORKFLOW_LABELS[name] for name in summary.index]
        results.append(_make_bar_chart(summary, f"Experiment 2 {ylabel} by Workflow", ylabel, FIGURE_OUTPUTS[filename], GROUP_COLORS["speech"]))

    paired = exp2_df[exp2_df["workflow_name"].isin(["ga20a_whisper_pipeline", "ga20c_whisper_direct"])].copy()
    delta_df = paired.pivot(index="sample_index", columns="workflow_name", values="wer").dropna()
    delta_df["wer_delta"] = delta_df["ga20a_whisper_pipeline"] - delta_df["ga20c_whisper_direct"]
    delta_df = delta_df.sort_values("wer_delta")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = [GROUP_COLORS["warning"] if value < 0 else GROUP_COLORS["mms"] for value in delta_df["wer_delta"]]
    ax.barh(delta_df.index.astype(str), delta_df["wer_delta"], color=colors, edgecolor="black")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Experiment 2 Whisper Interface Delta by Sample (WER)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pipeline WER - direct Whisper WER")
    ax.set_ylabel("Sample index")
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.01, 0.02, "Positive values favor cleaned direct Whisper.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    results.append(_save_figure(fig, FIGURE_OUTPUTS["whisper_interface_delta.png"]))
    return results


def generate_exp3_figures() -> list[Path]:
    exp3_df = _read_csv(GROUP_A_DIR / "exp3_asr_sensitivity" / "exp3_asr_sensitivity.csv").copy()
    for column in ["resampling_rate", "raw_wer", "clean_wer", "raw_cer", "clean_cer", "inference_time_sec"]:
        exp3_df[column] = pd.to_numeric(exp3_df[column], errors="coerce")

    exp3a = exp3_df[exp3_df["sub_experiment"] == "3A"].copy()
    exp3a_summary = exp3a.groupby(["workflow_name", "resampling_rate"], as_index=False).agg(
        clean_wer=("clean_wer", "mean"),
        inference_time_sec=("inference_time_sec", "mean"),
    )
    exp3a_summary["resampling_rate"] = exp3a_summary["resampling_rate"].astype(int)

    workflow_order = [name for name in ["ga20a_whisper_pipeline", "ga20b_wav2vec2_ctc", "ga20e_speecht5_asr"] if name in set(exp3a_summary["workflow_name"])]
    results = [
        _make_line_chart(
            exp3a_summary,
            "resampling_rate",
            "clean_wer",
            "workflow_name",
            [8000, 16000, 24000],
            workflow_order,
            WORKFLOW_LABELS,
            "Experiment 3A Mean WER vs Input Resampling Rate",
            "Mean WER",
            FIGURE_OUTPUTS["asr_sampling_rate_curve.png"],
        ),
        _make_line_chart(
            exp3a_summary,
            "resampling_rate",
            "inference_time_sec",
            "workflow_name",
            [8000, 16000, 24000],
            workflow_order,
            WORKFLOW_LABELS,
            "Experiment 3A Runtime vs Input Resampling Rate",
            "Mean runtime (s)",
            FIGURE_OUTPUTS["asr_sampling_rate_runtime.png"],
        ),
    ]

    exp3b = exp3_df[exp3_df["sub_experiment"] == "3B"].copy()
    exp3b["wer_gain"] = exp3b["raw_wer"] - exp3b["clean_wer"]
    exp3b = exp3b.sort_values("wer_gain")
    summary_means = pd.DataFrame(
        {
            "Metric": ["WER", "CER"],
            "Raw": [exp3b["raw_wer"].mean(), exp3b["raw_cer"].mean()],
            "Cleaned": [exp3b["clean_wer"].mean(), exp3b["clean_cer"].mean()],
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]})
    colors = [GROUP_COLORS["warning"] if value < 0 else GROUP_COLORS["mms"] for value in exp3b["wer_gain"]]
    axes[0].barh(exp3b["sample_index"].astype(int).astype(str), exp3b["wer_gain"], color=colors, edgecolor="black")
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_title("Experiment 3B GA20C Cleanup Delta by Sample", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Raw WER - cleaned WER")
    axes[0].set_ylabel("Sample index")
    axes[0].grid(axis="x", alpha=0.3)

    x_positions = np.arange(len(summary_means))
    width = 0.35
    raw_bars = axes[1].bar(x_positions - width / 2, summary_means["Raw"], width=width, label="Raw decode", color=GROUP_COLORS["warning"], edgecolor="black")
    clean_bars = axes[1].bar(x_positions + width / 2, summary_means["Cleaned"], width=width, label="Cleaned decode", color=GROUP_COLORS["speech"], edgecolor="black")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(summary_means["Metric"])
    axes[1].set_ylabel("Mean error rate")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=True)
    _annotate_vertical_bars(axes[1], raw_bars)
    _annotate_vertical_bars(axes[1], clean_bars)
    fig.tight_layout()
    results.append(_save_figure(fig, FIGURE_OUTPUTS["ga20c_cleanup_delta.png"]))
    return results


def generate_exp4_figures() -> list[Path]:
    exp4_df = _read_csv(GROUP_A_DIR / "exp4_longform_whisper" / "exp4_longform_whisper.csv").copy()
    for column in ["chunk_length_s", "batch_size", "num_segments", "transcript_similarity_to_baseline", "inference_time_sec"]:
        exp4_df[column] = pd.to_numeric(exp4_df[column], errors="coerce")
    exp4_df["condition"] = exp4_df.apply(lambda row: _format_condition_label(row["chunk_length_s"], row["batch_size"]), axis=1)

    segment_summary = pd.Series(exp4_df["num_segments"].to_numpy(), index=exp4_df["condition"])
    runtime_summary = pd.Series(exp4_df["inference_time_sec"].to_numpy(), index=exp4_df["condition"])
    results = [
        _make_bar_chart(segment_summary, "Experiment 4 Long-form Segment Count by Condition", "Segment count", FIGURE_OUTPUTS["longform_segment_counts.png"], GROUP_COLORS["speech"]),
        _make_bar_chart(runtime_summary, "Experiment 4 Long-form Runtime by Condition", "Runtime (s)", FIGURE_OUTPUTS["longform_runtime.png"], GROUP_COLORS["neutral"]),
    ]

    matrix = exp4_df[["transcript_similarity_to_baseline"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["vs 5s / bs8 baseline"])
    ax.set_yticks(range(len(exp4_df)))
    ax.set_yticklabels(exp4_df["condition"])
    ax.set_title("Experiment 4 Transcript Similarity to Corrected Baseline", fontsize=13, fontweight="bold")
    for row_index, value in enumerate(matrix[:, 0]):
        ax.text(0, row_index, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Similarity")
    fig.tight_layout()
    results.append(_save_figure(fig, FIGURE_OUTPUTS["longform_similarity_heatmap.png"]))
    return results


def generate_exp5_figures() -> list[Path]:
    exp5_df = _read_csv(GROUP_B_DIR / "exp5_speecht5_tts" / "exp5_speecht5_tts.csv").copy()
    exp5_df["waveform_duration_sec"] = pd.to_numeric(exp5_df["waveform_duration_sec"], errors="coerce")
    exp5_df["text_order"] = exp5_df["text_tag"].map({name: index for index, name in enumerate(TEXT_TAG_ORDER)})
    exp5_df["speaker_order"] = exp5_df["speaker_condition"].map({name: index for index, name in enumerate(SPEAKER_ORDER)})
    exp5_df = exp5_df.sort_values(["text_order", "speaker_order"])

    image_paths = [_resolve_path(value) for value in exp5_df["spectrogram_path"]]
    labels = [f"{TEXT_LABELS.get(text_tag, text_tag)}\n{SPEAKER_LABELS.get(speaker_condition, speaker_condition)}" for text_tag, speaker_condition in zip(exp5_df["text_tag"], exp5_df["speaker_condition"])]
    spectrogram_panel = _make_image_grid(image_paths, labels, "Experiment 5 SpeechT5 Spectrogram Panel", FIGURE_OUTPUTS["speecht5_spectrogram_panel.png"], ncols=2)

    pivot = exp5_df.pivot(index="text_tag", columns="speaker_condition", values="waveform_duration_sec").reindex(TEXT_TAG_ORDER)
    pivot.index = [TEXT_LABELS.get(name, name) for name in pivot.index]
    duration_summary = _make_grouped_bar_chart(
        pivot,
        "Experiment 5 SpeechT5 Waveform Duration by Text and Speaker Condition",
        "Duration (s)",
        FIGURE_OUTPUTS["speecht5_duration_summary.png"],
        "Speaker condition",
        label_map=SPEAKER_LABELS,
    )
    return [spectrogram_panel, duration_summary]


def generate_exp6_figures() -> list[Path]:
    exp6_df = _read_csv(GROUP_B_DIR / "exp6_mms_tts" / "exp6_mms_tts.csv").copy()
    exp6_df["seed"] = pd.to_numeric(exp6_df["seed"], errors="coerce").astype("Int64")
    exp6_df["speaking_rate"] = pd.to_numeric(exp6_df["speaking_rate"], errors="coerce")
    exp6_df["waveform_duration_sec"] = pd.to_numeric(exp6_df["waveform_duration_sec"], errors="coerce")

    exp6a = exp6_df[exp6_df["sub_experiment"] == "6A"].copy()
    exp6a_summary = exp6a.groupby(["text_tag", "seed"], as_index=False)["waveform_duration_sec"].mean()
    seed_figure = _make_line_chart(
        exp6a_summary,
        "seed",
        "waveform_duration_sec",
        "text_tag",
        SEED_ORDER,
        [tag for tag in TEXT_TAG_ORDER if tag in set(exp6a_summary["text_tag"])],
        {tag: TEXT_LABELS.get(tag, tag) for tag in TEXT_TAG_ORDER},
        "Experiment 6A MMS-TTS Duration Variability Across Seeds",
        "Duration (s)",
        FIGURE_OUTPUTS["mms_seed_duration.png"],
    )

    exp6b = exp6_df[exp6_df["sub_experiment"] == "6B"].copy()
    exp6b_summary = exp6b.groupby(["text_tag", "speaking_rate"], as_index=False)["waveform_duration_sec"].mean()
    rate_figure = _make_line_chart(
        exp6b_summary,
        "speaking_rate",
        "waveform_duration_sec",
        "text_tag",
        SPEAKING_RATE_ORDER,
        [tag for tag in TEXT_TAG_ORDER if tag in set(exp6b_summary["text_tag"])],
        {tag: TEXT_LABELS.get(tag, tag) for tag in TEXT_TAG_ORDER},
        "Experiment 6B MMS-TTS Duration vs Speaking Rate",
        "Duration (s)",
        FIGURE_OUTPUTS["mms_speaking_rate_duration.png"],
    )
    return [seed_figure, rate_figure]


def generate_exp7_figures() -> list[Path]:
    exp7_df = _read_csv(GROUP_C_DIR / "exp7_bark" / "exp7_bark.csv").copy()
    for column in ["seed", "model_sample_rate", "saved_sample_rate", "waveform_duration_sec", "generation_time_sec", "round_trip_wer"]:
        exp7_df[column] = pd.to_numeric(exp7_df[column], errors="coerce")

    exp7a = exp7_df[exp7_df["sub_experiment"] == "7A"].copy()
    exp7a["prompt_order"] = exp7a["prompt_mode"].map({name: index for index, name in enumerate(PROMPT_ORDER)})
    exp7a = exp7a.sort_values(["prompt_order", "seed"])
    bark_grid = _make_audio_spectrogram_grid(
        [_resolve_path(value) for value in exp7a["output_path"]],
        [f"{PROMPT_LABELS.get(mode, mode)} | seed {int(seed)}" for mode, seed in zip(exp7a["prompt_mode"], exp7a["seed"])],
        "Experiment 7A Bark Prompt-Style Spectrogram Grid",
        FIGURE_OUTPUTS["bark_prompt_style_grid.png"],
        ncols=2,
    )

    exp7b = exp7_df[exp7_df["sub_experiment"] == "7B"].copy()
    exp7b["voice_order"] = exp7b["voice_condition"].map({name: index for index, name in enumerate(VOICE_ORDER)})
    exp7b = exp7b.sort_values(["voice_order", "seed"])
    voice_means = exp7b.groupby("voice_condition", as_index=False).agg(
        waveform_duration_sec=("waveform_duration_sec", "mean"),
        round_trip_wer=("round_trip_wer", "mean"),
    )
    voice_means["voice_order"] = voice_means["voice_condition"].map({name: index for index, name in enumerate(VOICE_ORDER)})
    voice_means = voice_means.sort_values("voice_order")
    voice_positions = np.arange(len(voice_means))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    duration_bars = axes[0].bar(
        voice_positions,
        voice_means["waveform_duration_sec"],
        color=GROUP_COLORS["bark"],
        edgecolor="black",
    )
    _annotate_vertical_bars(axes[0], duration_bars, decimals=2)
    axes[0].set_xticks(voice_positions)
    axes[0].set_xticklabels([VOICE_LABELS.get(value, value) for value in voice_means["voice_condition"]])
    axes[0].set_title("Mean Bark duration")
    axes[0].set_ylabel("Duration (s)")
    axes[0].grid(axis="y", alpha=0.3)

    wer_bars = axes[1].bar(
        voice_positions,
        voice_means["round_trip_wer"],
        color=GROUP_COLORS["speech_alt"],
        edgecolor="black",
    )
    _annotate_vertical_bars(axes[1], wer_bars, decimals=3)
    axes[1].set_xticks(voice_positions)
    axes[1].set_xticklabels([VOICE_LABELS.get(value, value) for value in voice_means["voice_condition"]])
    axes[1].set_title("Mean Bark round-trip WER")
    axes[1].set_ylabel("WER")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 7B Bark Voice-Condition Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    voice_summary = _save_figure(fig, FIGURE_OUTPUTS["bark_voice_preset_summary.png"])

    exp7c = exp7_df[exp7_df["sub_experiment"] == "7C"].copy()
    exp7c["save_rate_mode"] = exp7c["save_rate_mode"].replace({"native": "model_native"})
    audit_means = exp7c.groupby("save_rate_mode", as_index=False).agg(
        waveform_duration_sec=("waveform_duration_sec", "mean"),
        round_trip_wer=("round_trip_wer", "mean"),
    )
    audit_means["mode_order"] = audit_means["save_rate_mode"].map({name: index for index, name in enumerate(SAVE_RATE_ORDER)})
    audit_means = audit_means.sort_values("mode_order")
    audit_positions = np.arange(len(audit_means))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    duration_bars = axes[0].bar(audit_positions, audit_means["waveform_duration_sec"], color=GROUP_COLORS["warning"], edgecolor="black")
    _annotate_vertical_bars(axes[0], duration_bars, decimals=2)
    axes[0].set_xticks(audit_positions)
    axes[0].set_xticklabels([SAVE_RATE_LABELS.get(value, value) for value in audit_means["save_rate_mode"]], rotation=10, ha="right")
    axes[0].set_title("Observed playback duration")
    axes[0].set_ylabel("Duration (s)")
    axes[0].grid(axis="y", alpha=0.3)

    wer_bars = axes[1].bar(audit_positions, audit_means["round_trip_wer"], color=GROUP_COLORS["speech_alt"], edgecolor="black")
    _annotate_vertical_bars(axes[1], wer_bars, decimals=3)
    axes[1].set_xticks(audit_positions)
    axes[1].set_xticklabels([SAVE_RATE_LABELS.get(value, value) for value in audit_means["save_rate_mode"]], rotation=10, ha="right")
    axes[1].set_title("Mean round-trip WER")
    axes[1].set_ylabel("WER")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 7C Bark Sample-Rate Audit", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    sample_rate_audit = _save_figure(fig, FIGURE_OUTPUTS["bark_sample_rate_audit.png"])
    return [bark_grid, voice_summary, sample_rate_audit]


def generate_exp8_figures() -> list[Path]:
    exp8_df = _read_csv(GROUP_C_DIR / "exp8_cross_tts" / "exp8_cross_tts.csv").copy()
    for column in ["saved_sample_rate", "waveform_duration_sec", "generation_time_sec", "round_trip_wer", "round_trip_cer"]:
        exp8_df[column] = pd.to_numeric(exp8_df[column], errors="coerce")

    results = []
    for metric, ylabel, filename in [
        ("round_trip_wer", "Mean round-trip WER", "tts_roundtrip_wer.png"),
        ("round_trip_cer", "Mean round-trip CER", "tts_roundtrip_cer.png"),
        ("generation_time_sec", "Mean generation time (s)", "tts_runtime_comparison.png"),
    ]:
        summary = exp8_df.groupby("system_id")[metric].mean().reindex(SYSTEM_ORDER)
        summary = summary.dropna()
        summary.index = [SYSTEM_LABELS.get(name, name) for name in summary.index]
        results.append(_make_bar_chart(summary, f"Experiment 8 {ylabel} by TTS Family", ylabel, FIGURE_OUTPUTS[filename], GROUP_COLORS["mms"]))

    summary_df = exp8_df.groupby("system_id", as_index=False).agg(
        mean_wer=("round_trip_wer", "mean"),
        mean_cer=("round_trip_cer", "mean"),
        mean_runtime_sec=("generation_time_sec", "mean"),
        mean_duration_sec=("waveform_duration_sec", "mean"),
        saved_rate_hz=("saved_sample_rate", "mean"),
        reused_rows=("reused_from_experiment", lambda values: int(values.fillna("").astype(str).str.len().gt(0).sum())),
    )
    summary_df["system_order"] = summary_df["system_id"].map({name: index for index, name in enumerate(SYSTEM_ORDER)})
    summary_df = summary_df.sort_values("system_order")
    table_df = pd.DataFrame(
        {
            "System": summary_df["system_id"].map(SYSTEM_LABELS),
            "Mean WER": summary_df["mean_wer"].map(lambda value: f"{value:.3f}"),
            "Mean CER": summary_df["mean_cer"].map(lambda value: f"{value:.3f}"),
            "Mean runtime (s)": summary_df["mean_runtime_sec"].map(lambda value: f"{value:.2f}"),
            "Mean duration (s)": summary_df["mean_duration_sec"].map(lambda value: f"{value:.2f}"),
            "Saved Hz": summary_df["saved_rate_hz"].map(lambda value: str(int(round(value)))),
            "Control mechanism": summary_df["system_id"].map(CONTROL_LABELS),
            "Reused rows": summary_df["reused_rows"].astype(int).astype(str),
        }
    )
    results.append(_make_table_figure(table_df, "Experiment 8 TTS Control and Trade-off Summary", FIGURE_OUTPUTS["tts_control_tradeoff_table.png"], font_size=8))
    return results


def copy_report_figures(figure_paths: list[Path]) -> None:
    REPORT_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    for figure_path in figure_paths:
        shutil.copy2(figure_path, REPORT_FIGS_DIR / figure_path.name)
        shutil.copy2(figure_path, LATEX_FIGS_DIR / figure_path.name)


def generate_all_figures() -> list[Path]:
    figure_paths = []
    figure_paths.append(generate_exp1_baseline_summary())
    figure_paths.extend(generate_exp2_figures())
    figure_paths.extend(generate_exp3_figures())
    figure_paths.extend(generate_exp4_figures())
    figure_paths.extend(generate_exp5_figures())
    figure_paths.extend(generate_exp6_figures())
    figure_paths.extend(generate_exp7_figures())
    figure_paths.extend(generate_exp8_figures())
    copy_report_figures(figure_paths)
    return figure_paths


def main() -> int:
    figure_paths = generate_all_figures()
    print(f"Generated {len(figure_paths)} figures.")
    print(f"Report figures directory: {REPORT_FIGS_DIR}")
    print(f"LaTeX figures directory: {LATEX_FIGS_DIR}")
    for figure_path in figure_paths:
        print(f" - {figure_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())