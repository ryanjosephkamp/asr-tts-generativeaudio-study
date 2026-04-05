"""Step 7: Cross-Experiment Analysis and Data Consolidation.

Generates:
  - hw14_experiments/consolidated_metrics.csv
  - hw14_experiments/summary_statistics.json
  - hw14_experiments/final_tradeoff_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT / "hw14_experiments"
REPORT_FIGS = PROJECT_ROOT / "hw14_reports" / "figures"

PRIMARY_CSVS = [
    "group_a_asr/exp1_baselines/exp1_baselines.csv",
    "group_a_asr/exp2_shortform_asr/exp2_shortform_asr.csv",
    "group_a_asr/exp3_asr_sensitivity/exp3_asr_sensitivity.csv",
    "group_a_asr/exp4_longform_whisper/exp4_longform_whisper.csv",
    "group_b_structured_tts/exp1_baselines/exp1_baselines.csv",
    "group_b_structured_tts/exp5_speecht5_tts/exp5_speecht5_tts.csv",
    "group_b_structured_tts/exp6_mms_tts/exp6_mms_tts.csv",
    "group_c_bark_cross/exp1_baselines/exp1_baselines.csv",
    "group_c_bark_cross/exp7_bark/exp7_bark.csv",
    "group_c_bark_cross/exp8_cross_tts/exp8_cross_tts.csv",
]


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Consolidated metrics CSV
    # ------------------------------------------------------------------
    frames = []
    for rel in PRIMARY_CSVS:
        csv_path = EXP_ROOT / rel
        df = pd.read_csv(csv_path)
        df["source_csv"] = rel
        frames.append(df)

    consolidated = pd.concat(frames, ignore_index=True, sort=False)
    consolidated.to_csv(EXP_ROOT / "consolidated_metrics.csv", index=False)
    print(f"consolidated_metrics.csv: {len(consolidated)} rows from {len(PRIMARY_CSVS)} CSVs")

    # ------------------------------------------------------------------
    # 2. Summary statistics JSON
    # ------------------------------------------------------------------
    group_a_summary = json.loads(
        (EXP_ROOT / "group_a_asr" / "full" / "group_a_full_summary.json").read_text()
    )
    group_b_summary = json.loads(
        (EXP_ROOT / "group_b_structured_tts" / "group_b_full_summary.json").read_text()
    )
    group_c_summary = json.loads(
        (EXP_ROOT / "group_c_bark_cross" / "group_c_full_summary.json").read_text()
    )
    best_asr = json.loads(
        (EXP_ROOT / "group_a_asr" / "full" / "best_asr_evaluator.json").read_text()
    )

    exp_counts = {}
    for rel in PRIMARY_CSVS:
        name = Path(rel).stem
        exp_counts[name] = len(pd.read_csv(EXP_ROOT / rel))

    # Exp2 ASR winner
    exp2 = pd.read_csv(EXP_ROOT / "group_a_asr/exp2_shortform_asr/exp2_shortform_asr.csv")
    for col in ["wer", "cer", "inference_time_sec"]:
        exp2[col] = pd.to_numeric(exp2[col], errors="coerce")

    # Exp8 TTS winner
    exp8 = pd.read_csv(EXP_ROOT / "group_c_bark_cross/exp8_cross_tts/exp8_cross_tts.csv")
    for col in ["generation_time_sec", "round_trip_wer", "round_trip_cer"]:
        exp8[col] = pd.to_numeric(exp8[col], errors="coerce")

    exp8_by_system = (
        exp8.groupby("system_id")
        .agg(
            mean_wer=("round_trip_wer", "mean"),
            mean_cer=("round_trip_cer", "mean"),
            mean_runtime=("generation_time_sec", "mean"),
        )
        .sort_values(["mean_wer", "mean_cer", "mean_runtime"])
    )
    exp8_winner = exp8_by_system.index[0]

    figure_count = len(list(REPORT_FIGS.glob("*.png")))

    summary_stats = {
        "project": "HW14 Speech AI",
        "total_primary_csvs": len(PRIMARY_CSVS),
        "consolidated_rows": int(len(consolidated)),
        "figure_count": figure_count,
        "experiment_row_counts": exp_counts,
        "group_summaries": {
            "group_a_asr": group_a_summary.get("counts", {}),
            "group_b_structured_tts": group_b_summary.get("counts", {}),
            "group_c_bark_cross": group_c_summary.get("counts", {}),
        },
        "best_asr_evaluator": best_asr,
        "exp2_asr_winner": {
            "workflow": best_asr["workflow_name"],
            "mean_wer": best_asr["mean_wer"],
            "mean_cer": best_asr["mean_cer"],
        },
        "exp8_tts_winner": {
            "system_id": exp8_winner,
            "mean_wer": float(exp8_by_system.loc[exp8_winner, "mean_wer"]),
            "mean_cer": float(exp8_by_system.loc[exp8_winner, "mean_cer"]),
            "mean_runtime_sec": float(exp8_by_system.loc[exp8_winner, "mean_runtime"]),
        },
        "research_questions_addressable": [
            "RQ1: Exp2 CSV (96 rows across 4 ASR workflows on 24 fixed MINDS-14 utterances)",
            "RQ2: Exp3 CSV (114 rows covering resampling-rate sensitivity and GA20C cleanup delta)",
            "RQ3: Exp4 CSV (6 long-form conditions with segment counts and similarity scores)",
            "RQ4: Exp5 CSV (6 SpeechT5 conditions with spectrogram and duration data)",
            "RQ5: Exp6 CSV (12 MMS-TTS conditions with seed and speaking-rate variation)",
            "RQ6: Exp7 CSV (20 Bark rows across prompt styles, voice presets, and export audit)",
            "RQ7: Exp8 CSV (9 cross-model rows) plus consolidated cross-analysis",
        ],
    }

    with open(EXP_ROOT / "summary_statistics.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"summary_statistics.json written ({len(summary_stats)} top-level keys)")

    # ------------------------------------------------------------------
    # 3. Final tradeoff summary CSV
    # ------------------------------------------------------------------
    asr_summary = (
        exp2.groupby("workflow_name")
        .agg(
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_runtime_sec=("inference_time_sec", "mean"),
            sample_count=("wer", "count"),
        )
        .reset_index()
    )
    asr_summary.rename(columns={"workflow_name": "system_id"}, inplace=True)
    asr_summary["domain"] = "ASR"
    asr_summary["source_experiment"] = "exp2_shortform_asr"

    tts_summary = (
        exp8.groupby("system_id")
        .agg(
            mean_wer=("round_trip_wer", "mean"),
            mean_cer=("round_trip_cer", "mean"),
            mean_runtime_sec=("generation_time_sec", "mean"),
            sample_count=("round_trip_wer", "count"),
        )
        .reset_index()
    )
    tts_summary["domain"] = "TTS"
    tts_summary["source_experiment"] = "exp8_cross_tts"

    tradeoff = pd.concat([asr_summary, tts_summary], ignore_index=True)
    tradeoff = tradeoff[
        ["domain", "system_id", "mean_wer", "mean_cer", "mean_runtime_sec", "sample_count", "source_experiment"]
    ]
    tradeoff = tradeoff.sort_values(["domain", "mean_wer"])
    tradeoff.to_csv(EXP_ROOT / "final_tradeoff_summary.csv", index=False)
    print(f"final_tradeoff_summary.csv: {len(tradeoff)} rows")

    print("\nAll Step 7 summary artifacts created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
