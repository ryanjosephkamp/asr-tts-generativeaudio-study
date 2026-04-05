from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FIGS = PROJECT_ROOT / "hw14_reports" / "figures"
LATEX_FIGS = PROJECT_ROOT / "hw14_reports" / "latex" / "figures"

EXPECTED_FIGURES = [
    "exp1_baseline_summary.png",
    "asr_family_wer_bar.png",
    "asr_family_cer_bar.png",
    "asr_family_runtime.png",
    "whisper_interface_delta.png",
    "asr_sampling_rate_curve.png",
    "asr_sampling_rate_runtime.png",
    "ga20c_cleanup_delta.png",
    "longform_segment_counts.png",
    "longform_runtime.png",
    "longform_similarity_heatmap.png",
    "speecht5_spectrogram_panel.png",
    "speecht5_duration_summary.png",
    "mms_seed_duration.png",
    "mms_speaking_rate_duration.png",
    "bark_prompt_style_grid.png",
    "bark_voice_preset_summary.png",
    "bark_sample_rate_audit.png",
    "tts_roundtrip_wer.png",
    "tts_roundtrip_cer.png",
    "tts_runtime_comparison.png",
    "tts_control_tradeoff_table.png",
]

KEY_FIGURES = [
    "exp1_baseline_summary.png",
    "asr_family_wer_bar.png",
    "asr_sampling_rate_curve.png",
    "longform_similarity_heatmap.png",
    "speecht5_spectrogram_panel.png",
    "mms_speaking_rate_duration.png",
    "bark_sample_rate_audit.png",
    "tts_roundtrip_wer.png",
    "tts_control_tradeoff_table.png",
]


def main() -> int:
    report_figs = list(REPORT_FIGS.glob("*.png"))
    latex_figs = list(LATEX_FIGS.glob("*.png"))

    assert len(report_figs) >= 20, f"Expected at least 20 report figures, found {len(report_figs)}"
    assert len(latex_figs) >= 20, f"Expected at least 20 LaTeX figures, found {len(latex_figs)}"

    for fig_name in EXPECTED_FIGURES:
        report_path = REPORT_FIGS / fig_name
        latex_path = LATEX_FIGS / fig_name
        assert report_path.exists(), f"Missing report figure: {fig_name}"
        assert latex_path.exists(), f"Missing LaTeX figure: {fig_name}"
        assert report_path.stat().st_size > 5_000, f"Suspiciously small report figure: {fig_name}"
        assert latex_path.stat().st_size > 5_000, f"Suspiciously small LaTeX figure: {fig_name}"

    for fig_name in KEY_FIGURES:
        assert (REPORT_FIGS / fig_name).exists(), f"Missing key figure: {fig_name}"

    print(f"Verified {len(EXPECTED_FIGURES)} expected figures in {REPORT_FIGS}")
    print(f"Verified {len(EXPECTED_FIGURES)} expected figures in {LATEX_FIGS}")
    print("STEP 6 VERIFICATION: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())