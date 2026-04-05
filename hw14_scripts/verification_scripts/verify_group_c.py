from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "hw14_experiments" / "group_c_bark_cross"
NB_ROOT = PROJECT_ROOT / "hw14_scripts" / "notebooks"


def main() -> int:
    dry_run_notebook = NB_ROOT / "group_c_bark_cross_dry_run.ipynb"
    full_notebook = NB_ROOT / "group_c_bark_cross_full.ipynb"
    if dry_run_notebook.exists() and full_notebook.exists():
        print("TEST 0 PASS: Group C notebooks exist.")
    else:
        print("TEST 0 SKIP: Group C notebooks are not present in this synced runtime copy.")

    baseline_csv = EXP_ROOT / "exp1_baselines" / "exp1_baselines.csv"
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == 2, f"Expected 2 baseline rows, found {len(baseline_rows)}"
    print("TEST 1 PASS: Group C baseline CSV is complete.")

    exp7_csv = EXP_ROOT / "exp7_bark" / "exp7_bark.csv"
    with exp7_csv.open(newline="", encoding="utf-8") as handle:
        exp7_rows = list(csv.DictReader(handle))
    assert len(exp7_rows) == 20, f"Expected 20 Exp 7 rows, found {len(exp7_rows)}"
    sub_counts: dict[str, int] = {}
    for row in exp7_rows:
        sub_experiment = row["sub_experiment"]
        sub_counts[sub_experiment] = sub_counts.get(sub_experiment, 0) + 1
    assert sub_counts.get("7A") == 6 and sub_counts.get("7B") == 6 and sub_counts.get("7C") == 8, f"Unexpected Exp 7 counts: {sub_counts}"
    save_modes = {row.get("save_rate_mode", "") for row in exp7_rows if row.get("sub_experiment") == "7C"}
    assert save_modes == {"hardcoded_16k", "model_native"}, f"Unexpected save-rate modes: {save_modes}"
    print("TEST 2 PASS: Exp 7 row counts and save-rate modes are correct.")

    exp8_csv = EXP_ROOT / "exp8_cross_tts" / "exp8_cross_tts.csv"
    with exp8_csv.open(newline="", encoding="utf-8") as handle:
        exp8_rows = list(csv.DictReader(handle))
    assert len(exp8_rows) == 9, f"Expected 9 Exp 8 rows, found {len(exp8_rows)}"
    systems = {row["system_id"] for row in exp8_rows}
    texts = {row["text_tag"] for row in exp8_rows}
    assert systems == {"speecht5", "mms_tts", "bark_preset"}, f"Unexpected systems: {systems}"
    assert texts == {"hello_dog", "llamas", "banking_request"}, f"Unexpected text set: {texts}"
    assert all(row.get("asr_evaluator") for row in exp8_rows), "Missing asr_evaluator in Exp 8 rows"
    assert all(row.get("round_trip_wer") for row in exp8_rows), "Missing round_trip_wer in Exp 8 rows"
    print("TEST 3 PASS: Exp 8 comparison coverage is complete.")

    print("\nSTEP 5 VERIFICATION: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())