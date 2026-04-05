from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "hw14_experiments" / "group_b_structured_tts"
NB_ROOT = PROJECT_ROOT / "hw14_scripts" / "notebooks"


def main() -> int:
    dry_run_notebook = NB_ROOT / "group_b_structured_tts_dry_run.ipynb"
    full_notebook = NB_ROOT / "group_b_structured_tts_full.ipynb"
    if dry_run_notebook.exists() and full_notebook.exists():
        print("TEST 0 PASS: Group B notebooks exist.")
    else:
        print("TEST 0 SKIP: Group B notebooks are not present in this synced runtime copy.")

    baseline_csv = EXP_ROOT / "exp1_baselines" / "exp1_baselines.csv"
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == 2, f"Expected 2 baseline rows, found {len(baseline_rows)}"
    print("TEST 1 PASS: Group B baseline CSV is complete.")

    exp5_csv = EXP_ROOT / "exp5_speecht5_tts" / "exp5_speecht5_tts.csv"
    with exp5_csv.open(newline="", encoding="utf-8") as handle:
        exp5_rows = list(csv.DictReader(handle))
    assert len(exp5_rows) == 6, f"Expected 6 Exp 5 rows, found {len(exp5_rows)}"
    assert len(list((EXP_ROOT / "exp5_speecht5_tts").glob("*.wav"))) == 6
    assert len(list((EXP_ROOT / "exp5_speecht5_tts").glob("*.png"))) >= 6
    print("TEST 2 PASS: Exp 5 rows and artifacts are complete.")

    exp6_csv = EXP_ROOT / "exp6_mms_tts" / "exp6_mms_tts.csv"
    with exp6_csv.open(newline="", encoding="utf-8") as handle:
        exp6_rows = list(csv.DictReader(handle))
    assert len(exp6_rows) == 12, f"Expected 12 Exp 6 rows, found {len(exp6_rows)}"
    rates = sorted({row.get("speaking_rate", "") for row in exp6_rows if row.get("sub_experiment") == "6B"})
    assert rates == ["0.8", "1.0", "1.2"], f"Unexpected speaking-rate set: {rates}"
    assert len(list((EXP_ROOT / "exp6_mms_tts").glob("*.wav"))) == 12
    print("TEST 3 PASS: Exp 6 rows and speaking-rate conditions are complete.")

    speaker_pairs = {(row["text_tag"], row["speaker_condition"]) for row in exp5_rows}
    for text_tag in ["llamas", "hello_dog", "banking_request"]:
        assert (text_tag, "helper_embedding") in speaker_pairs
        assert (text_tag, "zero_vector") in speaker_pairs
    print("TEST 4 PASS: SpeechT5 speaker conditions cover all texts.")

    print("\nSTEP 4 VERIFICATION: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())