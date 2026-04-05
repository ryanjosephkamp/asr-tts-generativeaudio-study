from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "hw14_experiments" / "group_a_asr"
NB_ROOT = PROJECT_ROOT / "hw14_scripts" / "notebooks"


def main() -> int:
    assert (NB_ROOT / "group_a_asr_dry_run.ipynb").exists()
    assert (NB_ROOT / "group_a_asr_full.ipynb").exists()
    print("TEST 0 PASS: Group A notebooks exist.")

    baseline_csv = EXP_ROOT / "exp1_baselines" / "exp1_baselines.csv"
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == 5, f"Expected 5 baseline rows, found {len(baseline_rows)}"
    assert any(
        "corrected" in row.get("notes", "").lower()
        for row in baseline_rows
        if row.get("script") == "GA20D"
    ), "GA20D corrected wrapper not labeled"
    print("TEST 1 PASS: Group A baseline CSV is complete.")

    exp2_csv = EXP_ROOT / "exp2_shortform_asr" / "exp2_shortform_asr.csv"
    with exp2_csv.open(newline="", encoding="utf-8") as handle:
        exp2_rows = list(csv.DictReader(handle))
    assert len(exp2_rows) == 96, f"Expected 96 Exp 2 rows, found {len(exp2_rows)}"
    workflow_counts: dict[str, int] = {}
    for row in exp2_rows:
        workflow_name = row["workflow_name"]
        workflow_counts[workflow_name] = workflow_counts.get(workflow_name, 0) + 1
    assert set(workflow_counts.values()) == {24}, f"Unexpected Exp 2 workflow counts: {workflow_counts}"
    print("TEST 2 PASS: Exp 2 row counts are correct.")

    exp3_csv = EXP_ROOT / "exp3_asr_sensitivity" / "exp3_asr_sensitivity.csv"
    with exp3_csv.open(newline="", encoding="utf-8") as handle:
        exp3_rows = list(csv.DictReader(handle))
    assert len(exp3_rows) == 114, f"Expected 114 Exp 3 rows, found {len(exp3_rows)}"
    sub_counts: dict[str, int] = {}
    for row in exp3_rows:
        sub_experiment = row["sub_experiment"]
        sub_counts[sub_experiment] = sub_counts.get(sub_experiment, 0) + 1
    assert sub_counts.get("3A") == 90 and sub_counts.get("3B") == 24, f"Unexpected Exp 3 counts: {sub_counts}"
    print("TEST 3 PASS: Exp 3 row counts are correct.")

    exp4_csv = EXP_ROOT / "exp4_longform_whisper" / "exp4_longform_whisper.csv"
    with exp4_csv.open(newline="", encoding="utf-8") as handle:
        exp4_rows = list(csv.DictReader(handle))
    assert len(exp4_rows) == 6, f"Expected 6 Exp 4 rows, found {len(exp4_rows)}"
    artifact_count = len(list((EXP_ROOT / "exp4_longform_whisper").glob("*.json")))
    assert artifact_count >= 6, f"Expected at least 6 long-form JSON artifacts, found {artifact_count}"
    print("TEST 4 PASS: Exp 4 outputs exist.")

    best_eval_path = EXP_ROOT / "full" / "best_asr_evaluator.json"
    assert best_eval_path.exists(), "best_asr_evaluator.json missing"
    with best_eval_path.open(encoding="utf-8") as handle:
        best_eval = json.load(handle)
    assert best_eval.get("workflow_name"), "workflow_name missing from best_asr_evaluator.json"
    print("TEST 5 PASS: Best ASR evaluator saved.")

    print("\nSTEP 3 VERIFICATION: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())