import json

import pandas as pd

from steel_rlvr.prepare_data import prepare_dataset
from steel_rlvr.schema import GRADE_COLUMN, PASS_BASE_FEATURES, TARGETS


def _toy_frame() -> pd.DataFrame:
    rows = []
    grades = ["COMMON_A"] * 4 + ["COMMON_B"] * 4 + ["RARE_C"] * 2
    for index, grade in enumerate(grades):
        row = {GRADE_COLUMN: grade}
        for column in {
            field
            for fields in PASS_BASE_FEATURES.values()
            for field in fields
        }:
            row[column] = float(index + 1)
        for offset, column in enumerate(TARGETS.values()):
            row[column] = (
                1_000_000.0 + offset
                if grade == "RARE_C"
                else float(index + offset) / 10.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_dataset_keeps_rare_grades_out_of_training(tmp_path) -> None:
    source = tmp_path / "toy.csv"
    output = tmp_path / "processed"
    frame = _toy_frame()
    sentinel = frame.iloc[0].copy()
    sentinel[GRADE_COLUMN] = " Steel_Grade "
    pd.concat([frame, sentinel.to_frame().T], ignore_index=True).to_csv(
        source,
        index=False,
    )

    report = prepare_dataset(
        input_path=source,
        output_dir=output,
        threshold=3,
        validation_fraction=0.25,
        seed=42,
        strict_contract=False,
    )

    train = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["steel_grade"] for row in train}.isdisjoint(
        {row["steel_grade"] for row in test}
    )
    assert {row["steel_grade"] for row in test} == {"RARE_C"}
    assert all(row["grade_frequency"] > 0 for row in train)
    assert all(row["grade_frequency"] == 0 for row in test)
    assert all(
        stats["upper_bound"] < 10.0
        for stats in report["train_only_statistics"].values()
    )
    assert report["formal_contract"]["status"] == "mismatch_allowed"
    assert report["split_protocol"]["test_labels_used_for_training_or_selection"] is False
    assert report["source"]["raw_rows"] == 11
    assert report["source"]["rows"] == 10
    assert report["source"]["excluded_invalid_grade_rows"] == 1
