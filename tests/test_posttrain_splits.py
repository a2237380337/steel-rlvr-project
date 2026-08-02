import json
from pathlib import Path

from steel_rlvr.io_utils import atomic_write_json, sha256_file, write_jsonl
from steel_rlvr.prepare_posttrain_splits import build_posttrain_splits


def _rows(prefix: str, per_pass: int) -> list[dict]:
    return [
        {"id": f"{prefix}-{pass_name}-{index}", "pass_name": pass_name}
        for pass_name in ("Pass1", "Pass2", "Pass3")
        for index in range(per_pass)
    ]


def test_posttrain_subsets_are_disjoint_and_balanced(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, _rows("train", 12))
    write_jsonl(validation_path, _rows("validation", 14))
    data_report = tmp_path / "data_report.json"
    atomic_write_json(
        data_report,
        {
            "files": {
                "train.jsonl": {"sha256": sha256_file(train_path)},
                "validation.jsonl": {"sha256": sha256_file(validation_path)},
            }
        },
    )

    report = build_posttrain_splits(
        train_file=train_path,
        validation_file=validation_path,
        data_report_file=data_report,
        output_dir=tmp_path,
        original_sft_train_samples=6,
        original_sft_validation_samples=6,
        pool_samples=12,
        preference_validation_samples=6,
        model_validation_samples=9,
        original_seed=1,
        posttrain_seed=7,
    )

    assert report["pairwise_disjoint"] is True
    for filename, expected in (
        ("posttrain_pool.jsonl", 12),
        ("posttrain_preference_validation.jsonl", 6),
        ("posttrain_model_validation.jsonl", 9),
    ):
        rows = [json.loads(line) for line in (tmp_path / filename).read_text().splitlines()]
        assert len(rows) == expected
        counts = {name: 0 for name in ("Pass1", "Pass2", "Pass3")}
        for row in rows:
            counts[row["pass_name"]] += 1
        assert max(counts.values()) - min(counts.values()) <= 1
