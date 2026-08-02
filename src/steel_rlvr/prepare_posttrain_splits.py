"""Create disjoint post-training and model-selection subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    verify_reported_file,
    write_jsonl,
)
from .sampling import balanced_indices


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if str(row["id"]) not in excluded_ids]
    indices = balanced_indices(
        [row["pass_name"] for row in eligible],
        count=count,
        seed=seed,
    )
    return [eligible[index] for index in indices]


def build_posttrain_splits(
    *,
    train_file: Path,
    validation_file: Path,
    data_report_file: Path,
    output_dir: Path,
    original_sft_train_samples: int,
    original_sft_validation_samples: int,
    pool_samples: int,
    preference_validation_samples: int,
    model_validation_samples: int,
    original_seed: int,
    posttrain_seed: int,
) -> dict[str, Any]:
    verify_reported_file(train_file, data_report_file)
    verify_reported_file(validation_file, data_report_file)
    train_rows = read_jsonl(train_file)
    validation_rows = read_jsonl(validation_file)

    original_train = _select_rows(
        train_rows,
        count=original_sft_train_samples,
        seed=original_seed,
        excluded_ids=set(),
    )
    original_validation = _select_rows(
        validation_rows,
        count=original_sft_validation_samples,
        seed=original_seed,
        excluded_ids=set(),
    )
    original_train_ids = {str(row["id"]) for row in original_train}
    original_validation_ids = {str(row["id"]) for row in original_validation}

    pool = _select_rows(
        train_rows,
        count=pool_samples,
        seed=posttrain_seed,
        excluded_ids=original_train_ids,
    )
    preference_validation = _select_rows(
        validation_rows,
        count=preference_validation_samples,
        seed=posttrain_seed + 1,
        excluded_ids=original_validation_ids,
    )
    preference_validation_ids = {
        str(row["id"]) for row in preference_validation
    }
    model_validation = _select_rows(
        validation_rows,
        count=model_validation_samples,
        seed=posttrain_seed + 2,
        excluded_ids=original_validation_ids | preference_validation_ids,
    )

    sets = {
        "original_sft_train": original_train_ids,
        "posttrain_pool": {str(row["id"]) for row in pool},
        "original_sft_validation": original_validation_ids,
        "preference_validation": preference_validation_ids,
        "model_validation": {str(row["id"]) for row in model_validation},
    }
    if sets["original_sft_train"] & sets["posttrain_pool"]:
        raise ValueError("post-training pool overlaps original SFT training rows")
    validation_names = (
        "original_sft_validation",
        "preference_validation",
        "model_validation",
    )
    for position, name in enumerate(validation_names):
        for other in validation_names[position + 1 :]:
            if sets[name] & sets[other]:
                raise ValueError(f"validation subsets overlap: {name} and {other}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "posttrain_pool.jsonl": pool,
        "posttrain_preference_validation.jsonl": preference_validation,
        "posttrain_model_validation.jsonl": model_validation,
    }
    files: dict[str, dict[str, Any]] = {}
    for filename, rows in outputs.items():
        path = output_dir / filename
        count = write_jsonl(path, rows)
        files[filename] = {"rows": count, "sha256": sha256_file(path)}
    report = {
        "schema_version": 1,
        "method": "disjoint balanced subsets for post-training and model selection",
        "source_files": {
            train_file.name: sha256_file(train_file),
            validation_file.name: sha256_file(validation_file),
        },
        "seeds": {"original_sft": original_seed, "posttrain": posttrain_seed},
        "excluded_original_sft": {
            "train_samples": len(original_train_ids),
            "validation_samples": len(original_validation_ids),
        },
        "pairwise_disjoint": True,
        "files": files,
    }
    atomic_write_json(output_dir / "posttrain_split_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file", type=Path, default=Path("data/processed/validation.jsonl")
    )
    parser.add_argument(
        "--data-report-file", type=Path, default=Path("data/processed/data_report.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pool-samples", type=int, default=9000)
    parser.add_argument("--preference-validation-samples", type=int, default=768)
    parser.add_argument("--model-validation-samples", type=int, default=1536)
    parser.add_argument("--original-seed", type=int, default=42)
    parser.add_argument("--posttrain-seed", type=int, default=137)
    args = parser.parse_args()
    report = build_posttrain_splits(
        train_file=args.train_file,
        validation_file=args.validation_file,
        data_report_file=args.data_report_file,
        output_dir=args.output_dir,
        original_sft_train_samples=3000,
        original_sft_validation_samples=768,
        pool_samples=args.pool_samples,
        preference_validation_samples=args.preference_validation_samples,
        model_validation_samples=args.model_validation_samples,
        original_seed=args.original_seed,
        posttrain_seed=args.posttrain_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
