"""Build leakage-safe SFT/Dr. GRPO JSONL from the private rolling dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_file, write_jsonl
from .schema import (
    GRADE_COLUMN,
    PASS_BASE_FEATURES,
    PASS_FEATURES,
    SYSTEM_PROMPT,
    TARGETS,
    add_derived_features,
    normalize_columns,
    record_prompt,
    schema_manifest,
    target_completion,
    validate_source_columns,
)
from .split import (
    assert_disjoint_grade_test,
    assert_disjoint_record_ids,
    split_common_indices,
    split_grade_names,
)

INVALID_GRADE_SENTINELS = frozenset({"steel_grade"})
EXPECTED_SOURCE_ROWS = 34_317
EXPECTED_UNIQUE_GRADES = 63
EXPECTED_RARE_GRADES = 19
EXPECTED_TEST_RECORDS_PER_PASS = 339


def load_source_frame(path: Path) -> Any:
    """Load one supported private table without guessing its sheet schema."""

    import pandas as pd

    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported data file type: {path.suffix}")


def _finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric value")
    return number


def robust_train_statistics(values: Any) -> dict[str, Any]:
    """Compute every reward/fallback statistic from training labels only."""

    targets = values.dropna().astype(float)
    if targets.empty:
        raise ValueError("training target is empty")
    center = float(targets.median())
    scale = float((targets - center).abs().median())
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = float(targets.std(ddof=0))
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    lower = float(targets.quantile(0.01))
    upper = float(targets.quantile(0.99))
    if not lower < upper:
        lower = float(targets.min())
        upper = float(targets.max())
    if not lower < upper:
        lower, upper = center - scale, center + scale
    return {
        "target_center": center,
        "target_scale": scale,
        "lower_bound": lower,
        "upper_bound": upper,
        "source": "train_labels_only",
        "scale_definition": "median_absolute_deviation_with_population_std_fallback",
        "range_definition": "train_label_q01_q99",
    }


def _record_id(source_digest: str, source_index: int, pass_name: str) -> str:
    material = f"{source_digest}:{source_index}:{pass_name}".encode()
    return hashlib.sha256(material).hexdigest()[:20]


def _conversation(prompt_text: str, target: float | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_completion(target)}],
            }
        )
    return messages


def _build_row(
    *,
    frame: Any,
    source_index: int,
    split_name: str,
    pass_name: str,
    source_digest: str,
    statistics: dict[str, Any],
    full_grade_counts: dict[str, int],
    train_grade_counts: Counter[str],
) -> dict[str, Any]:
    grade = str(frame.at[source_index, GRADE_COLUMN])
    target = _finite_float(frame.at[source_index, TARGETS[pass_name]])
    features = {
        column: _finite_float(frame.at[source_index, column])
        for column in PASS_FEATURES[pass_name]
    }
    prompt_text = record_prompt(pass_name, grade, features)
    return {
        "id": _record_id(source_digest, source_index, pass_name),
        "prompt": _conversation(prompt_text),
        "messages": _conversation(prompt_text, target),
        "target": target,
        "target_center": statistics["target_center"],
        "target_scale": statistics["target_scale"],
        "lower_bound": statistics["lower_bound"],
        "upper_bound": statistics["upper_bound"],
        "pass_name": pass_name,
        "steel_grade": grade,
        # This value is used by tail-aware reward training. It excludes the
        # validation portion and is zero for unseen test grades.
        "grade_frequency": int(train_grade_counts.get(grade, 0)),
        "full_grade_frequency": int(full_grade_counts[grade]),
        "split": split_name,
    }


def _contract_errors(
    *,
    source_rows: int,
    unique_grades: int,
    rare_grades: int,
    test_rows_by_pass: dict[str, int],
    expected_source_rows: int,
    expected_unique_grades: int,
    expected_rare_grades: int,
    expected_test_records_per_pass: int,
) -> list[str]:
    errors: list[str] = []
    expected_pairs = [
        ("source_rows", source_rows, expected_source_rows),
        ("unique_grades", unique_grades, expected_unique_grades),
        ("rare_grades", rare_grades, expected_rare_grades),
    ]
    for label, actual, expected in expected_pairs:
        if actual != expected:
            errors.append(f"{label}: expected {expected}, got {actual}")
    for pass_name, count in test_rows_by_pass.items():
        if count != expected_test_records_per_pass:
            errors.append(
                f"{pass_name} test records: expected {expected_test_records_per_pass}, got {count}"
            )
    return errors


def prepare_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    threshold: int,
    validation_fraction: float,
    seed: int,
    strict_contract: bool,
    expected_source_rows: int = EXPECTED_SOURCE_ROWS,
    expected_unique_grades: int = EXPECTED_UNIQUE_GRADES,
    expected_rare_grades: int = EXPECTED_RARE_GRADES,
    expected_test_records_per_pass: int = EXPECTED_TEST_RECORDS_PER_PASS,
) -> dict[str, Any]:
    """Prepare and persist all splits; returned report is safe to inspect."""

    import numpy as np
    import pandas as pd

    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"steel data file does not exist: {input_path}")
    frame = load_source_frame(input_path)
    frame.columns = normalize_columns(frame.columns)
    validate_source_columns(list(frame.columns))
    frame = frame.reset_index(drop=True)
    raw_source_rows = len(frame)
    frame[GRADE_COLUMN] = frame[GRADE_COLUMN].astype("string").str.strip()
    invalid_grade_mask = (
        frame[GRADE_COLUMN].str.casefold().isin(INVALID_GRADE_SENTINELS).fillna(False)
    )
    excluded_invalid_grade_rows = int(invalid_grade_mask.sum())
    # Preserve the original row indices so record IDs and every unaffected
    # train/validation sample stay byte-identical after removing sentinels.
    frame = frame.loc[~invalid_grade_mask].copy()

    numeric_source_columns = sorted(
        {
            *TARGETS.values(),
            *(column for features in PASS_BASE_FEATURES.values() for column in features),
        }
    )
    for column in numeric_source_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = add_derived_features(frame)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    grades = frame[GRADE_COLUMN].tolist()
    common_grades, rare_grades, full_grade_counts = split_grade_names(grades, threshold=threshold)
    common_indices = frame.index[frame[GRADE_COLUMN].isin(common_grades)].tolist()
    test_indices = sorted(frame.index[frame[GRADE_COLUMN].isin(rare_grades)].tolist())
    train_indices, validation_indices = split_common_indices(
        common_indices,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    source_digest = sha256_file(input_path)
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in split_indices
    }
    statistics_by_pass: dict[str, dict[str, Any]] = {}
    train_frequency_summary_by_pass: dict[str, dict[str, float]] = {}
    dropped_rows: dict[str, dict[str, int]] = {
        split_name: {} for split_name in split_indices
    }

    for pass_name, target_column in TARGETS.items():
        required = [target_column, *PASS_FEATURES[pass_name]]
        train_valid_mask = frame.loc[train_indices, required].notna().all(axis=1)
        pass_train_indices = train_valid_mask.index[train_valid_mask].tolist()
        pass_train_grade_counts: Counter[str] = Counter(
            str(value)
            for value in frame.loc[pass_train_indices, GRADE_COLUMN].tolist()
        )
        statistics = robust_train_statistics(
            frame.loc[pass_train_indices, target_column]
        )
        statistics_by_pass[pass_name] = statistics
        pass_frequencies = sorted(pass_train_grade_counts.values())
        train_frequency_summary_by_pass[pass_name] = {
            "grades": len(pass_frequencies),
            "minimum": min(pass_frequencies),
            "median": float(pd.Series(pass_frequencies).median()),
            "maximum": max(pass_frequencies),
        }
        for split_name, indices in split_indices.items():
            valid_mask = frame.loc[indices, required].notna().all(axis=1)
            valid_indices = valid_mask.index[valid_mask].tolist()
            dropped_rows[split_name][pass_name] = len(indices) - len(valid_indices)
            for source_index in valid_indices:
                rows_by_split[split_name].append(
                    _build_row(
                        frame=frame,
                        source_index=int(source_index),
                        split_name=split_name,
                        pass_name=pass_name,
                        source_digest=source_digest,
                        statistics=statistics,
                        full_grade_counts=full_grade_counts,
                        train_grade_counts=pass_train_grade_counts,
                    )
                )

    assert_disjoint_grade_test(
        rows_by_split["train"],
        rows_by_split["validation"],
        rows_by_split["test"],
    )
    assert_disjoint_record_ids(
        rows_by_split["train"],
        rows_by_split["validation"],
        rows_by_split["test"],
    )
    records_by_split_and_pass = {
        split_name: {
            pass_name: sum(row["pass_name"] == pass_name for row in rows)
            for pass_name in TARGETS
        }
        for split_name, rows in rows_by_split.items()
    }
    test_rows_by_pass = records_by_split_and_pass["test"]
    contract_errors = _contract_errors(
        source_rows=len(frame),
        unique_grades=len(full_grade_counts),
        rare_grades=len(rare_grades),
        test_rows_by_pass=test_rows_by_pass,
        expected_source_rows=expected_source_rows,
        expected_unique_grades=expected_unique_grades,
        expected_rare_grades=expected_rare_grades,
        expected_test_records_per_pass=expected_test_records_per_pass,
    )
    if strict_contract and contract_errors:
        raise ValueError("formal data contract failed: " + "; ".join(contract_errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split_name, rows in rows_by_split.items():
        path = output_dir / f"{split_name}.jsonl"
        count = write_jsonl(path, rows)
        files[path.name] = {"rows": count, "sha256": sha256_file(path)}

    schema = schema_manifest()
    schema["train_only_statistics"] = statistics_by_pass
    schema_path = output_dir / "schema.json"
    atomic_write_json(schema_path, schema)
    files[schema_path.name] = {"sha256": sha256_file(schema_path)}

    report: dict[str, Any] = {
        "schema_version": 3,
        "source": {
            "filename": input_path.name,
            "sha256": source_digest,
            "raw_rows": raw_source_rows,
            "rows": len(frame),
            "excluded_invalid_grade_rows": excluded_invalid_grade_rows,
            "invalid_grade_sentinels": sorted(INVALID_GRADE_SENTINELS),
        },
        "split_protocol": {
            "frequency_threshold": threshold,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "common_grade_rule": f"full frequency >= {threshold}",
            "test_grade_rule": f"full frequency < {threshold}",
            "test_labels_used_for_training_or_selection": False,
        },
        "unique_grades": len(full_grade_counts),
        "common_grades": len(common_grades),
        "rare_test_grades": len(rare_grades),
        "source_records_by_split": {
            split_name: len(indices) for split_name, indices in split_indices.items()
        },
        "records_by_split_and_pass": records_by_split_and_pass,
        "dropped_non_finite_rows": dropped_rows,
        "train_only_statistics": statistics_by_pass,
        "tail_weight_frequency_source": "per-pass valid training partition only",
        "train_frequency_summary_by_pass": train_frequency_summary_by_pass,
        "formal_contract": {
            "strict": strict_contract,
            "status": "passed" if not contract_errors else "mismatch_allowed",
            "errors": contract_errors,
            "expected": {
                "source_rows": expected_source_rows,
                "unique_grades": expected_unique_grades,
                "rare_grades": expected_rare_grades,
                "test_records_per_pass": expected_test_records_per_pass,
            },
        },
        "files": files,
    }
    report_path = output_dir / "data_report.json"
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=os.environ.get("STEEL_DATA_PATH"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help="Allow non-paper toy data; formal runs must omit this flag.",
    )
    args = parser.parse_args()
    if args.input is None:
        raise SystemExit("Pass --input or set STEEL_DATA_PATH")
    report = prepare_dataset(
        input_path=Path(args.input),
        output_dir=args.output_dir,
        threshold=args.threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        strict_contract=not args.allow_contract_mismatch,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
