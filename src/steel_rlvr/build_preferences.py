"""Build DPO chosen/rejected pairs from leakage-safe train and validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    verify_reported_file,
    write_jsonl,
)
from .preference import build_preference_pair


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["absolute_margin"]) for row in rows]
    normalized = [float(row["normalized_margin"]) for row in rows]
    return {
        "pairs": len(rows),
        "chosen_rejected_identical": sum(row["chosen"] == row["rejected"] for row in rows),
        "absolute_margin_mean": mean(margins),
        "absolute_margin_min": min(margins),
        "absolute_margin_max": max(margins),
        "normalized_margin_values": sorted(set(normalized)),
        "all_completions_strict_json": all(
            text.startswith('{"leveling":') and text.endswith("}")
            for row in rows
            for text in (row["chosen"], row["rejected"])
        ),
    }


def build_preference_files(
    *,
    train_file: Path,
    validation_file: Path,
    data_report_file: Path,
    output_dir: Path,
    seed: int,
    margin_multipliers: list[float],
) -> dict[str, Any]:
    verify_reported_file(train_file, data_report_file)
    verify_reported_file(validation_file, data_report_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, source_path in (
        ("train", train_file),
        ("validation", validation_file),
    ):
        output_rows[split_name] = [
            build_preference_pair(
                row,
                seed=seed,
                margin_multipliers=margin_multipliers,
            )
            for row in read_jsonl(source_path)
        ]
    files: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for split_name, rows in output_rows.items():
        path = output_dir / f"preferences_{split_name}.jsonl"
        count = write_jsonl(path, rows)
        files[path.name] = {"rows": count, "sha256": sha256_file(path)}
        summaries[split_name] = _summary(rows)
    report = {
        "schema_version": 1,
        "method": "observed target preferred over deterministic same-format numeric perturbation",
        "seed": seed,
        "margin_multipliers_in_train_mad_units": margin_multipliers,
        "test_labels_used": False,
        "source_files": {
            train_file.name: sha256_file(train_file),
            validation_file.name: sha256_file(validation_file),
        },
        "summaries": summaries,
        "files": files,
    }
    atomic_write_json(output_dir / "preference_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--data-report-file",
        type=Path,
        default=Path("data/processed/data_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin-multipliers", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    args = parser.parse_args()
    report = build_preference_files(
        train_file=args.train_file,
        validation_file=args.validation_file,
        data_report_file=args.data_report_file,
        output_dir=args.output_dir,
        seed=args.seed,
        margin_multipliers=list(args.margin_multipliers),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
