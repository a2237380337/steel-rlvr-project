"""Select a checkpoint on common-grade validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_file


def select_checkpoint(
    summaries: dict[str, dict[str, Any]],
    *,
    expected_validation_hash: str,
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    sample_hashes = {
        str(summary["sample_ids_sha256"]) for summary in summaries.values()
    }
    if len(sample_hashes) != 1:
        raise ValueError("candidate checkpoints were not evaluated on identical sample ids")
    for label, summary in summaries.items():
        if summary["split_sha256"] != expected_validation_hash:
            raise ValueError(f"{label} was not evaluated on the declared validation split")
        metrics = summary["metrics"]
        ranking.append(
            {
                "label": label,
                "model_or_checkpoint": summary["model_or_checkpoint"],
                "macro_mae": float(metrics["macro_mae"]),
                "mae": float(metrics["mae"]),
                "worst_group_mae": float(metrics["worst_group_mae"]),
                "strict_json_rate": float(metrics["strict_json_rate"]),
            }
        )
    if not ranking:
        raise ValueError("no checkpoint summaries were supplied")
    ranking.sort(
        key=lambda row: (
            row["macro_mae"],
            row["mae"],
            row["worst_group_mae"],
            -row["strict_json_rate"],
            row["label"],
        )
    )
    return {
        "schema_version": 1,
        "selection_protocol": {
            "split": "common-grade validation only",
            "split_sha256": expected_validation_hash,
            "sample_ids_sha256": next(iter(sample_hashes)),
            "formal_low_frequency_test_metrics_used": False,
            "primary_metric": "macro_mae_asc",
            "tie_breakers": [
                "mae_asc",
                "worst_group_mae_asc",
                "strict_json_rate_desc",
                "label_asc",
            ],
        },
        "selected": ranking[0],
        "ranking": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="LABEL=PATH; repeat for every candidate checkpoint.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries: dict[str, dict[str, Any]] = {}
    for item in args.summary:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"invalid --summary value: {item!r}")
        summaries[label] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    result = select_checkpoint(
        summaries,
        expected_validation_hash=sha256_file(args.validation_file),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
