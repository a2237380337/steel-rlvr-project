"""Combine Base/SFT/DPO/frequency-aware DPO evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json

REQUIRED_LABELS = ("base", "sft", "dpo", "frequency_aware_dpo")


def build_matrix(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("no evaluation summaries were supplied")
    split_hashes = {summary["split_sha256"] for summary in summaries.values()}
    if len(split_hashes) != 1:
        raise ValueError("evaluations do not use the same split artifact")
    sample_hashes = {summary["sample_ids_sha256"] for summary in summaries.values()}
    if len(sample_hashes) != 1:
        raise ValueError("evaluations do not use the same ordered sample ids")
    sample_counts = {int(summary["sample_count"]) for summary in summaries.values()}
    if len(sample_counts) != 1:
        raise ValueError("evaluations do not use the same sample count")
    base_revisions = {summary["base_revision"] for summary in summaries.values()}
    if len(base_revisions) != 1:
        raise ValueError("evaluations do not resolve to the same base model revision")
    source_hashes = {summary["source_tree_sha256"] for summary in summaries.values()}
    if len(source_hashes) != 1:
        raise ValueError("evaluation source code changed between model runs")
    rows: dict[str, Any] = {}
    for label, summary in summaries.items():
        metrics = summary["metrics"]
        rows[label] = {
            "model_or_checkpoint": summary["model_or_checkpoint"],
            "base_revision": summary["base_revision"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "macro_mae": metrics["macro_mae"],
            "worst_group_mae": metrics["worst_group_mae"],
            "strict_json_rate": metrics["strict_json_rate"],
            "valid_value_rate": metrics["valid_value_rate"],
            "passes": {
                pass_name: {
                    "mae": pass_metrics["mae"],
                    "r2": pass_metrics["r2"],
                    "macro_mae": pass_metrics["macro_mae"],
                    "worst_group_mae": pass_metrics["worst_group_mae"],
                }
                for pass_name, pass_metrics in metrics["passes"].items()
            },
            "peak_memory_mib": summary["hardware"]["peak_memory_mib"],
        }
    comparisons: dict[str, Any] = {}
    if "dpo" in rows and "frequency_aware_dpo" in rows:
        standard = rows["dpo"]
        frequency_aware = rows["frequency_aware_dpo"]
        comparisons["frequency_aware_dpo_minus_dpo"] = {
            "mae": frequency_aware["mae"] - standard["mae"],
            "macro_mae": frequency_aware["macro_mae"] - standard["macro_mae"],
            "worst_group_mae": (
                frequency_aware["worst_group_mae"] - standard["worst_group_mae"]
            ),
            "strict_json_rate": (
                frequency_aware["strict_json_rate"] - standard["strict_json_rate"]
            ),
        }
    return {
        "schema_version": 1,
        "split_sha256": next(iter(split_hashes)),
        "sample_ids_sha256": next(iter(sample_hashes)),
        "sample_count": next(iter(sample_counts)),
        "base_revision": next(iter(base_revisions)),
        "source_tree_sha256": next(iter(source_hashes)),
        "complete": all(label in summaries for label in REQUIRED_LABELS),
        "missing_labels": [label for label in REQUIRED_LABELS if label not in summaries],
        "models": rows,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="LABEL=PATH; repeat once per evaluated model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summaries: dict[str, dict[str, Any]] = {}
    for item in args.summary:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"invalid --summary value: {item!r}")
        path = Path(raw_path)
        summaries[label] = json.loads(path.read_text(encoding="utf-8"))
    result = build_matrix(summaries)
    if args.require_complete and not result["complete"]:
        raise ValueError(f"missing formal evaluations: {result['missing_labels']}")
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
