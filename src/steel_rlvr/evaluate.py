"""Run deterministic steel prediction and save leakage-safe metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .io_utils import (
    atomic_write_json,
    sha256_file,
    source_tree_sha256,
    verify_reported_file,
)
from .metrics import group_mae, mean_absolute_error, pass_grade_keys, r2_score
from .output_parsing import parse_prediction_detail
from .policy_loading import load_text_policy
from .sampling import select_balanced
from .training_logging import require_fresh_output_dir

LOGGER = logging.getLogger("steel_rlvr.evaluation")


def resolve_effective_config(
    config: dict[str, Any],
    model_override: str | None,
    output_dir_override: Path | None,
) -> dict[str, Any]:
    effective = dict(config)
    if model_override is not None:
        effective["model_or_checkpoint"] = model_override
    if output_dir_override is not None:
        effective["output_dir"] = str(output_dir_override)
    return effective


def _metric_block(rows: list[dict[str, Any]], *, grouped_by_pass: bool) -> dict[str, Any]:
    targets = [float(row["target"]) for row in rows]
    predictions = [
        float(row["prediction"])
        if row.get("prediction") is not None
        else float(row["target_center"])
        for row in rows
    ]
    valid_pairs = [
        (float(row["target"]), float(row["prediction"]))
        for row in rows
        if row.get("prediction") is not None
    ]
    if grouped_by_pass:
        groups = pass_grade_keys(
            [str(row["pass_name"]) for row in rows],
            [str(row["steel_grade"]) for row in rows],
        )
    else:
        groups = [str(row["steel_grade"]) for row in rows]
    grouped = group_mae(groups, targets, predictions)
    strict_count = sum(
        bool(row.get("strict_json", row.get("prediction") is not None))
        for row in rows
    )
    numeric_rows = [row for row in rows if row.get("prediction") is not None]
    physical_violations = sum(
        not (
            float(row["lower_bound"])
            <= float(row["prediction"])
            <= float(row["upper_bound"])
        )
        for row in numeric_rows
        if "lower_bound" in row and "upper_bound" in row
    )
    baseline_predictions = [float(row["target_center"]) for row in rows]
    mae = mean_absolute_error(targets, predictions)
    baseline_mae = mean_absolute_error(targets, baseline_predictions)
    return {
        "samples": len(rows),
        "valid_value_rate": len(valid_pairs) / len(rows),
        "strict_json_rate": strict_count / len(rows),
        "fallback_rate": (len(rows) - len(valid_pairs)) / len(rows),
        "physical_violation_rate_among_numeric": (
            physical_violations / len(numeric_rows) if numeric_rows else None
        ),
        "mae": mae,
        "r2": r2_score(targets, predictions),
        "valid_only_mae": (
            mean_absolute_error(
                [target for target, _ in valid_pairs],
                [prediction for _, prediction in valid_pairs],
            )
            if valid_pairs
            else None
        ),
        "macro_mae": mean(grouped.values()),
        "worst_group_mae": max(grouped.values()),
        "train_median_baseline_mae": baseline_mae,
        "mae_improvement_over_train_median": baseline_mae - mae,
        "groups": grouped,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score invalid generations with the train-only median instead of dropping them."""

    if not rows:
        raise ValueError("cannot summarize an empty prediction file")
    by_pass: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pass[str(row["pass_name"])].append(row)
    summary = _metric_block(rows, grouped_by_pass=True)
    summary["passes"] = {
        pass_name: _metric_block(subset, grouped_by_pass=False)
        for pass_name, subset in sorted(by_pass.items())
    }
    summary["failure_counts"] = dict(
        sorted(Counter(str(row.get("parse_status", "unknown")) for row in rows).items())
    )
    token_counts = [
        int(row["completion_tokens"])
        for row in rows
        if row.get("completion_tokens") is not None
    ]
    summary["generation"] = {
        "mean_completion_tokens": mean(token_counts) if token_counts else None,
        "max_completion_tokens": max(token_counts) if token_counts else None,
        "total_completion_tokens": sum(token_counts),
    }
    summary["invalid_output_policy"] = (
        "If no finite leveling value can be parsed, score the train-only target median "
        "stored in target_center. Never drop an invalid sample from main MAE/R2."
    )
    summary["macro_definition"] = "unweighted mean MAE across pass_name::steel_grade groups"
    return summary


def _trim_completion_ids(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids
    return token_ids[: index + 1]


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[list[dict[str, Any]]],
    max_new_tokens: int,
) -> list[tuple[str, int]]:
    import torch

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=False,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    inputs = {name: value.to("cuda") for name, value in inputs.items()}
    prompt_width = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    generated_batch = outputs[:, prompt_width:].tolist()
    results: list[tuple[str, int]] = []
    for generated in generated_batch:
        trimmed = _trim_completion_ids(generated, tokenizer.eos_token_id)
        results.append((tokenizer.decode(trimmed, skip_special_tokens=True), len(trimmed)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", help="Override model_or_checkpoint from YAML.")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir from YAML.")
    args = parser.parse_args()
    config = resolve_effective_config(
        yaml.safe_load(args.config.read_text(encoding="utf-8")),
        args.model,
        args.output_dir,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Evaluation requires ROCm/HIP PyTorch")
    split_file = Path(config["split_file"])
    verify_reported_file(split_file, Path(config["data_report_file"]))
    output_dir = Path(config["output_dir"])
    require_fresh_output_dir(output_dir)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    project_root = Path(__file__).resolve().parents[2]
    project_source_hash = source_tree_sha256(project_root)
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = output_dir / "evaluation_manifest.json"
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "config_sha256": config_hash,
        "source_tree_sha256": project_source_hash,
        "resolved_config_sha256": sha256_file(resolved_config_path),
        "resolved_config": str(resolved_config_path.resolve()),
        "split_sha256": sha256_file(split_file),
        "model_or_checkpoint": config["model_or_checkpoint"],
    }
    atomic_write_json(manifest_path, manifest)

    try:
        dataset = load_dataset("json", data_files=str(split_file), split="train")
        requested = int(config["max_samples"])
        if bool(config.get("balanced_by_pass", True)):
            dataset = select_balanced(
                dataset,
                count=requested,
                seed=int(config["seed"]),
            )
        else:
            if requested <= 0 or len(dataset) < requested:
                raise ValueError(
                    f"evaluation requested {requested} rows but split contains {len(dataset)}"
                )
            dataset = dataset.select(range(requested))
        sample_ids_sha256 = hashlib.sha256(
            "\n".join(str(value) for value in dataset["id"]).encode("utf-8")
        ).hexdigest()
        manifest["sample_ids_sha256"] = sample_ids_sha256
        manifest["sample_count"] = len(dataset)
        atomic_write_json(manifest_path, manifest)
        model, tokenizer, base_name, base_revision, _ = load_text_policy(
            str(config["model_or_checkpoint"]),
            config.get("model_revision"),
            trainable_adapter=False,
            mixed_precision=config["mixed_precision"],
            local_files_only=bool(config["local_files_only"]),
        )
        model.to("cuda")
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        rows: list[dict[str, Any]] = []
        predictions_path = output_dir / "predictions.jsonl"
        batch_size = int(config["batch_size"])
        logging_interval = int(config["logging_interval"])
        if batch_size <= 0 or logging_interval <= 0:
            raise ValueError("batch_size and logging_interval must be positive")
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for batch_start in range(0, len(dataset), batch_size):
                examples = [
                    dataset[index]
                    for index in range(
                        batch_start,
                        min(batch_start + batch_size, len(dataset)),
                    )
                ]
                generated = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[example["prompt"] for example in examples],
                    max_new_tokens=int(config["max_new_tokens"]),
                )
                for example, (completion, completion_tokens) in zip(
                    examples,
                    generated,
                    strict=True,
                ):
                    parsed = parse_prediction_detail(completion)
                    row = {
                        "id": example["id"],
                        "pass_name": example["pass_name"],
                        "steel_grade": example["steel_grade"],
                        "target": float(example["target"]),
                        "target_center": float(example["target_center"]),
                        "lower_bound": float(example["lower_bound"]),
                        "upper_bound": float(example["upper_bound"]),
                        "prediction": parsed.value,
                        "scored_prediction": (
                            parsed.value
                            if parsed.value is not None
                            else float(example["target_center"])
                        ),
                        "parse_status": parsed.status,
                        "strict_json": parsed.strict_json,
                        "physical_range_valid": (
                            parsed.value is not None
                            and float(example["lower_bound"])
                            <= parsed.value
                            <= float(example["upper_bound"])
                        ),
                        "completion": completion,
                        "completion_tokens": completion_tokens,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed = batch_start + len(examples)
                if completed % logging_interval == 0 or completed == len(dataset):
                    LOGGER.info("evaluation_progress=%d/%d", completed, len(dataset))

        torch.cuda.synchronize()
        metrics = summarize_rows(rows)
        summary = {
            "schema_version": 2,
            "model_or_checkpoint": config["model_or_checkpoint"],
            "base_model": base_name,
            "base_revision": base_revision,
            "source_tree_sha256": project_source_hash,
            "evaluated_split": str(split_file),
            "split_sha256": sha256_file(split_file),
            "sample_ids_sha256": sample_ids_sha256,
            "sample_count": len(dataset),
            "metrics": metrics,
            "hardware": {
                "device": torch.cuda.get_device_name(0),
                "hip": torch.version.hip,
                "peak_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2,
                    2,
                ),
            },
            "config": config,
        }
        atomic_write_json(output_dir / "summary.json", summary)
        manifest["status"] = "completed"
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.monotonic() - started, 3)
        manifest["summary"] = str((output_dir / "summary.json").resolve())
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.monotonic() - started, 3)
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
