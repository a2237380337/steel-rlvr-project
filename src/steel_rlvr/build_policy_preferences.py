"""Build policy-anchored hard negatives and a difficulty-selected train set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from .output_parsing import parse_prediction_detail
from .policy_loading import load_text_policy
from .preference import preference_indices, tail_sampling_weight
from .schema import target_completion


def _fallback_sign(identifier: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return -1.0 if digest[0] % 2 == 0 else 1.0


def policy_anchored_pair(
    row: dict[str, Any],
    completion: str,
    *,
    seed: int,
    minimum_margin_mad: float,
    maximum_margin_mad: float,
) -> dict[str, Any]:
    target = float(row["target"])
    scale = float(row["target_scale"])
    if not math.isfinite(target) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid target scale for {row['id']}")
    parsed = parse_prediction_detail(completion)
    if parsed.value is None:
        raw_error_mad = maximum_margin_mad
        direction = _fallback_sign(str(row["id"]), seed)
    else:
        delta = float(parsed.value) - target
        raw_error_mad = abs(delta) / scale
        direction = (
            math.copysign(1.0, delta)
            if abs(delta) > 1e-12
            else _fallback_sign(str(row["id"]), seed)
        )
    margin_mad = min(
        maximum_margin_mad,
        max(minimum_margin_mad, raw_error_mad),
    )
    rejected_value = round(target + direction * margin_mad * scale, 6)
    chosen = target_completion(target)
    rejected = target_completion(rejected_value)
    if chosen == rejected:
        rejected_value = round(target - direction * minimum_margin_mad * scale, 6)
        rejected = target_completion(rejected_value)
    if chosen == rejected:
        raise ValueError(f"could not construct a distinct hard negative: {row['id']}")
    grade_frequency = int(row["grade_frequency"])
    return {
        "id": str(row["id"]),
        "prompt": row["prompt"],
        "chosen": chosen,
        "rejected": rejected,
        "pass_name": str(row["pass_name"]),
        "steel_grade": str(row["steel_grade"]),
        "grade_frequency": grade_frequency,
        "tail_sampling_weight": tail_sampling_weight(grade_frequency),
        "policy_prediction": parsed.value,
        "policy_parse_status": parsed.status,
        "policy_error_mad": raw_error_mad,
        "hard_margin_mad": margin_mad,
        "difficulty_sampling_weight": 1.0 + min(raw_error_mad, 4.0),
    }


def _generate_completions(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    label: str,
) -> list[str]:
    import torch

    completions: list[str] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [row["prompt"] for row in batch]
            inputs = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                enable_thinking=False,
            )
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            prompt_width = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            generated = outputs[:, prompt_width:].tolist()
            completions.extend(
                tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in generated
            )
            completed = min(start + len(batch), len(rows))
            if completed % 300 == 0 or completed == len(rows):
                print(f"{label}_generation={completed}/{len(rows)}", flush=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    return completions


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["policy_error_mad"]) for row in rows]
    margins = [float(row["hard_margin_mad"]) for row in rows]
    return {
        "pairs": len(rows),
        "policy_numeric_rate": sum(row["policy_prediction"] is not None for row in rows)
        / len(rows),
        "policy_error_mad_mean": mean(errors),
        "policy_error_mad_max": max(errors),
        "hard_margin_mad_mean": mean(margins),
        "hard_margin_mad_min": min(margins),
        "hard_margin_mad_max": max(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="artifacts/checkpoints/sft-main")
    parser.add_argument(
        "--model-revision", default="a9a407bcae463285164cc9133995c515379cebe5"
    )
    parser.add_argument("--pool-file", type=Path, default=Path("data/processed/posttrain_pool.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/posttrain_preference_validation.jsonl"),
    )
    parser.add_argument(
        "--split-report-file",
        type=Path,
        default=Path("data/processed/posttrain_split_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--selected-train-samples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--minimum-margin-mad", type=float, default=0.1)
    parser.add_argument("--maximum-margin-mad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=149)
    args = parser.parse_args()

    verify_reported_file(args.pool_file, args.split_report_file)
    verify_reported_file(args.validation_file, args.split_report_file)
    pool_rows = read_jsonl(args.pool_file)
    validation_rows = read_jsonl(args.validation_file)

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("policy preference generation requires ROCm/HIP PyTorch")
    model, tokenizer, base_name, base_revision, is_adapter = load_text_policy(
        args.policy,
        args.model_revision,
        trainable_adapter=False,
        mixed_precision="bf16",
        local_files_only=True,
    )
    if not is_adapter:
        raise ValueError("policy-anchored preferences require an SFT adapter")
    model.to("cuda")
    model.eval()
    pool_completions = _generate_completions(
        rows=pool_rows,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        label="pool",
    )
    validation_completions = _generate_completions(
        rows=validation_rows,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        label="validation",
    )
    pool_pairs = [
        policy_anchored_pair(
            row,
            completion,
            seed=args.seed,
            minimum_margin_mad=args.minimum_margin_mad,
            maximum_margin_mad=args.maximum_margin_mad,
        )
        for row, completion in zip(pool_rows, pool_completions, strict=True)
    ]
    validation_pairs = [
        policy_anchored_pair(
            row,
            completion,
            seed=args.seed,
            minimum_margin_mad=args.minimum_margin_mad,
            maximum_margin_mad=args.maximum_margin_mad,
        )
        for row, completion in zip(
            validation_rows, validation_completions, strict=True
        )
    ]
    selected_indices = preference_indices(
        pool_pairs,
        count=args.selected_train_samples,
        seed=args.seed,
        tail_aware=False,
        weight_column="difficulty_sampling_weight",
    )
    selected_pairs = [pool_pairs[index] for index in selected_indices]
    pool_by_id = {str(row["id"]): row for row in pool_rows}
    selected_raw = [pool_by_id[str(row["id"])] for row in selected_pairs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "policy_preferences_pool.jsonl": pool_pairs,
        "posttrain_hard_preferences_train.jsonl": selected_pairs,
        "posttrain_hard_train.jsonl": selected_raw,
        "posttrain_hard_preferences_validation.jsonl": validation_pairs,
        "posttrain_hard_validation.jsonl": validation_rows,
    }
    files: dict[str, dict[str, Any]] = {}
    for filename, rows in outputs.items():
        path = args.output_dir / filename
        count = write_jsonl(path, rows)
        files[filename] = {"rows": count, "sha256": sha256_file(path)}
    report = {
        "schema_version": 1,
        "method": "SFT-policy-anchored clipped hard negatives with difficulty sampling",
        "base_model": base_name,
        "base_revision": base_revision,
        "policy": args.policy,
        "policy_adapter_sha256": sha256_file(Path(args.policy) / "adapter_model.safetensors"),
        "seed": args.seed,
        "minimum_margin_mad": args.minimum_margin_mad,
        "maximum_margin_mad": args.maximum_margin_mad,
        "test_labels_used": False,
        "summaries": {
            "pool": _summary(pool_pairs),
            "selected_train": _summary(selected_pairs),
            "validation": _summary(validation_pairs),
        },
        "files": files,
    }
    atomic_write_json(args.output_dir / "policy_preference_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
