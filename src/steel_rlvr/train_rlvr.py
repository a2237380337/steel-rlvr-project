"""Train frequency-robust online RLVR from the audited SFT checkpoint."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from .io_utils import verify_reported_file
from .policy_loading import load_text_policy
from .precision import precision_flags
from .reward import (
    physical_range_reward,
    strict_json_reward,
    tail_aware_value_reward,
)
from .reward_health import RewardHealthCallback
from .sampling import select_balanced
from .training_logging import (
    finish_training_run,
    log_dataset_summary,
    prepare_training_run,
    record_dataset_selection,
)

REWARD_COLUMNS = [
    "prompt",
    "target",
    "target_scale",
    "lower_bound",
    "upper_bound",
    "grade_frequency",
    "steel_grade",
    "pass_name",
    "id",
]


def _prepare_dataset(config: dict[str, Any]) -> Any:
    from datasets import load_dataset

    train_file = Path(config["train_file"])
    verify_reported_file(train_file, Path(config["data_report_file"]))
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    dataset = select_balanced(
        dataset,
        count=int(config["max_samples"]),
        seed=int(config["seed"]),
    )
    missing = sorted(set(REWARD_COLUMNS) - set(dataset.column_names))
    if missing:
        raise ValueError(f"RLVR dataset is missing columns: {missing}")
    dataset = dataset.select_columns(REWARD_COLUMNS)
    if any(int(value) <= 0 for value in dataset["grade_frequency"]):
        raise ValueError("RLVR rows must have positive training-only grade frequencies")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    generation_batch_size = int(config["gradient_accumulation_steps"])
    num_generations = int(config["num_generations"])
    if generation_batch_size % num_generations != 0:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by num_generations "
            "for single-device RLVR"
        )

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("online RLVR requires ROCm/HIP PyTorch")
    output_dir, run_manifest, metrics_callback = prepare_training_run(
        run_name=f"rlvr-{args.config.stem}",
        config_path=args.config,
        config=config,
    )

    from trl import GRPOConfig, GRPOTrainer

    dataset = _prepare_dataset(config)
    log_dataset_summary(
        train_dataset=dataset,
        validation_dataset=None,
        config=config,
    )
    record_dataset_selection(
        output_dir=output_dir,
        manifest=run_manifest,
        train_dataset=dataset,
        validation_dataset=None,
    )
    model, tokenizer, _, _, is_adapter = load_text_policy(
        str(config["model_name_or_path"]),
        config.get("model_revision"),
        trainable_adapter=True,
        mixed_precision=str(config["mixed_precision"]),
        local_files_only=bool(config["local_files_only"]),
    )
    if not is_adapter:
        raise ValueError("RLVR must start from the audited SFT adapter")
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    use_fp16, use_bf16 = precision_flags(str(config["mixed_precision"]))
    max_steps = int(config["max_steps"])
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_completion_length=int(config["max_completion_length"]),
        max_steps=max_steps,
        num_generations=int(config["num_generations"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(config["learning_rate"]),
        warmup_steps=max(0, round(max_steps * float(config["warmup_ratio"]))),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        max_grad_norm=float(config["max_grad_norm"]),
        temperature=float(config["temperature"]),
        fp16=use_fp16,
        bf16=use_bf16,
        use_vllm=False,
        beta=float(config["beta"]),
        loss_type=str(config["loss_type"]),
        scale_rewards=config["scale_rewards"],
        reward_weights=[float(value) for value in config["reward_weights"]],
        shuffle_dataset=False,
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        report_to=[],
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=int(config["num_completions_to_print"]),
        chat_template_kwargs={"enable_thinking": False},
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    numeric_reward = partial(
        tail_aware_value_reward,
        reference_frequency=int(config["tail_reference_frequency"]),
        maximum=float(config["tail_max_weight"]),
    )
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[numeric_reward, strict_json_reward, physical_range_reward],
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            metrics_callback,
            RewardHealthCallback(int(config["reward_zero_std_patience"])),
        ],
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    finish_training_run(
        trainer=trainer,
        train_result=train_result,
        output_dir=output_dir,
        manifest=run_manifest,
    )


if __name__ == "__main__":
    main()
