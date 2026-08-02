"""Continue the audited SFT adapter on policy-selected hard examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .io_utils import verify_reported_file
from .policy_loading import load_text_policy
from .precision import precision_flags
from .sampling import select_balanced
from .training_logging import (
    finish_training_run,
    log_dataset_summary,
    prepare_training_run,
    record_dataset_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_file = Path(config["train_file"])
    validation_file = Path(config["validation_file"])
    report_file = Path(config["selection_report_file"])
    verify_reported_file(train_file, report_file)
    verify_reported_file(validation_file, report_file)

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("continued SFT requires ROCm/HIP PyTorch")
    output_dir, run_manifest, metrics_callback = prepare_training_run(
        run_name=f"continued-sft-{args.config.stem}",
        config_path=args.config,
        config=config,
    )

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_dataset = Dataset.from_json(str(train_file))
    validation_dataset = Dataset.from_json(str(validation_file))
    seed = int(config["seed"])
    train_dataset = select_balanced(
        train_dataset,
        count=int(config["max_samples"]),
        seed=seed,
    )
    validation_dataset = select_balanced(
        validation_dataset,
        count=int(config["validation_max_samples"]),
        seed=seed + 1,
    )
    log_dataset_summary(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
    )
    record_dataset_selection(
        output_dir=output_dir,
        manifest=run_manifest,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    train_dataset = train_dataset.select_columns(["messages"])
    validation_dataset = validation_dataset.select_columns(["messages"])

    model, tokenizer, _, _, is_adapter = load_text_policy(
        str(config["model_name_or_path"]),
        config.get("model_revision"),
        trainable_adapter=True,
        mixed_precision=str(config["mixed_precision"]),
        local_files_only=bool(config["local_files_only"]),
    )
    if not is_adapter:
        raise ValueError("continued SFT must start from the audited SFT adapter")
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    use_fp16, use_bf16 = precision_flags(str(config["mixed_precision"]))
    max_steps = int(config["max_steps"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(config["max_length"]),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(config["learning_rate"]),
        warmup_steps=max(1, round(max_steps * float(config["warmup_ratio"]))),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        fp16=use_fp16,
        bf16=use_bf16,
        assistant_only_loss=True,
        completion_only_loss=False,
        packing=False,
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(config["logging_steps"]),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=[metrics_callback],
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
