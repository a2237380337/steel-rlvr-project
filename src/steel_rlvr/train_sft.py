"""Train the text-only Qwen3.5 policy with assistant-only LoRA SFT."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .io_utils import verify_reported_file
from .precision import precision_flags, torch_dtype
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
    report_file = Path(config["data_report_file"])
    verify_reported_file(train_file, report_file)
    verify_reported_file(validation_file, report_file)

    import torch

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("SFT requires ROCm/HIP PyTorch")
    output_dir, run_manifest, metrics_callback = prepare_training_run(
        run_name=f"sft-{args.config.stem}",
        config_path=args.config,
        config=config,
    )

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_dataset = load_dataset("json", data_files=str(train_file), split="train")
    validation_dataset = load_dataset(
        "json",
        data_files=str(validation_file),
        split="train",
    )
    seed = int(config["seed"])
    train_dataset = select_balanced(
        train_dataset,
        count=int(config["max_samples"]),
        seed=seed,
    )
    validation_dataset = select_balanced(
        validation_dataset,
        count=int(config["validation_max_samples"]),
        seed=seed,
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

    model = AutoModelForMultimodalLM.from_pretrained(
        config["model_name_or_path"],
        revision=config["model_revision"],
        dtype=torch_dtype(config["mixed_precision"]),
        attn_implementation="sdpa",
        local_files_only=bool(config["local_files_only"]),
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        revision=config["model_revision"],
        local_files_only=bool(config["local_files_only"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_fp16, use_bf16 = precision_flags(config["mixed_precision"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(config["max_length"]),
        max_steps=int(config["max_steps"]),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(config["learning_rate"]),
        warmup_steps=float(config["warmup_ratio"]),
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
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(config["logging_steps"]),
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules="all-linear",
        exclude_modules=r".*visual.*",
        task_type="CAUSAL_LM",
        revision=config["model_revision"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
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
