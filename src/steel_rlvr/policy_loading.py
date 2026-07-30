"""Load Qwen3.5 as a text-only policy while preserving the base revision."""

from __future__ import annotations

import json
from pathlib import Path

from .precision import torch_dtype


def resolve_policy_source(
    model_or_checkpoint: str,
    requested_revision: str | None,
) -> tuple[str, str, bool]:
    adapter_config = Path(model_or_checkpoint) / "adapter_config.json"
    if not adapter_config.exists():
        return model_or_checkpoint, requested_revision or "main", False
    metadata = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_model = str(metadata["base_model_name_or_path"])
    base_revision = requested_revision or metadata.get("revision") or "main"
    return base_model, str(base_revision), True


def load_text_policy(
    model_or_checkpoint: str,
    revision: str | None,
    *,
    trainable_adapter: bool,
    mixed_precision: str,
    local_files_only: bool,
):
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    base_model, base_revision, is_adapter = resolve_policy_source(
        model_or_checkpoint,
        revision,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        base_model,
        revision=base_revision,
        dtype=torch_dtype(mixed_precision),
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    )
    if is_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            model_or_checkpoint,
            is_trainable=trainable_adapter,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=base_revision,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, base_model, base_revision, is_adapter
