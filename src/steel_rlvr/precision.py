"""One-card ROCm precision policy shared by training and evaluation."""

from __future__ import annotations


def precision_flags(mixed_precision: str) -> tuple[bool, bool]:
    normalized = mixed_precision.casefold()
    if normalized == "fp16":
        return True, False
    if normalized == "bf16":
        return False, True
    raise ValueError(f"unsupported mixed_precision: {mixed_precision!r}")


def torch_dtype(mixed_precision: str):
    import torch

    use_fp16, use_bf16 = precision_flags(mixed_precision)
    return torch.float16 if use_fp16 else torch.bfloat16 if use_bf16 else None
