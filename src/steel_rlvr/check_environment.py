"""Print the ROCm and package state used by an experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-rocm", action="store_true")
    parser.add_argument("--run-kernel-smoke", action="store_true")
    args = parser.parse_args()
    packages = {}
    for package in ("torch", "transformers", "trl", "datasets", "peft", "pandas"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    try:
        import torch

        backend = "rocm" if torch.version.hip else ("cuda" if torch.version.cuda else "cpu")
        summary = {
            "packages": packages,
            "backend": backend,
            "hip_version": torch.version.hip,
            "accelerator_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
        if torch.cuda.is_available():
            summary["primary_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if args.run_kernel_smoke:
                torch.cuda.reset_peak_memory_stats()
                left = torch.randn(
                    (1024, 1024),
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                right = torch.randn(
                    (1024, 1024),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                loss = (left @ right).float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                summary["kernel_smoke"] = {
                    "status": "passed",
                    "dtype": "bfloat16",
                    "shape": [1024, 1024],
                    "loss_is_finite": bool(torch.isfinite(loss).item()),
                    "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                }
                del left, right, loss
                torch.cuda.empty_cache()
    except ImportError:
        summary = {"packages": packages, "backend": "none", "accelerator_available": False, "gpus": []}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_rocm and summary["backend"] != "rocm":
        raise SystemExit("ROCm/HIP PyTorch is required")
    if args.run_kernel_smoke and "kernel_smoke" not in summary:
        raise SystemExit("Kernel smoke requested but no PyTorch accelerator is available")


if __name__ == "__main__":
    main()
