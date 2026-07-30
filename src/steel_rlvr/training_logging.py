"""Durable run manifests and JSONL metric logs for SFT and GRPO."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from .io_utils import atomic_write_json, sha256_file, source_tree_sha256

LOGGER = logging.getLogger("steel_rlvr.training")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_value(cwd: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    names = ["torch", "transformers", "trl", "datasets", "accelerate", "peft", "pandas"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def require_fresh_output_dir(output_dir: Path) -> None:
    """Prevent checkpoints from two experiments being silently mixed."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Choose a new directory or explicitly remove the old run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if not key.startswith("_")}


def peak_accelerator_memory_mib(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    torch_module.cuda.synchronize()
    return round(float(torch_module.cuda.max_memory_allocated()) / 1024**2, 2)


class JsonlMetricsCallback:
    """A Transformers callback that retains every metric event."""

    def __init__(self, output_dir: Path, session_id: str) -> None:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, path: Path, active_session: str) -> None:
                self.path = path
                self.active_session = active_session

            def _append(self, event: str, state: Any, metrics: dict[str, Any] | None = None) -> None:
                row = {
                    "timestamp": utc_now(),
                    "session_id": self.active_session,
                    "event": event,
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "metrics": metrics or {},
                }
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

            def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                self._append("train_begin", state)

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                self._append("log", state, logs)

            def on_evaluate(
                self,
                args: Any,
                state: Any,
                control: Any,
                metrics: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                self._append("evaluate", state, metrics)

            def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                self._append("checkpoint_saved", state)

            def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                self._append("train_end", state)

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.callback = _Callback(logs_dir / "trainer_metrics.jsonl", session_id)


def prepare_training_run(
    *,
    run_name: str,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], Any]:
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    output_dir = Path(config["output_dir"])
    require_fresh_output_dir(output_dir)
    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[2]
    git_status = _git_value(project_root, "status", "--porcelain")
    data_artifacts: dict[str, dict[str, str]] = {}
    for key in ("train_file", "validation_file", "data_report_file"):
        raw_path = config.get(key)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file():
            data_artifacts[key] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
    session_id = f"{run_name}-{uuid.uuid4().hex[:12]}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "run_name": run_name,
        "status": "running",
        "started_at": utc_now(),
        "ended_at": None,
        "duration_seconds": None,
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_tree_sha256": source_tree_sha256(project_root),
        "resolved_config": str(resolved_config_path.resolve()),
        "git": {
            "commit": _git_value(project_root, "rev-parse", "HEAD"),
            "branch": _git_value(project_root, "branch", "--show-current"),
            "dirty": bool(git_status),
            "status": git_status.splitlines() if git_status else [],
        },
        "packages": _package_versions(),
        "data_artifacts": data_artifacts,
        "hardware": {
            "accelerator_available": torch.cuda.is_available(),
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gib": (
                round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    manifest["_monotonic_started"] = time.monotonic()
    manifest["_previous_excepthook"] = sys.excepthook
    atomic_write_json(output_dir / "run_manifest.json", _public_manifest(manifest))

    def record_uncaught_failure(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: Any,
    ) -> None:
        mark_training_run_failed(output_dir, manifest, exception)
        previous = manifest["_previous_excepthook"]
        previous(exception_type, exception, traceback)

    sys.excepthook = record_uncaught_failure
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    LOGGER.info("run_name=%s session_id=%s output_dir=%s", run_name, session_id, output_dir.resolve())
    LOGGER.info("config_sha256=%s", manifest["config_sha256"])
    LOGGER.info("hardware=%s", json.dumps(manifest["hardware"], ensure_ascii=False))
    return output_dir, manifest, JsonlMetricsCallback(output_dir, session_id).callback


def mark_training_run_failed(
    output_dir: Path,
    manifest: dict[str, Any],
    error: BaseException,
) -> None:
    if manifest.get("status") == "failed":
        return
    manifest["status"] = "failed"
    manifest["ended_at"] = utc_now()
    manifest["duration_seconds"] = round(
        time.monotonic() - float(manifest["_monotonic_started"]),
        3,
    )
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_write_json(output_dir / "run_manifest.json", _public_manifest(manifest))


def log_dataset_summary(
    *,
    train_dataset: Any,
    validation_dataset: Any | None,
    config: dict[str, Any],
) -> None:
    LOGGER.info(
        "dataset=train:%d validation:%s seed:%s max_steps:%s grad_accum:%s lr:%s",
        len(train_dataset),
        len(validation_dataset) if validation_dataset is not None else None,
        config.get("seed"),
        config.get("max_steps"),
        config.get("gradient_accumulation_steps"),
        config.get("learning_rate"),
    )


def record_dataset_selection(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    train_dataset: Any,
    validation_dataset: Any | None,
) -> None:
    """Persist selected IDs and pass balance without storing private samples."""

    def summarize(dataset: Any) -> dict[str, Any]:
        identifiers = [str(value) for value in dataset["id"]]
        return {
            "samples": len(dataset),
            "ordered_ids_sha256": hashlib.sha256(
                "\n".join(identifiers).encode("utf-8")
            ).hexdigest(),
            "pass_counts": dict(
                sorted(Counter(str(value) for value in dataset["pass_name"]).items())
            ),
        }

    manifest["dataset_selection"] = {
        "train": summarize(train_dataset),
        "validation": summarize(validation_dataset) if validation_dataset is not None else None,
    }
    atomic_write_json(output_dir / "run_manifest.json", _public_manifest(manifest))


def finish_training_run(
    *,
    trainer: Any,
    train_result: Any,
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    import torch

    metrics = dict(train_result.metrics)
    peak_memory = peak_accelerator_memory_mib(torch)
    if peak_memory is not None:
        metrics["peak_accelerator_memory_mib"] = peak_memory
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    manifest["status"] = "completed"
    manifest["ended_at"] = utc_now()
    manifest["duration_seconds"] = round(
        time.monotonic() - float(manifest["_monotonic_started"]),
        3,
    )
    manifest["train_metrics"] = metrics
    atomic_write_json(output_dir / "run_manifest.json", _public_manifest(manifest))
    sys.excepthook = manifest["_previous_excepthook"]
    LOGGER.info("train_metrics=%s", json.dumps(metrics, ensure_ascii=False, default=str))
