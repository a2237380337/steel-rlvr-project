"""Small deterministic filesystem helpers for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    """Hash project-owned code/config without reading data or checkpoints."""

    candidates: list[Path] = []
    for directory, patterns in {
        "src": ("*.py",),
        "tests": ("*.py",),
        "configs": ("*.yaml", "*.yml"),
        "scripts": ("*.sh",),
    }.items():
        base = project_root / directory
        for pattern in patterns:
            candidates.extend(base.rglob(pattern))
    candidates.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "requirements-rocm.txt",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def verify_reported_file(path: Path, report_path: Path) -> None:
    """Fail if a processed split no longer matches its data report."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report.get("files", {}).get(path.name, {}).get("sha256")
    if not expected:
        raise ValueError(f"{path.name} has no SHA256 entry in {report_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"processed data hash mismatch for {path}: expected {expected}, got {actual}"
        )
