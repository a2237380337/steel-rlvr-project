from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_MODELS = {"base", "sft", "drgrpo", "tail_aware"}
EXPECTED_SAMPLE_COUNT = 1017
EXPECTED_SPLIT_SHA256 = "6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49"
EXPECTED_SAMPLE_IDS_SHA256 = "16688d7338105327588dcafe04f738d2b5d59041203a86c1994d8630e8113139"
REQUIRED_PUBLIC_FILES = {
    ".gitignore",
    "DATA_CARD.md",
    "LICENSE",
    "MODEL_CARD.md",
    "README.md",
    "docs/experiment_protocol.md",
    "reports/experiment_matrix.csv",
    "reports/main_experiment_report.md",
    "reports/provenance.md",
    "reports/result_card.generated.md",
    "results/formal_evaluation_matrix.json",
}
FORBIDDEN_TRACKED_PREFIXES = ("artifacts/", "data/processed/")
FORBIDDEN_TRACKED_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".parquet")
# Split the literals so this audit module does not flag its own rule text.
LOCAL_PATH_MARKERS = ("/mnt/" + "d/", "D:" + "\\lessonFile", "D:" + "/lessonFile")


class ReleaseAuditError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReleaseAuditError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAuditError(message)


def validate_matrix(matrix: dict[str, Any]) -> None:
    _require(matrix.get("complete") is True, "Formal matrix is not marked complete.")
    _require(matrix.get("missing_labels") == [], "Formal matrix still has missing model labels.")
    _require(matrix.get("sample_count") == EXPECTED_SAMPLE_COUNT, "Unexpected formal sample count.")
    _require(matrix.get("split_sha256") == EXPECTED_SPLIT_SHA256, "Unexpected formal test hash.")
    _require(
        matrix.get("sample_ids_sha256") == EXPECTED_SAMPLE_IDS_SHA256,
        "Unexpected formal sample-ID hash.",
    )
    models = matrix.get("models")
    _require(isinstance(models, dict), "Formal matrix has no models object.")
    _require(set(models) == EXPECTED_MODELS, f"Unexpected model labels: {sorted(models)}")
    for label, summary in models.items():
        _require(isinstance(summary, dict), f"Invalid model summary: {label}")
        _require(summary.get("strict_json_rate") == 1.0, f"{label} is not 100% strict JSON.")
        _require(summary.get("valid_value_rate") == 1.0, f"{label} is not 100% numeric.")
        for metric in ("mae", "r2", "macro_mae", "worst_group_mae"):
            value = summary.get(metric)
            _require(
                isinstance(value, (int, float)) and math.isfinite(value),
                f"{label}.{metric} is not finite.",
            )


def validate_csv(matrix: dict[str, Any], csv_path: Path, tolerance: float = 5.1e-7) -> None:
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require({row["run_id"] for row in rows} == EXPECTED_MODELS, "CSV model labels differ from JSON.")
    for row in rows:
        model = matrix["models"][row["run_id"]]
        comparisons = {
            "overall_mae": model["mae"],
            "overall_r2": model["r2"],
            "pass1_mae": model["passes"]["Pass1"]["mae"],
            "pass1_r2": model["passes"]["Pass1"]["r2"],
            "pass2_mae": model["passes"]["Pass2"]["mae"],
            "pass2_r2": model["passes"]["Pass2"]["r2"],
            "pass3_mae": model["passes"]["Pass3"]["mae"],
            "pass3_r2": model["passes"]["Pass3"]["r2"],
            "macro_mae": model["macro_mae"],
            "worst_group_mae": model["worst_group_mae"],
        }
        for column, expected in comparisons.items():
            actual = float(row[column])
            _require(
                abs(actual - float(expected)) <= tolerance,
                f"CSV mismatch for {row['run_id']}.{column}: {actual} vs {expected}",
            )


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def tracked_files(root: Path) -> list[str]:
    result = _git(root, "ls-files", "-z")
    return [path for path in result.stdout.split("\0") if path]


def validate_public_tree(root: Path) -> None:
    _require((root / ".git").is_dir(), "Run the release audit inside the independent project Git repo.")
    tracked = set(tracked_files(root))
    missing = REQUIRED_PUBLIC_FILES - tracked
    _require(not missing, f"Required public files are not tracked: {sorted(missing)}")

    forbidden = [
        path
        for path in tracked
        if path.startswith(FORBIDDEN_TRACKED_PREFIXES) or path.lower().endswith(FORBIDDEN_TRACKED_SUFFIXES)
    ]
    _require(not forbidden, f"Private/generated files are tracked: {forbidden}")

    text_suffixes = {".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    local_path_hits: list[str] = []
    for relative in sorted(tracked):
        path = root / relative
        if path.suffix.lower() not in text_suffixes or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in content for marker in LOCAL_PATH_MARKERS):
            local_path_hits.append(relative)
    _require(not local_path_hits, f"Tracked files expose local absolute paths: {local_path_hits}")

    ignore_expectations = {
        "artifacts/checkpoints/example/adapter_model.safetensors": True,
        "data/processed/test.jsonl": True,
        "private.xlsx": True,
        "reports/result_card.generated.md": False,
        "results/formal_evaluation_matrix.json": False,
    }
    for relative, should_ignore in ignore_expectations.items():
        result = _git(root, "check-ignore", "-q", "--", relative, check=False)
        ignored = result.returncode == 0
        _require(ignored == should_ignore, f"Unexpected ignore status for {relative}: {ignored}")


def audit_release(root: Path) -> None:
    matrix_path = root / "results" / "formal_evaluation_matrix.json"
    matrix = _load_json(matrix_path)
    validate_matrix(matrix)
    validate_csv(matrix, root / "reports" / "experiment_matrix.csv")
    validate_public_tree(root)

    result_card = (root / "reports" / "result_card.generated.md").read_text(encoding="utf-8")
    _require(EXPECTED_SPLIT_SHA256 in result_card, "Result card does not contain the final test hash.")
    _require("SFT + Tail-aware Dr. GRPO" in result_card, "Result card is missing the final model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the public project release boundary and formal metrics.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit_release(root)
    print("Release audit passed.")
    print(f"root={root}")
    print(f"formal_samples={EXPECTED_SAMPLE_COUNT}")
    print(f"test_sha256={EXPECTED_SPLIT_SHA256}")


if __name__ == "__main__":
    main()
