from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from steel_rlvr.release_audit import (
    ReleaseAuditError,
    validate_csv,
    validate_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _matrix() -> dict:
    return json.loads((PROJECT_ROOT / "results" / "formal_evaluation_matrix.json").read_text(encoding="utf-8"))


def test_formal_matrix_and_csv_pass_release_checks() -> None:
    matrix = _matrix()
    validate_matrix(matrix)
    validate_csv(matrix, PROJECT_ROOT / "reports" / "experiment_matrix.csv")


def test_incomplete_matrix_is_rejected() -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["complete"] = False
    with pytest.raises(ReleaseAuditError, match="not marked complete"):
        validate_matrix(matrix)
