import pytest

from steel_rlvr.compare_evaluations import build_matrix


def _summary(path: str, macro: float) -> dict:
    pass_metrics = {
        name: {
            "mae": macro,
            "r2": 0.1,
            "macro_mae": macro,
            "worst_group_mae": macro + 1,
        }
        for name in ("Pass1", "Pass2", "Pass3")
    }
    return {
        "split_sha256": "same",
        "sample_ids_sha256": "same-samples",
        "sample_count": 1017,
        "model_or_checkpoint": path,
        "base_revision": "pinned",
        "source_tree_sha256": "same-source",
        "metrics": {
            "mae": macro,
            "r2": 0.1,
            "macro_mae": macro,
            "worst_group_mae": macro + 1,
            "strict_json_rate": 1.0,
            "valid_value_rate": 1.0,
            "passes": pass_metrics,
        },
        "hardware": {"peak_memory_mib": 100},
    }


def test_matrix_requires_identical_test_artifact() -> None:
    first = _summary("a", 1.0)
    second = _summary("b", 0.9)
    second["split_sha256"] = "different"
    with pytest.raises(ValueError, match="same split"):
        build_matrix({"base": first, "sft": second})


def test_matrix_reports_controlled_tail_delta() -> None:
    result = build_matrix(
        {
            "base": _summary("base", 1.2),
            "sft": _summary("sft", 1.0),
            "drgrpo": _summary("dr", 0.9),
            "tail_aware": _summary("tail", 0.8),
        }
    )
    assert result["complete"] is True
    assert result["comparisons"]["tail_aware_minus_drgrpo"]["macro_mae"] == pytest.approx(-0.1)
