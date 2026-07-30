from steel_rlvr.build_result_card import build_markdown


def _model(mae: float, macro: float, worst: float) -> dict:
    return {
        "mae": mae,
        "macro_mae": macro,
        "worst_group_mae": worst,
        "strict_json_rate": 1.0,
        "passes": {
            name: {"mae": mae, "r2": 0.0}
            for name in ("Pass1", "Pass2", "Pass3")
        },
    }


def test_result_card_names_best_model_and_tail_tradeoff() -> None:
    matrix = {
        "complete": True,
        "split_sha256": "split",
        "sample_ids_sha256": "ids",
        "source_tree_sha256": "source",
        "sample_count": 1017,
        "models": {
            "base": _model(1.0, 1.0, 2.0),
            "sft": _model(0.9, 1.1, 1.8),
            "drgrpo": _model(0.8, 0.9, 1.5),
            "tail_aware": _model(0.81, 0.89, 1.6),
        },
        "comparisons": {
            "tail_aware_minus_drgrpo": {
                "mae": 0.01,
                "macro_mae": -0.01,
                "worst_group_mae": 0.1,
            }
        },
    }

    markdown = build_markdown(matrix, paper_baselines=[])

    assert "| 方法 | 总体 MAE |" in markdown
    assert "总体最佳模型：**SFT + Dr. GRPO**" in markdown
    assert "总体 MAE 上升 1.25%" in markdown
    assert "Macro-MAE 下降 1.11%" in markdown
    assert "存在指标回退" in markdown


def test_result_card_promotes_tail_only_when_all_target_errors_improve() -> None:
    matrix = {
        "complete": True,
        "split_sha256": "split",
        "sample_ids_sha256": "ids",
        "source_tree_sha256": "source",
        "sample_count": 1017,
        "models": {
            "base": _model(1.0, 1.0, 2.0),
            "sft": _model(0.9, 0.95, 1.8),
            "drgrpo": _model(0.8, 0.9, 1.5),
            "tail_aware": _model(0.79, 0.89, 1.4),
        },
        "comparisons": {
            "tail_aware_minus_drgrpo": {
                "mae": -0.01,
                "macro_mae": -0.01,
                "worst_group_mae": -0.1,
            }
        },
    }

    markdown = build_markdown(matrix, paper_baselines=[])

    assert "总体最佳模型：**SFT + Tail-aware Dr. GRPO**" in markdown
    assert "作为本项目最终方法" in markdown
    assert "单个随机种子不足以判断统计显著性" in markdown
