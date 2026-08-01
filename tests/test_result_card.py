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


def _matrix(frequency_model: dict, standard_model: dict) -> dict:
    return {
        "complete": True,
        "split_sha256": "split",
        "sample_ids_sha256": "ids",
        "source_tree_sha256": "source",
        "sample_count": 1017,
        "models": {
            "base": _model(1.0, 1.0, 2.0),
            "sft": _model(0.9, 1.1, 1.8),
            "dpo": standard_model,
            "frequency_aware_dpo": frequency_model,
        },
        "comparisons": {
            "frequency_aware_dpo_minus_dpo": {
                "mae": frequency_model["mae"] - standard_model["mae"],
                "macro_mae": (
                    frequency_model["macro_mae"] - standard_model["macro_mae"]
                ),
                "worst_group_mae": (
                    frequency_model["worst_group_mae"]
                    - standard_model["worst_group_mae"]
                ),
            }
        },
    }


def test_result_card_names_best_model_and_frequency_tradeoff() -> None:
    standard = _model(0.8, 0.9, 1.5)
    frequency = _model(0.81, 0.89, 1.6)
    markdown = build_markdown(
        _matrix(frequency, standard), paper_baselines=[]
    )

    assert "| 方法 | 总体 MAE |" in markdown
    assert "总体最佳模型：**SFT + DPO**" in markdown
    assert "总体 MAE 上升 1.25%" in markdown
    assert "Macro-MAE 下降 1.11%" in markdown
    assert "存在指标回退" in markdown


def test_result_card_promotes_frequency_model_only_when_all_errors_improve() -> None:
    standard = _model(0.8, 0.9, 1.5)
    frequency = _model(0.79, 0.89, 1.4)
    markdown = build_markdown(
        _matrix(frequency, standard), paper_baselines=[]
    )

    assert "总体最佳模型：**SFT + 频次感知 DPO**" in markdown
    assert "至少有一项改进且没有回退" in markdown
    assert "单个随机种子不足以判断统计显著性" in markdown


def test_result_card_does_not_call_an_exact_tie_an_improvement() -> None:
    standard = _model(0.8, 0.9, 1.5)
    frequency = _model(0.8, 0.9, 1.5)
    markdown = build_markdown(
        _matrix(frequency, standard), paper_baselines=[]
    )

    assert "总体 MAE 持平 0.00%" in markdown
    assert "没有带来可测的生成增益" in markdown
