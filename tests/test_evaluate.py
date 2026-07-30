from steel_rlvr.evaluate import summarize_rows


def test_summarize_rows_uses_train_center_for_invalid_json() -> None:
    rows = [
        {
            "pass_name": "Pass1",
            "steel_grade": "A",
            "target": 1.0,
            "target_center": 0.0,
            "lower_bound": -1.0,
            "upper_bound": 1.5,
            "prediction": 1.0,
            "strict_json": True,
            "parse_status": "valid",
            "completion_tokens": 5,
        },
        {
            "pass_name": "Pass1",
            "steel_grade": "B",
            "target": 2.0,
            "target_center": 0.0,
            "lower_bound": -1.0,
            "upper_bound": 1.5,
            "prediction": None,
            "strict_json": False,
            "parse_status": "no_json_object",
            "completion_tokens": 3,
        },
    ]

    result = summarize_rows(rows)

    assert result["valid_value_rate"] == 0.5
    assert result["strict_json_rate"] == 0.5
    assert result["fallback_rate"] == 0.5
    assert result["mae"] == 1.0
    assert result["passes"]["Pass1"]["valid_only_mae"] == 0.0
    assert result["generation"]["total_completion_tokens"] == 8
    assert result["failure_counts"] == {"no_json_object": 1, "valid": 1}
