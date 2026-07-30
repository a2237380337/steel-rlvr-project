from steel_rlvr.metrics import (
    group_mae,
    mean_absolute_error,
    pass_grade_keys,
    r2_score,
)


def test_regression_metrics() -> None:
    assert mean_absolute_error([0.0, 2.0], [1.0, 2.0]) == 0.5
    assert r2_score([0.0, 2.0], [0.0, 2.0]) == 1.0
    grouped = group_mae(
        ["A", "A", "B"],
        [0.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
    )
    assert grouped == {"A": 0.5, "B": 2.0}


def test_overall_group_key_does_not_mix_passes() -> None:
    assert pass_grade_keys(["Pass1", "Pass2"], ["A", "A"]) == [
        "Pass1::A",
        "Pass2::A",
    ]
