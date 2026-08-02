from steel_rlvr.build_policy_preferences import policy_anchored_pair


def _row() -> dict:
    return {
        "id": "example",
        "prompt": [{"role": "user", "content": "x"}],
        "target": 1.0,
        "target_scale": 0.5,
        "pass_name": "Pass1",
        "steel_grade": "A",
        "grade_frequency": 100,
    }


def test_policy_pair_clips_large_errors_and_keeps_same_format() -> None:
    pair = policy_anchored_pair(
        _row(),
        '{"leveling":4.0}',
        seed=7,
        minimum_margin_mad=0.1,
        maximum_margin_mad=1.0,
    )
    assert pair["chosen"] == '{"leveling":1.0}'
    assert pair["rejected"] == '{"leveling":1.5}'
    assert pair["policy_error_mad"] == 6.0
    assert pair["hard_margin_mad"] == 1.0


def test_policy_pair_uses_minimum_margin_for_exact_prediction() -> None:
    pair = policy_anchored_pair(
        _row(),
        '{"leveling":1.0}',
        seed=7,
        minimum_margin_mad=0.1,
        maximum_margin_mad=1.0,
    )
    assert pair["chosen"] != pair["rejected"]
    assert pair["hard_margin_mad"] == 0.1
