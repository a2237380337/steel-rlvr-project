import math

from steel_rlvr.reward import (
    parse_leveling_prediction,
    parse_prediction_detail,
    physical_range_reward,
    strict_json_reward,
    tail_aware_value_reward,
    tail_weight,
    value_reward,
)


def test_prediction_parser_distinguishes_value_and_format() -> None:
    strict = parse_prediction_detail('{"leveling":-0.25}')
    assert strict.value == -0.25
    assert strict.strict_json is True
    assert strict.status == "valid"

    surrounded = parse_prediction_detail('answer: {"leveling": 1.5}')
    assert surrounded.value == 1.5
    assert surrounded.strict_json is False
    assert surrounded.status == "valid_value_format_violation"

    assert parse_leveling_prediction('{"wrong":1}') is None
    assert parse_leveling_prediction('{"leveling":true}') is None
    assert parse_leveling_prediction('{"leveling":"nan"}') is None


def test_numeric_reward_is_bounded_before_tail_weighting() -> None:
    rewards = value_reward(
        ['{"leveling":1.0}', '{"leveling":3.0}', "bad"],
        target=[1.0, 1.0, 1.0],
        target_scale=[1.0, 1.0, 1.0],
    )
    assert rewards[0] == 1.0
    assert rewards[1] == math.exp(-2.0)
    assert rewards[2] == 0.0


def test_format_and_physical_rewards() -> None:
    completions = ['{"leveling":1.0}', 'prefix {"leveling":2.0}']
    assert strict_json_reward(completions) == [1.0, 0.0]
    assert physical_range_reward(
        completions,
        lower_bound=[0.0, 0.0],
        upper_bound=[1.5, 1.5],
    ) == [1.0, 0.0]


def test_tail_aware_reward_uses_configured_training_frequency() -> None:
    assert tail_weight(50) == 2.0
    assert tail_weight(200) == 1.0
    assert tail_weight(800) == 1.0
    weighted = tail_aware_value_reward(
        ['{"leveling":1.0}'],
        target=[1.0],
        target_scale=[1.0],
        grade_frequency=[50],
        reference_frequency=200,
        maximum=2.0,
    )
    assert weighted == [2.0]
