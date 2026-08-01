from steel_rlvr.output_parsing import (
    parse_leveling_prediction,
    parse_prediction_detail,
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
