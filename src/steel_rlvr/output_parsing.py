"""Parse structured steel-leveling generations for evaluation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedPrediction:
    value: float | None
    status: str
    strict_json: bool


def parse_prediction_detail(text: str) -> ParsedPrediction:
    """Parse one finite value and classify every failure mode."""

    match = _JSON_OBJECT.search(text)
    if match is None:
        return ParsedPrediction(None, "no_json_object", False)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ParsedPrediction(None, "malformed_json", False)
    if not isinstance(payload, dict):
        return ParsedPrediction(None, "json_not_object", False)
    if "leveling" not in payload:
        return ParsedPrediction(None, "missing_leveling", False)
    if isinstance(payload["leveling"], bool):
        return ParsedPrediction(None, "non_numeric_leveling", False)
    try:
        value = float(payload["leveling"])
    except (TypeError, ValueError):
        return ParsedPrediction(None, "non_numeric_leveling", False)
    if not math.isfinite(value):
        return ParsedPrediction(None, "non_finite_leveling", False)
    native_json_number = isinstance(payload["leveling"], (int, float))
    strict = (
        _JSON_OBJECT.fullmatch(text.strip()) is not None
        and set(payload) == {"leveling"}
        and native_json_number
    )
    status = "valid" if strict else "valid_value_format_violation"
    return ParsedPrediction(value, status, strict)


def parse_leveling_prediction(text: str) -> float | None:
    return parse_prediction_detail(text).value
