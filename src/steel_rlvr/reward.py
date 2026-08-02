"""Verifiable numeric, format, physical, and tail-aware rewards for RLVR."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedPrediction:
    value: float | None
    status: str
    strict_json: bool


def completion_contents(completions: list[Any]) -> list[str]:
    """Normalize TRL's plain and conversational completion containers."""

    contents: list[str] = []
    for completion in completions:
        if isinstance(completion, str):
            contents.append(completion)
        elif isinstance(completion, dict):
            contents.append(str(completion.get("content", "")))
        elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
            contents.append(str(completion[0].get("content", "")))
        else:
            contents.append("")
    return contents


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


def value_reward(
    completions: list[Any],
    target: list[float],
    target_scale: list[float],
    **_: Any,
) -> list[float]:
    """Reward numeric accuracy using a train-only, group-normalized error scale."""

    rewards: list[float] = []
    for content, gold, scale in zip(
        completion_contents(completions),
        target,
        target_scale,
        strict=True,
    ):
        prediction = parse_leveling_prediction(content)
        if prediction is None:
            rewards.append(0.0)
            continue
        denominator = max(float(scale), 1e-6)
        rewards.append(math.exp(-abs(prediction - float(gold)) / denominator))
    return rewards


def tail_weight(
    grade_frequency: int,
    reference_frequency: int = 200,
    maximum: float = 2.0,
) -> float:
    """Upweight lower-frequency grades using training frequencies only."""

    if grade_frequency <= 0:
        raise ValueError("grade_frequency must be positive")
    if reference_frequency <= 0:
        raise ValueError("reference_frequency must be positive")
    if maximum < 1.0:
        raise ValueError("maximum must be at least one")
    return min(maximum, max(1.0, math.sqrt(reference_frequency / grade_frequency)))


def tail_aware_value_reward(
    completions: list[Any],
    target: list[float],
    target_scale: list[float],
    grade_frequency: list[int],
    *,
    reference_frequency: int = 200,
    maximum: float = 2.0,
    **kwargs: Any,
) -> list[float]:
    """Scale verifiable accuracy rewards toward lower-frequency grades."""

    base = value_reward(
        completions=completions,
        target=target,
        target_scale=target_scale,
        **kwargs,
    )
    return [
        reward
        * tail_weight(
            int(frequency),
            reference_frequency=reference_frequency,
            maximum=maximum,
        )
        for reward, frequency in zip(base, grade_frequency, strict=True)
    ]


def strict_json_reward(completions: list[Any], **_: Any) -> list[float]:
    return [
        1.0 if parse_prediction_detail(content).strict_json else 0.0
        for content in completion_contents(completions)
    ]


def physical_range_reward(
    completions: list[Any],
    lower_bound: list[float],
    upper_bound: list[float],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for content, lower, upper in zip(
        completion_contents(completions),
        lower_bound,
        upper_bound,
        strict=True,
    ):
        prediction = parse_leveling_prediction(content)
        rewards.append(
            1.0
            if prediction is not None and float(lower) <= prediction <= float(upper)
            else 0.0
        )
    return rewards
