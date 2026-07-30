"""Dependency-free regression metrics with pass-grade group summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def mean_absolute_error(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> float:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and have equal length")
    return sum(
        abs(float(target) - float(prediction))
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)


def r2_score(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and have equal length")
    numeric_targets = [float(value) for value in targets]
    numeric_predictions = [float(value) for value in predictions]
    mean_target = sum(numeric_targets) / len(numeric_targets)
    total = sum((target - mean_target) ** 2 for target in numeric_targets)
    residual = sum(
        (target - prediction) ** 2
        for target, prediction in zip(
            numeric_targets,
            numeric_predictions,
            strict=True,
        )
    )
    return 0.0 if total == 0 else 1.0 - residual / total


def group_mae(
    groups: Sequence[str],
    targets: Sequence[float],
    predictions: Sequence[float],
) -> dict[str, float]:
    if not (len(groups) == len(targets) == len(predictions)):
        raise ValueError("group, target and prediction lengths differ")
    grouped_targets: dict[str, list[float]] = defaultdict(list)
    grouped_predictions: dict[str, list[float]] = defaultdict(list)
    for group, target, prediction in zip(groups, targets, predictions, strict=True):
        key = str(group)
        grouped_targets[key].append(float(target))
        grouped_predictions[key].append(float(prediction))
    return {
        group: mean_absolute_error(grouped_targets[group], grouped_predictions[group])
        for group in sorted(grouped_targets)
    }


def pass_grade_keys(
    passes: Sequence[str],
    grades: Sequence[str],
) -> list[str]:
    if len(passes) != len(grades):
        raise ValueError("pass and grade lengths differ")
    return [
        f"{pass_name}::{grade}"
        for pass_name, grade in zip(passes, grades, strict=True)
    ]
