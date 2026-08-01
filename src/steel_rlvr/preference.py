"""Leakage-safe preference pairs and deterministic tail-aware sampling."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .schema import target_completion


def tail_sampling_weight(
    grade_frequency: int,
    *,
    reference_frequency: int = 200,
    maximum: float = 2.0,
) -> float:
    """Return the train-frequency weight used only for pair sampling."""

    if grade_frequency <= 0:
        raise ValueError("grade_frequency must be positive")
    if reference_frequency <= 0:
        raise ValueError("reference_frequency must be positive")
    if maximum < 1.0:
        raise ValueError("maximum must be at least one")
    return min(maximum, max(1.0, math.sqrt(reference_frequency / grade_frequency)))


def _stable_choice(identifier: str, seed: int, choices: Sequence[float]) -> float:
    if not choices:
        raise ValueError("choices must not be empty")
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return float(choices[int.from_bytes(digest[:8], "big") % len(choices)])


def build_preference_pair(
    row: dict[str, Any],
    *,
    seed: int,
    margin_multipliers: Sequence[float],
) -> dict[str, Any]:
    """Prefer the observed target over a same-format, controlled numeric negative."""

    identifier = str(row["id"])
    target = float(row["target"])
    scale = float(row["target_scale"])
    if not math.isfinite(target) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid target/scale for {identifier}")
    multiplier = _stable_choice(identifier, seed, margin_multipliers)
    if multiplier <= 0:
        raise ValueError("margin multipliers must be positive")
    sign = -1.0 if _stable_choice(identifier + ":sign", seed, (-1.0, 1.0)) < 0 else 1.0
    rejected_value = round(target + sign * multiplier * scale, 6)
    if rejected_value == round(target, 6):
        raise ValueError(f"preference pair has zero numeric margin: {identifier}")
    grade_frequency = int(row["grade_frequency"])
    if grade_frequency <= 0:
        raise ValueError(f"preference training row has non-positive grade frequency: {identifier}")
    chosen = target_completion(target)
    rejected = target_completion(rejected_value)
    if chosen == rejected:
        raise ValueError(f"preference pair has identical completions: {identifier}")
    return {
        "id": identifier,
        "prompt": row["prompt"],
        "chosen": chosen,
        "rejected": rejected,
        "pass_name": str(row["pass_name"]),
        "steel_grade": str(row["steel_grade"]),
        "grade_frequency": grade_frequency,
        "tail_sampling_weight": tail_sampling_weight(grade_frequency),
        "absolute_margin": abs(rejected_value - target),
        "normalized_margin": round(abs(rejected_value - target) / scale, 6),
    }


def _weighted_sample_indices(
    indices: Sequence[int],
    weights: Sequence[float],
    *,
    count: int,
    rng: random.Random,
) -> list[int]:
    """Efraimidis-Spirakis sampling without replacement."""

    if len(indices) != len(weights):
        raise ValueError("indices and weights must have the same length")
    if count < 0 or count > len(indices):
        raise ValueError("invalid sample count")
    ranked: list[tuple[float, int]] = []
    for index, weight in zip(indices, weights, strict=True):
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("sampling weights must be finite and positive")
        key = math.log(max(rng.random(), 1e-300)) / float(weight)
        ranked.append((key, int(index)))
    ranked.sort(reverse=True)
    return [index for _, index in ranked[:count]]


def preference_indices(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    tail_aware: bool,
) -> list[int]:
    """Select the same pass quota while optionally changing grade-frequency weights."""

    if count <= 0 or count > len(rows):
        raise ValueError(f"requested {count} rows from a dataset of {len(rows)}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["pass_name"])].append(index)
    names = sorted(grouped)
    base, remainder = divmod(count, len(names))
    rng = random.Random(seed)
    selected: list[int] = []
    for position, name in enumerate(names):
        quota = base + (1 if position < remainder else 0)
        candidates = grouped[name]
        if len(candidates) < quota:
            raise ValueError(f"pass {name!r} cannot supply {quota} preference pairs")
        weights = [
            float(rows[index]["tail_sampling_weight"]) if tail_aware else 1.0
            for index in candidates
        ]
        selected.extend(
            _weighted_sample_indices(candidates, weights, count=quota, rng=rng)
        )
    rng.shuffle(selected)
    return selected


def preference_selection_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frequencies = [int(row["grade_frequency"]) for row in rows]
    weights = [float(row["tail_sampling_weight"]) for row in rows]
    return {
        "samples": len(rows),
        "unique_grades": len({str(row["steel_grade"]) for row in rows}),
        "pass_counts": dict(sorted(Counter(str(row["pass_name"]) for row in rows).items())),
        "grade_frequency_mean": sum(frequencies) / len(frequencies),
        "tail_sampling_weight_mean": sum(weights) / len(weights),
        "tail_weighted_pair_rate": sum(weight > 1.0 for weight in weights) / len(weights),
    }
