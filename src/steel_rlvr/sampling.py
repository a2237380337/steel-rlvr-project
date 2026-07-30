"""Deterministic pass-balanced sampling for controlled training and evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def balanced_indices(
    groups: Sequence[object],
    *,
    count: int,
    seed: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if count > len(groups):
        raise ValueError(f"requested {count} rows but dataset contains only {len(groups)}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[str(group)].append(index)
    if not grouped:
        raise ValueError("cannot sample an empty dataset")
    names = sorted(grouped)
    base, remainder = divmod(count, len(names))
    rng = random.Random(seed)
    selected: list[int] = []
    for position, name in enumerate(names):
        requested = base + (1 if position < remainder else 0)
        candidates = grouped[name][:]
        if len(candidates) < requested:
            raise ValueError(
                f"group {name!r} has {len(candidates)} rows but balanced sampling needs {requested}"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:requested])
    rng.shuffle(selected)
    return selected


def select_balanced(
    dataset: Any,
    *,
    count: int,
    seed: int,
    column: str = "pass_name",
) -> Any:
    if column not in dataset.column_names:
        raise ValueError(f"balanced sampling column is missing: {column}")
    return dataset.select(
        balanced_indices(dataset[column], count=count, seed=seed)
    )
