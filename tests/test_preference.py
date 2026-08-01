from __future__ import annotations

from collections import Counter

import pytest

from steel_rlvr.preference import (
    build_preference_pair,
    preference_indices,
    tail_sampling_weight,
)


def _row(index: int, pass_name: str = "Pass1", frequency: int = 100) -> dict:
    return {
        "id": f"row-{index}",
        "prompt": [{"role": "user", "content": "predict"}],
        "target": 0.3,
        "target_scale": 0.5,
        "pass_name": pass_name,
        "steel_grade": f"grade-{index % 4}",
        "grade_frequency": frequency,
    }


def test_preference_pair_uses_same_format_and_positive_margin() -> None:
    pair = build_preference_pair(
        _row(1),
        seed=42,
        margin_multipliers=[0.5, 1.0, 1.5],
    )
    assert pair["chosen"].startswith('{"leveling":')
    assert pair["rejected"].startswith('{"leveling":')
    assert pair["chosen"] != pair["rejected"]
    assert pair["absolute_margin"] > 0
    assert pair["normalized_margin"] in {0.5, 1.0, 1.5}


def test_tail_weight_is_bounded_and_rejects_invalid_frequency() -> None:
    assert tail_sampling_weight(200) == 1.0
    assert tail_sampling_weight(50) == 2.0
    assert tail_sampling_weight(1) == 2.0
    with pytest.raises(ValueError):
        tail_sampling_weight(0)


def test_preference_sampling_preserves_pass_balance_and_is_deterministic() -> None:
    rows = [
        {
            **_row(index, pass_name=f"Pass{index % 3 + 1}", frequency=50 + index),
            "tail_sampling_weight": 2.0 if index % 2 else 1.0,
        }
        for index in range(60)
    ]
    first = preference_indices(rows, count=30, seed=7, tail_aware=True)
    second = preference_indices(rows, count=30, seed=7, tail_aware=True)
    assert first == second
    assert len(first) == len(set(first)) == 30
    assert Counter(rows[index]["pass_name"] for index in first) == {
        "Pass1": 10,
        "Pass2": 10,
        "Pass3": 10,
    }
