import pytest

from steel_rlvr.sampling import balanced_indices


def test_balanced_sampling_is_deterministic() -> None:
    groups = ["Pass1"] * 10 + ["Pass2"] * 10 + ["Pass3"] * 10
    first = balanced_indices(groups, count=12, seed=42)
    second = balanced_indices(groups, count=12, seed=42)
    assert first == second
    selected_groups = [groups[index] for index in first]
    assert selected_groups.count("Pass1") == 4
    assert selected_groups.count("Pass2") == 4
    assert selected_groups.count("Pass3") == 4


def test_balanced_sampling_rejects_undersized_group() -> None:
    with pytest.raises(ValueError, match="balanced sampling"):
        balanced_indices(["Pass1"] * 10 + ["Pass2"], count=6, seed=42)
