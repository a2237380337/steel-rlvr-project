import pytest

from steel_rlvr.split import assert_disjoint_grade_test, split_grade_names


def test_frequency_split() -> None:
    common, rare, counts = split_grade_names(["A"] * 3 + ["B"] * 2 + ["C"], threshold=3)
    assert common == {"A"}
    assert rare == {"B", "C"}
    assert counts["A"] == 3


def test_leakage_guard() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_grade_test(
            [{"steel_grade": "A"}],
            [{"steel_grade": "A"}],
            [{"steel_grade": "A"}],
        )
