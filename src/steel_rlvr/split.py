"""Frequency split and leakage checks for unseen low-frequency steel grades."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence


def split_grade_names(
    grades: Iterable[str],
    threshold: int = 50,
) -> tuple[set[str], set[str], dict[str, int]]:
    if threshold <= 1:
        raise ValueError("threshold must be greater than one")
    normalized = [str(grade).strip() for grade in grades]
    missing_markers = {"nan", "<na>", "none", "null"}
    if any(not grade or grade.casefold() in missing_markers for grade in normalized):
        raise ValueError("steel grade contains a blank or missing value")
    counts = Counter(normalized)
    common = {grade for grade, count in counts.items() if count >= threshold}
    rare = set(counts) - common
    if not common:
        raise ValueError("frequency split produced no common steel grades")
    if not rare:
        raise ValueError("frequency split produced no low-frequency test grades")
    return common, rare, dict(counts)


def split_common_indices(
    common_indices: Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Match the paper's deterministic 8:2 random row split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if len(common_indices) < 2:
        raise ValueError("at least two common-grade records are required")
    from sklearn.model_selection import train_test_split

    train, validation = train_test_split(
        list(common_indices),
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    return sorted(int(index) for index in train), sorted(int(index) for index in validation)


def assert_disjoint_grade_test(
    train_rows: Iterable[Mapping[str, object]],
    validation_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
) -> None:
    train_grades = {str(row["steel_grade"]) for row in train_rows}
    validation_grades = {str(row["steel_grade"]) for row in validation_rows}
    test_grades = {str(row["steel_grade"]) for row in test_rows}
    overlap = test_grades & (train_grades | validation_grades)
    if overlap:
        raise ValueError(f"rare-grade test leakage detected: {sorted(overlap)}")


def assert_disjoint_record_ids(
    train_rows: Iterable[Mapping[str, object]],
    validation_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
) -> None:
    split_ids = [
        {str(row["id"]) for row in rows}
        for rows in (train_rows, validation_rows, test_rows)
    ]
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise ValueError("record id leakage detected across data splits")
