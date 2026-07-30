"""Column, feature and prompt contract shared by every experiment stage."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 2
GRADE_COLUMN = "钢种"
TARGETS = {
    "Pass1": "第1道次操作工调平值",
    "Pass2": "第2道次操作工调平值",
    "Pass3": "第3道次操作工调平值",
}

# These features reproduce the temporal boundary used by the submitted CAC
# experiment. A pass never receives measurements produced after that pass.
PASS_BASE_FEATURES = {
    "Pass1": [
        "第1道次入口厚度",
        "第2道次入口厚度",
        "第3道次入口厚度",
        "第2道次出口宽度",
        "第3道次出口宽度",
    ],
    "Pass2": [
        "第1道次入口厚度",
        "第2道次入口厚度",
        "第3道次入口厚度",
        "第2道次出口宽度",
        "第3道次出口宽度",
        "第1道次操作侧轧制力",
        "第1道次传动侧轧制力",
        "第1道次出口头部长度L1",
        "第1道次出口头部偏移量",
        "第1道次出口尾部长度L2",
        "第1道次出口尾部偏移量",
    ],
    "Pass3": [
        "第1道次入口厚度",
        "第2道次入口厚度",
        "第3道次入口厚度",
        "第2道次出口宽度",
        "第3道次出口宽度",
        "第1道次操作侧轧制力",
        "第1道次传动侧轧制力",
        "第2道次操作侧轧制力",
        "第2道次传动侧轧制力",
        "第1道次出口头部长度L1",
        "第1道次出口头部偏移量",
        "第1道次出口尾部长度L2",
        "第1道次出口尾部偏移量",
        "第2道次出口头部长度L1",
        "第2道次出口头部偏移量",
        "第2道次出口尾部长度L2",
        "第2道次出口尾部偏移量",
    ],
}

PASS_DERIVED_FEATURES = {
    "Pass1": ["压下率_1to2", "压下率_2to3", "总压下率", "宽度收缩_2to3"],
    "Pass2": [
        "压下率_1to2",
        "压下率_2to3",
        "总压下率",
        "轧制力_1st_diff",
        "宽度收缩_2to3",
    ],
    "Pass3": [
        "压下率_1to2",
        "压下率_2to3",
        "总压下率",
        "轧制力_1st_diff",
        "轧制力_2nd_diff",
        "宽度收缩_2to3",
    ],
}

PASS_FEATURES = {
    pass_name: PASS_BASE_FEATURES[pass_name] + PASS_DERIVED_FEATURES[pass_name]
    for pass_name in TARGETS
}

SYSTEM_PROMPT = (
    "你是热轧精轧调平值预测模型。根据钢种、当前道次和此时可观测的工艺特征，"
    '只返回一个 JSON 对象，格式必须为 {"leveling": 数值}；不要解释，不要添加其他字段。'
)


def normalize_columns(columns: Sequence[object]) -> list[str]:
    """Strip whitespace while rejecting collisions introduced by stripping."""

    normalized = [str(column).strip() for column in columns]
    duplicates = sorted({column for column in normalized if normalized.count(column) > 1})
    if duplicates:
        raise ValueError(f"column names collide after trimming whitespace: {duplicates}")
    return normalized


def required_source_columns() -> set[str]:
    """Columns that must be present before derived features are constructed."""

    return {
        GRADE_COLUMN,
        *TARGETS.values(),
        *(column for features in PASS_BASE_FEATURES.values() for column in features),
    }


def validate_source_columns(columns: Sequence[str]) -> None:
    missing = sorted(required_source_columns() - set(columns))
    if missing:
        raise ValueError(f"source data is missing required columns: {missing}")


def add_derived_features(frame: Any) -> Any:
    """Return a copy with the leakage-safe features used by the CAC baseline."""

    result = frame.copy()
    result["压下率_1to2"] = (
        result["第1道次入口厚度"] - result["第2道次入口厚度"]
    ) / result["第1道次入口厚度"]
    result["压下率_2to3"] = (
        result["第2道次入口厚度"] - result["第3道次入口厚度"]
    ) / result["第2道次入口厚度"]
    result["总压下率"] = (
        result["第1道次入口厚度"] - result["第3道次入口厚度"]
    ) / result["第1道次入口厚度"]
    result["轧制力_1st_diff"] = result["第1道次操作侧轧制力"] - result["第1道次传动侧轧制力"]
    result["轧制力_2nd_diff"] = result["第2道次操作侧轧制力"] - result["第2道次传动侧轧制力"]
    result["宽度收缩_2to3"] = result["第2道次出口宽度"] - result["第3道次出口宽度"]
    return result


def _finite_number(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"feature {field!r} is not finite")
    return number


def record_prompt(pass_name: str, grade: str, features: Mapping[str, object]) -> str:
    """Serialize one record deterministically without exposing its target."""

    if pass_name not in PASS_FEATURES:
        raise ValueError(f"unsupported pass: {pass_name}")
    expected = PASS_FEATURES[pass_name]
    missing = [field for field in expected if field not in features]
    if missing:
        raise ValueError(f"{pass_name} prompt is missing features: {missing}")
    payload = {
        "pass": pass_name,
        "steel_grade": str(grade),
        "process_features": {
            field: round(_finite_number(features[field], field), 6)
            for field in expected
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def target_completion(target: float) -> str:
    number = _finite_number(target, "leveling")
    return json.dumps(
        {"leveling": round(number, 6)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def schema_manifest() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "grade_column": GRADE_COLUMN,
        "targets": TARGETS,
        "pass_base_features": PASS_BASE_FEATURES,
        "pass_derived_features": PASS_DERIVED_FEATURES,
        "system_prompt": SYSTEM_PROMPT,
        "completion_contract": {"leveling": "finite float"},
    }
