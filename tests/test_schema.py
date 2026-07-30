import pandas as pd
import pytest

from steel_rlvr.schema import (
    PASS_FEATURES,
    add_derived_features,
    normalize_columns,
    record_prompt,
)


def test_column_normalization_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="collide"):
        normalize_columns(["钢种", "钢种 "])


def test_derived_features_and_prompt_are_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {
                "第1道次入口厚度": 100.0,
                "第2道次入口厚度": 80.0,
                "第3道次入口厚度": 60.0,
                "第2道次出口宽度": 1200.0,
                "第3道次出口宽度": 1190.0,
                "第1道次操作侧轧制力": 1000.0,
                "第1道次传动侧轧制力": 900.0,
                "第2道次操作侧轧制力": 800.0,
                "第2道次传动侧轧制力": 700.0,
            }
        ]
    )
    derived = add_derived_features(frame)
    assert derived.at[0, "压下率_1to2"] == 0.2
    assert derived.at[0, "轧制力_1st_diff"] == 100.0

    features = {name: 1.0 for name in PASS_FEATURES["Pass1"]}
    prompt = record_prompt("Pass1", "Q235B", features)
    assert '"pass":"Pass1"' in prompt
    assert '"steel_grade":"Q235B"' in prompt
    assert "leveling" not in prompt
