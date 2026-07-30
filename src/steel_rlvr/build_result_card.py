"""Render the formal result card from completed machine-readable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LABELS = {
    "base": "Qwen3.5 Base",
    "sft": "LoRA-SFT",
    "drgrpo": "SFT + Dr. GRPO",
    "tail_aware": "SFT + Tail-aware Dr. GRPO",
}


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pass_cell(row: dict[str, Any], pass_name: str) -> str:
    metrics = row["passes"][pass_name]
    return f"{_number(metrics['mae'])} / {_number(metrics['r2'])}"


def _relative_change(reference: float, candidate: float) -> float:
    return 100.0 * (float(candidate) - float(reference)) / float(reference)


def _change_phrase(value: float) -> str:
    direction = "下降" if value < 0 else "上升"
    return f"{direction} {abs(value):.2f}%"


def _paper_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_markdown(
    matrix: dict[str, Any],
    *,
    paper_baselines: list[dict[str, str]],
) -> str:
    if not matrix.get("complete"):
        raise ValueError(
            "formal result card requires Base/SFT/DrGRPO/tail-aware evaluations; "
            f"missing {matrix.get('missing_labels')}"
        )
    models = matrix["models"]
    lines = [
        "# 项目 2 正式结果卡",
        "",
        "> 本页只由 `summary.json` 自动生成，不接受手工填写或估算数字。",
        "",
        f"- 测试集 SHA256：`{matrix['split_sha256']}`",
        f"- 测试样本 ID SHA256：`{matrix['sample_ids_sha256']}`",
        f"- 评测源码 SHA256：`{matrix['source_tree_sha256']}`",
        f"- 测试对象：19 个训练阶段未见的低频钢种，共 {matrix['sample_count']} 条道次样本",
        "- 数据审计：过滤 1 条钢种值为 `Steel_Grade`、板坯号为 `Slab_ID` 的占位记录",
        "- 主指标：无效输出以训练集目标中位数回填后计入 MAE/R²",
        "- Macro-MAE：对 `道次 × 钢种` 分组 MAE 做等权平均",
        "",
        "| 方法 | 总体 MAE | P1 MAE / R² | P2 MAE / R² | P3 MAE / R² | Macro-MAE | Worst-group | JSON 合法率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = models[label]
        lines.append(
            "| {name} | {mae} | {p1} | {p2} | {p3} | {macro} | {worst} | {json_rate} |".format(
                name=LABELS[label],
                mae=_number(row["mae"]),
                p1=_pass_cell(row, "Pass1"),
                p2=_pass_cell(row, "Pass2"),
                p3=_pass_cell(row, "Pass3"),
                macro=_number(row["macro_mae"]),
                worst=_number(row["worst_group_mae"]),
                json_rate=f"{100 * float(row['strict_json_rate']):.1f}%",
            )
        )
    best_label = min(LABELS, key=lambda label: float(models[label]["mae"]))
    best = models[best_label]
    base = models["base"]
    drgrpo = models["drgrpo"]
    tail_aware = models["tail_aware"]
    best_mae_change = _relative_change(base["mae"], best["mae"])
    best_worst_change = _relative_change(
        base["worst_group_mae"],
        best["worst_group_mae"],
    )
    tail_mae_change = _relative_change(drgrpo["mae"], tail_aware["mae"])
    tail_macro_change = _relative_change(
        drgrpo["macro_mae"],
        tail_aware["macro_mae"],
    )
    tail_worst_change = _relative_change(
        drgrpo["worst_group_mae"],
        tail_aware["worst_group_mae"],
    )
    tail_dominates = all(
        change <= 0
        for change in (tail_mae_change, tail_macro_change, tail_worst_change)
    )
    tail_conclusion = (
        "- 频次加权在预先指定的总体、Macro 和 Worst-group 三项误差上均优于"
        "普通 Dr. GRPO，因此作为本项目最终方法；单个随机种子不足以判断统计显著性。"
        if tail_dominates
        else "- 频次加权存在指标回退，因此只作为受控消融，不作为最终模型。"
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 总体最佳模型：**{LABELS[best_label]}**，MAE={_number(best['mae'], 4)}；"
            f"相对 Base {_change_phrase(best_mae_change)}。",
            f"- 该模型的 Worst-group MAE={_number(best['worst_group_mae'], 4)}；"
            f"相对 Base {_change_phrase(best_worst_change)}。",
            f"- Tail-aware 相对普通 Dr. GRPO：总体 MAE {_change_phrase(tail_mae_change)}，"
            f"Macro-MAE {_change_phrase(tail_macro_change)}，"
            f"Worst-group MAE {_change_phrase(tail_worst_change)}。",
            tail_conclusion,
        ]
    )
    comparison = matrix.get("comparisons", {}).get("tail_aware_minus_drgrpo", {})
    if comparison:
        lines.extend(
            [
                "",
                "## 受控对比",
                "",
                "Tail-aware 相对普通 Dr. GRPO 的差值（负数表示误差降低）：",
                "",
                f"- 总体 MAE：{_number(comparison['mae'], 4)}",
                f"- Macro-MAE：{_number(comparison['macro_mae'], 4)}",
                f"- Worst-group MAE：{_number(comparison['worst_group_mae'], 4)}",
            ]
        )
    lines.extend(
        [
            "",
            "## CAC submitted 论文基线",
            "",
            "| 方法 | P1 MAE/R² | P2 MAE/R² | P3 MAE/R² |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paper_baselines:
        lines.append(
            f"| {row['method']} | {row['pass1_mae']}/{row['pass1_r2']} | "
            f"{row['pass2_mae']}/{row['pass2_r2']} | "
            f"{row['pass3_mae']}/{row['pass3_r2']} |"
        )
    lines.extend(
        [
            "",
            "> 论文数字只作为相同工业场景的已投稿 baseline。RLVR 项目额外过滤了"
            "一条字段占位记录，因此二者不是完全相同的测试表，不能写成直接复现或公平胜负。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--paper-baselines",
        type=Path,
        default=Path("reports/paper_baselines.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/result_card.generated.md"),
    )
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    markdown = build_markdown(
        matrix,
        paper_baselines=_paper_rows(args.paper_baselines),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
