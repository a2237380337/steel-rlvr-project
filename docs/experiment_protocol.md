# 实验协议

## 研究问题

在完全不使用低频钢种标签训练的冷启动设置下，确定性数值奖励的 Dr. GRPO 是否比 LoRA-SFT 更好；进一步，按训练分区钢种频次放大低频区域的数值奖励，是否改善 Macro-MAE 和 Worst-group MAE。

本项目基于历史监督数据进行上下文策略优化，不属于在线轧机控制，也没有部署到生产线。数据中没有状态转移和干预反事实，因此未采用依赖闭环交互假设的 PPO/SAC。

## 固定数据协议

1. 清洗列名首尾空格，并验证论文使用的中文字段完整存在。
2. 复现论文的时序合法基础特征和六个派生特征。
3. 用完整数据中的钢种频次划分 common/rare。
4. common 记录用 seed 42 随机按 8:2 划分训练/验证；同一原始记录的三个道次不会跨 split。
5. rare 记录全部作为正式测试。
6. 训练中位数、MAD 和物理区间只从训练标签计算。
7. 低频感知权重的 `n_g` 只统计训练分区。

初始预注册合同：34,318 条记录、64 个类别值、20 个低频类别、每个道次 340 条测试记录。任一项不符，数据准备默认失败。

### 数据审计修订（2026-07-30）

第一次统一评测后按类别名做人工审计，发现其中 1 行的板坯号为 `Slab_ID`、钢种为 `Steel_Grade`，是字段占位记录而非真实钢种。该判断只依赖原始字段语义，不依赖模型误差。最终合同修订为：

- 原始 34,318 行，排除 1 条 sentinel 后为 34,317 条；
- 63 个真实钢种，其中 19 个低频钢种进入测试；
- 每个道次 339 条，共 1,017 条测试样本。

修订前后 `train.jsonl` SHA256 均为 `ef60984c...abc2`，`validation.jsonl` 均为 `7268251f...c1c1`，证明训练数据、验证数据和已训练 checkpoint 不受影响。旧测试结果归档，只重跑四模型评测。

## 固定对照

| 组别 | 方法 | 作用 |
|---|---|---|
| A | CAC submitted 论文模型 | 相同场景的外部 baseline |
| B | Qwen3.5 Base | 未后训练语言模型 |
| C | Qwen3.5 LoRA-SFT | 监督后训练基线 |
| D | SFT + Dr. GRPO | 普通数值奖励 |
| E | SFT + Tail-aware Dr. GRPO | 项目方法 |

D/E 共用 SFT checkpoint、1,500 条抽样上限、200 steps、4 completions、LoRA、优化器、随机种子和辅助奖励，只改变数值奖励是否乘以训练频次权重。

## 指标

主指标：

- 三个道次分别计算 MAE、R²；
- Macro-MAE：每个 `道次 × 钢种` 先算 MAE，再等权平均；
- Worst-group MAE：上述组中最大的 MAE。

完整性指标：

- 有限数值可解析率；
- 严格 JSON 合法率；
- 数值输出中的物理区间违规率；
- invalid-output fallback 后的主指标与 valid-only MAE；
- 生成 token、峰值显存、wall-clock、失败类型。

## Checkpoint 规则

- SFT 只根据 common-grade validation 的 `eval_loss` 选最优 adapter。
- 两个 Dr. GRPO 主实验使用预先固定的第 200 step 最终模型，不查看 test 选择 checkpoint。
- 如需比较中间 checkpoint，只能用 `eval_validation.yaml` 和 `select_checkpoint.py`；该工具会校验 validation 文件哈希。
- 正式测试只在模型和超参数冻结后运行。

## 结果解释与限制

- Tail-aware 可能降低 Worst-group 但提高总体 MAE，应如实报告这种 Pareto 取舍。
- LLM 不一定优于 FT-Transformer；负结果可以反映语言模型用于连续工业回归时的限制。
- 论文 baseline 只用于场景对比，与 RLVR 代码复现结果分开报告。
- 正式报告只采用 `summary.json` 和统一 matrix 中能够复核的结果。

## 正式结果

- Tail-aware Dr. GRPO 为总体 MAE 最优模型：0.6438，相对 Base 0.6864 降低 6.22%；
- Worst-group MAE 从 Base 的 1.6783 降到 1.3750，降低 18.07%；
- 相对普通 Dr. GRPO，Tail-aware 的总体/Macro/Worst-group MAE 分别降低 0.50%/0.63%/1.79%；
- 100% 输出可解析为严格 JSON，物理区间违规率为 0；
- Macro-MAE 仍比 Base 高 1.05%，R² 仍为负，报告中保留这些负结果。
