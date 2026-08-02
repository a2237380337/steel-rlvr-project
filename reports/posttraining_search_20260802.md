# SFT 后训练方案探索与验证（2026-08-02）

## 结论

本轮最终选择 **频次感知在线 RLVR**，而不是固定偏好对 DPO。该方法从
`Qwen3.5-0.8B-Base` 的正式 LoRA-SFT adapter 出发，每个 prompt 在线采样 4 个
候选，用可计算的数值误差、JSON 格式和物理范围作为奖励，并提高训练集中低频
钢种的数值奖励权重。

在 1,536 条未参与后训练的常见钢种验证样本上，RLVR 将 MAE 从 0.6501 降到
0.6350，下降 2.31%；在 1,017 条训练未见稀有钢种上，MAE 从 0.6486 降到
0.6438，下降 0.74%。两组严格 JSON 率均保持 100%。因此，可以说该方案在本次
固定数据与单种子实验中优于 SFT；不能据此声称已经达到生产部署要求或普遍优于
所有后训练方法。

## 为什么没有选择 DPO/MPO

先后验证了三个方向：

| 方法 | 常见钢种验证 MAE | 相对 SFT | 稀有钢种测试 MAE | 相对 SFT |
|---|---:|---:|---:|---:|
| SFT | 0.6501 | — | 0.6486 | — |
| 难例 continued-SFT | 0.6158 | -5.28% | 未作为最终候选测试 | — |
| 策略难负例 MPO | 0.6195 | -4.70% | 0.6899 | +6.37% |
| 频次感知在线 RLVR（最终） | 0.6350 | -2.31% | 0.6438 | -0.74% |

MPO 使用 SFT 当前策略的误差方向构造近邻负例，并联合优化 `SFT + 0.2 × DPO`
损失。它在常见钢种验证集上有效，但在全部由未见钢种构成的测试集上明显退化，
说明模型学到了现有钢种附近的修正方向，没有获得稳定的跨钢种迁移能力。在线
RLVR 不固定 rejected answer，而是在训练时持续从当前策略采样，并直接按任务误差
评分，更符合连续值生成任务的目标。

## 最终方法

对预测值 \(\hat y\) 和目标 \(y\)，数值奖励为：

```text
r_value = exp(-abs(y_hat - y) / max(train_MAD(pass, grade), 1e-6))
r_tail  = r_value * clip(sqrt(200 / train_grade_frequency), 1, 2)
r_total = r_tail + 0.05 * r_strict_json + 0.05 * r_physical_range
```

所有 MAD、物理边界和钢种频次只从训练分区计算。训练设置如下：

| 项目 | 设置 |
|---|---|
| 起点 | 正式 `sft-main` LoRA adapter |
| RL 优化 | Dr.GRPO loss，`beta=0`，不使用外部奖励模型 |
| 训练样本 | 1,500 条，Pass1/2/3 各 500 条 |
| 在线采样 | 每个 prompt 4 个 completion，temperature 0.7 |
| 训练步数 | 200 |
| 学习率 | `1e-6`，cosine decay，无 warmup |
| 精度 | BF16 |
| 单卡实测 | RX 7900 XT，571.7 秒，峰值 2,391.5 MiB |
| adapter SHA-256 | `5908d432dee4db77e1f2f2c0c1624fb2abfd1acc803a9558a6874c068c97725b` |

200 个训练步骤中 `frac_reward_zero_std` 未出现连续归零，严格 JSON 和物理范围
奖励始终为 1。训练 loss 接近 0 是 Dr.GRPO 组内优势正负抵消后的记录值；非零
梯度、非零 reward std 和冻结生成评测共同证明训练链路有效。

## 最终指标

### 未参与后训练的常见钢种验证集

| 方法 | MAE | R² | Macro-MAE | Worst-group MAE | 严格 JSON |
|---|---:|---:|---:|---:|---:|
| SFT | 0.6501 | -0.1728 | 0.6340 | 1.85 | 100% |
| SFT + RLVR | **0.6350** | **-0.1221** | **0.6210** | **1.80** | 100% |

三道次 MAE 均改善：Pass1 0.6293→0.6191，Pass2 0.6414→0.6295，Pass3
0.6795→0.6564。配对 20,000 次 bootstrap 的 MAE 差值为 -0.0150，95% CI
`[-0.0187, -0.0114]`；538 条改善、307 条变差、691 条不变。

### 训练未见稀有钢种测试集

| 方法 | MAE | R² | Macro-MAE | Worst-group MAE | 严格 JSON |
|---|---:|---:|---:|---:|---:|
| SFT | 0.6486 | -0.0255 | 0.7029 | **1.375** | 100% |
| SFT + RLVR | **0.6438** | **-0.0123** | **0.6959** | 1.400 | 100% |

配对 bootstrap 的 MAE 差值为 -0.00482，95% CI
`[-0.00895, -0.00069]`；259 条改善、210 条变差、548 条不变。Pass2 和 Pass3
改善，Pass1 从 0.6513 小幅变为 0.6534。最差分组也从 1.375 变为 1.40，说明
收益较小且仍存在分组波动。

该测试作为复现性描述而非完全盲测：同算法的历史 checkpoint 在本轮之前已有该
测试集结果，本轮又在 MPO 失败分析中查看过测试指标。正式结论应以验证集结果为
主，并补做多随机种子和新的外部时间切分。

## 统计解释与 11/11 谬误检查

置信等级：**CAUTION**。

- Simpson's paradox：常见钢种验证的三个道次方向一致；稀有钢种测试中 Pass1
  轻微退化、Pass2/3 改善，存在异质性但没有整体与所有分组完全反向。
- Ecological fallacy：结论限定在样本与钢种分组，不推断单条产线记录之外的个体。
- Berkson's paradox：测试集按低频钢种筛选，属于选择性分布；只声称低频留出集表现。
- Collider bias：没有加入由方法和结果共同导致的控制变量。
- Base-rate neglect：主指标是连续值误差，不使用敏感度/特异度类结论。
- Regression to the mean：低频组被强调，但有固定 SFT 对照和独立样本；仍不把组内变化解释为生产因果。
- Survivorship bias：主评测未丢弃无效输出；1,536/1,536 和 1,017/1,017 全部计分。
- Look-elsewhere effect：本轮探索了 continued-SFT、MPO 和 RLVR，存在多方案搜索，未做多重比较校正。
- Garden of forking paths：属于探索性实验；通过固定 hash、同集对照和失败结果留档降低但未消除风险。
- Correlation versus causation：固定数据上的受控训练对比支持本次算法差异，不外推为产线因果效果。
- Reverse causality：训练干预先于评测，方向不适用；真实生产部署仍需前瞻试验。

没有进行多种子重复，测试收益绝对值较小，又存在多方案搜索，因此不能把 bootstrap
尾概率当作未经校正的正式 p 值。下一步至少运行 3 个训练种子，并新增一个按时间或
钢种整组冻结的外部确认集。

## 复现命令

在 WSL Ubuntu 22.04 的 ROCm 环境中：

```bash
source /home/tomotake/venvs/rocm72-py310/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

bash scripts/run_rlvr_tail_smoke.sh
bash scripts/run_rlvr_tail_replication.sh

MODEL_OR_CHECKPOINT=artifacts/checkpoints/rlvr-tail-replication \
EVAL_OUTPUT_DIR=artifacts/evals-rlvr-replication/rlvr-tail-replication \
bash scripts/run_eval.sh configs/eval_posttrain_sft.yaml
```

训练入口为 `src/steel_rlvr/train_rlvr.py`，奖励定义为
`src/steel_rlvr/reward.py`，正式配置为
`configs/rlvr_tail_replication.yaml`。私有数据、逐样本预测与 checkpoint 受
`.gitignore` 保护，不应提交到公开仓库。
