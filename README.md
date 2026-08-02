# Qwen3.5 钢种调平数值生成：SFT + 在线 RLVR

本项目把 CAC submitted 论文 `KD-Steel` 的热轧调平场景改写为结构化数值生成
任务，研究小模型在 LoRA-SFT 之后如何通过后训练继续降低误差。最终方案使用
Qwen3.5-0.8B-Base、LoRA-SFT 和频次感知在线 RLVR，不涉及 RAG，也不把离线
预测描述成在线控制。

项目已在单张 AMD RX 7900 XT、WSL2 Ubuntu 22.04、PyTorch 2.9.1 ROCm 7.2
环境中完成训练与评测。原始产线数据、逐样本预测和 checkpoint 不公开；仓库保存
代码、固定配置、聚合指标和实验报告。

## 最终结果

| 数据 | 方法 | MAE | R² | Macro-MAE | Worst-group | 严格 JSON |
|---|---|---:|---:|---:|---:|---:|
| 1,536 条常见钢种验证 | SFT | 0.6501 | -0.1728 | 0.6340 | 1.85 | 100% |
| 1,536 条常见钢种验证 | SFT + RLVR | **0.6350** | **-0.1221** | **0.6210** | **1.80** | 100% |
| 1,017 条未见稀有钢种 | SFT | 0.6486 | -0.0255 | 0.7029 | **1.375** | 100% |
| 1,017 条未见稀有钢种 | SFT + RLVR | **0.6438** | **-0.0123** | **0.6959** | 1.400 | 100% |

RLVR 在常见钢种验证集上将 MAE 降低 2.31%，在未见稀有钢种上降低 0.74%。
收益不大，且稀有钢种的最差分组略有退化，因此结论限定为：在本次固定数据、
固定配置和单随机种子下，RLVR 的总体误差优于 SFT。完整失败实验、bootstrap
区间和局限见 [后训练方案探索报告](reports/posttraining_search_20260802.md)。

## 任务与数据

34,317 条有效产线记录各拆成 Pass1、Pass2、Pass3 三条时序合法样本。模型读取
钢种、道次和当时可观测的工艺特征，只输出：

```json
{"leveling": -0.37}
```

- 44 个常见钢种按 8:2 划分训练和验证，同一原始记录的三个道次不跨分区；
- 19 个频次低于 50 的钢种整组留作测试，共 1,017 条道次样本；
- 训练中位数、MAD、物理边界和钢种频次只从训练分区计算；
- 无效输出不会从 MAE/R² 中删除，而是按预先记录的训练中位数计分；
- 测试标签不参与训练、checkpoint 选择或超参数调整。

## 方法

最终 RLVR 从正式 SFT adapter 出发，每个 prompt 在线采样 4 个候选。数值奖励按
训练集 MAD 归一化，再提高低频钢种权重：

```text
r_value = exp(-abs(y_hat - y) / max(train_MAD(pass, grade), 1e-6))
r_tail  = r_value * clip(sqrt(200 / train_grade_frequency), 1, 2)
r_total = r_tail + 0.05 * r_strict_json + 0.05 * r_physical_range
```

正式配置使用 1,500 条样本（每个道次 500 条）、200 steps、4 generations、
BF16、学习率 `1e-6` 和 Dr.GRPO loss。单卡训练耗时 571.7 秒，峰值显存
2,391.5 MiB。

项目也保留了 continued-SFT、普通 DPO、频次感知 DPO 和策略难负例 MPO 的
实现。固定偏好对 DPO 几乎没有改变预测；MPO 虽在常见钢种验证集上改善，却在
未见稀有钢种上退化，因此没有作为最终方案。

## 主要文件

```text
configs/rlvr_tail_replication.yaml       最终 RLVR 配置
src/steel_rlvr/train_rlvr.py             在线 RLVR 训练入口
src/steel_rlvr/reward.py                 数值、格式、物理和低频奖励
scripts/run_rlvr_tail_replication.sh      正式训练脚本
reports/posttraining_search_20260802.md   完整实验与统计解释
results/posttraining_search_20260802.json 机器可读结果（本地）
```

`configs/`、`scripts/` 和 `src/steel_rlvr/` 中同时保留 SFT、DPO/MPO 对照和统一
评测实现。`tests/` 覆盖数据隔离、偏好构造、奖励函数和结果审计。

## 运行

```bash
git clone YOUR_REPOSITORY_URL steel-rlvr-project
cd steel-rlvr-project
bash scripts/install_rocm72_venv.sh
source "${ROCM_VENV_PATH:-${HOME}/venvs/rocm72-py310}/bin/activate"
bash scripts/setup_environment.sh
export STEEL_DATA_PATH="/path/to/private/steel_cleaned.xlsx"
```

准备数据和训练 SFT 后，运行 RLVR：

```bash
bash scripts/prepare_data.sh
bash scripts/run_sft_main.sh
bash scripts/run_rlvr_tail_smoke.sh
bash scripts/run_rlvr_tail_replication.sh
```

评测：

```bash
MODEL_OR_CHECKPOINT=artifacts/checkpoints/rlvr-tail-replication \
EVAL_OUTPUT_DIR=artifacts/evals-rlvr-replication/rlvr-tail-replication \
bash scripts/run_eval.sh configs/eval_posttrain_sft.yaml
```

提交代码前运行：

```bash
ruff check src tests
pytest -q
```

每个训练和评测输出目录必须为空，程序会拒绝把两次运行写入同一目录。私有数据、
checkpoint 和逐样本预测受 `.gitignore` 保护。
