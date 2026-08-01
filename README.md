# Qwen3.5 低频钢种数值 DPO

本项目把本人 CAC submitted 论文 `KD-Steel` 的热轧调平场景改写成结构化数值生成任务，研究 LoRA-SFT 之后的 DPO 是否能改善训练未见钢种的泛化。项目不使用 RAG，也不把离线预测描述为在线控制。

正式 SFT、普通 DPO、频次感知 DPO 和四模型冻结评测均已在单张 AMD RX 7900 XT 上完成。原始产线数据、逐样本预测和 checkpoint 不公开；仓库只提交代码、固定配置、聚合指标和实验报告。

## 任务与数据

34,317 条有效产线记录各拆成 Pass1/Pass2/Pass3 三条时序合法样本。模型读取钢种、道次和当时可观测的工艺特征，只输出：

```json
{"leveling":-0.37}
```

- 44 个常见钢种按 8:2 划分训练/验证，同一原始记录的三个道次不跨分区；
- 19 个频次低于 50 的钢种整组留作测试，共 1,017 条道次样本；
- 训练中位数、MAD、物理边界和钢种频次只从训练分区计算；
- 测试标签不参与训练、偏好对构造、checkpoint 选择或超参数调整；
- 数据审计过滤 1 条 `Slab_ID`/`Steel_Grade` 字段占位记录。

## 方法

```text
Qwen3.5-0.8B-Base（固定 revision）
  → 3,000 条 LoRA-SFT，300 steps，rank=16，alpha=32
  → 从 81,546 条训练偏好对中按道次均衡抽取 3,000 条
  → 普通 DPO / 频次感知 DPO，均为 200 steps，beta=0.1
  → 19 个训练未见钢种的冻结测试
```

偏好对的 `chosen` 是观测目标组成的严格 JSON；`rejected` 保持相同格式，只将数值按确定性规则偏移 `0.5/1.0/1.5 × train_MAD`。这样 DPO 学习的是数值排序，不会把格式差异当捷径。策略模型和显式冻结参考模型都从同一个 SFT adapter 加载。

频次感知组只改变训练对的抽样权重：

```text
w_g = clip(sqrt(200 / n_g_train), 1, 2)
```

普通组和频次感知组分别把加权样本占比从 7.93% 提高到 10.63%，验证集的 768 条偏好对及顺序完全一致。两组其余训练配置相同。

## RX 7900 XT 实测

环境为 WSL2 Ubuntu 22.04、PyTorch 2.9.1 ROCm 7.2、BF16、SDPA。

| 阶段 | 样本 / 步数 | 墙钟时间 | 峰值显存 |
|---|---:|---:|---:|
| LoRA-SFT | 3,000 / 300 | 约 62.9 分钟 | 3.43 GiB |
| 普通 DPO | 3,000 / 200 | 48分51秒 | 6.47 GiB |
| 频次感知 DPO | 3,000 / 200 | 49分15秒 | 6.47 GiB |

DPO 结束时，普通/频次感知组的验证损失为 0.2506/0.2503，偏好准确率为 86.07%/86.59%。这些值只用于训练诊断，不替代冻结测试指标。

## 冻结测试结果

四组模型使用相同测试文件、相同 1,017 个样本 ID、贪心生成和相同评测源码哈希。

| 方法 | 总体 MAE | R² | Macro-MAE | Worst-group | 严格 JSON |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 Base | 0.6864 | -0.2313 | **0.6895** | 1.6783 | 100.00% |
| LoRA-SFT | **0.6486** | **-0.0255** | 0.7029 | **1.3750** | **100.00%** |
| SFT + DPO | 0.6488 | -0.0263 | 0.7028 | **1.3750** | 99.12% |
| SFT + 频次感知 DPO | 0.6488 | -0.0263 | 0.7028 | **1.3750** | 99.02% |

结论：

- SFT 相对 Base 将总体 MAE 降低 5.52%，Worst-group MAE 降低 18.07%；
- 普通 DPO 相对 SFT 的总体 MAE 回退 0.03%，没有继续提高回归精度；
- 两个 DPO adapter 权重不同，但 1,017 条测试样本的贪心数值输出完全一致，频次感知抽样没有带来可测增益；
- 四组模型数值可解析率均为 100%，物理区间违规率为 0；
- 所有总体 R² 仍为负，且 SFT/DPO 的 Macro-MAE 均差于 Base。项目不声称超过论文中的 KD-Steel，也不声称具备在线控制能力。

这组负结果说明：偏好准确率上升不等于连续值生成误差下降；固定偏移负样本对当前策略过于容易，DPO 只改变了 1,017 个测试预测中的 4 个。更合理的后续方向是使用当前策略生成的近邻难负样本、扩大种子实验并在验证集上预注册选择规则。

机器可读结果见 `results/formal_evaluation_matrix.json`，自动结果卡见 `reports/result_card.generated.md`，完整分析见 `reports/main_experiment_report.md`。

## 目录

```text
configs/                 SFT、DPO 与统一评测配置
data/                    私有数据目录；processed 不提交
docs/                    实验协议与实现状态
reports/                 论文 baseline、结果表与实验报告
results/                 公开的聚合正式指标
scripts/                 环境、数据、训练和评测入口
src/steel_rlvr/          数据、偏好对、训练和评测实现
tests/                   数据隔离、DPO 协议和结果审计测试
```

## 运行

```bash
git clone YOUR_REPOSITORY_URL steel-rlvr-project
cd steel-rlvr-project
bash scripts/install_rocm72_venv.sh
source "${ROCM_VENV_PATH:-${HOME}/venvs/rocm72-py310}/bin/activate"
bash scripts/setup_environment.sh
export STEEL_DATA_PATH="/path/to/private/steel_cleaned.xlsx"
```

先做 smoke：

```bash
bash scripts/run_gpu_smoke.sh
```

完整实验：

```bash
bash scripts/prepare_data.sh
bash scripts/build_preferences.sh
bash scripts/run_sft_main.sh
bash scripts/run_dpo_baseline.sh
bash scripts/run_dpo_tail_aware.sh
bash scripts/run_all_evals.sh
```

也可以用 `bash scripts/run_formal_experiment.sh` 串行执行。每个输出目录必须为空，程序会拒绝混用两次运行的 checkpoint 或指标。

发布前执行：

```bash
bash scripts/verify_release.sh
```

该命令检查代码、测试、JSON/CSV 一致性、公开文件完整性、忽略规则和本机绝对路径。
