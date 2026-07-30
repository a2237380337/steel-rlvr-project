# Qwen3.5 低频钢种调平值 RLVR

本项目沿用 CAC submitted 论文 `KD-Steel` 的工业场景，把调平值预测改写成一个不使用 RAG 的可验证奖励后训练（RLVR）实验。

当前状态：正式 SFT、两组 Dr. GRPO 和四模型统一评测均已在 RX 7900 XT 上完成。机器可读结果见 `results/formal_evaluation_matrix.json`，自动结果卡见 `reports/result_card.generated.md`，完整实验报告见 `reports/main_experiment_report.md`，脱敏运行身份与哈希索引见 `reports/provenance.md`。

公开仓库只包含训练/评测代码、配置、聚合指标和实验报告。原始产线数据、
逐样本 Prompt/预测、checkpoint 与本机环境日志均由 `.gitignore` 隔离。

## 任务

每条原始产线记录拆成三个样本：

- Pass1：只使用第一道次决策时可观测的特征；
- Pass2：增加第一道次轧制力和出口偏移等结果；
- Pass3：再增加第二道次结果。

模型接收钢种、道次和结构化工艺特征，只输出：

```json
{"leveling":-0.37}
```

三个道次独立预测一个调平值，避免把后续测量泄漏给前一道次。

## 方法

```text
Qwen3.5-0.8B-Base（固定 revision）
  → 3,000 条结构化样本 LoRA-SFT
  → 普通 Dr. GRPO
  → 低频感知 Dr. GRPO
  → 19 个训练阶段未见的低频钢种统一测试
```

数值奖励：

```text
r_value = exp(-abs(prediction - target) / train_MAD)
```

总奖励由数值准确性、严格 JSON 和训练集物理区间三部分组成：

```text
r = r_value + 0.05 * r_format + 0.05 * r_physical
```

低频感知版本只改变数值奖励：

```text
w_g = clip(sqrt(200 / n_g_train), 1, 2)
r_tail = w_g * r_value + 0.05 * r_format + 0.05 * r_physical
```

这里的 `n_g_train` 只统计训练分区，不使用低频测试钢种标签。普通和低频感知 Dr. GRPO 从同一个 SFT checkpoint 出发，除 `tail_aware` 开关和输出目录外配置完全一致。

## 数据隔离

正式数据契约固定为：

- 原始 Excel 有 34,318 行；数据审计过滤 1 条 `Slab_ID`/`Steel_Grade` 字段占位记录，保留 34,317 条真实记录、63 个钢种；
- 全数据频次不少于 50 的钢种随机按 8:2 分成训练/验证；
- 频次少于 50 的 19 个真实钢种只进入测试，每个道次 339 条，共 1,017 条；
- 测试标签不参与 SFT、GRPO 奖励、归一化、物理范围或 checkpoint 选择；
- 训练中位数、MAD、1%/99% 分位区间全部由训练标签计算；
- `data_report.json` 保存原始文件、三个 split 和 schema 的 SHA256。

正式模式如果不能复现这些数量会直接失败；toy data 需显式传入 `--allow-contract-mismatch`。

## RX 7900 XT 运行配置

- WSL2 + ROCm PyTorch 2.9.1；
- 单卡 RX 7900 XT 20 GB；
- Qwen3.5-0.8B-Base，BF16，SDPA，LoRA；
- 不安装 CUDA FlashAttention、bitsandbytes、DeepSpeed 或 vLLM；
- SFT：按三个道次均衡抽取 3,000 条、300 steps、LoRA rank 16；
- 每个 Dr. GRPO：按三个道次均衡抽取 1,500 条、200 steps、4 completions、128-token 上限；
- `loss_type=dr_grpo`、`scale_rewards=false`。

## 正式结果

四组模型使用同一测试文件 SHA256 `6922b193...53a49` 和同一组 1,017 个样本 ID。所有模型的数值可解析率、严格 JSON 合法率均为 100%，物理区间违规率为 0。

| 方法 | 总体 MAE | R² | Macro-MAE | Worst-group MAE |
|---|---:|---:|---:|---:|
| Qwen3.5 Base | 0.6864 | -0.2313 | 0.6895 | 1.6783 |
| LoRA-SFT | 0.6485 | -0.0256 | 0.7026 | 1.3750 |
| SFT + Dr. GRPO | 0.6470 | -0.0186 | 0.7011 | 1.4000 |
| SFT + Tail-aware Dr. GRPO | **0.6438** | **-0.0133** | **0.6967** | **1.3750** |

尾部感知奖励相对普通 Dr. GRPO：

- 总体 MAE 降低 0.50%；
- Macro-MAE 降低 0.63%；
- Worst-group MAE 降低 1.79%。

最终模型相对 Base 的总体 MAE 降低 6.22%，Worst-group MAE 降低 18.07%，但 Macro-MAE 仍比 Base 高 1.05%，所有模型的总体 R² 仍为负。这说明频次加权 RLVR 改善了当前离线测试中的整体误差和低频组误差，但结果不能替代论文中的 KD-Steel，也不足以支持在线控制。

正式资源实测：

- SFT：300 steps，约 63 分钟墙钟时间，峰值显存 3.43 GiB；
- 普通 Dr. GRPO：200 steps，约 11.7 分钟，峰值显存 2.37 GiB；
- Tail-aware Dr. GRPO：200 steps，约 11.4 分钟，峰值显存 2.34 GiB；
- 23 项 CPU 单元测试与 BF16 ROCm kernel smoke 全部通过。

## 目录

```text
configs/                 固定 SFT、GRPO、评测配置
data/                    私有数据目录；processed 不提交
docs/                    实验协议与实现说明
reports/                 论文 baseline 和结果卡模板
results/                 自动汇总的正式指标
scripts/                 环境、数据、训练、评测入口
src/steel_rlvr/          数据、奖励、训练、评测实现
tests/                   23 项 CPU 单元测试
DATA_CARD.md             数据来源、切分、泄漏控制与公开范围
MODEL_CARD.md            模型训练、结果、用途与限制
LICENSE                  仅覆盖本仓库原创代码和文档
```

## 运行顺序

在 WSL2 Ubuntu 22.04 中创建独立 ROCm 7.2 环境并安装项目依赖：

```bash
git clone YOUR_REPOSITORY_URL steel-rlvr-project
cd steel-rlvr-project
bash scripts/install_rocm72_venv.sh
source "${ROCM_VENV_PATH:-${HOME}/venvs/rocm72-py310}/bin/activate"
bash scripts/setup_environment.sh
```

原始数据不随仓库发布。获得合法授权后，把环境变量指向本地 Excel：

```bash
export STEEL_DATA_PATH="/path/to/private/steel_cleaned.xlsx"
bash scripts/prepare_data.sh
```

先跑 smoke，只验证链路：

```bash
bash scripts/run_gpu_smoke.sh
```

正式实验：

```bash
bash scripts/run_sft_main.sh
bash scripts/run_grpo_baseline.sh
bash scripts/run_grpo_tail_aware.sh
bash scripts/run_all_evals.sh
```

也可以使用 `bash scripts/run_formal_experiment.sh` 串行完成全部步骤。每个输出目录必须为空，代码会拒绝把两次实验的 checkpoint 或指标混在一起。

## 正式评测

Base、SFT、普通 Dr. GRPO、低频感知 Dr. GRPO 统一读取同一个 `test.jsonl` 哈希。主要报告：

- Pass1/2/3 的 MAE、R²；
- 对 `道次 × 钢种` 等权平均的 Macro-MAE；
- Worst-group MAE；
- 数值可解析率、严格 JSON 合法率和物理越界率；
- 峰值显存、生成 token 和逐样本失败类型。

无法解析出有限数值的输出不会被删除，而是用该道次的训练集目标中位数回填后计入主 MAE/R²，同时单独报告 valid-only MAE。

`steel_rlvr.build_result_card` 只有在四组正式评测全部存在且测试集哈希一致时才生成结果卡，因此不会用占位值或论文数字伪造项目结果。

准备公开或投递前运行：

```bash
bash scripts/verify_release.sh
```

该命令会重新运行代码检查和测试，并确认正式 JSON/CSV 一致、关键报告已被
Git 跟踪、私有数据与 checkpoint 被忽略、公开文件不含本机绝对路径。

正式实验早于本目录的独立 Git 初始化。运行 manifest 如实记录了空 Git commit
与 dirty 状态；本仓库不会用事后 commit 冒充训练时 commit。训练时的配置、
源码树、数据和样本顺序哈希已脱敏整理到 `reports/provenance.md`。

第一次正式评测发现一条字段占位记录被误当作低频钢种；旧的 1,020 样本结果已完整归档在 `artifacts/archive/pre-sentinel-cleanup-20260730/`。修复后训练与验证 JSONL 的 SHA256 均未变化，所以原 checkpoint 仍有效，只重新运行了测试评测。
