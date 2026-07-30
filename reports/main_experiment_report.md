# Qwen3.5 低频钢种数值决策 RLVR：正式实验报告

## 结论先行

项目已在单张 AMD Radeon RX 7900 XT 上完成 Qwen3.5-0.8B-Base 的 LoRA-SFT、普通 Dr. GRPO、钢种频次感知 Dr. GRPO，以及四模型统一评测。

最终 Tail-aware Dr. GRPO 在 19 个训练阶段未见低频钢种、1,017 条道次样本上取得：

- 总体 MAE 0.6438，较 Base 0.6864 降低 6.22%；
- Worst-group MAE 1.3750，较 Base 1.6783 降低 18.07%；
- 严格 JSON 合法率 100%，物理区间违规率 0；
- 相对普通 Dr. GRPO，总体/Macro/Worst-group MAE 分别降低 0.50%/0.63%/1.79%。

最终 Macro-MAE 0.6967 仍比 Base 0.6895 高 1.05%，总体 R² 仍为负，且只运行一个随机种子。这些结果不足以证明统计显著，也不足以支持模型替代 KD-Steel 或用于在线轧机控制。

## 研究问题与受控变量

研究问题有两层：

1. 在完全不使用低频测试钢种标签训练的条件下，数值可验证奖励 Dr. GRPO 能否继续改善 LoRA-SFT？
2. 在相同 SFT 起点、训练样本顺序、seed、步数、优化器和辅助奖励下，只把数值奖励乘以训练钢种频次权重，能否改善总体、Macro 和 Worst-group MAE？

普通与 Tail-aware Dr. GRPO 的唯一方法差异为：

```text
w_g = clip(sqrt(200 / n_g_train), 1, 2)
r_tail = w_g * r_value + 0.05 * r_json + 0.05 * r_physical
```

其中 `n_g_train` 只来自每个道次的有效训练分区；测试钢种在训练中的频次为 0，标签从未进入奖励、归一化、物理范围或模型选择。

## 数据与审计

原始文件共有 34,318 行。第一次统一评测后做类别语义审计，发现 1 行的板坯号为 `Slab_ID`、钢种为 `Steel_Grade`，是字段占位记录。最终数据合同为：

- 34,317 条真实记录、63 个真实钢种；
- 44 个常见钢种按 seed 42 做 8:2 训练/验证；
- 19 个频次小于 50 的钢种整组留作测试；
- 训练/验证/测试原始记录为 27,182/6,796/339；
- 每条记录拆为三个道次样本，最终 JSONL 为 81,546/20,388/1,017 条。

占位记录只在测试集。修复前后：

- `train.jsonl` SHA256：`ef60984c117b8f5b6118e3e2272e160864e95254c20b44919e6d8af18fa0abc2`；
- `validation.jsonl` SHA256：`7268251f77d0fe159602ba8a88c6b4511ea43538706388fd9086b7ce007561c1`。

二者完全不变，因此训练 checkpoint 可复用；旧 1,020 样本评测完整归档到 `artifacts/archive/pre-sentinel-cleanup-20260730/`，只重新运行了测试推理。

## 环境与训练配置

| 项目 | 配置 |
|---|---|
| GPU | AMD Radeon RX 7900 XT，19.94 GiB |
| 系统 | WSL2 Ubuntu-22.04，ROCm/HIP 7.2 |
| PyTorch | 2.9.1+ROCm 7.2 |
| Transformers / TRL / PEFT | 5.15.0.dev0 / 1.9.2 / 0.20.0 |
| 基座模型 | Qwen/Qwen3.5-0.8B-Base |
| 固定 revision | `a9a407bcae463285164cc9133995c515379cebe5` |
| 精度与注意力 | BF16，SDPA，text-only |
| SFT | 3,000 条、300 steps、LoRA rank 16、grad accumulation 16 |
| 两组 Dr. GRPO | 各 1,500 条、200 steps、4 completions、`beta=0`、`scale_rewards=false` |

实测资源：

| 阶段 | 墙钟时间 | 峰值显存 |
|---|---:|---:|
| LoRA-SFT | 约 62.9 分钟 | 3.43 GiB |
| 普通 Dr. GRPO | 约 11.7 分钟 | 2.37 GiB |
| Tail-aware Dr. GRPO | 约 11.4 分钟 | 2.34 GiB |

SFT 的验证 loss 从 step 50 的 0.2600 逐步降至最终约 0.2447。两组 RL 的 200 个 step 均没有零方差奖励组：

| 训练组 | 平均总奖励 | 平均 reward std | 平均 JSON 奖励 | JSON 非满分 step | 平均完成长度 |
|---|---:|---:|---:|---:|---:|
| 普通 Dr. GRPO | 0.4660 | 0.2088 | 0.9788 | 15/200 | 12.16 |
| Tail-aware Dr. GRPO | 0.4778 | 0.2135 | 0.9888 | 9/200 | 11.70 |

## 正式测试结果

四模型使用相同测试文件和相同样本 ID：

- 测试文件 SHA256：`6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49`；
- 样本 ID SHA256：`16688d7338105327588dcafe04f738d2b5d59041203a86c1994d8630e8113139`；
- 评测时源码 SHA256：`ff61ca60d7a37395ae88a75692861d744990fa72b63602c1f0efc60838f2e3a2`。

| 方法 | 总体 MAE | R² | Macro-MAE | Worst-group | JSON |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 Base | 0.6864 | -0.2313 | 0.6895 | 1.6783 | 100% |
| LoRA-SFT | 0.6485 | -0.0256 | 0.7026 | 1.3750 | 100% |
| SFT + Dr. GRPO | 0.6470 | -0.0186 | 0.7011 | 1.4000 | 100% |
| SFT + Tail-aware Dr. GRPO | **0.6438** | **-0.0133** | **0.6967** | **1.3750** | **100%** |

分道次 MAE：

| 方法 | Pass1 | Pass2 | Pass3 |
|---|---:|---:|---:|
| Base | 0.7063 | 0.6792 | 0.6738 |
| SFT | 0.6513 | 0.6522 | 0.6419 |
| Dr. GRPO | 0.6537 | 0.6504 | 0.6369 |
| Tail-aware Dr. GRPO | **0.6519** | **0.6501** | **0.6292** |

## 结果解释

1. SFT 提供主要增益：它先学会稳定 JSON 输出和粗粒度数值关系。
2. 普通 Dr. GRPO 相对 SFT 只带来小幅总体收益，并使 Worst-group 略有回退，说明只优化逐样本数值奖励不足以保证组间公平。
3. 频次加权在严格控制变量下同时改善普通 Dr. GRPO 的总体、Macro 和 Worst-group MAE，增益虽小但方向一致。
4. Tail-aware 的 Macro-MAE 仍未优于 Base，说明在常见钢种内部做频次重加权，不能完全解决对训练未见钢种的冷启动泛化。
5. CAC submitted 论文中的 KD-Steel 三道次 MAE/R² 仍明显更好。它是同场景外部 baseline，不是本项目复现结果；这也说明工业连续值任务中，专用表格模型/知识融合仍有优势。

## 实验流程

### 输入

- 私有原始 Excel：仅本地读取，不进入 Git；
- 数据报告：`data/processed/data_report.json`；
- 固定模型 revision 和 YAML 配置：`configs/`。

### 处理

- `src/steel_rlvr/prepare_data.py`：字段恢复、派生特征、sentinel 清洗、整钢种隔离；
- `src/steel_rlvr/train_sft.py`：assistant-only LoRA-SFT；
- `src/steel_rlvr/train_grpo.py`：普通与 Tail-aware Dr. GRPO；
- `src/steel_rlvr/evaluate.py`：greedy、同批量、逐样本可审计评测。

### 产物

- 正式 checkpoint：`artifacts/checkpoints/sft-main/`、`grpo-baseline/`、`grpo-tail-aware/`；
- 四模型预测与 summary：`artifacts/evals/`；
- 机器可读矩阵：`results/formal_evaluation_matrix.json`；
- 自动结果卡：`reports/result_card.generated.md`；
- 控制台与逐 step 日志：`artifacts/logs/` 和各 checkpoint 的 `logs/`。

### 验证

- `ruff check src tests` 通过；
- `pytest -q`：23 项通过（含正式结果一致性与发布检查）；
- BF16 ROCm kernel smoke 通过；
- SFT 20-step、Dr. GRPO 10-step 和端到端 32 样本 smoke 通过；
- 四模型 matrix 校验 base revision、测试 SHA256、样本 ID SHA256 和样本数一致。
- `bash scripts/verify_release.sh` 通过：23 项测试、JSON/CSV 一致性、公开文件
  完整性、Git 忽略规则和本机路径扫描均通过。

### 已知限制

- 单 GPU、单随机种子，未做置信区间；
- 测试为历史离线数据，不含在线干预反事实；
- 19 个测试钢种训练期完全未见，钢种名称本身无法提供稳定材料语义；
- Qwen3.5 结果未超过 KD-Steel，不能包装成工业 SOTA；
- 原始产线数据和逐条预测不得公开。
