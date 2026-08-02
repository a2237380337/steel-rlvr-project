# 实现状态

更新日期：2026-08-02

已完成：

- 真实中文字段恢复、派生特征、字段占位记录清洗和严格数据合同；
- common/rare 整钢种隔离，训练统计量与测试标签防泄漏；
- Qwen3.5-0.8B-Base 的 3,000 条 BF16 LoRA-SFT；
- 81,546/20,388 条训练/验证偏好对构造与哈希报告；
- 显式 SFT 冻结参考模型的普通 DPO 与频次感知 DPO；
- 两组各 3,000 条、200 steps 的正式单卡训练；
- Base/SFT/DPO/频次感知 DPO 的 1,017 条统一冻结评测；
- Macro/Worst-group、无效输出回填、严格 JSON 和物理越界统计；
- 运行 manifest、配置/数据/源码/样本顺序 SHA256、显存和训练日志；
- 机器可读结果矩阵、自动结果卡、主实验报告与发布审计；
- ROCm kernel smoke、DPO 端到端 smoke、ruff 和 pytest 验证；
- 策略难负例 MPO、难例 continued-SFT 与频次感知在线 RLVR 的后训练比较；
- 1,500 条均衡样本、4 路在线采样、200-step Dr.GRPO 正式训练；
- RLVR 与 SFT 的独立验证、稀有钢种复现评测及 20,000 次配对 bootstrap；
- 私有数据、逐样本预测、checkpoint 和环境日志的 Git 隔离。

最终选择 LoRA-SFT + 频次感知在线 RLVR。它在 1,536 条独立验证样本上将 MAE
从 0.6501 降至 0.6350，在 1,017 条训练未见稀有钢种上从 0.6486 降至
0.6438；严格 JSON 率均为 100%。固定偏移 DPO 和策略难负例 MPO 作为负结果
保留，完整证据与限制见 `reports/posttraining_search_20260802.md`。
