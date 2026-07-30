# 实现状态

更新日期：2026-07-30

正式完成：

- 真实中文字段恢复、派生特征和严格数据合同；
- common/rare 整组隔离与训练统计量防泄漏；
- Qwen3.5-0.8B-Base BF16 LoRA-SFT；
- 普通 Dr. GRPO 与低频感知 Dr. GRPO；
- 数值、严格 JSON、物理范围奖励；
- 同文件 Base/SFT/Dr. GRPO/Tail-aware 推理评测；
- invalid output 回填、Macro/Worst-group、失败分类；
- 运行 manifest、配置/数据 SHA256、显存和 token 记录；
- validation checkpoint 选择工具；
- 四组结果矩阵和防虚构结果卡；
- 23 项 CPU 单元测试（含发布审计）、BF16 ROCm kernel smoke 和端到端 GPU smoke；
- 3,000 条 LoRA-SFT（300 steps）；
- 普通/低频感知 Dr. GRPO（各 1,500 条、200 steps、4 completions）；
- Base/SFT/Dr. GRPO/Tail-aware 四模型 1,017 条统一测试；
- 自动结果矩阵、结果卡和主实验报告；
- 独立 Git 仓库、数据卡、模型卡、MIT 代码许可证、自包含 ROCm 环境脚本；
- 发布审计：公开聚合结果纳入版本控制，私有数据、逐样本预测、checkpoint、
  环境归档和本机绝对路径不得进入公开提交。

数据审计额外识别并过滤了 1 条 `Slab_ID`/`Steel_Grade` 占位记录。过滤前后的训练与验证 JSONL SHA256 完全一致，因此未重训 checkpoint；旧评测完整归档后，仅重跑干净测试集。最终模型为 Tail-aware Dr. GRPO，总体 MAE 0.6438。
