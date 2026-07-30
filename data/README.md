# 私有数据说明

原始产线 Excel、逐条工艺参数、钢种订单信息和任何可反推生产状态的样本都不得提交。

`STEEL_DATA_PATH` 应指向本地清洗文件。`prepare_data.py` 会在 `data/processed/` 生成：

- `train.jsonl`：common-grade 训练样本；
- `validation.jsonl`：common-grade 验证样本；
- `test.jsonl`：训练阶段未见的 low-frequency-grade 测试样本；
- `schema.json`：字段、Prompt、训练统计量；
- `data_report.json`：数量合同和 SHA256。

整个 `data/processed/` 已加入 `.gitignore`，因为 Prompt 和标签仍包含私有工艺数值。

公开材料只允许保留：

- 字段名称和时序合法性说明；
- 频次切分、奖励、训练和评测代码；
- 聚合后的样本数、钢种数和模型指标；
- 不包含真实订单或工艺值的人工 toy data。
