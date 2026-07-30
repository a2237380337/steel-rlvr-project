# 结果卡入口

项目 2 已正式完成。机器可读矩阵为 `../results/formal_evaluation_matrix.json`，自动生成的可读结果卡为 `result_card.generated.md`。

重新生成命令：

```bash
bash scripts/run_all_evals.sh
```

脚本会生成或刷新：

- `results/formal_evaluation_matrix.json`
- `reports/result_card.generated.md`

生成器会检查四组结果是否齐全、测试集 SHA256 和样本 ID SHA256 是否一致；任一条件不满足都会失败。CAC submitted 论文数字只从 `paper_baselines.csv` 读取并明确标为外部场景 baseline。
