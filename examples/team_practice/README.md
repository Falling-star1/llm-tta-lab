# 四人答题正确率统计（协作练习）

这是 Issue #6 的玩具练习，**不是科研指标**，不属于正式复现实验。不要用它推断任何 TTA 方法的效果，也不要把结果写进论文或实验结论。

## 运行

```bash
python examples/team_practice/toy_accuracy.py --output-dir /tmp/toy-accuracy
```

固定样例：

- 预测答案：`["A", "B", "C", "A"]`
- 正确答案：`["A", "B", "D", "A"]`

预期输出：

```text
{
  "correct": 3,
  "total": 4,
  "accuracy": 0.75
}
```

同时会在 `/tmp/toy-accuracy/metrics.json` 写入：

```json
{
  "correct": 3,
  "total": 4,
  "accuracy": 0.75,
  "quality_status": "NOT_APPLICABLE",
  "note": "Toy collaboration exercise; not a scientific metric."
}
```

`quality_status` 固定为 `NOT_APPLICABLE`，因为本练习不评价任何科研质量。

## 测试

```bash
python -m unittest discover -s tests -p test_toy_accuracy.py
```

覆盖正常输入、空输入和两个列表长度不一致三种情况，并检查命令行写出的 `metrics.json`。
