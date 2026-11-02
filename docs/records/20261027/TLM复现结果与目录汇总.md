# 复现结果与服务器目录汇总

主评测快照：2026-10-27 21:24（北京时间）。这是当前可用结果索引；未完成的评测不填最终分数。模型均为本轮实际使用的3B版本。

## Source、TENT与TLM主实验

Source和此前TENT六组均已完成5000条。TLM六组各5000条适应训练已完成，5/6组全量生成与评分完成。

Geography/Dolly为ROUGE-Lsum×100，GSM8K为准确率%。Source为未适应模型，TLM复用其相同输入的Source。TENT与TLM训练配方有差异，此表用于汇总实际跑出结果，不能把全部分差都归因于损失函数。

| 模型 | 数据集 | Source | 此前TENT | TLM | TLM评测状态 |
|---|---|---:|---:|---:|---|
| Llama-3.2-3B | Geography | 23.98 | 1.21 | 29.32 | 5000/5000，评测完成 |
| Llama-3.2-3B | Dolly | 33.33 | 0.045 | 24.07 | 5000/5000，评测完成 |
| Llama-3.2-3B | GSM8K | 77.60% | 78.04% | 77.98% | 5000/5000，评测完成 |
| Qwen2.5-3B | Geography | 24.94 | 0.80 | 29.64 | 5000/5000，评测完成 |
| Qwen2.5-3B | Dolly | 27.65 | 0.009 | 25.69 | 5000/5000，评测完成 |
| Qwen2.5-3B | GSM8K | 80.08% | 83.62% | 未出最终分数 | 4725/5000，评测中 |

TENT四组Geography/Dolly结果严重退化，5000条生成均命中512 token上限；执行记录的PASS只表示运行结束，不代表质量或论文效果复现成功。TLM两组Geography提高，两组Dolly下降；Llama GSM8K仅小幅提高。原始结果均保留。

## 每组TLM结果的准确目录

- Llama-3.2-3B / Geography：`/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal/llama3b__geosignal_5k`
- Llama-3.2-3B / Dolly：`/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal/llama3b__dolly_5k`
- Llama-3.2-3B / GSM8K：`/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal/llama3b__gsm8k_5k`
- Qwen2.5-3B / Geography：`/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal/qwen3b__geosignal_5k`
- Qwen2.5-3B / Dolly：`/data/chuwenjing/results/tlm_qwen_dolly_recovery_v6/run_20261026_014911_83ebf2/formal/qwen3b__dolly_5k`
- Qwen2.5-3B / GSM8K：`/data/chuwenjing/results/tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal/qwen3b__gsm8k_5k`

Qwen Dolly的当前有效结果在独立恢复目录 `tlm_qwen_dolly_recovery_v6`，应使用上面列出的完整路径。

每组目录内：

| 文件/目录 | 用途 |
|---|---|
| `metrics.json` | 5000条完成后的最终分数；未完成组暂时没有此文件 |
| `tlm.jsonl` | 每条实际回答、source_index、分数、生成token等；运行中持续增加 |
| `final_adapter/adapter_model.safetensors` | 5000条适应后的LoRA权重，不含完整底模 |
| `trainer/checkpoint-5000/` | 训练检查点，含优化器、调度器等恢复资料 |
| `training_complete.json` | 只证明适应训练完成，不代表生成评测完成 |
| `adaptation.json` | 训练访问顺序、损失、筛选权重及学习率记录 |
| `source_reuse.json` | 本组复用Source的实际路径和SHA256 |
| `run_manifest.json` | 配置身份和执行阶段；PASS不能单独当作效果提高 |
| `quality_latest.json` | 最近窗口的质量检查，不代替全量质量结论 |

六组评测协调器的总状态：

```text
/data/chuwenjing/results/tlm_eval_complete_v6/run_20261026_052804_45ad81/status.json
```

该协调器目录存放总进度与接续日志，回答与最终分数仍在上面的各组目录。

## Source与此前TENT的位置

共同根目录：

```text
/data/chuwenjing/results/tent_formal_v3_6gpu/run_20261020_001210_900089
```

六个子目录依次为 `llama3b__geosignal_5k`、`llama3b__dolly_5k`、`llama3b__gsm8k_5k`、`qwen3b__geosignal_5k`、`qwen3b__dolly_5k`、`qwen3b__gsm8k_5k`。每组的 `source.jsonl` 是Source回答，`tent.jsonl` 是TENT回答，`source_metrics.json` 是Source分数，`metrics.json` 含Source和TENT两部分分数。JSON索引列出了每一组的完整路径。

## 已完成的机制对照

以下不属于原文5000条全量复现成绩，不能并入上面的主实验。

**Dolly 1000步：两模型各四组控制。** 固定100条下一token探针，以及8条完整生成。标题/首步结束明显减少，但完整回答仍有重复和答非所问；8条ROUGE不能估计全量结果。

| 组别 | 首token为格式标题 | 首token为结束符 | 8条ROUGE×100 |
|---|---:|---:|---:|
| llama3b__all | 84/100 | 0/100 | 32.19 |
| llama3b__role_masked | 0/100 | 0/100 | 34.80 |
| llama3b__wrapper_masked | 0/100 | 0/100 | 37.23 |
| llama3b__both_masked | 0/100 | 0/100 | 37.72 |
| qwen3b__all | 3/100 | 40/100 | 21.45 |
| qwen3b__role_masked | 0/100 | 0/100 | 40.46 |
| qwen3b__wrapper_masked | 0/100 | 0/100 | 30.56 |
| qwen3b__both_masked | 0/100 | 0/100 | 40.76 |

训练、检查点与原始失败记录：`/data/chuwenjing/results/tlm_template_matrix/run_20261026_160452_836775`。

五组因AMP实际LR差异而被早期严格门禁拒绝，之后只加载保存权重补评测；补评测结果在 `/data/chuwenjing/results/tlm_template_eval_recovery/run_20261027_004343_f67018`。原失败和LR差异仍保留。

**GSM8K检查点探针。** 原0/1000/2500/5000步，在固定100题强制给出开场白后，结束符argmax分别为0/4/14/24；这是条件探针，不是实际空答率。结果：`/data/chuwenjing/results/tlm_gsm_milestones/run_20261027_001700_8aff5a`。

**GSM8K三组2500步目标屏蔽控制。** 训练、完整检查点、最终adapter及条件探针在 `/data/chuwenjing/results/tlm_gsm_transition_controls/run_20261027_003255_5dcc17`。子目录为 `hint_masked`、`eot_masked`、`both_masked`。固定100题条件结束符argmax从原2500步14/100降至2/0/0，不能据此宣布全量质量修复。

**16道真实故障题回放。** 原5000步350条开场白空壳中固定抽16题，再比较同题2500步权重；Source同题答对13题。

| 组别 | 只说开场白 | 正确题数 |
|---|---:|---:|
| original_step2500 | 5/16 | 8/16 |
| hint_masked | 0/16 | 14/16 |
| eot_masked | 0/16 | 13/16 |
| both_masked | 0/16 | 14/16 |

结果：`/data/chuwenjing/results/tlm_gsm_failure_replay/run_20261027_074127_85f7d3`。每组 `summary.json` 存汇总，`answers.json` 存完整生成。16题来自既有故障，不能称全量准确率。

## 正在运行的200题验证

```text
/data/chuwenjing/results/tlm_gsm_paired_validation/run_20261027_125214_86cafa
```

| 组别 | 已生成 | 状态 |
|---|---:|---|
| original_step2500 | 112/200 | EVALUATING |
| hint_masked | 106/200 | EVALUATING |
| eot_masked | 121/200 | EVALUATING |
| both_masked | 117/200 | EVALUATING |

四组比较已有2500步权重，不重训。200题按固定种子从原5000题中排除此前116条诊断题后抽取，仍来自原适应集合；注册文件为根目录的 `registration.json`。每组的 `status.json` 看进度，`answers.json` 看当前回答，全部完成后才生成 `summary.json`。收齐前不报告前缀分数或选最高分组。三组屏蔽目标属于方法变体，最终也不能替换原文基线。

## 数据、底模与主训练代码

- 准备输入根目录：`/data/chuwenjing/tta_offline/tlm_inputs_v1`，三个数据集子目录各含 `inputs.jsonl`、`references.jsonl`、`manifest.json`。
- Llama底模：`/data/chuwenjing/models/Llama-3.2-3B-Instruct`。
- Qwen底模：`/data/chuwenjing/models/Qwen2.5-3B-Instruct`。
- 冻结主运行代码：`/home/chuwenjing/TTA_LLM/TLM/local_repro/tlm_offline_b6da067b55f5`。
- [机器可读的逐组结果和完整路径](tlm-reproduction-results-index.json)。
- [原始核查快照](tlm-reproduction-results-snapshot.json)。
- [原因与对照解释](TLM_Dolly与GSM8K排查进展.md)。
- [200题验证范围与接手命令](TLM_GSM8K修正版200题配对验证.md)。
