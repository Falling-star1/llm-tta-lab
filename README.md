# llm-tta-lab · 大语言模型测试时自适应

在固定 TLM 源码上，把 Tent、EATA、SAR、DeYO 四个视觉 TTA 方法移植到 LLM 的文本任务，做同协议对照；基线可靠后再研究 OPUS proxy pool 的样本选择。旧 OTTA 自研选样计划已归档。

## 当前进度

更新日期：2026-11-18（服务器实测）。**五方法的横向基线表以[基线总表](docs/基线总表.md)为准**；过程细节与负结果见[实验记录](docs/实验记录.md)，任务状态见[团队看板](https://github.com/users/Falling-star1/projects/1)。

| 工作 | 状态 |
|---|---|
| 模型与数据 | 已就绪：Llama-3.2-3B-Instruct、Qwen2.5-3B-Instruct；三套数据各 5000 条，SHA256 已核 |
| Source（适应前） | **六格全量完成**，与论文同格差 ≤ 0.0045；存于服务器 `results/source/` |
| Tent（`norm_gamma_only`） | 六组 5000 条完成，全部 `PASS`；llama3b/dolly 塌缩 |
| EATA（`lora_qv`） | 七组完成（六组有效 + 一组旧 `d_margin` 口径）；geo/dolly 四组塌缩 |
| TLM 方法复现 | Gaussian v6 六组各 5000 条完成（qwen/dolly 用补救跑）；v7 只作独立诊断 |
| SAR / DeYO | **各六组完成**（`norm_gamma_only`）；另有 SAR `reset_ratio` 敏感性六组 |
| OPUS proxy pool | 待核定：池来源、划分与候选流 disjoint 规则未定 |

**已知负结果**（完整保留，见[实验记录](docs/实验记录.md)）：

- Tent 与 DeYO 在 llama3b/Dolly 上塌成同一个常量拒答串（Δ ≈ −0.2985，唯一回答 20~24/5000）。
- EATA 在 geo/dolly 四组撞生成上限 4999~5000/5000，退化为重复串——与论文自身的 EATA 数字（0.0064 / 0.0090）同一现象。
- EATA 的 `d_margin=1.0` 会使 4067/5000 个 batch 被整批筛空、静默冻结（`llama3b/Dolly` 0.0910）；关闭筛选后重跑得 0.0184。
- SAR 是唯一没让 llama3b/Dolly 塌缩的方法（−0.0059），但增益也被压平（`Σ|Δ|` 0.0565 vs Tent 0.4426）。

Qwen2.5-7B 全量不排入计划（2026-11-01 决定）：现有正式结果全部是 3B，7B 权重也已不在服务器。

## 从这里开始

- [基线总表](docs/基线总表.md)：**五方法 × 六格的冻结基线**（Source / Δ / `quality` 状态 / 配置 / 已知缺口）。后续研究引这一个。
- [实验记录](docs/实验记录.md)：每次实验的数字、结论与负结果。
- [复现协议](docs/复现协议.md)：冻结口径——两种适配面、80/512 token 预算、五个方法的超参、与论文的已知偏离。
- [TLM 原版结果与官方核对](docs/records/20261028/TLM原版复现与官方实现对照.md)：六组完整分数、实现差异与复现限制。
- [TLM 复现实验方案](docs/TLM复现实验方案.md)：固定设置、历史阶段记录和队友接手说明。
- [文献与方法核对](docs/文献与方法核对.md)：论文 → 官方代码 → 我们的实现三方对照（Tent / EATA / TLM）。
- [环境与运行](docs/环境与运行.md)：服务器环境、激活方式、五个方法的运行入口。
- [复现结果汇总（2026-11-12）](docs/复现结果汇总_20261112.md)：历史快照，含 Tent-LoRA v3 作废组的完整对照，**分数引用以基线总表为准**。
- [records/](docs/records/)：原始记录（正式结果登记、失败与停摆、目录整理）。
- [项目总览](项目总览.md)：研究范围与验收标准。
- [团队分工与协作](团队分工与协作.md)：提交、评审与交接规则。

## 仓库结构

```text
local_repro/     自建方法代码（Tent / EATA / SAR / DeYO 的只读快照，含哈希清单）
docs/            协议、基线总表、实验记录、原始记录、环境说明
offline_tta/     各阶段冻结离线包与 TLM 诊断代码，SHA256SUMS 保持原样
scripts/         仓库完整性检查与 CPU 测试入口
archive/         原 GitHub 主线、旧交接文档
```

`local_repro/` 是**只读快照**：代码在服务器上改，改完重拷并更新哈希表；不要在这里直接改，服务器跑的不是这些文件。详见 [local_repro/README.md](local_repro/README.md)。

`offline_tta/tlm_tent_full_3gpu_det_bundle` 是 Qwen2.5-7B 三卡严格确定性包，验证的是 7B 可行性与确定性，**与现行 3B 单卡结果不是同一协议**，包内 README 是交付时的冻结快照。

模型、数据集、依赖安装包、缓存、checkpoint 与完整预测留在服务器；仓库只存代码、配置、结果摘要与运行记录。不要 `git add .` 收录整个本地研究目录。

## 本地核对

```bash
python scripts/verify_repository.py   # 只用标准库，校验仓库完整性与哈希
python scripts/run_cpu_tests.py       # 需要 torch / transformers 4.46.1 / peft 0.12.0 / numpy
```

本仓库不会自动启动 GPU 实验。历史版本与回溯方式见[归档说明](archive/README.md)。


## 三人协作入口

[团队看板](https://github.com/users/Falling-star1/projects/1) · [快速开始](docs/team-workflow.md) · [Agent 约定](AGENTS.md)。新任务通过 Issue 分工，以 PR 交接；新实验用统一入口关联 MLflow，现有实验沿用原运行路径。
