# llm-tta-lab · 大语言模型测试时自适应

在固定 TLM 源码上，把 Tent、EATA、SAR、DeYO 四个视觉 TTA 方法移植到 LLM 的文本任务，做同协议对照；基线可靠后再研究 OPUS proxy pool 的样本选择。旧 OTTA 自研选样计划已归档。

## 当前进度

更新日期：2026-11-01（服务器实测）。数字与证据见[实验记录](docs/实验记录.md)，任务状态见[项目看板](项目看板.md)。

| 工作 | 状态 |
|---|---|
| 模型与数据 | 已就绪：Llama-3.2-3B-Instruct、Qwen2.5-3B-Instruct；三套数据各 5000 条，SHA256 已核 |
| Tent（`norm_gamma_only`） | **六组 5000 条完成**（两模型 × 三数据集），全部 `PASS` |
| EATA（`lora_qv`） | **三组完成**（llama/dolly、llama/gsm8k、qwen/gsm8k）；**四组评测中** |
| TLM 方法复现 | Gaussian v6 六组各 5000 条适应和评测已完成；正式结果及与官方实现的差异见[核对记录](docs/records/20261028/TLM原版复现与官方实现对照.md)。v7 等修订只作独立诊断，不替换原版结果 |
| SAR / DeYO | 待开展：文本适配方案未定 |
| OPUS proxy pool | 待核定：池来源、划分与候选流 disjoint 规则未定 |

**已知负结果**（完整保留，见实验记录 §4）：Tent 在 llama3b/Dolly 上 0.3333 → 0.0347（熵塌到 0.00047）；EATA 在 llama3b/Dolly 上 0.3333 → 0.0910（`d_margin=1.0` 下 4067/5000 个 batch 被整批筛掉）。

Qwen2.5-7B 全量不排入计划（2026-11-01 决定）：现有正式结果全部是 3B，7B 权重也已不在服务器。

## 从这里开始

- [TLM 原版结果与官方核对](docs/records/20261028/TLM原版复现与官方实现对照.md)：六组完整分数、实现差异与复现限制。
- [TLM 复现实验方案](docs/TLM复现实验方案.md)：固定设置、历史阶段记录和队友接手说明。
- [团队看板](https://github.com/users/Falling-star1/projects/1)：任务、状态和负责人；[仓库索引](项目看板.md) 保留历史里程碑。
- [三人协作快速开始](docs/team-workflow.md)：认领任务、agent 分工、PR 评审、实验运行与交接。
- [实验记录](docs/实验记录.md)：结果数字与结论。
- [复现协议](docs/复现协议.md)：冻结口径——两种适配面、80/512 token 预算、各方法超参、与论文的已知偏离。
- [文献与方法核对](docs/文献与方法核对.md)：论文 → 官方代码 → 我们的实现三方对照。
- [环境与运行](docs/环境与运行.md)：服务器环境、激活方式、运行入口。
- [records/](docs/records/)：原始记录（正式结果登记、失败与停摆、目录整理）。
- [项目总览](项目总览.md)：研究范围与验收标准。
- [团队分工与协作](团队分工与协作.md)：提交、评审与交接规则。

## 仓库结构

```text
local_repro/     自建方法代码（Tent / EATA 的只读快照，含哈希清单）
docs/            协议、实验记录、原始记录、环境说明
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

协作启动器另有不依赖训练框架的 CPU 检查：`python -m unittest discover -s tests -p test_labctl.py`。

本仓库不会自动启动 GPU 实验。历史版本与回溯方式见[归档说明](archive/README.md)。

## 三人协作入口

[团队看板](https://github.com/users/Falling-star1/projects/1) · [快速开始](docs/team-workflow.md) · [Agent 约定](AGENTS.md)。新任务通过 Issue 分工，以 PR 交接；新实验用统一入口关联 MLflow，现有实验沿用原运行路径。
