# llm-tta-lab · 大语言模型测试时自适应

当前工作是基于固定 TLM 源码，在文本任务上复现与适配 Tent、EATA、SAR、DeYO，并在基线可靠后研究 OPUS proxy pool。旧 OTTA 自研选样计划已归档，不再作为当前执行要求。

## 当前进度

更新日期：2026-10-18（北京时间）。这是已收集证据的快照，不是服务器实时监控。

| 工作 | 当前状态 |
|---|---|
| Qwen2.5-7B-Instruct、固定 TLM 源码和三套数据 | 已准备，Source 小样本生成和评分通过 |
| Tent-text-LoRA 单步真实 GPU 验证 | 已通过，基础权重冻结、LoRA 更新和重置通过 |
| 三数据集各 16 条 Source/Tent offline 对照 | 已通过流程验收；GSM8K 得分下降，完整保留 |
| 三卡确定性长输入与断点恢复预检 | 三组全部通过，adapter / Adam / RNG 恢复一致 |
| 三数据集各 5000 条 Source/Tent 全量对照 | 启动命令已准备；尚无已登记的启动信息和完整结果 |
| Llama-3.2-3B-Instruct | 访问权限、模型准备及同协议实验待完成 |
| EATA / SAR / DeYO、OPUS proxy pool | 待实现与验收 |

**当前优先继续 Tent 全量对照，不把预检通过写成论文复现完成。** 当前使用 LoRA 的文本工程适配，尚未证明与作者 Tent 基线全部细节一致。

## 从这里开始

- [项目总览](项目总览.md)：研究范围、阶段目标与验收标准。
- [项目看板](项目看板.md)：当前任务、认领方式和阻塞事项。
- [团队分工与协作](团队分工与协作.md)：提交、评审和交接规则。
- [环境与运行](docs/环境与运行.md)：新队友接手与全量运行入口。
- [复现协议](docs/复现协议.md)：固定模型、数据、长度、学习率与标签访问边界。
- [实验记录](docs/实验记录.md)：已验证结果和下一次运行登记位置。
- [文献与方法核对](docs/文献与方法核对.md)：官方源码、方法对齐与历史阅读材料。

## 仓库结构

```text
offline_tta/     已交付的轻量源码包，SHA256SUMS 保持原样
docs/           当前协议、运行说明、记录和整理设计
scripts/        仓库完整性检查与 CPU 测试入口
archive/        原 GitHub 主线、旧修复分支和过期交接文档
```

当前有效全量入口为 [三卡确定性全量包](offline_tta/tlm_tent_full_3gpu_det_bundle/README.md)。包内 README 是交付当时的冻结快照；最新完成状态以本首页和实验记录为准，不改包内文件以免破坏哈希及服务器代码一致性。

模型、数据集、依赖安装包、缓存、checkpoint 和完整预测留在服务器或共享存储。仓库提交代码、配置、结果摘要、校验信息与运行记录；不直接执行 `git add .` 收录整个本地研究目录。

## 本地核对

```bash
python scripts/verify_repository.py
python scripts/run_cpu_tests.py
```

第一条只使用 Python 标准库；第二条要求已有 torch、Transformers 4.46.1、PEFT 0.12.0、NumPy。服务器环境与 CPU 验证的边界见[环境说明](docs/环境与运行.md)。本仓库不会自动启动 GPU 实验。

历史版本和回溯方式见 [归档说明](archive/README.md)。


## 三人协作入口

[团队看板](https://github.com/users/Falling-star1/projects/1) · [快速开始](docs/team-workflow.md) · [Agent 约定](AGENTS.md)。新任务通过 Issue 分工，以 PR 交接；新实验用统一入口关联 MLflow，现有实验沿用原运行路径。
