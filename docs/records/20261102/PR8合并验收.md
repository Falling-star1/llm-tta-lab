# PR #8 合并验收

日期：2026-11-02（北京时间）。用户授权解决 [PR #8](https://github.com/Falling-star1/llm-tta-lab/pull/8)，通过核验后合并协作模板与启动器。

## 合并内容

- 原 PR head：`ef6a6d1ae05a047741099a10770ff463490dd593`；纳入主分支：`e14c0b6aa400944f66fb31bf096886e95cf66bf9`，已包含 PR #3 的复现成果。
- README 冲突已解决：保留最新 TLM / Tent / EATA 说明和历史结果导航，补入团队看板、快速开始和启动器 CPU 检查入口。
- AGENTS、Issue 表单、PR 模板、CI、启动器和测试代码保持原 PR 已评审版本，未修改研究方法、冻结包或原始结果。

## 本轮实际验证

- Windows / Python 3.11：20 项中 19 项通过，跳过 1 项 Linux 进程组专用检查。
- 服务器 Linux / Python 3.12.4：隔离临时目录中的 20 项全部通过，`CUDA_VISIBLE_DEVICES` 为空，未安装依赖。
- 仓库完整性检查通过：26 份冻结包、242 个文件校验；202 份源码语法通过，包括 `local_repro` 及新增启动器。
- 启动器、smoke 脚本及测试与原 head 字节一致；服务器已部署启动器 SHA256 为 `363f39c6a96753c4c7f2a25266643a74c800ffadc03f8c808c47f4f8695b11e1`，与候选源码一致。
- 只读访问 MLflow，原 CPU smoke run `b2053f23f43f4011b6e0f181d2b46668` 仍为 FINISHED / NOT_APPLICABLE；未重跑该任务。
- 独立只读复核确认 README 没有遗失主分支内容，协作说明的 21 个本地链接存在，实验授权及平台边界描述一致，未发现合并阻断问题。

Linux 测试日志保留在 `/tmp/llm-tta-pr8-merge-7oocs_3y` 对应隔离测试目录，原部署及初次验证见[协作入口验证记录](../20261101/team-workflow-validation.md)。最终 PR 提交的 GitHub CI 另在 PR 检查页核验。

本次合并只是将已部署、已验证的工具及协作模板收录主分支，没有启动真实模型训练或生成、修改运行中的实验、开放 MLflow 端口或增加其他成员权限。其余成员的账号和任务认领仍需按实际情况配置。
