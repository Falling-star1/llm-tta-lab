# 协作入口 v1 验证记录

日期：2026-11-01。关联 [Issue #5](https://github.com/Falling-star1/llm-tta-lab/issues/5)、[PR #8](https://github.com/Falling-star1/llm-tta-lab/pull/8)。

- 本地 Windows：`python -m unittest discover -s tests -p test_labctl.py`，17 项通过。
- 服务器 Linux：同一组 17 项通过，未安装新依赖。
- GitHub Actions：Linux 和 Windows 均通过，[本轮运行](https://github.com/Falling-star1/llm-tta-lab/actions/runs/41292111631)。
- 部署入口：`/data/chuwenjing/lab/bin/labctl.py`；新目录独立于既有复现包。
- CPU 请求：`/data/chuwenjing/lab/requests/team-smoke-20261101.json`；方法 workflow-smoke，不加载模型、数据或 GPU。
- 实跑：执行 SUCCEEDED，退出码 0，质量 NOT_APPLICABLE，MLflow SYNCED；`smoke_check=1` 只证明工作流贯通。
- MLflow experiment：`Team-experiments`；run：`b2053f23f43f4011b6e0f181d2b46668`。
- 输出：`/data/chuwenjing/lab/runs/team-smoke-20261101/output`；request/state/console.log 位于上一级。
- 再次提交同请求被拒绝，未重跑。
- 对空闲 GPU 2 短暂取得中央锁及两套旧锁，同一锁不能再次取得，检查后释放；未启动 GPU 计算。
- 验证时原 4 个 EATA 进程仍在 GPU 0、1、3、4 运行，未重启或迁移。

源码 commit 声明：`a7e5f037b558f0b61daa6ae05269f23d546f1d0a`。入口由该提交文件部署，代码与输入按请求指纹校验。PR 的后续纯文档提交不改变本次 smoke 来源。

本记录不表示科研结果验收通过，也不表示未来新增训练已经获准。任务认领、评审和下一次实验接入分别见看板及 Issue #6、#7。

## 独立评审后的修订

独立代码评审发现并修复两项问题：主进程先退出时需等待同组 worker 才能释放锁；同一任务的 MLflow 同步需互斥以免重复创建 run。自动同步遇到人工同步竞争时保留已完成的执行状态。

新增三项回归用例。Linux 上旧版本的 worker 测试失败（约 0.54 秒提前结束），新版等 worker 完成，20 项全部通过。Windows 20 项中 19 项通过，跳过 1 项 Linux 进程组专用检查。修复经独立复核，未发现剩余关键或重要问题。

本次修订不重跑已完成的 CPU smoke，原始请求、结果、release 和 MLflow run 保持可追溯。
