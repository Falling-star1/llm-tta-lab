# 三人协作快速开始

[团队看板](https://github.com/users/Falling-star1/projects/1) · [任务列表](https://github.com/Falling-star1/llm-tta-lab/issues) · [协作约定](../AGENTS.md)

采用 GitHub Issue/Project/PR + 独立 worktree + 服务器统一实验入口 + MLflow。三人无需另维护一份任务表。

## 每天怎么用

1. 从“待做”认领一个 Issue，填 owner、评审者、允许文件范围和验收，移到“进行中”。每人一项主任务；多个 agent 只拆互不重叠的子任务。
2. 在独立 worktree 开发；分支例如 `codex/issue-6-onboarding`。不让几个 agent 同时改同一份工作目录。读代码、查资料任务可以共享只读上下文。
3. 推送 PR，填验证结果并移到“待评审”。另一位成员评审。代码任务验收后到“完成”；需要运行实验的任务先到“待实验”。
4. 实验获准后使用统一入口。回填 MLflow run、输出路径、执行状态和质量结论，再完成 Issue。评审通过不自动代表获准使用 GPU。

owner 对结果负责；评审者检查代码与协议；第三位成员可核对实验结果。角色按任务轮换。没人认领时留空，不虚构三位成员的账号或职责。

## 服务器入口

部署根目录 `/data/chuwenjing/lab`。入口仅用标准库，不给现有训练环境安装包。

```bash
PY=/home/chuwenjing/miniconda3/bin/python
LAB=/data/chuwenjing/lab/bin/labctl.py
$PY "$LAB" status
$PY "$LAB" plan /data/chuwenjing/lab/requests/你的请求.json
# 仅执行已获准的请求
$PY "$LAB" run /data/chuwenjing/lab/requests/你的请求.json
$PY "$LAB" status 你的实验id
# MLflow 恢复连接后补同步，无需重跑训练
$PY "$LAB" sync 你的实验id
```

`plan` 只校验，不登记任务和占 GPU。`run` 前台执行；可在 tmux 中运行。每次提交都会登记 ID 和实验身份，改名也不能绕过重复检查。失败也保留登记；需要复验时先评审新的 protocol_id，不删除历史登记。

CPU 演示请求保存在服务器 `requests/team-smoke-20261101.json`，已执行过，**只能 plan/status/sync，不要再次 run**。它不加载模型和数据，不代表科研指标。

新 GPU 请求先将现有脚本与依赖放进独立 release 目录，以 `fingerprint <code_root>` 获取 code_files，填写下面字段。请求放在代码目录外，避免请求自身参与哈希循环。站点配置示例见 [lab-site.example.json](../configs/lab-site.example.json)。

```json
{
  "schema_version": 1,
  "experiment_id": "唯一小写英文数字横线id",
  "owner": "负责人账号",
  "issue": "https://github.com/Falling-star1/llm-tta-lab/issues/7",
  "method": "方法名",
  "model": "模型名及版本",
  "dataset": "数据集名及版本",
  "protocol_id": "已评审协议版本",
  "seed": 42,
  "commit": "40位完整Git提交SHA",
  "code_root": "/data/chuwenjing/lab/releases/新版本",
  "code_files": {"entry.py": "实际SHA256"},
  "entrypoint": "entry.py",
  "inputs": {
    "dataset": {"path": "/data/chuwenjing/实际数据文件", "sha256": "实际SHA256"},
    "model_contract": {"path": "/data/chuwenjing/模型契约.json", "sha256": "实际SHA256"}
  },
  "argv": ["{python}", "{entrypoint}", "--output-dir", "{output}"],
  "gpu_count": 1,
  "gpu_candidates": [0, 1, 2, 3, 4, 5, 6, 7],
  "timeout_seconds": 3600,
  "queue_timeout_seconds": 600,
  "legacy_lock_policy": "wrapper"
}
```

这是字段说明，不可原样运行；CLI 参数必须匹配该方法的真实入口。多输入文件应逐项记录。模型契约应包含 checkpoint 路径、revision、权重/配置校验清单和环境版本；入口只校验契约文件本身，不重新读取所有模型权重。完整 commit 是来源声明，实际运行源码由 code_files 校验；本工具不替你证明二者一致，需要评审 release 的生成过程。模型、数据不会自动上传 GitHub 或 MLflow。

## 运行时如何协调

- 任务必须带 Issue、owner、协议、seed、commit、代码和输入指纹。排队结束后再次检查指纹，修改后不能悄悄执行。
- 输出统一放 `runs/<experiment_id>/output`；旁边保存 request.json、state.json、console.log。禁止复用或覆盖冻结结果目录。
- 选择 nvidia-smi 显示空闲的 GPU，再取中央文件锁和已知 TLM/TENT 锁，并二次检查占用。已运行的旧实验继续按旧方式工作。
- 默认 wrapper 持有兼容锁；若前台子程序自身取得旧锁，必须用 child 策略：入口只探测旧锁，子程序自行持有，中央锁仍保留。对自行后台化或绕过约定的入口不提供完整互斥保证；这不是 Slurm，也不是权限沙箱。
- Linux 上超时/中断只向本任务子进程组发送 SIGTERM，等待组内存活 worker 全部退出才释放锁（包括已关闭继承锁 FD 的 worker）；不自动 SIGKILL。若子程序不退出，保持 STOPPING，需要 owner 排查。不要终止其他人的进程。控制器被 SIGKILL 后，状态可能需要人工核对，不能仅凭状态文件判定 GPU 空闲。Windows 的 CPU 开发检查只管理直接子进程，正式多进程实验在 Linux 服务器运行。
- CPU 请求设置 gpu_count=0，CUDA_VISIBLE_DEVICES 为空。GPU 可见设备会被重编号；一般子程序使用 `cuda:0`，`{gpu}` 表示物理编号，仅供确实需要物理编号的旧入口。

## MLflow 与质量验收

新入口写入独立 experiment `Team-experiments`，已有历史结果仍保留。成功结束不等于科学有效：`execution_status` 和 `quality_status` 分开。应用在 output/metrics.json 中写数值指标和 quality_status；未提供质量检查就是 NOT_REPORTED，演示为 NOT_APPLICABLE。

MLflow 不通时，实验结果与日志留在服务器，state 标记 PENDING，恢复后 sync。同一任务的 sync 使用文件锁互斥，正在同步时再次手动同步会报告 Already leased；自动同步遇到竞争则保留已完成状态，不覆盖正在同步者的记录。MLflow 记录 commit、协议、owner、Issue、实验身份和产物路径；本版不上传大文件/原始预测，需通过服务器路径查看。

仅通过 SSH 隧道访问 MLflow，例如本机将远端 localhost:5000 转发到 localhost:15000，然后打开 http://127.0.0.1:15000。无需公开服务器端口。

## 验证与迭代边界

```bash
python -m unittest discover -s tests -p test_labctl.py
```

GitHub Actions 在 Linux 和 Windows 跑相同 CPU 检查。没有 GPU 训练自动触发器，也没有自动合并规则。先用 Issue #6 完成一次三人交接，再按 Issue #7 接入下一项已批准的小实验；不把现有 EATA 中途迁移。
