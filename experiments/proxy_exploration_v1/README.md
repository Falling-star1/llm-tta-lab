# 六组代理方向探索代码

关联 [Issue #14](https://github.com/Falling-star1/llm-tta-lab/issues/14)。执行定义见 [PROTOCOL.md](PROTOCOL.md)。本目录为待评审实现，当前验证为 CPU/合成样例，真实模型 GPU 预检另行登记。

## 文件与入口

| 文件 | 职责 |
|---|---|
| inventory.py / prepare.py | 盘点冻结缓存，校验哈希，只用无答案输入生成共同分割 |
| analysis.py | 实验一的六组四分位、区间、分布与 Markdown 表 |
| backend.py | 复用并验证既有 v6 模型、编码、CE、生成和评分 |
| geometry.py / collect.py | 实验二的梯度、低/随机池、Gram 特征分解和独立诊断 |
| reviews.py | 隐去模型/CE 的诊断题包、判定导入、四组对比与可选图 |
| training.py / updates.py | AdamW 位移投影、同幅度对照、逐访问记录、检查点和配对评分 |
| entry.py | 显式阶段命令，输出身份和单写入锁 |
| make_requests.py | 生成请求及各组合的输出索引，供 labctl plan/run |

## 准备清单

在新 release 中复制本目录，保持代码文件完整；配置与请求放在 release 外。准备完成后提交本目录并固定完整 Git commit，review release 的实际文件与提交一致。运行时路径取现有冻结包，inventory 会记录它全部受保护文件和 code_id，Backend 在载入前验证。

在服务器 release 目录执行以下准备命令。路径名称可换成该次新请求的独立名称。

    python inventory.py --output /data/chuwenjing/lab/requests/proxy-assets-v1.json
    python prepare.py --assets /data/chuwenjing/lab/requests/proxy-assets-v1.json --output /data/chuwenjing/lab/requests/proxy-config-v1.json

这两步只冻结输入清单和指纹，不聚合正式回答成绩。Dolly 抽样顺序与旧试点使用相同的归一化去重、Random(42) 和 1,344 样本分割逻辑。输入每套共 5,000 条，建池/诊断各 256、适应 500、评测 300，备用前缀 32。

## 生成与检查请求

make_requests 的 --code-root 指部署后的本目录，--commit 必须是当前实现的真实 40 位提交。下例中的代码路径、提交和超时须用本次已核验值替换。CPU 和 GPU 运行统一由 labctl 启动。

    python make_requests.py --stage analyze --config /data/chuwenjing/lab/requests/proxy-config-v1.json --code-root /data/chuwenjing/lab/releases/本次代码 --commit 本次完整提交 --prefix proxy-e1-本次唯一标识 --timeout 600 --output /data/chuwenjing/lab/requests/本次CPU请求
    python /data/chuwenjing/lab/bin/labctl.py plan /data/chuwenjing/lab/requests/本次CPU请求/生成的实验ID.json

明确授权后使用同一个请求 run。请求生成器不会自动执行。preflight 生成六个单 GPU 请求，填写短预检超时。六请求可放在各自服务器 tmux 会话中，由中央锁排队，最多并行六组合。

完成全部短预检后，用 --preflight-index 指向其请求目录中的 index.json：

- --stage gradients 只采集六组梯度及无标签子空间。
- --stage updates 加 --geometry-index 指向梯度请求的 index.json，只做五条适应分支与六条件结果。
- --stage cell 在每个组合里连续做梯度采集和更新，仍需相同代码和配置的预检凭证。

请求生成时与实际入口都核对预检身份。每个组合只看见 labctl 分配的一张 GPU，使用 wrapper 锁策略，内部不获取重复 GPU 锁。数据/模型合同/配置和预检、子空间等依赖工件的 SHA256 写入请求。运行时使用固定 CUBLAS 和离线 NLTK 路径。

## 质量分组

用 CPU 导出某组合诊断集的盲评包：

    python entry.py --stage review-export --config 配置路径 --cell llama3b__dolly_5k --output 新盲评目录

blind.csv 交给评审者，private_key.json 留在分析端。每行填写 verdict（可接受／不可接受／未决）和 reason。字段内容校验会拒绝题目、回答或模型包错配，缺行、重复 ID 和未填写判定。此题包来自独立 diagnostic，两端各 64 题，与旧 history 的盲评不同。

    python entry.py --stage quality-report --config 配置路径 --cell llama3b__dolly_5k --geometry-output 该组合已完成的梯度输出/gradients --review-csv 已填写CSV --review-key 对应private_key.json --reviewer 评审者标识 --reviewer-type human --plots --output 新质量分析目录

GSM8K 使用同一 quality-report 命令，省略 review 相关参数，按冻结 Source EM 自动分组。文本组合缺少判定时输出 PENDING_DIAGNOSTIC_REVIEW。模型辅助评价必须显式用 model_assisted。人工来源由调用者声明，代码无法代替身份核验。

两端四组的实际数量、等量抽样 ID、余弦、谱、保留率与长度/上下文完整保存。m<2 或零梯度单列，热图屏蔽零梯度，图需要 matplotlib。质量标签及按质量划分的子空间均不进入更新算法。

`quality_groups.json` 的 `between.full` 和 `between.balanced` 记录好答与差答之间的平均余弦及有效配对数，零梯度不参与余弦。盲评排列采用 seed 421116 与组合名的 SHA256 派生种子，各组合的高低 CE 对应位置独立生成并可复现。

## 更新及恢复

六个条件为 Source、plain、plain_b、random、low、matched。plain_b 固定 A，仅应用普通 AdamW 的 B 位移；matched 也固定 A，并将其自身普通 B 位移缩放到 low 同访问位置的实际范数。低池先执行，matched 使用其逐访问实际位移范数。投影对象是 AdamW 的候选参数位移。每个分支重新加载初值并清空优化器，继承相同输入顺序与原协议门控和权重。

匹配以写回参数后的实际位移为准，目标非零时相对范数误差容限为 0.001；目标为零时实际位移应为零。逐访问记录 `match_failure`、绝对及相对误差，汇总未匹配位置。正目标因 FP32 舍入变成零位移会标记未匹配。

修订预检为 `proxy-six-preflight-v4`，更新为 `proxy-e3-six-adam-step-v3`，联合阶段为 `proxy-e2-v3-e3-v3`。每组合预检共 16 次访问，在固定的高 CE 适应输入上分别进行 plain、plain_b、project、matched 各 4 次加权更新；最长输入另作一次原始 CE 梯度检查。采用对应数据集的正式学习率。预检五维正交基由最长输入 B 梯度与四个 B 坐标方向构建，仅验证实际投影路径，运行后丢弃。检查有效更新、有限状态、投影残差不超过 0.001、幅度匹配及初始化恢复后签发凭证。正式历史池 U 仍由独立梯度采集产生。

训练每 25 次访问保存独立的 checkpoint 文件，再原子更新 checkpoint.json 指针。清单校验文件 SHA256、代码/配置身份和访问前缀；保存参数、Adam 状态、访问调度、GradScaler 和 Python/NumPy/Torch/CUDA RNG。推理 journal 保留完整行，中断的尾行在本次恢复中截断，逐题校验索引、参数身份和评分。

同一输出内的显式恢复使用 --resume。已有失败 labctl 实验 ID 不再次 run，跨运行恢复创建新的 updates 请求，使用相同代码、配置和 --geometry-index，再加 --resume-index 指向原请求索引。恢复会将各分支最新已提交检查点及预测 journal 复制到新输出，并保留旧运行；代码或配置身份改变时拒绝续接。阶段二逐题梯度缓存也校验身份和哈希，同一输出可继续采集。

## 验证和解释范围

    python -m unittest discover -s experiments/proxy_exploration_v1 -p "test_*.py"

从仓库根执行。测试使用临时的六组合成缓存和小型可微模型，覆盖真实保存/恢复与重复评测路径。真实 v6 FP16 路径、源预测重放、最长输入和显存须通过服务器短预检，CPU 结果单独报告。

源状态 Gaussian A / 零 B 会产生零 A 梯度。固定在该状态估计的 U 对 A 分量也为零，持续投影会限制 A 的更新。低池和随机池受相同约束，此结构及 Adam 位移投影与 Lifelong 原方法的差异应写入科学报告。用 low/random 检验池选择，low/plain_b 比较同一 B 参数范围中的约束效果，low/matched 比较同幅度方向，plain_b/plain 描述固定 A 的影响。Source、plain 与投影组的终点差异按整体效果解释。各固定 A 分支在实际写回后验证 A 位移为零；检查点保存冻结索引并拒绝错配续接。

当前实现保存全部梯度和检查点供审计。Llama 每条 FP32 LoRA 梯度约 9 MB，六组合梯度及检查点可能占数十 GB，正式准备时先测实际可用磁盘、内存和吞吐，再登记小时预算。运行输出、题包、模型、梯度和缓存均留在实验目录，不进入 Git。

六组修订执行预算为 15,000 次适应访问和 9,000 次新生成，比原预算增加 25%。代码已准备，新的执行预算须在正式请求中登记并获准。当前未启动正式运行。

真实预检数值修订：百万维参数列采用 CPU FP64 QR 后转存 FP32，优化器构造时用 FP64 Gram 检查正交性，保持原有误差容限。短预检 GradScaler 初始 scale=1024，避免默认 65536 的连续溢出耗尽四次访问；正式 500 次更新仍沿用默认 GradScaler。失败的首轮预检输出保留，新代码/新实验 ID 下复验六组。
