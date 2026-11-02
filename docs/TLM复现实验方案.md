# TLM 复现实验方案

<!-- original-results-policy-20261028 -->
正式结果采用修订前 Gaussian v6 六组 5000 条完整数据。v7 与屏蔽目标等修订实验作为独立诊断保留，不替换原版分数。最新六组结果及数值核查见[原版复现与官方实现对照](records/20261028/TLM原版复现与官方实现对照.md)。

截至 2026-10-28，六组各 5000 条适应及评测均已完成；最终分数以开头的官方核对记录为准。下面保留 2026-10-27 阶段进度，不代表仍需启动或接续实验。

Source、TENT、TLM 的历史成绩及每组服务器完整路径见[结果与目录汇总](records/20261027/TLM复现结果与目录汇总.md)。该记录早于六组最终完成时点。

### 2026-10-27 阶段快照

更新：2026-10-27 20:54（北京时间）。六组5000条适应训练已完成，5/6组全量评测已完成。Qwen Dolly已完成5000条，TLM 25.69；Qwen GSM8K生成到4667/5000。

Llama Dolly为Source33.33→TLM24.07，GSM8K为77.60%→77.98%；这两项没有复现论文中的提升。Dolly格式损失和GSM开场白/结束符对照、真实故障配对回放均已完成，确认部分行为问题，但尚未证明全量质量修复。最新结果、差异和证据见[排查进展](records/20261027/TLM_Dolly与GSM8K排查进展.md)。

原文基线保持原评分口径继续完成剩余评测。机制对照单独保存，不改名并入论文原方法，也不按分数改参数。旧Kaiming任务已停，旧结果保留；Qwen Dolly旧指纹异常的原因仍未确定，当前使用重新核验规范底模后的独立训练。历史运行与回执见[六组评测接续](records/20261026/TLM六组全量评测接续.md)和[底模核验记录](records/20261026/TLM_Qwen_Dolly底模指纹拦截与恢复.md)。

另已启动GSM8K四组200题配对验证，只评测现有2500步权重，不重训或替换原文基线；范围、注册与进度见[修正版验证记录](records/20261027/TLM_GSM8K修正版200题配对验证.md)。

## 1. 这轮要完成什么

在两种 3B 模型、三个数据集上补齐 TLM。每格都是独立实验：从原模型新建 Gaussian LoRA，用 5000 条问题适应一遍，再回答同样 5000 条问题。

| 模型 | GeoSignal | Dolly | GSM8K |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct | 5000 条 | 5000 条 | 5000 条 |
| Qwen2.5-3B-Instruct | 5000 条 | 5000 条 | 5000 条 |

适应只读问题和上下文，参考答案留到评分时使用。不同数据集之间不继承适应权重。旧 Kaiming 结果单独记录，不能改名并入这轮 Gaussian 主实验。

本轮是 **3B / FP16 条件下的方法复现**。论文 Qwen 主结果使用 7B，官方示例使用 BF16，和当前 2080 Ti 的条件不同；结果中会写明这些差异。

## 2. TLM 怎么更新模型

对一条问题，依次做四件事：

1. 套用模型的固定提示模板，得到输入 token。
2. 关闭 LoRA，用冻结的底模计算输入平均交叉熵 `c0`。
3. 只选中 `c0 > 3` 的样本，权重为 `0.1 × exp(c0 - 3)`；`c0` 和权重不参与反向传播。
4. 打开 LoRA，计算当前模型在同一输入上的平均交叉熵，乘上权重并更新 LoRA。

前一个位置预测下一个输入 token，排除 padding；真实 EOS 不因 pad 与 EOS 同 ID 而被删掉。未选中的样本保留零损失图和固定源码中的优化器调用；已有 Adam 动量仍可能推动参数变化。

损失和 offline 流程核对官方提交 `496c9ed97d0fb8bc4854ed2479259e7e9e70a790`。初始化遵照论文，显式使用 Gaussian A、零 B。可核对[论文](https://arxiv.org/html/2505.20633v1)、[官方配置](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/examples/train_lora/offline_ttl.yaml)和[官方损失实现](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/train/ttl/trainer.py)。

## 3. 固定设置

| 项目 | 本轮设置 |
|---|---|
| 数据 | 固定版本的三个数据集，各 5000 条，核验文件哈希 |
| LoRA | q_proj、v_proj；rank=8、alpha=16、dropout=0 |
| 初始化 | Gaussian A、零 B；seed=42 |
| 数值类型 | 冻结底模 FP16，LoRA 参数和 CE 为 FP32；实际 Trainer 使用 AMP 与 GradScaler |
| 输入 | 最多 4096 token；裁剪用户内容尾部后重建完整模板，不读取答案决定截断 |
| 优化器 | AdamW；betas=(0.9, 0.999)、eps=1e-8、weight decay=0 |
| 学习率 | Geo/Dolly：5e-5；GSM8K：1e-6 |
| 调度 | cosine；warmup 前 10%；梯度裁剪 1.0 |
| 训练 | 一轮，batch=1，梯度累积=1；Trainer 按 seed=42 采样 |
| 生成 | 固定模板与停止符，greedy，最多 512 个新 token |
| 指标 | Geo/Dolly：ROUGE-Lsum；GSM8K：固定官方答案提取器和 Exact Match |
| 保存 | 完整检查点每 25 步一次、保留最近两个；另存 100/500/1000/2500/5000 步 adapter |

模型与哈希见 [profiles.json](../offline_tta/tlm_offline_v6_bundle/profiles.json)，数据与哈希见 [datasets.json](../offline_tta/tlm_offline_v6_bundle/datasets.json)。Torch 2.5.1+cu118、Transformers 4.46.1、PEFT 0.12.0、Accelerate 1.0.1 等版本由运行时核验。

## 4. 为什么有 v6

2080 Ti 显存有限。之前把输出层矩阵乘法分块，真实 GPU 反向梯度与整块实现不完全一致。v6 保留一次完整的 FP16 输出层计算，交叉熵分块计算，完整 FP16 logits 暂存到 CPU，反向时逐块取回。原始输出层矩阵形状、因果目标、TLM 权重和学习率保持一致。

此前 v6 的六组内部数值预检曾通过；其中 Dolly 旧底模指纹随后被撤出主实验，须以新的单组回执为准。历史内部检查结果为：零 B 和非零 B、不同长度共 32 组比较，LoRA 梯度及输出层反向梯度逐值一致；六组断点恢复一致；最长输入都完成 16 步实际混合精度训练，4096 条件下分配显存峰值约 8.0 GiB。12 份最终检查点共 120 个文件的哈希已独立核验。

这些内部数值检查不能替代规范底模身份检查，也不等于 5000 条成绩已经出来。六组训练均已完成，最新全量评测进度见本文开头。详细证据见 [真实 GPU 预检记录](records/20261026/tlm-v6-real-preflight.json)。

## 5. 当前怎么运行

训练包固定为 [tlm_offline_v6_bundle](../offline_tta/tlm_offline_v6_bundle/README.md)，代码身份：

```text
b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999
```

本轮已经走完预检、百条验证及正式 5000 条适应。Qwen Dolly 原全量启动前被底模检查拦截，后来在规范底模上重新核验并独立完成训练；不能使用旧回执替代这次记录。百条验证权重没有接入正式训练。

当前完成剩余原文基线评测，同时保留独立机制对照和真实故障回放。[评测接续入口](../offline_tta/tlm_evaluate_all_v1/README.md) 保留四组原进程，并加载两组 Dolly 的最终权重继续评测。两组均通过已保存首尾回答的逐 token 重放核验，已有预测不重写。

```bash
TLM_EVAL=/data/chuwenjing/results/tlm_eval_complete_v6/run_20261026_052804_45ad81
cat "$TLM_EVAL/status.json"
nvidia-smi
```

这个状态文件登记六组实际输出路径，当前使用 GPU 0、1、2、3、6、5。不要重复启动。旧控制器中保留的 BLOCKED 是历史停评记录，实时进度以新目录为准。

每组 `training_complete.json` 表示适应训练完成，`tlm.jsonl` 每行是一条实际生成的回答。只有每组生成并评分 5000 条后，才算全量实验完成。质量报警与执行状态分别记录。

今后从头新建实验可参考[统一控制器 v2](../offline_tta/tlm_full_run_gate_v2/README.md)，其中原有“质量报警即停止”的规则属于工程检查，不是论文要求。当前这一轮不用它重启，按上述接续入口完成所有样本评测。

## 6. 怎么判断结果能用

每组要有完整的 5000 条适应记录和 5000 条连续索引的回答；训练值、adapter 和 optimizer 有限，底模哈希不变，评分文件能从已保存回答重新计算。最终汇总同时给出 Source、TENT、TLM 和差值，负结果照实保留。

复用的 Source 来自已完成的同模型 TENT 运行。六份 Source 均有完整 5000 条，预检生成样例一致，输入身份、原始提示哈希和使用 token 数会继续核对。只复用底模 Source，不复用 TENT 适应权重。另外用 CPU 逐条核对了六组共 30000 条输入：token ID 与冻结的 TENT 编码器相同，所有 Source 提示元信息一致。见 [完整输入核对](records/20261026/tlm-v6-source-input-audit.json)。

生成每满 100 条，和配对 Source 检查空答及明显循环重复。当前工程报警规则是空答/重复超过 Source 5 个百分点，或同一非空答案占一半以上且超过 Source 10 个百分点。报警后保留真实 ALERT、权重、预测和日志，继续完成全部评测，不删坏样本或自动改参数重跑。身份或数值错误仍报错；输出质量下降本身不触发调参。

FP16 初始缩放过大时，GradScaler 可能跳过更新；因此记录 global_step 之外的实际成功步数。出现非有限 adapter/optimizer、长期没有有效更新、输入或代码身份变化，都会报错。单条日志里的初期 grad_norm 非有限值，须结合 GradScaler 回退和最终有限性检查判断。

## 7. 时间和论文差异

训练已完成，剩余时间主要花在Qwen生成。最新条数见本文开头；原文基线评测与诊断是独立进程，诊断完成不代表剩余主评测已完成。早期估时保存在历史记录中，不能当作当前剩余时间。

需要在结果里保留四点说明：Qwen 使用 3B；2080 Ti 使用 FP16；输入截断不受答案长度影响；CPU 暂存 logits 是工程省显存措施。已有 TENT 与本轮 TLM 的训练配方也有差异，分数变化不能全部归因于损失函数。

短验证检查是否能稳定运行，不保证全量会提高分数。此前 Qwen v4 百条结果为 TLM 27.05、Source 28.01，未触发退化报警；本轮 Gaussian 初始化和论文学习率不按这次分数取舍。

## 历史记录

- [旧运行核查与论文差异](records/20261025/TLM运行核查与风险_20261025.md)
- [Gaussian 初始化与 v3 验证](records/20261025/TLM修复选择与v3验证.md)
- [完整梯度诊断](records/20261025/TLM完整梯度诊断结果_20261025.md)
- [实际 GradScaler 与 16 步更新](records/20261025/TLM缩放与16步更新结果_20261025.md)
- [v4 预检及 Qwen 百条入口](records/20261026/TLM_v4预检结果与Qwen试跑.md)
- [v5 候选的当时状态](records/20261026/TLM_v5候选与Qwen试跑进度.md)
- [v6 六组实测与统一启动](records/20261026/TLM_v6预检与全量准备.md)

历史文件保留当时的结论，当前状态以本文开头和服务器运行文件为准。
