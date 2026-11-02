# 原版 TLM 复现结果与官方实现直接对照

记录时间：2026-10-28 10:03（北京时间）。正式表采用修订前的 Gaussian v6 六组结果；不使用 v7、格式清理、token 屏蔽或挑选较优检查点后的分数。既有结果和诊断实验均保留。

这里的“原版”指按论文 Gaussian 设置、沿用公开代码损失的 v6 实验，不表示它与官方默认配置、BF16 环境或论文全部条件严格相同。

## 六组最终结果

每组均完成 5000 条适应访问及 5000 条评测。Geography / Dolly 为 ROUGE-Lsum × 100，GSM8K 为 EM 百分比；差值是同组 TLM 减 Source。

| 模型 | 数据集 | Source | 原版 TLM | 差值 | 论文同模型 Source → TLM |
|---|---|---:|---:|---:|---|
| Llama3.2-3B | Geography | 23.98 | 29.32 | +5.34 | 23.95 → 28.93 |
| Llama3.2-3B | Dolly | 33.33 | 24.07 | -9.26 | 33.78 → 34.70 |
| Llama3.2-3B | GSM8K | 77.60 | 77.98 | +0.38 | 77.56 → 90.96 |
| Qwen2.5-3B | Geography | 24.94 | 29.64 | +4.70 | 论文为 7B，不直接对照 |
| Qwen2.5-3B | Dolly | 27.65 | 25.69 | -1.96 | 论文为 7B，不直接对照 |
| Qwen2.5-3B | GSM8K | 80.08 | 81.42 | +1.34 | 论文为 7B，不直接对照 |

## 核查范围和结果

| 核查项 | 论文或固定官方代码 | 当前原版实验 | 核查结论 |
|---|---|---|---|
| 源码身份 | 官方提交 `496c9ed97d0fb8bc4854ed2479259e7e9e70a790` | 对官方包、评分脚本和配置共 123 个文件核对 Git-blob；六份关键文件另用 SHA256 核对 | 123 / 123 一致，六份关键文件一致 |
| 模型 | Llama3.2-3B、Qwen2.5-7B 等 | Llama3.2-3B、Qwen2.5-3B | Llama 型号相同；Qwen 为规模扩展 |
| 原始数据 | 固定 AdaptEval 5k 文件 | 三份源文件哈希匹配，训练只使用问题与上下文 | 没有换题或加入参考答案训练 |
| 输入 token | 官方 SelfSupervisedDatasetProcessor | 同一官方原生模板，答案无关截断 | 30000 组模型×题目序列中 29995 组一致；5 组长输入有差异，详见下表 |
| 训练目标 token | 官方预处理设 labels=input_ids | labels=input_ids，padding 用 -100 | 包括问题内的格式文字和原生对话标记；没有套用后续屏蔽修订 |
| 筛选和权重 | 底模 CE > 3；0.1×exp(底模 CE−3) | 关闭 LoRA 计算底模 CE，按同样规则筛选加权 | 六组共 30000 条训练记录：筛选、权重与记录损失公式错误均为 0 |
| 损失形式 | 正文写加权 perplexity；公开 trainer 实际用 one-hot KL，即 CE 形式 | 加权输入 CE | 对照公开代码；正文与代码形式的差别仍需作者澄清 |
| LoRA 结构 | q_proj / v_proj，rank 8；代码默认 alpha 16、dropout 0 | 相同，仅 LoRA 参数可训练 | 结构一致、底模未变 |
| LoRA 初始化 | 论文写 Gaussian A、零 B；正常公开代码分支不显式设置初始化 | 显式 Gaussian，PEFT 标准差 1/r=0.125，B=0 | 符合论文分布类型；当前 PEFT 的官方默认分支实际走 Kaiming，二者不等价；作者 Gaussian 方差未说明 |
| 学习率与调度 | 论文 Geo/Dolly 5e-5，GSM8K 1e-6；官方示例一轮、batch 1、cosine、warmup 0.1 | 相同 | 配置一致；官方示例是 Geography / Llama3-8B，不是论文全部任务的完整配置 |
| 优化器 | 论文称 Adam；当前官方框架默认 AdamW、weight decay 0 | AdamW、weight decay 0，β=(0.9,0.999)，ε=1e-8 | 记录具体实现；不能据名称断言数值路径逐位等价 |
| 输出层和 CE | 官方完整 forward、one-hot KL | 完整输出层 GEMM，CPU 暂存 logits，分块 FP32 CE | 已做下述两模型各 8 步真实 Trainer 对照；不能外推全部训练轨迹 |
| 精度和环境 | 论文 A800 80GB / CUDA 12.1 / PyTorch 2.5.1；示例 BF16 | RTX 2080 Ti / CUDA 11.8 / PyTorch 2.5.1+cu118，FP16 底模与 FP32 LoRA | 存在差异，本次未做 A800 / BF16 实测 |
| 实际参数更新次数 | BF16 示例没有 FP16 GradScaler 同类跳步 | 六组分别跳过 4、3、0、6、5、3 次，共 21 次；5000 条均访问 | 必须报告；Llama GSM8K 为 0 次跳步，不能用跳步直接解释其分数差距 |
| 生成 | greedy、512 tokens；官方 EOS 加 additional_special_tokens_ids | greedy、512 tokens；模板停止符 | Llama 去重后相同；Qwen 停止符集合存在差异，尚未做完整官方解码对照 |
| 评分与工件 | ROUGE-Lsum / GSM8K 固定提取器 EM | 同指标、同官方提取器 | 本次每组约 32 条评分抽查无差异；Source 哈希不变、最终 adapter 与 checkpoint-5000 逐张量一致 |

## 逐 token 输入比较

直接调用已核验的官方处理器，使用原始参考答案长度复现其截断预算；当前输入由已冻结的 v6 编码器产生。这里读取参考答案仅为审计官方预处理，GPU 训练对照只读取严格无答案的 inputs.jsonl。

| 模型 | 数据集 | 完全相同的 token 序列 | 不同 |
|---|---|---:|---:|
| Qwen2.5-3B | Geography | 4998 / 5000 | 2 |
| Qwen2.5-3B | Dolly | 4999 / 5000 | 1 |
| Qwen2.5-3B | GSM8K | 5000 / 5000 | 0 |
| Llama3.2-3B | Geography | 4999 / 5000 | 1 |
| Llama3.2-3B | Dolly | 4999 / 5000 | 1 |
| Llama3.2-3B | GSM8K | 5000 / 5000 | 0 |

差异集中在长输入：Llama Geography 索引 3396、Dolly 索引 1404；Qwen Geography 索引 3131/3396、Dolly 索引 1404。不能因此断言它们对整体训练毫无影响，但 GSM8K 两组各 5000 条均一致，输入截断不构成该任务差距的直接解释。

## 前 8 步真实参数更新对照

两条路径从同一 Gaussian A / 零 B 初值开始，均使用完整 5000 条 GSM8K 无答案输入、seed 42、batch 1、lr 1e-6 和原 5000 步学习率调度，以回调在第 8 步停止。未缩短输入、未强行选择样本、未重置成 8 步调度。

官方路径调用固定 CustomSeq2SeqTrainer 的原始 compute_loss / cal_ce / cal_kl，当前路径调用冻结 v6 TLMTrainer。额外包装只记录索引、数值和回调停止；两边共享当前 FP16 模型加载、Gaussian 初值与运行环境。这是受控的实现比较，不是官方默认初始化或完整 TTL 工作流的重新运行。

| 模型 | 每条路径步数 | 样本顺序 | 损失最大绝对差 | 梯度最大绝对差 | 梯度最大相对 L2 差 | adapter 更新最大相对 L2 差 | adapter 更新 |
|---|---:|---|---:|---:|---:|---:|---|
| Llama3.2-3B | 8 | 一致 | 2.9802322e-08 | 0 | 0 | 0 | 逐值一致 |
| Qwen2.5-3B | 8 | 一致 | 0.00027221441 | 0.00059977174 | 0.010644234 | 0.047446968 | 有浮点差异 |

相对 L2 差以官方路径的范数为分母；adapter 比较相对于初始权重的更新量，避免被初始 A 的大小掩盖。梯度为 GradScaler unscale 及 clip 后、进入优化器前的梯度。每步 optimizer 状态、scaler 和具体数值在证据 JSON 中保留。

Llama：8 步的梯度、adapter 更新、optimizer 状态及 scaler 全部逐值一致；损失仅有最多 2.98e-8 的浮点归约差异。

Qwen：不能判为数值一致。第 1 步损失值相同，但梯度相对 L2 差已为 0.2407%；第 1 步学习率为 0，权重还未变化，但 Adam 状态已经产生差异。第 2 步 adapter 更新的相对 L2 差最大为 4.7447%，该时刻更新很小，最大绝对差仅 2.68e-9。到第 8 步，梯度相对 L2 差为 1.0644%，adapter 累积更新相对差为 0.3176%，最大绝对差为 1.26e-8。应同时看相对值和绝对值，不能据 4.74% 宣称全部权重偏差 4.74%，也不能因绝对值小就判为无影响。

两边的当前路径均精确重放原版最初 8 条训练历史。上述 Qwen 差异是这次直接对照新确认的实现路径差异；它在初始化、输入、底模打分、筛选和学习率相同的条件下出现，尚未进一步定位到损失归约、反向计算或其他具体算子。不能据此归因整个全量表现，更不能拿它解释 Llama 的 GSM8K 差距。

此前 full/chunk 预检以整块 CE 为参考；本次增加了实际官方 one-hot KL 和官方 Trainer 的直接比较，验证范围不同。此前 CE 路径预检通过，不能抵消本次观察到的 Qwen 差异。

- Llama3.2-3B：与原版前 8 条历史记录相比，索引、筛选、学习率均一致；底模 CE 最大绝对差 0，当前 CE 最大绝对差 0，损失最大绝对差 0。
- Qwen2.5-3B：与原版前 8 条历史记录相比，索引、筛选、学习率均一致；底模 CE 最大绝对差 0，当前 CE 最大绝对差 0，损失最大绝对差 0。

CPU 隔离比较还验证了实际官方 CE/KL 方法与当前 CE：batch 1 损失绝对差 0、梯度绝对差 0；带 padding 的 batch 2 损失绝对差 5.96e-8、梯度绝对差 0。正式实验 batch 为 1。

## 能得出什么结论

核查结果用于判断已验证范围内的数值实现是否存在差异，不能证明性能已复现。尤其不能把“前 8 步相近”解释成“BF16 与 FP16 等价”“官方默认初始化一致”或“5000 步轨迹相同”。

本次没有修改原版实现或覆盖结果。三次诊断入口接入失败的输出目录和日志也保留，分别发生在 GPU 空闲检查、官方构造参数和不完整参数对象检查阶段，均未执行 Trainer 的训练步骤；补充 CPU 小模型两路径各 8 步的集成检查后，再完成上表真实模型实测。正式报告只使用完成完整性检查的最后一轮诊断。

原版负结果保留。GSM8K 输入序列已逐条核对、训练筛选和权重记录已逐条检查；初始化默认分支、精度环境、完整训练轨迹及作者未提供的具体任务配置仍应列为复现限制，不能从本次对照中指定其中一项为已确认根因。

报告中建议写：“在本文记录的 FP16 环境和 Gaussian v6 配置下，Geography 复现了提升趋势；Dolly 出现退化；Llama GSM8K 未复现论文报告的提升。我们保留全部原始结果，并核查了公开实现与本地计算路径的差异。”

完整证据：[官方实现对照数据](tlm-official-comparison-evidence.json)。短步数诊断脚本：[说明](../../../offline_tta/tlm_official_comparison_v1/README.md)。

来源：[论文表 2/3、公式 5/6 与附录 C](https://arxiv.org/html/2505.20633v1)、[固定官方配置](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/examples/train_lora/offline_ttl.yaml)、[官方损失](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/train/ttl/trainer.py)、[官方输入处理](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/data/processor/self_supervised.py)、[官方 LoRA 初始化分支](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/model/adapter.py)。
