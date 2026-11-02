# 当前六组与 TLM 原文对照

核查时间：2026-10-26 18:57 左右（北京时间）。只读取服务器记录，在 CPU 上检查，没有暂停任务、占用 GPU、重训或修改配置。

六组训练都完成了，各 5000 条。评测仍在进行。训练配方的关键项能与原文及固定官方代码对上，但输出表现没有全部复现论文：Geo 两组提高，GSM8K 暂时只小幅提高，Dolly 两组下降。本轮仍不能称为严格原条件或论文性能复现。

## 已完成样本的实际表现

每行只比较同一组、同一批已完成索引的 Source 与 TLM。Geo / Dolly 为 ROUGE-Lsum 乘 100，GSM8K 为 EM 百分比。不能拿这些前缀分数当最终 5000 条成绩。

| 模型 | 数据集 | 已生成 | 配对 Source | 当前 TLM | 差值（百分点） |
|---|---|---:|---:|---:|---:|
| llama3b | GeoSignal | 3603/5000 | 24.01 | 29.53 | +5.53 |
| llama3b | Dolly | 2151/5000 | 32.86 | 23.61 | -9.25 |
| llama3b | GSM8K | 2342/5000 | 76.64 | 77.46 | +0.81 |
| qwen3b | GeoSignal | 2933/5000 | 25.10 | 29.46 | +4.36 |
| qwen3b | Dolly | 1104/5000 | 27.00 | 24.44 | -2.56 |
| qwen3b | GSM8K | 1188/5000 | 80.89 | 81.31 | +0.42 |

Llama Dolly 已完成 2151 条中，161 条被固定的重复检测规则标记（7.48%），Source 同批为 0。Qwen Dolly 已完成 1104 条中，134 条空答（12.14%）、58 条被标记重复（5.25%）；两类计数不能直接相加当成独立错误率。Qwen 的这些空答均为第一步生成 `<|im_end|>` 正常结束符，并非未写入回答文件。

原文 Table 2 / 3 的 Llama3.2-3B 主指标为：Geo 23.95 → 28.93，Dolly 33.78 → 34.70，GSM8K 77.56% → 90.96%。目前 Geo 的提升方向相近；Dolly 方向相反；GSM8K 远未显示原文报告的提升幅度。当前是前缀，模型数值条件和部分预处理不同，不能据此把差距直接归因于某一项。

## 训练与评分核查

- 六组均为 Gaussian A / 零 B，q_proj、v_proj，rank 8、alpha 16、dropout 0、seed 42。没有接入旧 Kaiming 权重或百条验证权重。
- 每组适应记录恰好 5000 条、5000 个唯一索引。关闭 LoRA 计算底模输入 CE，严格用 `c0 > 3` 筛选，权重 `0.1 * exp(c0 - 3)`，当前输入 CE 乘权重；逐条核对未发现阈值、权重或损失公式不符。
- Geo / Dolly 学习率 5e-5，GSM8K 1e-6；batch 1、一轮、cosine、warmup 0.1。实际为 AdamW、weight decay 0；与论文所写 Adam 在相同其他设置下更新公式等价。官方损失实现使用 CE / one-hot KL，本轮按该实现核对，不把公式中的 perplexity 另改成新的指数损失。
- 六组最终检查点文件哈希全部匹配；最终 adapter 与 checkpoint-5000 逐张量一致，adapter、optimizer 和记录中的数值有限。训练完成记录均确认底模未变，底模身份记录符合规范指纹。
- 六组 FP16 GradScaler 跳步分别为 4、3、0、6、5、3，共 21 次。样本仍全部访问；这不等于六组都有 5000 次成功参数更新，也是与 BF16 条件的差异。
- 回答索引连续，配对 Source 文件哈希未变；每组约 32 条抽样重新评分，Source 和 TLM 均未发现不符。主指标与原文一致：Geo / Dolly 为 ROUGE-Lsum，GSM8K 为固定官方答案提取器后的 EM。

## 已确认的条件与实现差异

| 项目 | 原文或固定官方实现 | 当前 |
|---|---|---|
| 模型 | 含 Llama3.2-3B、Qwen2.5-7B 等 | Llama3.2-3B 相同；Qwen 为 3B |
| 硬件与精度 | 原文 A800 80GB / CUDA 12.1；官方示例 BF16 | 2080 Ti / CUDA 11.8 / FP16，LoRA 与 CE 为 FP32 |
| 覆盖范围 | 四模型、十个数据集 | 两模型、三个数据集，共六组 |
| 长输入 | 官方处理器结合输入和参考答案长度分配 4096 预算 | 不读取答案决定输入截断；重建完整提示模板 |
| Qwen 停止符 | EOS 加 tokenizer 的 additional_special_tokens_ids | 仅模板停止符 151645 |
| 输出层与 CE | 官方整块前向与损失 | v6 保留完整输出层 GEMM，CPU 暂存 logits、分块 FP32 CE；先前预检有梯度逐值一致证据 |

当前实际裁剪输入数：Llama Geo / Dolly 各 1，GSM8K 为 0；Qwen Geo 2、Dolly 1、GSM8K 为 0。这是当前规则的裁剪数量，不等于与官方 token 序列不同的总条数，后者未在本次逐条重跑官方预处理。

Qwen 停止符这项此前没有充分列明，本次 CPU 加载实际 tokenizer 后已确认。Llama 两种停止符写法去重后相同；Qwen 官方额外包括 im_start 等特殊符号。后续只读扫描时，Qwen TLM 的 Geo 2944 条、Dolly 1113 条、GSM8K 1195 条均未生成这些额外停止符，所以截至该扫描时没有发现它改变已保存的 TLM 回答。Source 未保存生成 token ID，不能按相同办法逐 token 验证。它解释不了已查的 134 条 Dolly 空答，也不能据此宣称全量均无影响。

## 这轮如何处理

继续完成现有六组评测，所有负结果和 ALERT 保留，不按分数调初始化、学习率或检查点。本次没有把配置差异悄悄改进运行中。Qwen 停止符差异需在最终协议中明确；如补官方解码对照，可复用已训练 adapter 单独评测并单列结果，不需要为这个生成设置差异重训。

重复和空答的原因尚未定位。当前检查支持“记录与固定计算配方一致”，不支持“已证明实现每一处都与官方原条件相同”，也不支持“性能已经复现”。

证据见[实时配置、权重和输出核查](tlm-paper-live-comparison.json)。

来源：[原文及附录 C](https://arxiv.org/html/2505.20633v1)、[固定官方训练配置](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/examples/train_lora/offline_ttl.yaml)、[官方损失](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/train/ttl/trainer.py)、[官方生成流程](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/train/ttl/workflow.py)、[官方输入处理](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/src/llamafactory/data/processor/self_supervised.py)、[官方评分脚本](https://github.com/Fhujinwu/TLM/blob/496c9ed97d0fb8bc4854ed2479259e7e9e70a790/scripts/eval/eval_similarity.py)。
