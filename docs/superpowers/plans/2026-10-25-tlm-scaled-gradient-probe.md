# TLM梯度缩放与实际更新验证计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐已获用户同意的真实GradScaler验证，判断未缩放梯度差异是否进入实际参数更新，不改变正式算法或放宽容差。

**Architecture:** 新建独立包，依赖冻结v3与第一轮完整诊断包；不修改两份旧包。先在相同初始权重、同一258-token输入上比较完整前向和真实Trainer的缩放、反传、还原梯度，再使用相同四条真实长输入做16步实际Trainer更新对照。只在梯度非有限时按GradScaler原backoff规则降低scale，不根据误差大小挑scale。正式预检和全量仍由后续独立验收决定。

**Tech Stack:** 服务器Torch2.5.1+cu118、Transformers4.46.1、PEFT0.12.0、Accelerate1.0.1；2080 Ti FP16。本地CPU机制测试使用已有兼容依赖和Torch2.14。

用户在前一轮结果解释后要求“继续”，批准沿已说明的GradScaler、分块归约与重计算排查顺序推进。服务器GPU任务仍由用户执行命令。

## 1. 边界与数据

- Gaussian A、零B、seed42，保留临时B=.001工程梯度探针。
- 真实GradScaler对象的配置作为起点；每对路径使用独立但相同状态的scaler。用`accelerator.backward(loss)`，再`scaler.unscale_(optimizer)`比较FP32 LoRA梯度。
- 只有完整路径和分块路径均有限时才比较，停止继续降低scale。非有限结果保留；不能选使误差更小的scale。
- 追加关闭投影checkpoint的控制项；原checkpoint对象始终在finally恢复。
- 更新对照沿用原16步工程配方、补偿筛选权重、cosine/warmup、lr=1e-6。使用最长四条真实输入的至多258-token前缀，重复访问只用于工程对照，不能写作正式离线一轮实验。
- 保存每步真实optimizer skip、scale、学习率、输入访问顺序与adapter更新差异。更新结果用于诊断，不据此生成正式PASS收据。
- 恢复权重、模块模式、RNG、原scaler对象及其状态、forward包装、函数替换、缓存和梯度。

## 2. 文件与步骤

新包路径：`offline_tta/tlm_scaled_precision_bundle`。

- [x] `tests/test_scaled_probe.py`：先验证缺失功能失败，再覆盖真实CPU GradScaler还原、溢出backoff、相同scale比较、异常恢复和真实Trainer多步对照。
- [x] `probe_support.py`：加载并校验旧包；保存/恢复初始状态；汇总梯度与参数变化。
- [x] `scaled_probe.py`：固定两种B状态，完整/Trainer双路径采用实际缩放反传，比较全部梯度；有限scale下追加无checkpoint对照。
- [x] `trajectory.py`：两次真实Trainer短程训练，完整前向与原分块分别16步，同起点同数据；恢复forward包装，逐步比较参数及实际变化量。
- [x] `diagnose_scaled.py`、`check.py`、`README.md`、`SHA256SUMS`：只提供有边界的诊断命令，不自动进入全量。
- [x] 独立复核、CPU回归、文件哈希和文档链接检查；发布到原PR，给用户服务器命令。

CPU数值机制用例：

```python
p = torch.nn.Parameter(torch.tensor([1.]))
loss = (p.half().float() * 1e-8).sum()
scaler.scale(loss).backward()
scaler.unscale_(torch.optim.AdamW([p]))
assert 0 < float(p.grad[0]) < 2e-8
```

该用例证明测试能覆盖FP16中间梯度下溢与缩放还原机制，不代表服务器TLM差异已经归因于下溢。

## 3. 验收

CPU通过、真实FP16诊断完成、正式预检通过分别记录。新报告必须标记非预检收据。若真实缩放后仍不满足原梯度标准，保留MISMATCH并继续定位；若更新轨迹不同，记录大小和条件，不能直接称训练等价或Dolly已修复。
