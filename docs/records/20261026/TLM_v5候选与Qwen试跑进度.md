# Llama v5候选与Qwen v4试跑进度

## Qwen保持v4

v4完整预检PASS后，用户启动100条Qwen＋Dolly试跑，目录为`/data/chuwenjing/results/tlm_offline_v4_pilot/qwen_dolly_100_20261025_170101`。2026-10-26 01:11核查：100条适应完成，100个索引各一次，81条选中、日志数值有限、基座未变；生成23/100条，暂无空答或高重复，4条达到512-token上限。23条前缀不能代替100条质量检查或5000条最终结果，不据前缀分数选参。新全量尚未启动。

## Llama使用独立候选

v4的Llama仍有13个张量含超原逐元素容差的元素。v5不改已通过和正在运行的v4，只测试一个训练后端策略：关闭FP16 GEMM的低精度中间归约。前向、反向和checkpoint重算在同一训练设置下完成；生成阶段临时使用已有Source的True设置，成功或异常后恢复训练值。两种设置写入model contract和精度报告，CPU回归覆盖异常恢复。

依据是[PyTorch 2.5.1数值精度说明](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst)。它说明某些GPU GEMM为了性能会截断中间累加，并提供关闭选项；它没有证明本次Llama残差一定由此造成。v5仍是待真实GPU验证的候选，不按CPU结果宣布修好。

69项CPU检查通过，独立复核无问题。该策略不是论文指定超参数，属于3B/FP16条件下的工程数值策略；TLM公式、Gaussian、阈值、学习率、权重dtype和原容差均不变。

只在空闲GPU6上运行Llama的v5完整预检，命令见[v5运行说明](../../../offline_tta/tlm_offline_v5_bundle/README.md)。Qwen继续v4试跑，不切换版本。不同代码身份和数值策略需在结果表分别登记，不能作为完全同配置的消融。

记录：[v5候选验证](tlm-v5-candidate-validation.json)、[Qwen试跑快照](qwen-v4-pilot-progress.json)。
