# TLM v4真实GPU预检结果

核查时间：2026-10-26 00:10—00:14，北京时间。运行目录使用服务器UTC日期命名：`/data/chuwenjing/results/tlm_offline_v4_preflight/run_20261025_160256`。

## 两组状态

| 检查 | Qwen3B＋Dolly | Llama3B＋Dolly |
|---|---|---|
| Gaussian、代码身份 | 匹配 | 匹配 |
| 所测输入的full/chunk损失与梯度 | 完全一致，144张量最大差0 | 13/112张量仍含超逐元素容差的元素 |
| 最差参数相对L2 | 0 | 0.77975%，低于1%限值，但逐元素仍失败 |
| 16步保存恢复 | EXACT_MATCH | 未执行 |
| Source样例核对、生成及基座冻结 | 通过 | 整套尚未通过 |
| 完整预检收据 | PASS | 无PASS收据 |

Qwen预检16次访问中实际更新13次、GradScaler跳过3次；连续与恢复运行的参数、优化器、调度器、RNG和scaler一致，满足原预检要求。另独立复核full和partial最终检查点各10个文件，全部哈希匹配commit，commit哈希匹配收据。峰值allocated约7396MiB、reserved约8152MiB。

这些结果证明Qwen这组已通过所定义的工程预检，不代表5000条适应后输出仍正常，也不是最终论文复现成绩。Llama仍明确失败，不能因相对L2较小而忽略逐元素失败。

## Qwen下一步：100条试跑

使用已通过预检的冻结v4，不重新上传或修改包内文件，避免改变代码身份使收据失效。试跑采用正式Dolly学习率5e-5、Gaussian、seed42、chunk128，先对前100条适应，再评测同100条。100条覆盖既有空答、重复和同一回答塌缩检查窗口。小试跑通过也不自动启动全量，结果先复核。

确认GPU7空闲后，服务器执行：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v4_bundle
mkdir -p /data/chuwenjing/results/tlm_offline_v4_pilot
QWEN_PILOT=/data/chuwenjing/results/tlm_offline_v4_pilot/qwen_dolly_100_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=7 nohup python -B -u run.py pilot   --model qwen3b --dataset dolly_5k --lora-init gaussian --seed 42   --chunk-size 128 --samples 100   --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k   --receipt /data/chuwenjing/results/tlm_offline_v4_preflight/run_20261025_160256/qwen/receipt.json   --output "$QWEN_PILOT"   > "${QWEN_PILOT}.log" 2>&1 < /dev/null &

echo "$QWEN_PILOT"
```

试跑结束需核对`run_manifest.json`为PASS、`metrics.json`计数100、`quality_latest.json`无报警、全部预测完整。若报警，保留`quality_alert.json`、权重和预测，先检查原因。参考答案只参与评分，不用于选学习率或替换最终模型。

Llama继续修数值路径，不给它的全量启动命令。其他旧任务沿既有安排，不自动转换为新Gaussian结果。

机器可读证据：[v4真实GPU预检记录](tlm-v4-real-gpu-preflight.json)。
