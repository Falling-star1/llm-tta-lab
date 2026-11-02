# TLM v4：直接修订训练分块计算

本包修改正式训练使用的分块损失实现。旧v1/v2/v3、两份独立诊断包和已有结果保持原样。初始化继续Gaussian A、零B；TLM目标、阈值3、lambda=0.1、学习率、数据和生成设置不变。

**本地67项CPU检查通过不代表真实GPU修复完成。先运行下面两组v4完整预检，当前不提供全量启动命令。**

## 实际改动

1. 原来末块可能只有一个预测位置，head矩阵形状从128变为1。v4只在head内部给末块补零hidden，并为虚拟位置设置ignore标签，使其采用正常tile形状。虚拟位置无损失、无参数梯度，不改变decoder输入、因果位移、真实token数量或模板。整段不足一个块的短输入不额外拉长到128。
2. 原来各块先求和，再把块和相加。v4让checkpoint返回每个真实token的CE，拼接后按完整路径一次FP32求和，再除以原来的有效token数。只多保留很小的token CE向量，仍不保留整段词表logits。
3. 正式预检改用实际Accelerate backward与GradScaler unscale比较梯度。双方使用相同scale；仅出现非有限梯度时按原backoff共同降低，到第一个共同有限scale就比较，不按误差挑scale。原atol=1e-5、rtol=1e-2、逐参数relative L2≤1%保持不变，临时非零B覆盖A/B两类梯度。结束和异常均恢复原adapter、scaler、随机状态和模块模式。
4. 失败时保存完整参数统计；缺失梯度立即报错。预检身份新增chunk_size，防止换块大小后混用收据。

CPU回归已直接捕获旧实现的尾块形状变化和FP32归约顺序差异；修订后这些回归通过。真实2080 Ti上是否消除了剩余梯度差异，必须以本轮GPU结果为准。当前仍不能宣称Dolly退化已经修好。

## 上传与检查

本机PowerShell：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_offline_v4_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

服务器：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v4_bundle
python -B check.py
nvidia-smi
```

需看到67项检查和OK。这是本训练包原60项加7项回归；前一轮72项是训练包加两份独立诊断包的合集，仍分别保留，不是删掉测试。依赖继续使用Transformers4.46.1、PEFT0.12.0、Accelerate1.0.1。

## 两组Dolly完整预检

确认GPU6、7空闲，再运行。使用新结果目录，不接旧检查点，不自动进入全量。

```bash
mkdir -p /data/chuwenjing/results/tlm_offline_v4_preflight
TLM_V4_ROOT=/data/chuwenjing/results/tlm_offline_v4_preflight/run_$(date +%Y%m%d_%H%M%S)
mkdir "$TLM_V4_ROOT"

CUDA_VISIBLE_DEVICES=6 nohup python -B -u run.py preflight \
  --model llama3b --dataset dolly_5k --lora-init gaussian \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_V4_ROOT/llama" \
  > "$TLM_V4_ROOT/llama.log" 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=7 nohup python -B -u run.py preflight \
  --model qwen3b --dataset dolly_5k --lora-init gaussian \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_V4_ROOT/qwen" \
  > "$TLM_V4_ROOT/qwen.log" 2>&1 < /dev/null &

echo "$TLM_V4_ROOT"
```

预检不仅比较梯度，还运行原来的最长输入、真实更新、16步保存恢复、基座冻结与Source样例/生成检查。只有整套完成才写PASS收据；通过梯度一步不等于完整预检通过。

```bash
tail -n 30 "$TLM_V4_ROOT/llama.log"
tail -n 30 "$TLM_V4_ROOT/qwen.log"
```

数值细节保存在各组`resume_probe/full/trainer/precision_path_check.json`；总失败见`failure.json`，完整通过见`receipt.json`。任一失败保留原记录，不改阈值追求PASS。完整预检通过后还需正式小试跑，再按相同代码、模型、数据、seed、Gaussian和chunk_size登记全量。

质量检查继续沿用v3：每100条生成检查空答、内部循环和同一回答塌缩，报警时停止该组并保留权重和预测。它不改变训练公式、不删除异常样本，也不替换最终模型。
