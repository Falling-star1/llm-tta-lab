# TLM真实梯度缩放与16步更新对照

上一轮已确认：完整前向与单块计算一致，128-token多块计算开始出现未缩放梯度差异。本包补实际GradScaler路径，再检查差异是否进入真实参数更新。

本包只做有边界的诊断，不启动5000条全量、不生成评测回答、不生成正式预检收据。新增短训练结束后恢复内存中的初始LoRA，旧权重文件和已有运行均不修改。

## 这次查什么

1. 固定Gaussian A、seed42，分别检查原始B=0及临时B=0.001。调用真实`accelerator.backward(loss)`，再用真实`GradScaler.unscale_(optimizer)`还原FP32 LoRA梯度；不做优化器更新。
2. 完整与分块路径从实际GradScaler的相同初始状态出发。如果任一路径出现非有限梯度，双方一起按原backoff factor降低scale；到第一个双方均有限的scale就停止降低。不会按误差大小挑scale，也不改变原梯度容差。
3. 在该scale下关闭投影checkpoint做额外控制，区分重计算与分块的影响。
4. 从相同原始零B权重开始，完整计算和原分块计算各用实际Trainer跑16步。记录每一步scale、是否跳过优化器更新、学习率、输入顺序和参数变化差异。

更新对照使用原工程预检配方：lr=1e-6、AdamW、weight decay=0、cosine、warmup=0.1、clip=1。选最长四条真实输入的至多258-token前缀，重复访问用于16步工程对照。临时强制筛选并补偿权重。它不属于正式“5000条访问一轮”实验，也不改Dolly正式学习率5e-5。

即使缩放后梯度MATCH，也仍需把实际数值路径纳入正式预检，完成最长输入、保存恢复、Source核对和试跑。报告中`ready_for_full_training`始终为false。

## 上传

服务器需已保留原v3包和上一轮独立诊断包。本机PowerShell：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_scaled_precision_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

## CPU检查

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_scaled_precision_bundle
python -B check.py
nvidia-smi
```

需看到72项检查和OK：原v3的60项、上一轮诊断5项、本轮7项。新增测试覆盖真实CPU GradScaler、FP16中间梯度、联合backoff、Llama/Qwen、真实Trainer更新、异常恢复与forward包装清理。CPU结果不能代替下面的2080 Ti FP16运行。

## 两组服务器诊断

确认GPU6、7空闲。GPU锁仍沿用原运行器；忙卡会拒绝启动。

```bash
mkdir -p /data/chuwenjing/results/tlm_scaled_precision
TLM_SCALED_ROOT=/data/chuwenjing/results/tlm_scaled_precision/run_$(date +%Y%m%d_%H%M%S)
mkdir "$TLM_SCALED_ROOT"

CUDA_VISIBLE_DEVICES=6 nohup python -B -u diagnose_scaled.py \
  --model llama3b \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_SCALED_ROOT/llama" \
  > "$TLM_SCALED_ROOT/llama.log" 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=7 nohup python -B -u diagnose_scaled.py \
  --model qwen3b \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_SCALED_ROOT/qwen" \
  > "$TLM_SCALED_ROOT/qwen.log" 2>&1 < /dev/null &

echo "$TLM_SCALED_ROOT"
```

每个模型除有边界的梯度探针外，仅进行两条16步工程轨迹，不接入旧检查点或旧结果。

## 结果文件

- `scaled_report.json`：实际scale、每次非有限检查、还原后的梯度、完整/分块/无checkpoint控制比较。
- `trajectory_report.json`：每步参数值和相对初始参数的变化量差异、实际优化器计数、访问顺序、scale和跳步信息。
- `run_summary.json`：诊断是否记录完成、基座未变和初始adapter恢复等汇总；不是正式PASS收据。
- `failure.json`：执行错误；已完成的部分报告仍保留。

若退出码为2，先保留报告；它表示仍有待查项，不应通过放宽阈值或重复原命令追求0。退出码0也不表示可以直接跑全量。日志查看：

```bash
tail -n 20 "$TLM_SCALED_ROOT/llama.log"
tail -n 20 "$TLM_SCALED_ROOT/qwen.log"
```

旧包固定身份：v3代码`267c3593e1d2c62f7d741168dcfd82ff76dc60f5458a06e481beca570997915d`；上一轮诊断SHA256SUMS哈希`7d606a2b95c62e4e2468ea59b1addfc1a7daf1a5cc470260b775dee6a6a1b2fd`。三份交付均校验文件哈希，输入也核对原始数据身份。
