# TLM真实GPU梯度诊断

v3两组Dolly预检停在梯度比较，原检查只报告第一个失败张量。本包一次记录全部LoRA参数，并比较五条计算路径，帮助查清误差从哪里出现。

本包不更新参数，不生成回答，不进行5000条适应，不生成正式预检收据。即使报告为MATCH，也还要经过原有真实模型预检、保存恢复和小试跑，才能启动主实验。旧v1/v2/v3代码和旧结果均不修改。

## 比较什么

初始化固定为Gaussian，seed=42。同一模型、同一输入分别比较原始B=0和临时B=0.001；后一项覆盖A、B两类梯度，与此前失败的v3检查相同。诊断结束后恢复原始参数、各模块训练模式和随机状态。

| 路径 | 用来区分什么 |
|---|---|
| full | 完整模型前向，作为参考 |
| direct_full_head | 直接调用decoder，head仍投影全部N个位置；检查外层模型路径的影响 |
| direct_single | head只投影N-1个预测位置，放在一个块里；检查投影形状变化 |
| direct_multi | 将N-1个预测位置按128分块；检查分块和累加影响 |
| trainer_multi | 调用冻结v3的真实compute_loss；检查实际训练接线与独立计算是否一致 |

五条路径均显式使用同一Accelerate autocast上下文，并记录decoder是否进入autocast。使用最长真实输入的至多258-token前缀；默认覆盖128+128+1个预测位置。读取5000条输入是为了按原预检规则选输入，不是训练5000条。

报告保留基座CE、当前CE、样本权重、加权损失，decoder hidden states、原始head logits及hidden states的反向梯度，以及每个LoRA梯度的范数、差值范数、相对L2、方向相似性、近零比例和超容差数量。除与full比较外，也记录与上一条路径的比较。原梯度容差atol=1e-5、rtol=1e-2、相对L2上限1%保持不变。

这些比较帮助定位来源，不自动判定哪种实现正确，也不证明Dolly退化的原因。临时B不是训练配置修改；正式初始化仍为B=0。仅诊断临时强制筛选，并用0.1×exp(-3)抵消阈值变化，使非零权重幅度与正式配方相同。

诊断捕获包含GPU到CPU的数据复制和同步，耗时不能拿来估算正式训练性能。

## 上传

本机PowerShell执行，只上传新诊断包。服务器应已有未修改的`tlm_offline_v3_bundle`：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_precision_diagnostic_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

## 先做CPU检查

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_precision_diagnostic_bundle
python -B check.py
nvidia-smi
```

需看到65项检查和OK。它覆盖原v3的60项和新增5项诊断测试，包含故意注入梯度误差、Inf以及异常退出，确认完整留证且恢复状态。CPU通过不能代替下面的真实FP16检查。

## 两组真实GPU诊断

先确认GPU6、7空闲。程序还会沿用v3的GPU锁和占用检查；若卡上有任务会拒绝启动。不要停止其他组来腾卡。

```bash
mkdir -p /data/chuwenjing/results/tlm_precision_diagnosis
TLM_DIAG_ROOT=/data/chuwenjing/results/tlm_precision_diagnosis/run_$(date +%Y%m%d_%H%M%S)
mkdir "$TLM_DIAG_ROOT"

CUDA_VISIBLE_DEVICES=6 nohup python -B -u diagnose.py \
  --model llama3b \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_DIAG_ROOT/llama" \
  > "$TLM_DIAG_ROOT/llama.log" 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=7 nohup python -B -u diagnose.py \
  --model qwen3b \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_DIAG_ROOT/qwen" \
  > "$TLM_DIAG_ROOT/qwen.log" 2>&1 < /dev/null &

echo "$TLM_DIAG_ROOT"
```

每个模型只做两种B状态、五条路径，共10次有梯度计算，不执行optimizer.step。两组均固定Gaussian，避免同时更换初始化和其他参数。

## 看结果

```bash
tail -n 20 "$TLM_DIAG_ROOT/llama.log"
tail -n 20 "$TLM_DIAG_ROOT/qwen.log"
```

每完成一条路径就更新`模型目录/report.json`，不会在首个数值不一致处退出。最终`comparison_status`为MATCH或MISMATCH；MISMATCH的进程退出码为2，报告仍完整保留。若发生执行错误，保存`failure.json`和已完成部分的报告，退出码为1。

报告不是`receipt.json`，不能交给正式运行器使用。最终MATCH只说明这些有限诊断比较通过，不代表原来的完整预检通过。下一步根据报告找到首次差异位置，再决定修正方式，不直接放宽阈值。

本包只接受原v3代码身份`267c3593e1d2c62f7d741168dcfd82ff76dc60f5458a06e481beca570997915d`。两份交付文件均校验SHA256SUMS；输入也使用原数据哈希和纯输入schema验证。
