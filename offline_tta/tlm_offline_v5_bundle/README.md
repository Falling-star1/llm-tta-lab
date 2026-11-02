# TLM v5：Llama的FP16归约精度候选

Qwen的v4完整预检已通过，正在做100条试跑。Llama的v4仍有13个张量不满足原逐元素容差。此包保留v4的末块对齐、单次FP32求和和实际GradScaler检查，只增加一个训练数值策略，不修改已通过的v4或现有Qwen进程。

## 改动与范围

训练进程设置`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False`。权重仍为FP16、LoRA和损失归约仍为FP32，改动的是是否允许FP16 GEMM中间归约发生低精度截断。[PyTorch 2.5.1数值精度文档](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst)说明了该开关。

| 阶段 | v4 | v5 |
|---|---|---|
| 适应训练及数值检查 | 允许低精度中间归约 | 禁止低精度中间归约 |
| 生成回答 | 原Source的True设置 | 临时恢复原Source的True设置，结束或异常后恢复训练值 |

策略写入model contract和数值报告。CUDA数值检查要求训练策略为False；代码身份改变，不能复用v4收据。公式、有效token、Gaussian、学习率、阈值、数据、生成参数和原验收容差不变。

这是显存受限条件下的工程数值策略，不是论文明确规定的超参数，也不代表已证实Llama残差由此造成。仍属于3B/FP16条件下的方法复现，不能称为原条件严格复现。是否解决GPU差异，必须看下面的真实完整预检；失败仍记FAIL，不放宽门禁。

## 上传和CPU检查

本机PowerShell，只上传新包：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_offline_v5_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

服务器：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v5_bundle
python -B check.py
nvidia-smi
```

本包69项CPU检查包含v4的67项和两项新增策略测试：训练值为False，生成值为True，成功及异常均恢复训练值。CPU不执行真实CUDA矩阵计算，不能证明关闭该开关在2080 Ti上的数值效果。

## 只跑Llama完整预检

确认GPU6空闲，GPU7上的Qwen试跑继续使用v4。

```bash
mkdir -p /data/chuwenjing/results/tlm_offline_v5_preflight
LLAMA_V5=/data/chuwenjing/results/tlm_offline_v5_preflight/llama_dolly_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6 nohup python -B -u run.py preflight \
  --model llama3b --dataset dolly_5k --lora-init gaussian --seed 42 \
  --chunk-size 128 \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$LLAMA_V5" \
  > "${LLAMA_V5}.log" 2>&1 < /dev/null &

echo "$LLAMA_V5"
```

该命令只做完整预检，不启动5000条全量。需要整套数值、最长输入、保存恢复、基座冻结、Source样例及生成检查通过，才有PASS收据。失败细节仍保存在`resume_probe/full/trainer/precision_path_check.json`及`failure.json`。

所有旧运行、旧模型和冻结代码保留。Qwen的v4收据不能用于v5，也不能将不同训练数值策略混称为同一实现；结果记录应带代码身份和model contract。
