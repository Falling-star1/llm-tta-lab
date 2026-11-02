# TLM v3候选修订：Gaussian、AMP与输出质量检查

本包修正与官方混合精度训练路径的差异，默认使用论文Gaussian初始化，并在正式评测中增加质量预警。旧v1/v2和已有结果不改；两组Dolly仍暂停，其他四组按原要求继续，作为旧数值路径的工程对照。

**本地60项CPU检查通过；v3尚未通过服务器真实FP16预检，更没有证明能修复Dolly退化。先做下面的诊断，不直接重跑六组全量。**

## 改动

1. `init_lora_weights='gaussian'`为默认值，A为Gaussian、B为零。`--lora-init kaiming`保留旧默认初始化作为显式对照；选择写入运行身份和模型记录，预检收据不能混用。
2. 分块计算直接调用decoder，绕开Accelerate包装的外层forward。v3对基座评分和当前损失显式进入`accelerator.autocast()`。预检身份同时绑定seed和初始化；数值检查使用跨多个投影块的输入。LoRA参数和CE归约仍为FP32，不代表所有矩阵运算都使用FP32。
3. 真实预检增加整段前向与分块路径的损失和未缩放梯度比较，并确认双方均进入autocast。逐参数容差为atol=1e-5、rtol=1e-2，损失atol=1e-5、rtol=1e-4，非零梯度的逐参数相对L2差异不超过1%。检查临时设置非零B来覆盖A/B两种梯度，随后恢复原始权重，不污染训练历史和RNG；GradScaler另由16步保存恢复测试验证。
4. 评测每100条检查最近100条，与同索引Source比较空答和高重复数量。任一比例高出Source超过5个百分点时，停止该组，保留预测和权重，写入`quality_alert.json`。另检查跨问题回答塌缩：同一非空回答占窗口至少50%，且其频率比同索引Source高出超过10个百分点，也停止该组。这些是工程预警，不是论文阈值，不删异常样本、不替换最终模型。高重复使用双方解码文本的四元组重复比例≥0.5，至少32个词/标点token。

CPU使用BF16验证AMP包装机制，覆盖LoRA B非零后的梯度。它不能代替2080 Ti上的FP16数值验证。Gaussian本身也不能保证不退化，精度、Qwen3B规模和输入预处理与原文的差异仍存在。

## 上传与CPU检查

本机PowerShell：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_offline_v3_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

服务器：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v3_bundle
python -B check.py
nvidia-smi
```

看到`Ran 60 tests`和`OK`再执行真实预检。先确认GPU6、7空闲，避免碰到现有四组或暂停的Dolly进程。

## 两组Dolly只做真实预检

```bash
mkdir -p /data/chuwenjing/results/tlm_v3_diagnosis
TLM_FIX_ROOT=/data/chuwenjing/results/tlm_v3_diagnosis/run_$(date +%Y%m%d_%H%M%S)
mkdir "$TLM_FIX_ROOT"

CUDA_VISIBLE_DEVICES=6 nohup python -B -u run.py preflight \
  --model llama3b --dataset dolly_5k --lora-init gaussian \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_FIX_ROOT/llama_gaussian" \
  > "$TLM_FIX_ROOT/llama_gaussian.log" 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=7 nohup python -B -u run.py preflight \
  --model qwen3b --dataset dolly_5k --lora-init gaussian \
  --inputs /data/chuwenjing/tta_offline/tlm_inputs_v1/dolly_5k \
  --output "$TLM_FIX_ROOT/qwen_gaussian" \
  > "$TLM_FIX_ROOT/qwen_gaussian.log" 2>&1 < /dev/null &

echo "$TLM_FIX_ROOT"
```

查看：

```bash
tail -n 40 "$TLM_FIX_ROOT/llama_gaussian.log"
tail -n 40 "$TLM_FIX_ROOT/qwen_gaussian.log"
cat "$TLM_FIX_ROOT"/*/receipt.json
```

预期收据含`precision_path_check.status=PASS`、恢复一致、真实参数更新和Source样例一致。任一失败先看错误，不放宽容差或删除检查来追求PASS。这些命令只做预检，不启动5000条正式训练。

## 定位退化的顺序

先确认数值路径，再在相同配置和访问顺序下分开比较：旧v2的Kaiming路径、v3修正AMP后的Kaiming、v3修正AMP后的Gaussian。前两者检查数值路径差异，后两者检查初始化差异，避免同时改多个因素后直接归因。

历史Dolly的原模型、100/500/1000/2500步LoRA和最终LoRA还需在固定输入上做诊断，查看空答/循环从哪个阶段出现；不按测试答案挑选最好检查点覆盖本轮结果。16或100条小试跑只查工程问题，不能证明5000条适应后仍正常，所以正式模型的100条输出质量检查仍必须保留。

v3不调整正式阈值3、lambda=0.1、学习率、输入长度、生成长度、停止符或重复惩罚；这些设置若需改变，单列为新实验。发生质量预警时，原final adapter和已有预测保留，重新resume也会重新检查已保存窗口，不自动绕过预警。
