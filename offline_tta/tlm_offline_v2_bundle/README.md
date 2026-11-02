# TLM v2：修正FP16预检，单独重跑失败任务

更新：2026-10-25。v1中Llama＋GSM8K已完成5000条适应，正在评测。其余五组停在4步FP16恢复预检。v2保留正式训练、生成和评分设置，修正预检并增加跳过指定任务的入口。

## 问题与改动

v1预检把筛选阈值从3改成0来强制选中样本，却同时把指数权重放大了exp(3)，约20倍。默认FP16梯度缩放从65536开始，前几步出现非有限梯度后会跳过优化器更新。4步预算太短，五组都没有产生参数变化；其中一组仅执行了一次学习率为0的优化器步骤。

v2的处理：

- 强制选中样本时同步调整lambda，权重仍为0.1×exp(base_ce−3)。该调整仅用于工程预检。
- 真实Trainer恢复测试从4步延长到16步，第8步中断并恢复，让动态梯度缩放有时间调整。
- 仍要求参数真正变化、至少两次成功优化器步骤、权重和优化器/调度器/RNG/GradScaler恢复后一致。预检收据记录成功与跳过步数。
- 新增`--skip`，跳过指定任务并保留其他任务原来的物理卡映射。

正式训练仍使用threshold=3、lambda=0.1、一轮5000条。v1原包和运行中的快照保持原样。

本地使用Accelerate 1.0.1通过50项CPU检查，包含真实CPU FP16梯度缩放回退用例。**v2尚未通过服务器真实模型预检**，启动后逐组验证，失败仍会停止该组。

## 上传

本机PowerShell：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_offline_v2_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

## 检查并重跑失败的五组

服务器上执行：

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v2_bundle
python -B check.py
nvidia-smi
```

看到`Ran 50 tests`和`OK`，确认GPU 0、1、3、4、5空闲后，再执行以下整段。GPU 2上的原任务继续评测。

```bash
mkdir -p /data/chuwenjing/results/tlm_offline_v2
TLM_RUN_V2=/data/chuwenjing/results/tlm_offline_v2/run_$(date +%Y%m%d_%H%M%S)
nohup python -B -u launch.py \
  --gpus 0,1,2,3,4,5 \
  --skip llama3b__gsm8k_5k \
  --mode all --output "$TLM_RUN_V2" \
  > "${TLM_RUN_V2}.log" 2>&1 < /dev/null &
echo "结果目录：$TLM_RUN_V2"
echo "启动进程：$!"
```

跳过任务发生在创建子进程之前，Llama＋GSM8K不会重跑或申请GPU 2。其他五组各自重新进行预检、16条试跑、5000条适应和5000条评测。有效输入副本复用，不复用失败的预检收据。

全新六组实验才省略`--skip`，并确认六张卡均空闲。也可使用`--mode preflight`只跑预检；后续all使用另一个新结果目录。

## 查看进度

```bash
tail -n 40 "${TLM_RUN_V2}.log"
cat "$TLM_RUN_V2"/*.status.json
tail -n 50 "$TLM_RUN_V2/qwen3b__gsm8k_5k.preflight.log"
```

另开终端需要重新设置`TLM_RUN_V2`。结果在`$TLM_RUN_V2/formal/模型__数据集/`。最终汇总包含原v1的Llama＋GSM8K和v2的其他五组，分别登记代码身份及运行目录。

“全量”包括适应与评测两个阶段：先学习5000条问题，再让最终模型回答这5000条问题；训练不读参考答案，评分时才读取。因此评测进度显示5000条也是正式全量实验的一部分。

中断后使用原安装快照、原参数和显式完整检查点恢复，不在仍运行的任务上执行第二次恢复。检查点包含adapter、优化器、调度器、RNG、GradScaler及文件哈希，见`train_core.verify_checkpoint`。本包只适用于团队现有服务器部署路径。
