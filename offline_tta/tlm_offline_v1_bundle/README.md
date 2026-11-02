# TLM：六组并行运行

这是2026-10-25新写的TLM包。旧Tent已经完成，结果保留在原目录。本包只创建新的代码快照、输入副本和结果。

目前完成了本地CPU测试；**还没有在服务器真实3B模型上运行本包**。下面的启动程序会先做真实预检，失败的任务不会继续全量训练。

## 1. 本机上传

在Windows PowerShell执行：

```powershell
scp -r "D:\大创\进度\offline_tta\tlm_offline_v1_bundle" chuwenjing@10.126.62.55:/data/chuwenjing/tta_offline/
```

首次上传使用该目录。已有任务运行时不要直接改安装快照；启动程序会把代码复制到带哈希的新目录，以免之后重新上传影响运行。

## 2. 服务器检查

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /data/chuwenjing/tta_offline/tlm_offline_v1_bundle
python -B check.py
nvidia-smi
```

`check.py`核验文件SHA256并运行CPU测试，不下载、不安装包、不占GPU。要看到测试结尾的`OK`，再启动。仅使用现有项目环境，不用base环境。

2026-10-25兼容修订：测试使用`seed=42`，不再传Accelerate 1.0.1不支持的独立`data_seed`；旧libc缺少`renameat2`导出时直接调用Linux内核接口，仍拒绝覆盖已有目录。本机使用Accelerate 1.0.1通过45项CPU检查，Linux实际接口由服务器上的`check.py`验证。遇到旧包的这两类报错，重新上传整个目录后再检查，无需升级服务器依赖。

## 3. 六组一起跑

确认GPU0–5空闲后执行下面整段。若卡号不同，把六个数字换成实际空闲卡号。

```bash
mkdir -p /data/chuwenjing/results/tlm_offline_v1
TLM_RUN=/data/chuwenjing/results/tlm_offline_v1/run_$(date +%Y%m%d_%H%M%S)
nohup python -B -u launch.py --gpus 0,1,2,3,4,5 --mode all --output "$TLM_RUN" > "${TLM_RUN}.log" 2>&1 < /dev/null &
echo "结果目录：$TLM_RUN"
echo "启动进程：$!"
```

六个任务分别为Llama3B和Qwen3B，各自的GeoSignal、Dolly、GSM8K。每个任务独占一张卡，按下面顺序自动运行：

1. 检查模型与数据文件、输入、最长输入训练、512-token生成，以及真实FP16 Trainer保存/重载后的续算。
2. 跑16条，检查训练、生成、评分能否完整完成。
3. 从原模型重新开始，跑5000条适应与最终评测。

每组通过自己的前一步后才继续；一组失败会停止该组，其余组继续。不会把16条结果接到全量训练上。GPU6、7可留给其他工作；六组任务不需要额外占卡。

若只想先跑预检，把`--mode all`改为`--mode preflight`并使用另一个新结果目录。以后再启动all时，也使用新目录；有效输入副本会复用，但本版本会重新预检。

## 4. 看进度

在启动用的同一个终端：

```bash
tail -n 40 "${TLM_RUN}.log"
cat "$TLM_RUN"/*.status.json
nvidia-smi
```

另开终端后，先把`TLM_RUN`设为实际输出目录。每个任务有三个日志，例如：

```bash
tail -n 50 "$TLM_RUN/llama3b__geosignal_5k.preflight.log"
tail -n 50 "$TLM_RUN/llama3b__geosignal_5k.pilot.log"
tail -n 50 "$TLM_RUN/llama3b__geosignal_5k.formal.log"
```

后一个阶段没开始时，对应文件尚不存在。完整结果在`$TLM_RUN/formal/模型__数据集/`，查看`run_manifest.json`和`metrics.json`。失败时保留日志和`failure.json`，不要直接重开同一输出目录。

## 方法与边界

- 固定TLM提交：496c9ed97d0fb8bc4854ed2479259e7e9e70a790。
- 使用输入文本的平均next-token交叉熵；关闭adapter的基座评分大于3才选中，权重0.1×exp(base_ce−3)。未选中时保留官方零损失的优化器行为。
- q/v LoRA，rank8、alpha16、dropout0，PEFT默认初始化。原模型FP16、adapter与loss为FP32。官方BF16示例在这里明确改为FP16。
- AdamW、cosine、10%warmup、一轮、batch1。GeoSignal/Dolly峰值学习率5e-5，GSM8K为1e-6。
- 输入4096，评测最多生成512；没有Tent的80-token适应生成阶段。
- 每组原始5000条，训练顺序由固定seed42的Trainer采样决定，评测按原索引输出。
- 输入处理独立于参考答案，去掉了固定官方处理器中答案长度影响截断的路径；训练只读清洗后的inputs.jsonl，评分时才读references.jsonl。
- Source仅在原预测完整、全部输入身份一致且预检生成样例相同后复用；不满足时停止并提示，需要另外准备同口径Source。
- 分块覆盖整段输入；最长输入必须通过真实预检。当前实现为单卡，不自动换量化、缩短输入或跨卡。
- 每25步保存完整Trainer检查点，额外保留里程碑adapter；检查点包含显式GradScaler状态及文件摘要，恢复前核验。

## 中断后的单组恢复

保留整个结果目录。先找该组`trainer/checkpoint-N/commit.json`完整且校验通过的检查点，再使用**启动日志中的代码快照**，显式指定`run.py formal --resume /完整/checkpoint-N`以及原来的model、dataset、inputs、output、receipt和GPU。不要在运行中的任务上执行第二次resume。

例如，原始参数和路径都能在`launch.json`、状态文件和该组run_manifest.json中找到。`run.py`不接受布尔形式的“自动恢复最新”，避免误读未写完的检查点。评测阶段中断时也保留原输出和显式完整检查点，加载最终adapter继续未完成的预测。

## 本地复查

本机已有兼容库时：

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& .\.venv\Scripts\python.exe -B offline_tta/tlm_offline_v1_bundle/check.py --libs .tools/tent-compat
```

该环境变量只用于本机较新PyTorch读取本测试新生成的可信Trainer检查点；服务器固定PyTorch2.5.1无需设置。
