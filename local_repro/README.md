# 自建方法代码（local_repro/）

服务器上的原始位置：`/home/chuwenjing/TTA_LLM/TLM/local_repro/`。本目录是逐文件拷贝，内容与服务器一致，可用于复核与重新部署。

## 快照规则（重要）

本目录是**只读快照**，不是开发分支：

- 代码只在服务器上改，改完重新拷过来并更新下面的哈希表；
- **不要直接在本目录改代码**——服务器上跑的不是这里的文件，改了也不会生效，只会让两边悄悄分叉；
- 判断是否过期：把服务器上的文件哈希和下面表格比一下，任何一行不同就说明快照过期了。

```bash
# 服务器侧一行校验（输出应与本 README 的哈希表逐行一致）
cd /home/chuwenjing/TTA_LLM/TLM/local_repro && sha256sum eata_text/*.py tent_paper/*.py tent_paper/tests/*.py sar_deyo/*.py
```

保留快照的理由：服务器上的源码不是 git 仓库（ZIP 解压），若仓库里也不存，自建方法代码就**没有任何版本历史**，一旦被误删或被他人改动就不可恢复；且每个 `run_manifest.json` 里的 `code_id` 是代码哈希，没有代码就无法核对结果对应哪个版本。

## 目录

```
local_repro/
├── tent_paper/        Tent 实现 + 统一运行入口（EATA/SAR/DeYO 也依赖它）
│   ├── run.py         主入口：加载/编码/source/适应/评测/manifest
│   ├── entropy.py     FP32 全词表熵与掩码
│   ├── norm_affine.py RMSNorm gamma 适配面
│   ├── build_groups.py  Dolly 输出分层抽样
│   ├── diag_nan.py / diag_steps.py  诊断脚本
│   └── tests/test_cpu.py  CPU 单元测试
├── eata_text/         EATA 实现
│   ├── eata_alg.py    式(3) 熵筛选、式(5) 冗余筛选、Fisher 惩罚、损失
│   ├── run_eata.py    gamma 适配面 + 适配循环本体
│   ├── run_eata_lora.py LoRA 适配面，复用 run_eata.adapt_eata
│   └── resume_tent.py 断点续跑工具
└── sar_deyo/          SAR（ICLR 2023）与 DeYO（ICLR 2024）
    ├── sam.py         SAM：epsilon = rho*g/||g||_2，全局归一化，无梯度合并
    ├── deyo_alg.py    PLPD：块重排视图 + 逐位置伪标签概率
    └── run_tta.py     统一入口 --method sar|deyo --surface gamma|lora
```

`run_eata*.py` 通过 `import tent_run` 使用 `tent_paper/run.py` 的 encode / source / 评分 / manifest，**部署时两个目录必须放在同一父目录下**，且父目录要能 `import tent_paper.run`（服务器上是把 `local_repro` 加入 `sys.path`）。

## 文件哈希（sha256 前 20 位，与服务器一致）

| 文件 | 字节 | sha256 |
|---|---:|---|
| `eata_text/eata_alg.py` | 8389 | `005c861b3f290948be30` |
| `eata_text/resume_tent.py` | 1371 | `f30e0e4489710100a8f2` |
| `eata_text/run_eata.py` | 12631 | `7b4d6e109e23d9cd3c50` |
| `eata_text/run_eata_lora.py` | 8307 | `fe40a44b1f29f2ad210e` |
| `sar_deyo/deyo_alg.py` | 5210 | `3b7f2060ff5d672cdfb4` |
| `sar_deyo/run_tta.py` | 24155 | `8670b0c0a428bcad2df8` |
| `sar_deyo/sam.py` | 3241 | `56118ea2a1c75a8b5e49` |
| `tent_paper/build_groups.py` | 8248 | `a70dedfa4baddba3fa36` |
| `tent_paper/diag_nan.py` | 3238 | `fde00f2fc320cf4d8206` |
| `tent_paper/diag_steps.py` | 3274 | `8e5d9bd566fb910077b9` |
| `tent_paper/entropy.py` | 4578 | `89e463d414ddd07cb3e8` |
| `tent_paper/norm_affine.py` | 4017 | `9d0717f14267ec1af8bf` |
| `tent_paper/run.py` | 29705 | `1f6f7361c1cbaa02b8ae` |
| `tent_paper/tests/test_cpu.py` | 6894 | `469d7107450c4b4674ed` |

每个 run 的 `run_manifest.json` 里还有一个 `code_id`，是目录内所有 `*.py` 的 sha256 串联哈希，用于把结果和代码版本对上。

## 运行方式

```bash
source /home/chuwenjing/TTA_LLM/TLM/local_repro/stage1/activate_tlm.sh
cd /home/chuwenjing/TTA_LLM/TLM/local_repro

# Tent
python tent_paper/run.py formal --model llama3b --dataset dolly_5k --gpu 0

# EATA（LoRA 适配面）
python eata_text/run_eata_lora.py formal --model llama3b --dataset dolly_5k --gpu 1 --d-margin 0

# EATA（gamma 适配面）
python eata_text/run_eata.py formal --model llama3b --dataset dolly_5k --gpu 2

# DeYO / SAR（gamma 适配面，与 Tent 同条件）
python sar_deyo/run_tta.py formal --method deyo --surface gamma --model llama3b \
    --dataset dolly_5k --gpu 3 --batch 2 --lr 1.5625e-5 --sched const
python sar_deyo/run_tta.py formal --method sar --surface gamma --model llama3b \
    --dataset dolly_5k --gpu 4 --batch 2 --lr 1.5625e-5 --sched const
```

`--d-margin 0` 表示**关闭**式(5)；不要用 `1.0` 代替，理由见 [实验记录 §4](../docs/实验记录.md)。

`--sched` 默认 `cosine`（线性 warmup 10% + 余弦衰减到 0），与官方 `examples/train_lora/offline_ttl.yaml` 一致。要复现 2026-11-14 之前的旧结果（常数 lr），加 `--sched const`。评测结束后 manifest 里的 `quality.status` 会给出 `OK` / `COLLAPSED`；`COLLAPSED` 时顶层 `status` 也是 `COLLAPSED`，不再写 `PASS`。

`run_tta.py` 的阈值：`--tau-ent-ratio`/`--ent0-ratio` 乘 `lnV` 是论文形式，但 `0.4·lnV ≈ 4.7` 远高于本任务实测的逐 token 熵（0.16~1.1 nats），照用会让 `α = exp(Ent0−Ent) ≈ 77`、`τ_PLPD = 0.2` 筛掉约 87% 样本。因此 `--calib auto`（默认）在 source 模型上取 100 条输入的熵与 PLPD 中位数替换 `Ent0` 与 `τ_PLPD`，使 `α` 回到 1 附近；`--calib none` 恢复论文常数。标定值写进 manifest 的 `calibration` 段。

## 与上游的关系

`tent_paper/run.py` 复用了 TLM 提交 `496c9ed9` 的模型加载器、提示模板与评测函数，但不修改上游 `src/llamafactory/` 任何文件。上游代码未纳入本仓库，按 `docs/复现协议.md` §1 的固定提交自行拉取。
