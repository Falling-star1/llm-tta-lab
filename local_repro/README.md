# 自建方法代码（local_repro/）

服务器上的原始位置：`/home/chuwenjing/TTA_LLM/TLM/local_repro/`。本目录是逐文件拷贝，内容与服务器一致，可用于复核与重新部署。

## 快照规则（重要）

本目录是**只读快照**，不是开发分支：

- 代码只在服务器上改，改完重新拷过来并更新下面的哈希表；
- **不要直接在本目录改代码**——服务器上跑的不是这里的文件，改了也不会生效，只会让两边悄悄分叉；
- 判断是否过期：把服务器上的文件哈希和下面表格比一下，任何一行不同就说明快照过期了。

```bash
# 服务器侧一行校验（输出应与本 README 的哈希表逐行一致）
cd /home/chuwenjing/TTA_LLM/TLM/local_repro && sha256sum eata_text/*.py tent_paper/*.py tent_paper/tests/*.py
```

保留快照的理由：服务器上的源码不是 git 仓库（ZIP 解压），若仓库里也不存，自建方法代码就**没有任何版本历史**，一旦被误删或被他人改动就不可恢复；且每个 `run_manifest.json` 里的 `code_id` 是代码哈希，没有代码就无法核对结果对应哪个版本。

## 目录

```
local_repro/
├── tent_paper/        Tent 实现 + 统一运行入口（EATA 也依赖它）
│   ├── run.py         582 行，主入口：加载/编码/source/适应/评测/manifest
│   ├── entropy.py     FP32 全词表熵与掩码
│   ├── norm_affine.py RMSNorm gamma 适配面
│   ├── build_groups.py  Dolly 输出分层抽样
│   ├── diag_nan.py / diag_steps.py  诊断脚本
│   └── tests/test_cpu.py  CPU 单元测试
└── eata_text/         EATA 实现
    ├── eata_alg.py    式(3) 熵筛选、式(5) 冗余筛选、Fisher 惩罚、损失
    ├── run_eata.py    gamma 适配面 + 适配循环本体
    ├── run_eata_lora.py LoRA 适配面，复用 run_eata.adapt_eata
    └── resume_tent.py 断点续跑工具
```

`run_eata*.py` 通过 `import tent_run` 使用 `tent_paper/run.py` 的 encode / source / 评分 / manifest，**部署时两个目录必须放在同一父目录下**，且父目录要能 `import tent_paper.run`（服务器上是把 `local_repro` 加入 `sys.path`）。

## 文件哈希（sha256 前 16 位，与服务器一致）

| 文件 | 字节 | sha256 |
|---|---:|---|
| `eata_text/eata_alg.py` | 8389 | `005c861b3f290948be30` |
| `eata_text/resume_tent.py` | 1371 | `f30e0e4489710100a8f2` |
| `eata_text/run_eata.py` | 12088 | `e125a7ff0ed59505844f` |
| `eata_text/run_eata_lora.py` | 7963 | `069df65695470ca06f19` |
| `tent_paper/build_groups.py` | 8248 | `a70dedfa4baddba3fa36` |
| `tent_paper/diag_nan.py` | 3238 | `fde00f2fc320cf4d8206` |
| `tent_paper/diag_steps.py` | 3274 | `8e5d9bd566fb910077b9` |
| `tent_paper/entropy.py` | 4578 | `89e463d414ddd07cb3e8` |
| `tent_paper/norm_affine.py` | 4017 | `9d0717f14267ec1af8bf` |
| `tent_paper/run.py` | 26464 | `b77447d70b5fb6df1cfe` |
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
```

`--d-margin 0` 表示**关闭**式(5)；不要用 `1.0` 代替，理由见 [实验记录 §4](../docs/实验记录.md)。

## 与上游的关系

`tent_paper/run.py` 复用了 TLM 提交 `496c9ed9` 的模型加载器、提示模板与评测函数，但不修改上游 `src/llamafactory/` 任何文件。上游代码未纳入本仓库，按 `docs/复现协议.md` §1 的固定提交自行拉取。
