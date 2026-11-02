# GSM8K 修正版200题配对验证

启动：2026-10-27 20:53:35（北京时间）；状态快照：2026-10-27 20:56:12。用户已授权现在运行较大固定题集验证。

本轮复用已有2500步权重，只做生成与评分，不重训六组。原文复现主实验、TENT/Source和此前诊断结果保持原样。四组使用相同输入和512 token贪心生成，比较原2500步与三种目标屏蔽控制。

| 组别 | GPU | 已生成 | 状态 |
|---|---:|---:|---|
| 原目标全部保留 | 0 | 2/200 | EVALUATING |
| 屏蔽推理开场白 | 1 | 2/200 | EVALUATING |
| 屏蔽随后用户EOT | 2 | 2/200 | EVALUATING |
| 同时屏蔽两者 | 3 | 2/200 | EVALUATING |

## 选题与判据

固定seed=20261027，从原5000题排除此前100条探针和16条故障回放后，盲抽200题。选题不读取答案或成绩，四组完全配对；它们仍来自原适应集合，不是新的独立测试集，也不能将200题分数称为全量5000题分数。

收齐后同时检查官方EM、空字符串、仅输出开场白、生成上限及相对原2500步的新增正确/新增错误。`stub_removed`只表示空壳形式消失，不能据此断言恢复正确作答。Source和原5000步的同题分数为补充比较，主要基线为同训练步数的原2500步。不根据前缀分数调参，不自动选择最高分组。

## 与原文复现的关系

这轮验证不会改变原文复现运行和分数。三组目标屏蔽改变了原方法的训练目标，属于消融或修正版验证；即使提高分数，也必须单独报告，不能替代原TLM的论文复现成绩。这里尚未训练或评测修正版的5000步全量模型。

## 核验与运行位置

10项CPU测试本地/服务器均通过，独立复核GO；已注册37项原资料和四份权重的SHA256。每组在200题生成前，须精确重放原5000步2题与自身2500步旧2题。输出目录保护在创建前检查，GPU锁防止占用其他任务；启动回执锁防止重复派发，部分派发必须先检查，不盲目重跑。

- 运行目录：`/data/chuwenjing/results/tlm_gsm_paired_validation/run_20261027_125214_86cafa`
- 冻结入口：`/data/chuwenjing/tta_offline/tlm_gsm_paired_validation_8e828e3d6d44`
- 注册SHA256：`0460a809b56aa8e43620442e165306a2311577883e846a1d35f5ab0d4ae7a2ae`
- [固定选题与资料注册](tlm-gsm-paired-validation-registration.json)
- [启动和进度证据](tlm-gsm-paired-validation-launch-evidence.json)
- [独立推理代码](../../../offline_tta/tlm_gsm_paired_validation_v1/README.md)
- [先前原因与故障回放](TLM_Dolly与GSM8K排查进展.md)

服务器查看进度，直接读取本轮的新目录：

```bash
python - <<'PY'
import json
from pathlib import Path
root=Path('/data/chuwenjing/results/tlm_gsm_paired_validation/run_20261027_125214_86cafa')
for mode in ['original_step2500','hint_masked','eot_masked','both_masked']:
    folder=root/mode
    result={'mode':mode}
    for name in ['status.json','failure.json','summary.json']:
        if (folder/name).exists():result[name]=json.loads((folder/name).read_text())
    print(json.dumps(result,ensure_ascii=False))
PY
```
