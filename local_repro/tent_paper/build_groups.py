"""P0: build Good/Hard strata from frozen Source predictions on Dolly (llama3b).

Rules are pre-registered here and applied before any adapted-model evaluation:
  has_substance = not (empty | echo-of-prompt | repetition-loop | title-shell)
  correct       = rougeLsum >= THRESH   (THRESH from a fixed ladder, see pick_threshold)
Groups:
  A  = has_substance and correct          (model can answer)
  C  = not has_substance                  (model cannot produce content)
  D  = has_substance and not correct      (content but wrong)
  X  = excluded (hit_generation_limit and nothing else classified)
Splits (disjoint, seed fixed):
  A_train (500), C_train (500), R_rand (500, sampled from A_train ∪ C_train),
  E_eval (1000, sampled from neither A_train nor C_train)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[a-z0-9']+")
SENT_END = (".", "!", "?", ".\"", "!\"", "?\"")


def norm_words(text: str) -> list:
    return WORD.findall((text or "").lower())


def is_empty(text: str) -> bool:
    return len(norm_words(text)) == 0


def is_echo(pred: str, prompt: str, jaccard_max: float = 0.9) -> bool:
    """Output is (almost) the prompt itself, or a contiguous slice of it."""
    p = " ".join(norm_words(pred))
    q = " ".join(norm_words(prompt))
    if not p:
        return False
    if p and p in q:
        return True
    ps, qs = set(p.split()), set(q.split())
    if not ps or not qs:
        return False
    return len(ps & qs) / len(ps | qs) >= jaccard_max


def is_repetition(text: str, ngram: int = 4, repeat: int = 4, run: int = 8) -> bool:
    """Invalid loop: same n-gram too often, or one token repeated many times in a row."""
    w = norm_words(text)
    if len(w) >= run:
        cur = best = 1
        for i in range(1, len(w)):
            cur = cur + 1 if w[i] == w[i - 1] else 1
            best = max(best, cur)
        if best >= run:
            return True
    if len(w) >= ngram * repeat:
        grams = Counter(tuple(w[i:i + ngram]) for i in range(len(w) - ngram + 1))
        if grams and grams.most_common(1)[0][1] >= repeat:
            return True
    return False


def is_title_shell(text: str) -> bool:
    """Short noun-phrase-like output with no sentence ending (heuristic, flagged for review)."""
    w = norm_words(text)
    t = (text or "").strip()
    return 0 < len(w) <= 4 and not t.endswith(SENT_END)


def classify(rec: dict) -> str:
    pred, prompt = rec.get("predict") or "", rec.get("prompt") or ""
    if is_empty(pred):
        return "empty"
    if is_repetition(pred):
        return "repetition"
    if is_echo(pred, prompt):
        return "echo"
    if is_title_shell(pred):
        return "title_shell"
    return "substantive"


def pick_threshold(scores: list, need: int) -> tuple:
    """Fixed ladder, smallest threshold that still yields >= need correct samples."""
    ladder = [0.30, 0.25, 0.20, 0.15, 0.10]
    for t in ladder:
        if sum(1 for s in scores if s >= t) >= need:
            return t, ladder
    return ladder[-1], ladder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-a", type=int, default=500)
    ap.add_argument("--n-c", type=int, default=500)
    ap.add_argument("--n-e", type=int, default=1000)
    ap.add_argument("--n-r", type=int, default=500)
    args = ap.parse_args()

    src = Path(args.source)
    raw = [json.loads(l) for l in open(src, encoding="utf-8")]
    print(f"source records: {len(raw)}  sha256={hashlib.sha256(src.read_bytes()).hexdigest()[:16]}")

    kinds = Counter()
    rows = []
    for r in raw:
        k = classify(r)
        kinds[k] += 1
        rows.append({"source_index": r["source_index"], "kind": k,
                     "score": r["score"], "gen_tokens": r.get("generated_tokens"),
                     "hit_limit": bool(r.get("hit_generation_limit"))})
    print("classification:", dict(kinds))

    scores = [r["score"] for r in rows if r["kind"] == "substantive"]
    scores_sorted = sorted(scores)
    deciles = [round(scores_sorted[int(len(scores_sorted) * q)], 4) for q in
               (0.1, 0.25, 0.5, 0.75, 0.9)]
    print(f"substantive n={len(scores)}  deciles(p10,p25,p50,p75,p90)={deciles}  mean={statistics.mean(scores):.4f}")
    thresh, ladder = pick_threshold(scores, args.n_a)
    print(f"threshold ladder={ladder} -> chosen THRESH={thresh}")

    for r in rows:
        if r["kind"] == "substantive":
            r["group"] = "A" if r["score"] >= thresh else "D"
        else:
            r["group"] = "C"
    counts = Counter(r["group"] for r in rows)
    print("groups A/C/D:", dict(counts))

    rng = random.Random(args.seed)
    pool_a = [r["source_index"] for r in rows if r["group"] == "A"]
    pool_c = [r["source_index"] for r in rows if r["group"] == "C"]
    if len(pool_c) < args.n_c:
        n_both = min(len(pool_a), len(pool_c))
        print(f"[warn] C only has {len(pool_c)}; shrinking both groups to {n_both} (no reuse)")
        n_a = n_c = n_both
    else:
        n_a, n_c = args.n_a, args.n_c
    a_train = rng.sample(pool_a, n_a)
    c_train = rng.sample(pool_c, n_c)
    r_rand = rng.sample(a_train + c_train, min(args.n_r, len(a_train) + len(c_train)))
    used = set(a_train) | set(c_train)
    rest = [r["source_index"] for r in rows if r["source_index"] not in used]
    e_eval = rng.sample(rest, min(args.n_e, len(rest)))

    splits = {
        "seed": args.seed, "threshold": thresh, "threshold_ladder": ladder,
        "classify_rules": {
            "empty": "no alphanumeric tokens",
            "repetition": "same 4-gram >=4 times or one token repeated >=8 in a row",
            "echo": "output is a contiguous slice of prompt, or token Jaccard vs prompt >= 0.9",
            "title_shell": "<=4 words and no sentence-ending punctuation (heuristic, needs review)",
        },
        "counts": {"A_pool": len(pool_a), "C_pool": len(pool_c), "D_pool": counts.get("D", 0)},
        "A_train": sorted(a_train), "C_train": sorted(c_train),
        "E_eval": sorted(e_eval), "R_rand": sorted(r_rand),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "source_size": len(raw),
    }
    assert not (set(a_train) & set(c_train)), "A/C overlap"
    assert not (set(e_eval) & used), "E overlaps train"
    assert len(set(e_eval)) == len(e_eval)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "groups.json").write_text(json.dumps(
        {"per_sample": rows, "counts": dict(counts), "threshold": thresh}, ensure_ascii=False, indent=1))
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2))

    summary = f"""# Dolly 样本分层（P0 产物）

来源：`{src}`
sha256：`{splits['source_sha256']}`
记录数：{len(raw)}

## 分类结果

| 类别 | 条数 |
|---|---:|
| substantive（有实质内容） | {len(scores)} |
| empty | {kinds.get('empty',0)} |
| repetition | {kinds.get('repetition',0)} |
| echo（复述 prompt） | {kinds.get('echo',0)} |
| title_shell（疑似纯标题） | {kinds.get('title_shell',0)} |

substantive 的 ROUGE-Lsum 分位（p10/p25/p50/p75/p90）= {deciles}，均值 {statistics.mean(scores):.4f}
正确性阈值：**{thresh}**（固定梯队 {ladder}，取"仍能满足 A≥{args.n_a} 条"的最小值）

## 分组

| 组 | 定义 | 池子大小 | 实际抽样 |
|---|---|---:|---:|
| A | 有实质内容 且 score ≥ {thresh} | {len(pool_a)} | {len(a_train)} |
| C | 无实质内容（empty/rep/echo/title） | {len(pool_c)} | {len(c_train)} |
| D | 有实质内容 但 score < {thresh} | {counts.get('D',0)} | 备用，未抽 |
| E | held-out 评测集（与 A/C 不相交） | {len(rest)} | {len(e_eval)} |
| R | 随机对照（从 A∪C 中不分类型抽） | {len(a_train)+len(c_train)} | {len(r_rand)} |

种子：{args.seed}
"""
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    print("written:", out)


if __name__ == "__main__":
    main()
