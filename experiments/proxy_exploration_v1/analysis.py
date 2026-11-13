"""Frozen-source six-cell analysis. No model training or label-based selection."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(path, records=False):
    data = read(path) if Path(path).suffix == ".json" else [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if records:
        data = data["records"]
    result = {row["source_index"]:row for row in data}
    if len(result) != len(data) or set(result) != set(range(5000)):
        raise ValueError("Incomplete or duplicate indices: "+str(path))
    return result


def quartiles(ids, ce):
    if len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("Expected 256 unique IDs")
    if any(not math.isfinite(ce[i]) or ce[i] < 0 for i in ids):
        raise ValueError("CE must be finite and nonnegative")
    ordered = sorted(ids, key=lambda i:(ce[i], i))
    return [ordered[k:k+64] for k in range(0,256,64)]


def validate_split(split):
    ids = sum(split.values(), [])
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping split")
    if len(split["history"]) != 256:
        raise ValueError("History must contain 256 records")


def score_mean(scores):
    if not scores or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError("Scores must be finite probabilities")
    return statistics.mean(scores)*100


def summarize(config, output):
    from common import validate_config
    validate_config(config)
    summaries = []
    for cell in config["cells"]:
        name = cell["name"]
        directory = output/name
        directory.mkdir(parents=True, exist_ok=True)
        split = config["splits"][cell["dataset"]]
        validate_split(split)
        source = indexed(cell["source"])
        adaptation = indexed(cell["adaptation"])
        tokens = indexed(cell["tokens"], records=True)
        inputs = indexed(cell["inputs"])
        manifest = read(cell["training_manifest"])["config"]
        if manifest["model"] != cell["model"] or manifest["dataset"] != cell["dataset"]:
            raise ValueError("Cached model identity mismatch")
        for i in source:
            required={"metric","original_prompt_sha256","original_prompt_tokens","used_prompt_tokens",
                      "input_truncated","score","predict","hit_generation_limit"}
            if not required.issubset(source[i]):
                raise ValueError("Incomplete frozen Source schema in "+name+" at "+str(i))
            if source[i]["metric"] != cell["metric"]:
                raise ValueError("Wrong metric in "+name)
            score_mean([source[i]["score"]])
            if cell["metric"]=="exact_match" and source[i]["score"] not in (0,1):
                raise ValueError("Non-binary exact-match score")
            for key in ("original_prompt_sha256","original_prompt_tokens","used_prompt_tokens","input_truncated"):
                if source[i][key] != tokens[i][key]:
                    raise ValueError("Cached token identity mismatch")
            if not math.isfinite(adaptation[i]["base_ce"]) or adaptation[i]["base_ce"] < 0:
                raise ValueError("Invalid CE")
        ce = {i:r["base_ce"] for i,r in adaptation.items()}
        groups = quartiles(split["history"], ce)
        rows, descriptions = [], []
        for q, ids in enumerate(groups, 1):
            scores = [source[i]["score"] for i in ids]
            item = dict(quartile=q, n=64, score=score_mean(scores),
                        ce_min=min(ce[i] for i in ids), ce_max=max(ce[i] for i in ids),
                        mean_tokens=statistics.mean(tokens[i]["used_prompt_tokens"] for i in ids),
                        context_rate=statistics.mean(bool(inputs[i]["input"].strip()) for i in ids),
                        empty_rate=statistics.mean(not source[i]["predict"].strip() for i in ids),
                        hit_limit_rate=statistics.mean(source[i]["hit_generation_limit"] for i in ids))
            descriptions.append(item)
            for i in ids:
                rows.append(dict(source_index=i, quartile=q, ce=ce[i], score=source[i]["score"],
                                 tokens=tokens[i]["used_prompt_tokens"],has_context=bool(inputs[i]["input"].strip())))
        rng = random.Random(421114)
        a,b = [[source[i]["score"]*100 for i in group] for group in (groups[0],groups[-1])]
        boot = sorted((sum(rng.choices(a,k=64))-sum(rng.choices(b,k=64)))/64 for _ in range(10000))
        result = dict(name=name, metric=cell["metric"], quartiles=descriptions,
                      delta=statistics.mean(a)-statistics.mean(b), interval=[boot[249],boot[9749]],
                      interval_method="conditional on fixed quartile membership; independent within-quartile bootstrap, seed 421114")
        save(directory/"summary.json",result)
        save(directory/"rows.json",rows)
        summaries.append(result)
        print(name, [round(x["score"],3) for x in descriptions], flush=True)
    save(output/"six_cells.json",summaries)
    save(output/"frozen_config.json",config)
    save(output/"metrics.json", dict(quality_status="AUTOMATIC_COMPLETE",cells=6,n_per_cell=256,
                                    deltas={x["name"]:x["delta"] for x in summaries}))
    lines = ["# 实验一六组自动分析", "", "|组合|指标|Q1|Q2|Q3|Q4|Q1−Q4|95% 区间|",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in summaries:
        values = "|".join(f'{q["score"]:.2f}' for q in r["quartiles"])
        lines.append(f'|{r["name"]}|{r["metric"]}|{values}|{r["delta"]:.2f}|{r["interval"][0]:.2f}, {r["interval"][1]:.2f}|')
    lines += ["", "分数均乘 100。各指标分别解释。四分位按各模型自己的底模输入 CE 排名。",
              "区间以本次已固定的四分位成员为条件，不包含重新抽题并划分四分位的不确定性。",
              "这些结果估计冻结 Source 的指标关联。长度、上下文分布见各组 summary 和 rows，人工盲评另计。"]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from common import output_lock, bind_output, experiment_identity
    config=read(args.config)
    with output_lock(args.output):
        bind_output(args.output,experiment_identity(config,None,"analyze"),"analyze")
        summarize(config,args.output)
