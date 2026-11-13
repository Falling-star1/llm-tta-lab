"""Five-condition experiment 3 with per-arm journals and resumable optimizer state."""
import hashlib
import json
import random
import statistics
import shutil
from pathlib import Path
import torch

from analysis import save, read, score_mean
from common import sha, bind_output
from collect import artifact
from training import ProjectedAdamW, train_sequence, load_checkpoint, flat

ARMS = ("low","plain","random","matched")


def seed_resume(previous, output, identity):
    """Copy only committed branch state into a new run; preserve previous outputs."""
    previous,output=Path(previous).resolve(),Path(output).resolve()
    if previous==output:
        raise ValueError("Cross-run recovery needs a distinct new output")
    if read(previous/"identity.json")!={"identity":identity,"stage":"updates"}:
        raise ValueError("Previous update identity differs")
    bind_output(output,identity,"updates")
    if any((output/arm).exists() for arm in ARMS):
        raise ValueError("New recovery output already contains branch state")
    for arm in ARMS:
        directory=previous/arm
        if not (directory/"checkpoint.json").exists():
            continue
        load_checkpoint(directory/"checkpoint.json",identity+":"+arm)
        manifest=read(directory/"checkpoint.json")
        relatives=["checkpoint.json",manifest["file"]]
        if (directory/"predictions.jsonl").exists():
            relatives.append("predictions.jsonl")
        for relative in relatives:
            original=directory/relative
            source=original.resolve()
            if not source.is_relative_to(directory) or original.is_symlink():
                raise ValueError("Unsafe recovery artifact")
            target=output/arm/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            before=sha(source)
            shutil.copy2(source,target)
            if sha(source)!=before or sha(target)!=before:
                raise ValueError("Previous run changed while copying recovery state")


def vector_digest(backend):
    vector = flat(backend.parameters).cpu().contiguous()
    return hashlib.sha256(vector.numpy().tobytes()).hexdigest()


def read_journal(path, ids, identity, adapter):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = len(data) if data.endswith(b"\n") else data.rfind(b"\n")+1
    rows = [json.loads(line) for line in data[:end].splitlines()]
    if len(rows)>len(ids) or [r["source_index"] for r in rows]!=ids[:len(rows)]:
        raise ValueError("Prediction order/length mismatch")
    if any(r["identity"]!=identity or r["adapter_sha256"]!=adapter for r in rows):
        raise ValueError("Prediction identity mismatch")
    if end!=len(data):
        with path.open("r+b") as stream:
            stream.truncate(end)
    return rows


def evaluate(backend, output, identity):
    output = Path(output)
    ids = backend.split["evaluation"]
    adapter = vector_digest(backend)
    path = output/"predictions.jsonl"
    rows = read_journal(path,ids,identity,adapter)
    references = backend.references()
    source = backend.source()
    scorer = backend.scorer()
    for row in rows:
        i = row["source_index"]
        if abs(scorer.score(row["predict"],references[i]["output"])-row["score"])>1e-12:
            raise ValueError("Cached prediction score mismatch")
    with path.open("a",encoding="utf-8",newline="\n") as stream:
        for i in ids[len(rows):]:
            result = backend.generate(i)
            row = dict(result,source_index=i,identity=identity,adapter_sha256=adapter,
                       score=scorer.score(result["predict"],references[i]["output"]),
                       source_score=source[i]["score"])
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            stream.flush()
            rows.append(row)
            if len(rows)%25==0:
                print({"event":"evaluation","done":len(rows),"total":len(ids)},flush=True)
    return rows


def paired_difference(a,b):
    if not a or len(a)!=len(b):
        raise ValueError("Paired sample lengths differ")
    differences = [(x-y)*100 for x,y in zip(a,b)]
    rng = random.Random(421115)
    boot = sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    return {"difference":statistics.mean(differences),"interval":[boot[249],boot[9749]],
            "n":len(a),"method":"paired question bootstrap, seed 421115"}


def run_updates(backend, geometry_output, output, identity, geometry_identity,
                resume=False, stop_after=None):
    output = Path(output)
    geometry_output = Path(geometry_output)
    bind_output(output,identity,"updates")
    completed = read(geometry_output/"complete.json")
    if completed["identity"]!=geometry_identity:
        raise ValueError("Geometry belongs to a different execution")
    bases = {name:artifact(geometry_output/(name+"_basis.pt"),geometry_identity)["basis"]
             for name in ("low","random")}
    for name in bases:
        if sha(geometry_output/(name+"_basis.pt"))!=completed[name+"_basis_sha256"]:
            raise ValueError("Basis changed since collection")
    sources = backend.source()
    references = backend.references()
    scorer = backend.scorer()
    baseline = []
    for i in backend.split["evaluation"]:
        score = scorer.score(sources[i]["predict"],references[i]["output"])
        if abs(score-sources[i]["score"])>1e-12:
            raise ValueError("Frozen Source score replay differs")
        baseline.append(score)
    save(output/"source.json",{"identity":identity,"indices":backend.split["evaluation"],
                               "scores":baseline,"score_x100":score_mean(baseline)})
    traces, answers = {},{}
    for arm in ARMS:
        directory = output/arm
        directory.mkdir(exist_ok=True,parents=True)
        arm_identity = identity+":"+arm
        backend.reset()
        basis = bases["low" if arm=="low" else "random"].to(backend.device) if arm in ("low","random") else None
        mode = "project" if basis is not None else ("matched" if arm=="matched" else "plain")
        lr = 1e-6 if backend.cell["dataset"]=="gsm8k_5k" else 5e-5
        controller = ProjectedAdamW(backend.parameters,lr=lr,mode=mode,basis=basis)
        checkpoint = directory/"checkpoint.json"
        if checkpoint.exists() and not resume:
            raise ValueError("Checkpoint exists; use explicit --resume")
        state = load_checkpoint(checkpoint,arm_identity) if checkpoint.exists() else None
        scaler = torch.amp.GradScaler("cuda") if backend.use_amp else None
        targets = [r["applied_norm"] for r in traces["low"]] if arm=="matched" else None
        records = train_sequence(controller,lambda i:backend.loss(i,weighted=True),
            backend.split["adaptation"],directory,identity=arm_identity,resume=state,
            target_norms=targets,scaler=scaler,stop_after=stop_after)
        traces[arm] = records
        save(directory/"visits.json",records)
        if len(records)<len(backend.split["adaptation"]):
            return {"status":"INTERRUPTED","arm":arm}
        backend.verify_frozen()
        answers[arm] = evaluate(backend,directory,arm_identity)
        save(directory/"complete.json",{"identity":arm_identity,"visits":len(records),
             "adapter_sha256":vector_digest(backend),"predictions_sha256":sha(directory/"predictions.jsonl"),
             "n":len(answers[arm]),"score_x100":score_mean([r["score"] for r in answers[arm]])})
        del controller
    low = [r["score"] for r in answers["low"]]
    comparisons = {arm:paired_difference(low,[r["score"] for r in answers[arm]])
                   for arm in ("plain","random","matched")}
    comparisons["source"] = paired_difference(low,baseline)
    result = {"identity":identity,"n":len(baseline),"metric":backend.cell["metric"],
              "source":score_mean(baseline),
              "arms":{arm:score_mean([r["score"] for r in rows]) for arm,rows in answers.items()},
              "low_minus":comparisons,
              "visits":{arm:{"n":len(rows),"nonzero":sum(r["applied_norm"]>0 for r in rows),
                                  "amp_skips":sum(r["skipped"] for r in rows),
                                  "unmatched":sum(not r["matched"] for r in rows),
                                  "max_projection_residual":max((r["projection_residual"] or 0 for r in rows),default=0),
                                  "max_norm_error":max((r.get("norm_error",0) for r in rows),default=0)}
                         for arm,rows in traces.items()}}
    warnings = [arm+":zero_effective_updates" for arm,x in result["visits"].items() if x["nonzero"]==0]
    warnings += [arm+":unmatched_positions" for arm,x in result["visits"].items() if x["unmatched"]>0]
    result["warnings"] = warnings
    save(output/"summary.json",result)
    quality = "COMPLETE_WITH_UPDATE_WARNINGS" if warnings else "COMPLETE_PENDING_SCIENTIFIC_REVIEW"
    save(output/"metrics.json",{"quality_status":quality,**result})
    return result
