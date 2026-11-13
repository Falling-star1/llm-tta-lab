"""Generate labctl requests and an output index. Does not submit or start jobs."""
import argparse
import re
from pathlib import Path

from analysis import read, save
from common import sha, validate_config, experiment_identity


def code_snapshot(root):
    files={}
    for path in sorted(root.rglob("*")):
        relative=path.relative_to(root)
        if any(part in {".git","__pycache__",".venv",".pytest_cache"} for part in relative.parts):
            continue
        if path.is_file() and (path.suffix in (".py",".json",".yaml",".yml",".toml",".sh",".md")
                               or path.name=="SHA256SUMS"):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Linked source file")
            files[relative.as_posix()]=sha(path)
            if len(files)>5000:
                raise ValueError("Code snapshot too large")
    return files


def generate(args):
    config=validate_config(read(args.config))
    if not re.fullmatch("[0-9a-f]{40}",args.commit) or args.commit=="0"*40:
        raise ValueError("A real full source commit is required")
    if not re.fullmatch("[a-z0-9][a-z0-9-]{2,35}",args.prefix):
        raise ValueError("Invalid experiment prefix")
    if not 0<args.timeout<=14*86400:
        raise ValueError("Set an explicit bounded timeout")
    root=args.code_root.resolve()
    if args.output.resolve().is_relative_to(root) or args.config.resolve().is_relative_to(root):
        raise ValueError("Keep requests and config outside code_root")
    if args.output.exists():
        raise FileExistsError("Choose a new request directory")
    files=code_snapshot(root)
    if "entry.py" not in files:
        raise ValueError("code_root must be the released proxy_exploration_v1 directory")
    previous=read(args.preflight_index) if args.preflight_index else None
    geometry=read(args.geometry_index) if args.geometry_index else None
    recovery=read(args.resume_index) if args.resume_index else None
    if recovery and args.stage!="updates":
        raise ValueError("Cross-run recovery requests use the updates stage")
    outputs={}
    requests={}
    for cell in ([None] if args.stage=="analyze" else config["cells"]):
        name=cell["name"] if cell else "six"
        job=args.prefix+"-"+name.replace("_","-")
        if len(job)>80:
            raise ValueError("Experiment ID too long")
        output=args.lab_root/"runs"/job/"output"
        argv=["{python}","{entrypoint}","--stage",args.stage,"--config",str(args.config),"--output","{output}"]
        inputs={"frozen_config":{"path":str(args.config),"sha256":sha(args.config)}}
        for k,c in enumerate([cell] if cell else config["cells"]):
            for label in ("inputs","references","source","adaptation","tokens","training_manifest",
                          "model_contract","dataset_manifest","source_reuse"):
                field={"inputs":"dataset"}.get(label,label)
                if cell is None:
                    field=f"cell{k}_"+field
                inputs[field]={"path":c[label],"sha256":c["assets"][c[label]]}
        if cell:
            argv+=["--cell",name]
        if args.stage in ("gradients","updates","cell"):
            if not previous:
                raise ValueError("GPU experiment requests need a completed preflight index")
            receipt=Path(previous["outputs"][name]["output"])/"metrics.json"
            record=read(receipt)
            if (record.get("quality_status")!="PREFLIGHT_COMPLETE"
                    or record.get("identity")!=experiment_identity(config,name,"preflight")):
                raise ValueError("Preflight identity/quality mismatch")
            argv+=["--preflight",str(receipt)]
            inputs["preflight"]={"path":str(receipt),"sha256":sha(receipt)}
        if args.stage=="updates":
            if not geometry:
                raise ValueError("Updates need --geometry-index from a finished collection")
            path=Path(geometry["outputs"][name]["geometry"])
            if read(path/"complete.json")["identity"]!=experiment_identity(config,name,"gradients"):
                raise ValueError("Geometry index does not match configuration")
            argv+=["--geometry-output",str(path)]
            for file in ("complete.json","low_basis.pt","low_basis.json","random_basis.pt","random_basis.json"):
                inputs["geometry_"+file.replace(".","_")]={"path":str(path/file),"sha256":sha(path/file)}
            if recovery:
                recovery_output=Path(recovery["outputs"][name]["updates"])
                if read(recovery_output/"identity.json")!={"identity":experiment_identity(config,name,"updates"),"stage":"updates"}:
                    raise ValueError("Recovery source belongs to a different protocol or code")
                argv+=["--resume-from",str(recovery_output)]
                inputs["recovery_identity"]={"path":str(recovery_output/"identity.json"),"sha256":sha(recovery_output/"identity.json")}
                for arm in ("low","plain","plain_b","random","matched"):
                    manifest=recovery_output/arm/"checkpoint.json"
                    if not manifest.exists():
                        continue
                    file=(manifest.parent/read(manifest)["file"]).resolve()
                    if not file.is_relative_to(manifest.parent.resolve()):
                        raise ValueError("Recovery checkpoint outside previous output")
                    paths=[manifest,file]
                    journal=recovery_output/arm/"predictions.jsonl"
                    if journal.exists():
                        paths.append(journal)
                    for k,path in enumerate(paths):
                        inputs[f"recovery_{arm}_{k}"]={"path":str(path),"sha256":sha(path)}
        protocols={"analyze":"proxy-e1-six-source-v1","preflight":"proxy-six-preflight-v4",
                   "gradients":"proxy-e2-six-gradient-v3","updates":"proxy-e3-six-adam-step-v3",
                   "cell":"proxy-e2-v3-e3-v3"}
        request={"schema_version":1,"experiment_id":job,"owner":args.owner,
                 "issue":"https://github.com/Falling-star1/llm-tta-lab/issues/14",
                 "method":"proxy-"+args.stage,"model":cell["model"] if cell else "llama-qwen-3b",
                 "dataset":cell["dataset"] if cell else "dolly-geosignal-gsm8k",
                 "protocol_id":protocols[args.stage],"seed":42,"commit":args.commit,
                 "code_root":str(root),"code_files":files,"entrypoint":"entry.py","inputs":inputs,
                 "argv":argv,"gpu_count":0 if args.stage=="analyze" else 1,
                 "gpu_candidates":list(range(8)),"timeout_seconds":args.timeout,
                 "queue_timeout_seconds":600,"legacy_lock_policy":"wrapper"}
        requests[job]=request
        outputs[name]={"request":str(args.output/(job+".json")),"output":str(output),
                       "geometry":str(output/"gradients"),"updates":str(output/"updates")}
    args.output.mkdir(parents=True)
    for job,request in requests.items():
        save(args.output/(job+".json"),request)
    save(args.output/"index.json",{"stage":args.stage,"outputs":outputs})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("analyze","preflight","gradients","updates","cell"),required=True)
    for name in ("config","code-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--lab-root",type=Path,default=Path("/data/chuwenjing/lab"))
    parser.add_argument("--commit",required=True)
    parser.add_argument("--prefix",required=True)
    parser.add_argument("--owner",default="Falling-star1")
    parser.add_argument("--timeout",type=int,required=True)
    parser.add_argument("--preflight-index",type=Path)
    parser.add_argument("--geometry-index",type=Path)
    parser.add_argument("--resume-index",type=Path)
    generate(parser.parse_args())
