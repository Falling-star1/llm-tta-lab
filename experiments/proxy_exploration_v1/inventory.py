"""Read-only cache inventory. Produces local asset hashes, not analysis results."""
import argparse
import json
from pathlib import Path

from analysis import read, save
from common import MODELS, DATASETS, sha, digest


def inventory(output):
    if Path(output).exists():
        raise FileExistsError("Choose a new inventory manifest")
    root = Path("/data/chuwenjing/results")
    ordinary = root/"tlm_offline_v6_ready/run_20261026_000443_2ca89d/formal"
    recovery = root/"tlm_qwen_dolly_recovery_v6/run_20261026_014911_83ebf2/formal"
    runtime = Path("/home/chuwenjing/TTA_LLM/TLM/local_repro/tlm_offline_b6da067b55f5")
    runtime_files = {p.relative_to(runtime).as_posix():sha(p)
                     for p in sorted(runtime.rglob("*")) if p.is_file()
                     and p.suffix in (".py",".json",".sh",".md") and "__pycache__" not in p.parts}
    import hashlib
    code_id = hashlib.sha256(json.dumps(runtime_files,sort_keys=True).encode()).hexdigest()
    if code_id!="b6da067b55f5c65241ae6fbd8c706b88523f64b27dd9568ad5661255b89d7999":
        raise ValueError("Pinned runtime has changed")
    cells = []
    for model in MODELS:
        for dataset in DATASETS:
            name = model+"__"+dataset
            directory = (recovery if name=="qwen3b__dolly_5k" else ordinary)/name
            prepared = Path("/data/chuwenjing/tta_offline/tlm_inputs_v1")/dataset
            reuse = read(directory/"source_reuse.json")
            if not reuse["sample_generation_match"] or reuse["matched_input_records"]!=5000:
                raise ValueError("Unverified Source reuse")
            paths = {"inputs":prepared/"inputs.jsonl","references":prepared/"references.jsonl",
                     "dataset_manifest":prepared/"manifest.json","source":Path(reuse["path"]),
                     "adaptation":directory/"adaptation.json","tokens":directory/"input_tokens.json",
                     "training_manifest":directory/"run_manifest.json",
                     "model_contract":directory/"model_contract.json","source_reuse":directory/"source_reuse.json"}
            hashes = {str(path):sha(path) for path in paths.values()}
            if hashes[str(paths["source"])]!=reuse["sha256"]:
                raise ValueError("Source prediction hash mismatch")
            cells.append(dict(name=name,model=model,dataset=dataset,
                              metric="exact_match" if dataset=="gsm8k_5k" else "rougeLsum",
                              assets=hashes,**{k:str(v) for k,v in paths.items()}))
    save(output,{"cells":cells,"runtime":{"path":str(runtime),"files":runtime_files,
                                        "code_id":code_id,"nltk_data":"/data/chuwenjing/cache/nltk_data"}})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    inventory(parser.parse_args().output)
