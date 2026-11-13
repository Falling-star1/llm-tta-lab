import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from analysis import summarize, read, save
from common import MODELS, DATASETS, sha, validate_config
from prepare import split_inputs


def fixture(root):
    config={"seed":42,"pool_size":64,"rank":5,"splits":{},"cells":[],"runtime":{}}
    for dataset in DATASETS:
        directory=root/dataset
        directory.mkdir()
        inputs=[{"source_index":i,"instruction":f"synthetic-{i}","input":"context" if i%2 else ""}
                for i in range(5000)]
        path=directory/"inputs.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in inputs),encoding="utf-8")
        references=directory/"references.jsonl"
        references.write_text("".join(json.dumps({"source_index":i,"output":"answer"})+"\n"
                                     for i in range(5000)),encoding="utf-8")
        manifest=directory/"manifest.json"
        save(manifest,{"files":{"inputs.jsonl":sha(path)}})
        config["splits"][dataset]=split_inputs({r["source_index"]:r for r in inputs})
        for model in MODELS:
            cell_dir=directory/model
            cell_dir.mkdir()
            tokens=[{"source_index":i,"original_prompt_sha256":str(i),
                     "original_prompt_tokens":i%400+10,"used_prompt_tokens":i%400+10,
                     "input_truncated":False} for i in range(5000)]
            token_file=cell_dir/"tokens.json"
            save(token_file,{"records":tokens})
            metric="exact_match" if dataset=="gsm8k_5k" else "rougeLsum"
            rows=[dict(r,metric=metric,score=float(r["source_index"]%2),
                       predict="answer",label="answer",hit_generation_limit=False) for r in tokens]
            source=cell_dir/"source.jsonl"
            source.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
            adaptation=cell_dir/"adaptation.json"
            save(adaptation,[{"source_index":i,"base_ce":1+i/2000} for i in range(5000)])
            training=cell_dir/"run_manifest.json"
            save(training,{"config":{"model":model,"dataset":dataset,"input_sha256":sha(path)}})
            contract=cell_dir/"contract.json";save(contract,{"fixture":True})
            reuse=cell_dir/"reuse.json";save(reuse,{"fixture":True})
            paths={"inputs":str(path),"references":str(references),"source":str(source),
                   "adaptation":str(adaptation),"tokens":str(token_file),
                   "training_manifest":str(training),"dataset_manifest":str(manifest),
                   "model_contract":str(contract),"source_reuse":str(reuse)}
            config["cells"].append(dict(name=model+"__"+dataset,model=model,dataset=dataset,
                                         metric=metric,assets={p:sha(p) for p in paths.values()},**paths))
    return config


class SixCellTests(unittest.TestCase):
    def test_all_six_cells_are_reported_and_each_model_uses_same_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(config,root/"output")
            rows=read(root/"output/six_cells.json")
            self.assertEqual(len(rows),6)
            self.assertEqual({r["metric"] for r in rows},{"exact_match","rougeLsum"})
            for dataset in DATASETS:
                chosen=[]
                for model in MODELS:
                    records=read(root/"output"/(model+"__"+dataset)/"rows.json")
                    chosen.append([r["source_index"] for r in records])
                self.assertEqual(chosen[0],chosen[1])
            config["cells"][0]["metric"]="exact_match"
            with self.assertRaises(ValueError):
                validate_config(config,files=False)


if __name__=="__main__":
    unittest.main()
