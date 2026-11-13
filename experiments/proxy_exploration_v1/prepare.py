"""Freeze common dataset splits before parsing cached prediction scores."""
import argparse
import random
import unicodedata
from pathlib import Path

from analysis import indexed, read, save
from common import DATASETS, validate_config, sha


def split_inputs(inputs):
    unique = {}
    for i,row in sorted(inputs.items()):
        key = tuple(" ".join(unicodedata.normalize("NFC",row[k]).split()) for k in ("instruction","input"))
        unique.setdefault(key,i)
    ids = random.Random(42).sample(list(unique.values()),1344)
    return {"warmup":ids[:32],"history":ids[32:288],"diagnostic":ids[288:544],
            "adaptation":ids[544:1044],"evaluation":ids[1044:]}


def prepare(assets, output):
    if output.exists():
        raise FileExistsError("Choose a new preparation manifest")
    config = {"seed":42,"pool_size":64,"rank":5,
              "projection":"adamw_candidate_displacement","cells":assets["cells"],
              "runtime":assets["runtime"],"splits":{}}
    for dataset in DATASETS:
        cells = [x for x in config["cells"] if x["dataset"]==dataset]
        if len(cells)!=2 or cells[0]["inputs"]!=cells[1]["inputs"]:
            raise ValueError("Models must share a dataset input file")
        inputs = indexed(cells[0]["inputs"])
        config["splits"][dataset] = split_inputs(inputs)
    for cell in config["cells"]:
        for path,expected in cell["assets"].items():
            if sha(path)!=expected:
                raise ValueError("Source cache changed before preparation")
    validate_config(config)
    save(output,config)
    return config


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    prepare(read(args.assets),args.output)
