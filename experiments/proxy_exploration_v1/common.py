"""Frozen inputs and execution identities shared by CPU/GPU commands."""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from analysis import indexed, read, validate_split, save

MODELS = ("llama3b","qwen3b")
DATASETS = ("dolly_5k","geosignal_5k","gsm8k_5k")
SPLIT_SIZES = {"warmup":32,"history":256,"diagnostic":256,"adaptation":500,"evaluation":300}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_config(config, files=True):
    expected = {(m,d) for m in MODELS for d in DATASETS}
    cells = config["cells"]
    if len(cells)!=6 or {(x["model"],x["dataset"]) for x in cells}!=expected:
        raise ValueError("Exactly six unique model/dataset cells are required")
    if config.get("seed")!=42 or config.get("pool_size")!=64 or config.get("rank")!=5:
        raise ValueError("Configuration differs from registered protocol")
    for dataset,split in config["splits"].items():
        validate_split(split)
        if {k:len(v) for k,v in split.items()}!=SPLIT_SIZES:
            raise ValueError("Wrong split sizes")
        if any(type(i) is not int or not 0<=i<5000 for ids in split.values() for i in ids):
            raise ValueError("Invalid source index")
    if set(config["splits"])!=set(DATASETS):
        raise ValueError("Missing dataset splits")
    for cell in cells:
        if cell["name"]!=cell["model"]+"__"+cell["dataset"]:
            raise ValueError("Cell name/model mismatch")
        expected_metric = "exact_match" if cell["dataset"]=="gsm8k_5k" else "rougeLsum"
        if cell["metric"]!=expected_metric:
            raise ValueError("Wrong dataset metric")
        if files:
            for path,expected_hash in cell["assets"].items():
                if sha(path)!=expected_hash:
                    raise ValueError("Input fingerprint mismatch: "+path)
            required = ("inputs","source","adaptation","tokens","training_manifest","references","dataset_manifest")
            if any(cell[key] not in cell["assets"] for key in required):
                raise ValueError("Unfingerprinted input")
    return config


def cell_data(config, name):
    validate_config(config)
    cell = next(x for x in config["cells"] if x["name"]==name)
    inputs = indexed(cell["inputs"])
    if any(set(row)!={"source_index","instruction","input"} for row in inputs.values()):
        raise ValueError("Training inputs contain unexpected fields")
    tokens = indexed(cell["tokens"],records=True)
    adaptation = indexed(cell["adaptation"])
    manifest = read(cell["training_manifest"])["config"]
    if manifest["model"]!=cell["model"] or manifest["dataset"]!=cell["dataset"]:
        raise ValueError("Cached CE model/dataset mismatch")
    if manifest["input_sha256"]!=sha(cell["inputs"]):
        raise ValueError("CE/input manifest mismatch")
    return cell,inputs,tokens,adaptation,config["splits"][cell["dataset"]]


def experiment_identity(config,cell,stage):
    code = {p.name:sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return digest({"config":config,"cell":cell,"stage":stage,"code":code})


def bind_output(output, identity, stage):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    marker = output/"identity.json"
    expected = {"identity":identity,"stage":stage}
    if marker.exists():
        if read(marker)!=expected:
            raise ValueError("Output belongs to a different execution identity")
    else:
        if any(p.name!=".writer.lock" for p in output.iterdir()):
            raise ValueError("Refusing unidentified nonempty output")
        save(marker,expected)


def torch_save(path,value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value,temporary)
    os.replace(temporary,path)


@contextmanager
def output_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/".writer.lock").open("a+b") as stream:
        if os.name=="nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name=="nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
