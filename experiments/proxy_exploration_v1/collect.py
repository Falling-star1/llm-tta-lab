"""Experiment 2 collection from unlabeled history and independent diagnostics."""
import random
from pathlib import Path
import torch

from analysis import save, read, quartiles
from common import sha, torch_save
from geometry import fit_basis, retention, structure


def artifact(path, identity):
    metadata = read(path.with_suffix(".json"))
    if metadata["identity"]!=identity or sha(path)!=metadata["sha256"]:
        raise ValueError("Gradient artifact identity/hash mismatch")
    value = torch.load(path,map_location="cpu",weights_only=True)
    if value["identity"]!=identity:
        raise ValueError("Gradient artifact content mismatch")
    return value


def put_artifact(path, value):
    torch_save(path,value)
    save(path.with_suffix(".json"),{"identity":value["identity"],"sha256":sha(path)})


def collect(backend, output, identity, rank=5):
    output = Path(output)
    backend.reset()
    ce = {i:r["base_ce"] for i,r in backend.adaptation.items()}
    low = quartiles(backend.split["history"],ce)[0]
    random_ids = random.Random(42).sample(backend.split["history"],64)
    needed = sorted(set(low+random_ids+backend.split["diagnostic"]))
    save(output/"pool_ids.json",{"low":low,"random":random_ids,"overlap":len(set(low)&set(random_ids)),
                                "diagnostic":backend.split["diagnostic"],"identity":identity})
    for count,i in enumerate(needed,1):
        path = output/"gradients"/f"{i:05}.pt"
        if path.exists() and path.with_suffix(".json").exists():
            cached = artifact(path,identity)
            if cached["source_index"]!=i:
                raise ValueError("Wrong cached gradient index")
        else:
            gradient,metadata = backend.gradient(i)
            put_artifact(path,dict(identity=identity,source_index=i,gradient=gradient,metadata=metadata))
        if count%25==0:
            print({"event":"gradients","done":count,"total":len(needed)},flush=True)
    reports = {}
    bases = {}
    for name,ids in (("low",low),("random",random_ids)):
        matrix = torch.stack([artifact(output/"gradients"/f"{i:05}.pt",identity)["gradient"] for i in ids])
        basis,fit = fit_basis(matrix,rank)
        bases[name] = basis
        half = list(ids)
        random.Random(42).shuffle(half)
        positions = {i:k for k,i in enumerate(ids)}
        u,_ = fit_basis(matrix[[positions[i] for i in half[:32]]],rank)
        v,_ = fit_basis(matrix[[positions[i] for i in half[32:]]],rank)
        reports[name] = dict(structure=structure(matrix,rank),fit=fit,
                             half_overlap=float((u.T@v).square().sum())/rank)
        put_artifact(output/(name+"_basis.pt"),dict(identity=identity,basis=basis,ids=ids))
        del matrix
    rows = []
    for i in backend.split["diagnostic"]:
        vector = artifact(output/"gradients"/f"{i:05}.pt",identity)["gradient"]
        rows.append(dict(source_index=i,ce=ce[i],gradient_norm=float(vector.norm()),
                         low_retention=retention(bases["low"],vector),
                         random_retention=retention(bases["random"],vector)))
    save(output/"diagnostic.json",rows)
    save(output/"geometry.json",reports)
    backend.verify_frozen()
    save(output/"complete.json",{"identity":identity,"gradient_count":len(needed),
                                "low_basis_sha256":sha(output/"low_basis.pt"),
                                "random_basis_sha256":sha(output/"random_basis.pt")})
    save(output/"metrics.json",{"quality_status":"GRADIENTS_COMPLETE_QUALITY_LABELS_SEPARATE",
                               "gradient_count":len(needed),"geometry":reports})
    return reports
