"""Independent diagnostic review export/import and four-group gradient analysis."""
import csv
import random
from pathlib import Path
import torch

from analysis import read, save, quartiles, indexed
from common import cell_data, experiment_identity, sha, digest
from collect import artifact
from geometry import compare_quality, retention

VERDICTS = {"可接受":"good","不可接受":"bad","未决":"undecided"}


def export_reviews(config,name,output):
    cell,inputs,_,adaptation,split = cell_data(config,name)
    identity = experiment_identity(config,name,"quality")
    source = indexed(cell["source"])
    ce = {i:r["base_ce"] for i,r in adaptation.items()}
    groups = quartiles(split["diagnostic"],ce)
    selection = [(i,level) for ids,level in ((groups[0],"low"),(groups[3],"high")) for i in ids]
    # Cell-specific reproducibility prevents blind IDs sharing a CE-group map.
    random.Random(digest({"seed":421116,"cell":name})).shuffle(selection)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"blind.csv").exists() or (output/"private_key.json").exists():
        raise FileExistsError("Review package already exists")
    key = []
    with (output/"blind.csv").open("x",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream,quoting=csv.QUOTE_ALL)
        writer.writerow(["blind_id","instruction","context","response","reference","verdict","reason"])
        for k,(i,level) in enumerate(selection,1):
            blind = f"R{k:03}"
            fields = [inputs[i]["instruction"],inputs[i]["input"],source[i]["predict"],source[i]["label"]]
            fields = ["'"+x if x.lstrip().startswith(("=","+","-","@")) else x for x in fields]
            writer.writerow([blind,*fields,"",""])
            key.append({"blind_id":blind,"source_index":i,"ce_group":level,
                        "content_digest":digest([blind,*fields])})
    save(output/"private_key.json",{"identity":identity,"cell":name,"items":key,
                                   "blind_sha256":sha(output/"blind.csv")})


def import_reviews(csv_path,key,reviewer,reviewer_type):
    if reviewer_type not in ("human","model_assisted") or not reviewer.strip():
        raise ValueError("Explicit reviewer and reviewer type are required")
    with Path(csv_path).open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows)!=len(key["items"]) or len({r["blind_id"] for r in rows})!=len(rows):
        raise ValueError("Missing or duplicate review rows")
    by_id = {r["blind_id"]:r for r in rows}
    if set(by_id)!={r["blind_id"] for r in key["items"]}:
        raise ValueError("Review IDs differ from fixed package")
    labels = {}
    for item in key["items"]:
        row = by_id[item["blind_id"]]
        content = [row[k] for k in ("blind_id","instruction","context","response","reference")]
        if digest(content)!=item["content_digest"]:
            raise ValueError("Review text belongs to a different or edited package")
        if row["verdict"].strip() not in VERDICTS or not row["reason"].strip():
            raise ValueError("Every row requires an explicit verdict and reason")
        labels[item["source_index"]] = VERDICTS[row["verdict"].strip()]
    return {"identity":key["identity"],"labels":labels,"reviewer":reviewer,
            "reviewer_type":reviewer_type,"csv_sha256":sha(csv_path)}


def quality_report(config,name,geometry_output,output,review_csv=None,key_path=None,
                   reviewer="",reviewer_type="human",plots=False):
    cell,inputs,tokens,adaptation,split = cell_data(config,name)
    source = indexed(cell["source"])
    geometry_output,output = Path(geometry_output),Path(output)
    identity = experiment_identity(config,name,"gradients")
    if read(geometry_output/"complete.json")["identity"]!=identity:
        raise ValueError("Wrong geometry identity")
    ce = {i:r["base_ce"] for i,r in adaptation.items()}
    groups = quartiles(split["diagnostic"],ce)
    ids = groups[0]+groups[3]
    if cell["metric"]=="exact_match":
        if any("extracted_answer" not in source[i] for i in ids):
            raise ValueError("Frozen Source lacks GSM8K extraction records")
        if any(source[i]["score"] not in (0,1) for i in ids):
            raise ValueError("GSM8K scores must be binary")
        labels = {i:("good" if source[i]["score"]==1 else "bad") for i in ids}
        label_source = {"reviewer_type":"automatic_em","metric":"exact_match",
                        "extraction_failures":sum(source[i].get("extracted_answer") is None for i in ids)}
    else:
        if not review_csv or not key_path:
            save(output/"metrics.json",{"quality_status":"PENDING_DIAGNOSTIC_REVIEW","required":128})
            return
        key = read(key_path)
        if key["identity"]!=experiment_identity(config,name,"quality") or key["cell"]!=name:
            raise ValueError("Wrong blind-review identity")
        imported = import_reviews(review_csv,key,reviewer,reviewer_type)
        labels = imported.pop("labels")
        if set(labels)!=set(ids):
            raise ValueError("Review contains different diagnostic samples")
        label_source = imported
    gradients = {i:artifact(geometry_output/"gradients"/f"{i:05}.pt",identity)["gradient"] for i in ids}
    bases = {s:artifact(geometry_output/(s+"_basis.pt"),identity)["basis"] for s in ("low","random")}
    report = {"cell":name,"identity":identity,"label_source":label_source,"groups":{},"rows":[]}
    for level,selected in (("low",groups[0]),("high",groups[3])):
        report["groups"][level] = compare_quality(selected,gradients,labels,rank=5,seed=42)
        for i in selected:
            report["rows"].append({"source_index":i,"ce_group":level,"quality":labels[i],
                "tokens":tokens[i]["used_prompt_tokens"],"has_context":bool(inputs[i]["input"].strip()),
                "gradient_norm":float(gradients[i].norm()),
                "low_retention":retention(bases["low"],gradients[i]),
                "random_retention":retention(bases["random"],gradients[i])})
    save(output/"quality_groups.json",report)
    save(output/"metrics.json",{"quality_status":"DIAGNOSTIC_COMPLETE",
                                "group_counts":{k:v["counts"] for k,v in report["groups"].items()}})
    if plots:
        plot_quality(report,gradients,output)


def plot_quality(report, gradients, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = []
    labels = []
    for level in ("low","high"):
        for quality in ("good","bad"):
            sample = report["groups"][level]["balanced_ids"][quality]
            ids.extend(sample)
            labels.append((f"{level}/{quality}",len(sample)))
    if not ids:
        return
    matrix = torch.stack([gradients[i] for i in ids])
    norms = matrix.norm(dim=1,keepdim=True)
    units = matrix/norms.clamp_min(1e-30)
    cosines = units@units.T
    zero = norms[:,0]==0
    cosines[zero,:] = float("nan")
    cosines[:,zero] = float("nan")
    fig,axis = plt.subplots(figsize=(8,7))
    im=axis.imshow(cosines.numpy(),vmin=-1,vmax=1,cmap="coolwarm")
    offset=0
    for label,n in labels:
        if n:
            axis.text(-.01,offset+n/2,label,transform=axis.get_yaxis_transform(),
                      va="center",ha="right",fontsize=8)
            offset+=n
            axis.axhline(offset-.5,color="black",linewidth=.5)
            axis.axvline(offset-.5,color="black",linewidth=.5)
    axis.set_title("Diagnostic gradient cosine; balanced groups; zero gradients masked")
    fig.colorbar(im,ax=axis)
    fig.tight_layout()
    fig.savefig(Path(output)/"cosine.png",dpi=160)
    plt.close(fig)
    fig,axis = plt.subplots(figsize=(7,4))
    for level in ("low","high"):
        for quality in ("good","bad"):
            item = report["groups"][level]["balanced"][quality]
            if item and item["cumulative_energy"]:
                energy = item["cumulative_energy"]
                axis.plot(range(1,len(energy)+1),energy,label=f"{level}/{quality}")
    axis.set(xlabel="Component",ylabel="Cumulative uncentered gradient energy",ylim=(0,1.01))
    if axis.lines:
        axis.legend()
    fig.tight_layout()
    fig.savefig(Path(output)/"energy.png",dpi=160)
    plt.close(fig)
