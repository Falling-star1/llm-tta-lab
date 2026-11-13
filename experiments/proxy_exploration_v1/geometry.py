"""Uncentered gradient geometry. Rows are samples; columns are fixed parameters."""
import random
import torch


def require_matrix(gradients):
    gradients = gradients.detach().cpu().float()
    if gradients.ndim != 2 or not gradients.numel() or not torch.isfinite(gradients).all():
        raise ValueError("Expected a finite nonempty gradient matrix")
    return gradients


def orthonormalize(columns):
    """Use FP64 for the tall QR, then store the parameter basis in FP32."""
    return torch.linalg.qr(columns.double(),mode="reduced").Q.float().contiguous()


def fit_basis(gradients, rank):
    gradients = require_matrix(gradients)
    if type(rank) is not int or not 1 <= rank <= min(gradients.shape):
        raise ValueError("Invalid subspace rank")
    scale = float(gradients.abs().max())
    if scale == 0:
        raise ValueError("All gradients are zero")
    scaled = gradients / scale
    # Sample Gram matrix avoids an SVD of a multi-million-column dense matrix.
    gram = (scaled @ scaled.T).double()
    values, vectors = torch.linalg.eigh(gram)
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    tolerance = max(1e-12, float(values[0])*1e-6)
    available = int((values > tolerance).sum())
    if available < rank:
        raise ValueError(f"Requested rank {rank}, numerical rank {available}")
    basis = scaled.T @ (vectors[:, :rank] / values[:rank].sqrt()).float()
    basis = orthonormalize(basis)
    energy = (values / values.sum()).cumsum(0).tolist()
    return basis, {"rank":rank, "numerical_rank":available,
                   "cumulative_energy":energy, "gradient_scale":scale}


def retention(basis, gradient):
    gradient = gradient.detach().to(device=basis.device, dtype=basis.dtype)
    norm = float(gradient.norm())
    return None if norm == 0 else float((basis.T @ gradient).norm())/norm


def structure(gradients, rank=5):
    gradients = require_matrix(gradients)
    norms = gradients.norm(dim=1)
    nonzero = gradients[norms > 0]
    report = {"n":len(gradients), "zero_count":int((norms==0).sum()),
              "mean_norm":float(norms.mean()), "mean_cosine":None, "cumulative_energy":[]}
    if not len(nonzero):
        return report
    units = nonzero / nonzero.norm(dim=1, keepdim=True)
    cosines = (units @ units.T).clamp(-1,1)
    if len(nonzero)>1:
        report["mean_cosine"] = float((cosines.sum()-cosines.diag().sum())/
                                      (len(nonzero)*(len(nonzero)-1)))
    gram = (nonzero/nonzero.abs().max()) @ (nonzero/nonzero.abs().max()).T
    values = torch.linalg.eigvalsh(gram.double()).flip(0).clamp_min(0)
    report["cumulative_energy"] = (values/values.sum()).cumsum(0).tolist()
    report["requested_rank"] = min(rank,len(gradients),gradients.shape[1])
    return report


def compare_quality(ids, gradients, labels, rank=5, seed=42):
    groups = {name:[i for i in ids if labels.get(i,"undecided")==name]
              for name in ("good","bad","undecided")}
    count = min(len(groups["good"]),len(groups["bad"]))
    rng = random.Random(seed)
    balanced = {name:sorted(rng.sample(groups[name],count)) for name in ("good","bad")}
    report = {"counts":{k:len(v) for k,v in groups.items()}, "balanced_n":count,
              "balanced_ids":balanced, "full":{}, "balanced":{}, "between":{}}
    for scope,selection in (("full",groups),("balanced",balanced)):
        units = {}
        for name in ("good","bad"):
            matrix = torch.stack([gradients[i] for i in selection[name]]) if selection[name] else None
            if matrix is not None:
                norms = matrix.norm(dim=1)
                units[name] = matrix[norms>0]/norms[norms>0,None]
        pairs = len(units.get("good",[]))*len(units.get("bad",[]))
        mean = float((units["good"]@units["bad"].T).clamp(-1,1).mean()) if pairs else None
        report["between"][scope] = {"mean_cosine":mean,"valid_pairs":pairs}
    for name in ("good","bad"):
        for source,output in ((groups,"full"),(balanced,"balanced")):
            chosen = source[name]
            report[output][name] = structure(torch.stack([gradients[i] for i in chosen]),rank) if chosen else None
    return report
