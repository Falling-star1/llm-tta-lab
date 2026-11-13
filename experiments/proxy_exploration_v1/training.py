"""AdamW candidate-displacement projection and explicit checkpointed input visits."""
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch

from analysis import save

# Relative to the actual applied target, including sub-ULP positive targets.
MATCH_RELATIVE_TOLERANCE = 1e-3


def flat(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def assign(parameters, vector):
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.copy_(vector[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
    if offset != vector.numel():
        raise ValueError("Parameter vector dimensions differ")


class ProjectedAdamW:
    def __init__(self, parameters, lr, mode="plain", basis=None):
        self.parameters = list(parameters)
        if not self.parameters or mode not in ("plain","project","matched"):
            raise ValueError("Unknown update mode or empty parameters")
        if any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("Trainable parameters must be FP32")
        self.base_lr = lr
        self.mode = mode
        self.basis = basis
        if mode == "project":
            if basis is None or basis.shape[0] != sum(p.numel() for p in self.parameters):
                raise ValueError("Projection basis dimensions differ")
            gram = basis.T @ basis
            if not torch.allclose(gram,torch.eye(len(gram),device=gram.device),atol=1e-4,rtol=1e-4):
                raise ValueError("Basis is not orthonormal")
        self.optimizer = torch.optim.AdamW(self.parameters,lr=lr,betas=(.9,.999),
                                           eps=1e-8,weight_decay=0)

    def step(self, target_norm=None, scaler=None):
        before = flat(self.parameters).clone()
        scale_before = float(scaler.get_scale()) if scaler else None
        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        candidate = flat(self.parameters)-before
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite optimizer displacement")
        row = {"candidate_norm":float(candidate.norm()), "skipped":bool(
            scaler and scaler.get_scale()<scale_before), "matched":True}
        displacement = candidate
        if self.mode == "project":
            displacement = self.basis @ (self.basis.T @ candidate)
        if self.mode == "matched":
            if target_norm is None or not math.isfinite(target_norm) or target_norm<0:
                raise ValueError("Expected finite nonnegative target norm")
            displacement = torch.zeros_like(candidate)
            if row["candidate_norm"]>0:
                displacement = candidate*(target_norm/row["candidate_norm"])
            elif target_norm>0:
                row["matched"] = False
            row["target_norm"] = target_norm
        if self.mode != "plain":
            assign(self.parameters,before+displacement)
        actual = flat(self.parameters)-before
        row["applied_norm"] = float(actual.norm())
        row["rounding_error"] = float((actual-displacement).norm())
        row["projection_residual"] = None
        if self.mode == "project":
            row["projection_residual"] = float((actual-self.basis@(self.basis.T@actual)).norm())/max(row["applied_norm"],1e-30)
        if self.mode == "matched":
            row["norm_error"] = abs(row["applied_norm"]-target_norm)
            row["relative_norm_error"] = row["norm_error"]/target_norm if target_norm else row["norm_error"]
            row["matched"] = row["relative_norm_error"] <= MATCH_RELATIVE_TOLERANCE
            row["match_failure"] = None
            if not row["matched"]:
                row["match_failure"] = ("zero_candidate" if row["candidate_norm"]==0 else
                                        "rounded_to_zero" if row["applied_norm"]==0 else "norm_tolerance")
        return row


def linear_schedule(step, total, base_lr):
    """Match a 10% linear warmup followed by cosine decay over input visits."""
    warmup = math.ceil(total*.1)
    if step < warmup:
        return base_lr*step/max(1,warmup)
    return base_lr*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))


def rng_state():
    state = {"python":random.getstate(), "numpy":np.random.get_state(), "torch":torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(path, controller, records, identity, scaler):
    path = Path(path)
    file = path/"checkpoints"/f"visit-{len(records):05}.pt"
    file.parent.mkdir(parents=True,exist_ok=True)
    temporary = file.with_suffix(".tmp")
    state = {"identity":identity, "parameters":flat(controller.parameters).cpu(),
             "optimizer":controller.optimizer.state_dict(), "records":records, "rng":rng_state(),
             "scaler":scaler.state_dict() if scaler else None,
             "mode":controller.mode, "base_lr":controller.base_lr}
    torch.save(state,temporary)
    os.replace(temporary,file)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    save(path/"checkpoint.json",{"file":str(file.relative_to(path)), "sha256":digest,
                                 "identity":identity,"visits":len(records)})


def load_checkpoint(manifest, identity):
    manifest = Path(manifest)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    original = manifest.parent/record["file"]
    file = original.resolve()
    if not file.is_relative_to(manifest.parent.resolve()) or original.is_symlink():
        raise ValueError("Checkpoint outside output")
    if record["identity"] != identity or hashlib.sha256(file.read_bytes()).hexdigest()!=record["sha256"]:
        raise ValueError("Checkpoint identity/hash mismatch")
    # Only experiment-owned files validated against their committed manifest.
    state = torch.load(file,map_location="cpu",weights_only=False)
    if state["identity"]!=identity or len(state["records"])!=record["visits"]:
        raise ValueError("Checkpoint content mismatch")
    return state


def train_sequence(controller, loss_fn, ids, output, identity, checkpoint_every=25,
                   resume=None, target_norms=None, scaler=None, stop_after=None):
    output = Path(output)
    records = []
    if resume:
        if (resume["identity"]!=identity or resume["mode"]!=controller.mode
                or resume["base_lr"]!=controller.base_lr):
            raise ValueError("Resume configuration mismatch")
        assign(controller.parameters,resume["parameters"].to(controller.parameters[0].device))
        controller.optimizer.load_state_dict(resume["optimizer"])
        records = resume["records"]
        if [r["source_index"] for r in records] != ids[:len(records)]:
            raise ValueError("Resume input order mismatch")
        if bool(resume["scaler"]) != bool(scaler):
            raise ValueError("GradScaler differs on resume")
        if scaler:
            scaler.load_state_dict(resume["scaler"])
        restore_rng(resume["rng"])
    for position in range(len(records),len(ids)):
        if stop_after is not None and position>=stop_after:
            break
        controller.optimizer.zero_grad(set_to_none=True)
        lr = linear_schedule(position,len(ids),controller.base_lr)
        for group in controller.optimizer.param_groups:
            group["lr"] = lr
        value = loss_fn(ids[position])
        loss, metadata = value if isinstance(value,tuple) else (value,{})
        if loss.numel()!=1 or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(controller.optimizer)
        else:
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(controller.parameters,1.0,error_if_nonfinite=not bool(scaler))
        row = controller.step(target_norms[position] if target_norms is not None else None,scaler)
        if any(not torch.isfinite(p).all() for p in controller.parameters):
            raise FloatingPointError("Nonfinite parameters")
        if any(not torch.isfinite(value).all() for state in controller.optimizer.state.values()
               for value in state.values() if torch.is_tensor(value)):
            raise FloatingPointError("Nonfinite optimizer state")
        row.update(metadata)
        row.update(visit=position+1, source_index=ids[position], loss=float(loss.detach()),
                   lr=lr, gradient_norm=float(norm) if torch.isfinite(norm) else None)
        records.append(row)
        if len(records)%checkpoint_every==0 or len(records)==len(ids) or (
                stop_after is not None and len(records)==stop_after):
            checkpoint(output,controller,records,identity,scaler)
            print(json.dumps({"event":"checkpoint","visits":len(records),"identity":identity[:12]}),flush=True)
    return records
