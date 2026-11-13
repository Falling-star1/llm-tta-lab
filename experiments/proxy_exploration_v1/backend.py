"""Adapter to the fingerprinted v6 runtime; no edits to its frozen source."""
import contextlib
import sys
from pathlib import Path
import torch

from analysis import read, indexed
from common import sha, cell_data
from training import flat, assign


class Backend:
    def __init__(self, config, name):
        runtime = config["runtime"]
        directory = Path(runtime["path"]).resolve()
        for relative,expected in runtime["files"].items():
            path = (directory/relative).resolve()
            if not path.is_relative_to(directory) or sha(path)!=expected:
                raise ValueError("Frozen runtime fingerprint mismatch")
        sys.path.insert(0,str(directory))
        import runtime as pinned
        if pinned.code_identity()!=runtime["code_id"]:
            raise ValueError("Frozen runtime identity mismatch")
        from tlm_loss import model_input_cross_entropy
        self.loss_impl = model_input_cross_entropy
        self.pinned = pinned
        self.cell,self.inputs,self.token_records,self.adaptation,self.split = cell_data(config,name)
        self.loaded = pinned.LoadedModel(self.cell["model"],self.cell["dataset"],seed=config["seed"])
        self.model = self.loaded.model
        self.names,self.parameters = zip(*[(n,p) for n,p in self.model.named_parameters() if p.requires_grad])
        self.initial = flat(self.parameters).cpu().clone()
        self.initial_digest = pinned.frozen_digest(self.model)
        expected_contract = read(self.cell["model_contract"])
        for key in ("model_id","seed","frozen_sha256","trainable_parameters","files"):
            if self.loaded.contract[key]!=expected_contract[key]:
                raise ValueError("Model contract mismatch: "+key)
        self.encoded = self.loaded.tokens([self.inputs[i] for i in range(5000)])
        for i,(_,metadata) in enumerate(self.encoded):
            if metadata!=self.token_records[i]:
                raise ValueError("Tokenization differs at "+str(i))
        self.device = self.parameters[0].device
        self.use_amp = True

    def reset(self):
        assign(self.parameters,self.initial.to(self.device))
        self.model.zero_grad(set_to_none=True)
        self.pinned.configure(42)

    def train_mode(self):
        self.model.train()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    def loss(self, index, weighted=False):
        self.train_mode()
        ids = torch.tensor([self.encoded[index][0]],device=self.device)
        with torch.autocast("cuda",dtype=torch.float16):
            ce = self.loss_impl(self.model,ids,torch.ones_like(ids),128).mean()
        base = self.adaptation[index]["base_ce"]
        selected = base>3
        weight = .1*__import__("math").exp(base-3)
        value = ce*weight*selected if weighted else ce
        return value,{"base_ce":base,"current_ce":float(ce.detach()),"selected":selected,"weight":weight}

    def gradient(self,index):
        for scale in (1024.,128.,16.,1.):
            self.model.zero_grad(set_to_none=True)
            loss,_ = self.loss(index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite input CE")
            (loss*scale).backward()
            parts = [torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1)/scale
                     for p in self.parameters]
            gradient = torch.cat(parts)
            if torch.isfinite(gradient).all():
                result=gradient.cpu()
                by_kind={}
                for kind in ("lora_A","lora_B"):
                    values=[part.square().sum() for name,part in zip(self.names,parts) if kind in name]
                    by_kind[kind+"_norm"]=float(torch.stack(values).sum().sqrt()) if values else 0.0
                self.model.zero_grad(set_to_none=True)
                return result,{"loss":float(loss.detach()),"backward_scale":scale,**by_kind}
        raise FloatingPointError("Nonfinite gradients at every declared loss scale")

    def source(self):
        return indexed(self.cell["source"])

    def references(self):
        return indexed(self.cell["references"])

    def generate(self,index):
        return self.loaded.generate(self.encoded[index][0],max_new_tokens=512)

    def scorer(self):
        return self.pinned.Scorer(self.cell["metric"])

    def verify_frozen(self):
        if self.pinned.frozen_digest(self.model)!=self.initial_digest:
            raise ValueError("Frozen base changed")

    def source_replay(self):
        self.reset()
        source = self.source()
        for i in self.split["history"][:3]:
            if self.generate(i)["predict"]!=source[i]["predict"]:
                raise ValueError("Source replay mismatch at "+str(i))
        self.reset()
