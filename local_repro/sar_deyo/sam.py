"""Sharpness-Aware Minimization, exactly as SAR uses it.

Reference: https://github.com/mr-eggplant/SAR  (sam.py)
commit 20f6e24b17525f34503510afccedc0629b67b7c4.

Kept identical to the official implementation in the three points that matter:

* epsilon = rho * g / ||g||_2, where ||g||_2 is a **global** norm over every adapted
  tensor (sam.py:_grad_norm).  rho is therefore an absolute L2 budget in parameter
  space, not a per-layer ratio.
* the first gradient is discarded (first_step(zero_grad=True)); the update uses only
  the gradient taken at Theta + epsilon, applied back at Theta (Eqn. 5's first-order
  approximation).  There is no gradient averaging.
* adaptive=False.  adaptive=True scales the perturbation by p**2, which is zero for a
  zero-initialised LoRA B matrix and would silently disable the perturbation.

rho is a hyperparameter of the *scale* of the adapted surface, so the caller passes an
absolute value computed from the surface norm (see run_tta.py --rho-rel) rather than
copying the paper's 0.05, which was tuned on ResNet GroupNorm parameters.
"""
from __future__ import annotations

import torch


class SAM:
    """Minimal SAM wrapper: same interface as an optimizer plus first_step/second_step."""

    def __init__(self, params, base_cls, rho: float, adaptive: bool = False, **kwargs):
        assert not adaptive, "adaptive=True is a no-op on zero-initialised LoRA B"
        self.base = base_cls(params, **kwargs)
        self.param_groups = self.base.param_groups
        self.rho = float(rho)
        self.adaptive = adaptive
        self._old = {}

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        grads = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        if not grads:
            return None
        shared = grads[0].grad.device
        norms = [((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(2).to(shared)
                 for p in grads]
        return torch.norm(torch.stack(norms), 2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        gnorm = self._grad_norm()
        if gnorm is None:
            return None
        scale = self.rho / (gnorm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._old[p] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p.dtype)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
        return float(gnorm)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p in self._old:
                    p.data = self._old[p]
        self.base.step()
        if zero_grad:
            self.zero_grad()

    def state_dict(self):
        return {"base": self.base.state_dict(), "rho": self.rho}

    def load_state_dict(self, sd):
        self.base.load_state_dict(sd["base"])
        self.rho = sd.get("rho", self.rho)
