"""Temporary diagnostic FP32 projection; the frozen FP16 module is restored."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def fp32_head(head):
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError('The output head must be frozen')
    previous=head.forward
    weight=head.weight.detach().float()
    bias=head.bias.detach().float() if getattr(head,'bias',None) is not None else None
    def forward(hidden):
        with torch.autocast(hidden.device.type,enabled=False):
            return F.linear(hidden.float(),weight,bias)
    head.forward=forward
    try:
        yield
    finally:
        head.forward=previous

