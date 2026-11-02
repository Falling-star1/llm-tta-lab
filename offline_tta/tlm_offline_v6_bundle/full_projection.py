"""Preserve whole FP16 head GEMMs; bound FP32 CE buffers with recomputation."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,labels,counts,chunk_size):
        from tlm_loss import _check_finite
        parts=[]
        with torch.autocast(logits.device.type,enabled=False):
            for start in range(0,labels.shape[1],chunk_size):
                stop=min(start+chunk_size,labels.shape[1])
                values=logits[:,start:stop,:].float()
                _check_finite(values,'whole-head CE logits')
                losses=F.cross_entropy(values.reshape(-1,values.shape[-1]),labels[:,start:stop].reshape(-1),
                                       reduction='none',ignore_index=-100)
                parts.append(losses.reshape(labels.shape[0],-1))
            result=torch.cat(parts,dim=1).sum(-1)/counts
        if ctx.needs_input_grad[0]:
            # The full FP16 logits and their full gradient need not coexist on
            # the GPU. Preserve the exact forward values on CPU; reload only
            # each CE tile during backward. Original parameters stay on GPU.
            ctx.save_for_backward(logits.detach().to('cpu',copy=True),labels,counts)
        ctx.logits_device=logits.device
        ctx.chunk_size=chunk_size
        _check_finite(result,'whole-head input CE')
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        logits,labels,counts=ctx.saved_tensors
        gradient=torch.empty(logits.shape,device=ctx.logits_device,dtype=logits.dtype)
        scale=grad_output/counts
        # One contiguous gradient enters the original whole-head GEMM backward.
        # Creating slice autograd graphs of the whole logits would allocate a
        # full gradient for every tile; these independent local graphs avoid it.
        with torch.enable_grad(),torch.autocast(ctx.logits_device.type,enabled=False):
            for start in range(0,labels.shape[1],ctx.chunk_size):
                stop=min(start+ctx.chunk_size,labels.shape[1])
                values=logits[:,start:stop,:].to(device=ctx.logits_device,dtype=torch.float32).detach().requires_grad_(True)
                losses=F.cross_entropy(values.reshape(-1,values.shape[-1]),labels[:,start:stop].reshape(-1),
                                       reduction='none',ignore_index=-100).reshape(labels.shape[0],-1)
                derivative=torch.autograd.grad(losses,values,grad_outputs=scale[:,None].expand_as(losses),
                                               create_graph=False)[0]
                gradient[:,start:stop,:].copy_(derivative)
                del derivative,losses,values
        gradient[:,labels.shape[1]:,:].zero_()
        return gradient,None,None,None


def full_projection_cross_entropy(model,input_ids,attention_mask=None,chunk_size=128):
    from tlm_loss import _targets
    if isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('CE chunk size must be a positive integer')
    labels,counts=_targets(input_ids,attention_mask)
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if attention_mask is None:attention_mask=torch.ones_like(input_ids)
    hidden=base.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True).last_hidden_state
    # Keep the otherwise-unused final position so the GEMM shape matches the
    # pinned original model.forward exactly, in both forward and backward.
    logits=base.get_output_embeddings()(hidden)
    return _ChunkedCE.apply(logits,labels.to(logits.device),counts.to(logits.device),chunk_size)
