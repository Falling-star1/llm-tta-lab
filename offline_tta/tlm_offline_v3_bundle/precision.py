"""Compare full-forward and direct-decoder loss/gradients under actual AMP."""
import torch
from tlm_loss import input_cross_entropy,weighted_input_loss

def compare_precision_paths(trainer,feature):
    if trainer.history or trainer.base_scores:
        raise ValueError('Precision check requires a fresh diagnostic trainer')
    model=trainer.model
    batch=trainer._prepare_inputs(trainer.data_collator([feature]))
    ids,mask=batch['input_ids'],batch['attention_mask']
    device=ids.device.type
    parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
    initial={name:p.detach().cpu().clone() for name,p in parameters.items()}
    training=model.training
    cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None
    trace=[]
    hook=model.get_base_model().model.register_forward_pre_hook(
        lambda module,inputs:trace.append(torch.is_autocast_enabled(device)))
    try:
        model.zero_grad(set_to_none=True)
        # Exercise both A and B gradients, not only the zero-B initial state.
        with torch.no_grad():
            for name,p in parameters.items():
                if 'lora_B' in name:
                    p.fill_(1e-3)
        model.eval()
        with torch.no_grad(),model.disable_adapter(),trainer.accelerator.autocast():
            base=input_cross_entropy(model(input_ids=ids,attention_mask=mask,use_cache=False).logits,ids,mask)
        model.train()
        with trainer.accelerator.autocast():
            current=input_cross_entropy(model(input_ids=ids,attention_mask=mask,use_cache=False).logits,ids,mask)
        reference=weighted_input_loss(current,base,trainer.threshold,trainer.lamb).sum()
        reference.backward()
        gradients={name:p.grad.detach().cpu().clone() for name,p in parameters.items()}
        reference_value=float(reference.detach())
        del reference,current
        model.zero_grad(set_to_none=True)
        actual=trainer.compute_loss(model,batch)
        actual.backward()
        actual_value=float(actual.detach())
        torch.testing.assert_close(torch.tensor(actual_value),torch.tensor(reference_value),atol=1e-5,rtol=1e-4)
        max_abs,max_relative=0.,0.
        nonzero=False
        factor_norms={'lora_A':0.,'lora_B':0.}
        for name,p in parameters.items():
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError('Nonfinite/missing precision-check gradient: '+name)
            expected=gradients[name]
            got=p.grad.detach().cpu()
            torch.testing.assert_close(got,expected,atol=1e-5,rtol=1e-2,msg=lambda msg:name+' '+msg)
            difference=(got-expected).float()
            max_abs=max(max_abs,float(difference.abs().max()))
            norm=float(expected.float().norm())
            for factor in factor_norms:
                if factor in name:
                    factor_norms[factor]+=norm
            nonzero=nonzero or norm>0
            if norm>1e-8:
                max_relative=max(max_relative,float(difference.norm())/norm)
        if not trace or not all(trace):
            raise ValueError('Full and direct decoder paths must both enter autocast')
        if not nonzero or not all(factor_norms.values()) or max_relative>.01:
            raise ValueError('AMP gradient comparison is zero or differs by more than 1% in relative L2')
        return {'status':'PASS','device':device,'tokens':ids.shape[1],
                'projection_chunks':(ids.shape[1]-2)//trainer.chunk_size+1,
                'probe_adapter_state':'temporary B=1e-3; original A/B restored',
                'nonzero_gradient_factors':sorted(factor_norms),
                'reference_loss':reference_value,'chunk_loss':actual_value,
                'compared_parameters':len(parameters),'max_gradient_absolute_difference':max_abs,
                'max_gradient_relative_l2_difference':max_relative,'all_decoder_calls_under_autocast':True,
                'note':'Unscaled gradients isolate forward/backward precision; GradScaler is checked by disk resume probe'}
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(initial[name].to(p.device))
        trainer.history.clear()
        trainer.base_scores.clear()
        model.train(training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
