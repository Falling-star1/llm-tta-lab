"""Remove selected target terms without rescaling retained token gradients."""
import torch


def masked_input_cross_entropy(model,input_ids,attention_mask,target_mask,chunk_size=128):
    from tlm_loss import _targets
    from full_projection import _ChunkedCE
    if target_mask.shape!=input_ids.shape or target_mask.dtype!=torch.bool:
        raise ValueError('Expected a boolean mask for every input token')
    labels,original_counts=_targets(input_ids,attention_mask)
    labels.masked_fill_(~target_mask[:,1:].to(labels.device),-100)
    if not bool((labels!=-100).any(-1).all()):raise ValueError('No retained next-token targets')
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    hidden=base.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True).last_hidden_state
    logits=base.get_output_embeddings()(hidden)
    return _ChunkedCE.apply(logits,labels.to(logits.device),original_counts.to(logits.device),chunk_size)
