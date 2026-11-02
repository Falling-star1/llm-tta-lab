"""Validate the unchanged cosine schedule against actual AMP update counts."""
import math


def rate_at(updates,base_lr,warmup,total):
    if not 0<=updates<=total or not 0<=warmup<total or not math.isfinite(base_lr) or base_lr<=0:
        raise ValueError('Invalid scheduler definition')
    if updates<warmup:return base_lr*updates/max(1,warmup)
    return base_lr*max(0.,.5*(1.+math.cos(math.pi*(updates-warmup)/max(1,total-warmup))))


def verify_lr_trace(history,optimizer_steps,scheduler_steps,base_lr,warmup,total):
    if not history or optimizer_steps!=scheduler_steps:raise ValueError('Scheduler and optimizer counts differ')
    current=0;skips=[]
    for i,row in enumerate(history):
        lr=row['lr']
        if not math.isfinite(lr):raise ValueError('Nonfinite recorded learning rate')
        possibilities=[0] if i==0 else [current,current+1]
        matches=[n for n in possibilities if math.isclose(lr,rate_at(n,base_lr,warmup,total),rel_tol=1e-12,abs_tol=1e-16)]
        if len(matches)!=1:raise ValueError('Recorded LR is unexplained or ambiguous at visit '+str(i+1))
        updated=matches[0]
        if i and updated==current:skips.append(i)
        current=updated
    if optimizer_steps==current:skips.append(len(history))
    elif optimizer_steps!=current+1:raise ValueError('Final optimizer step count does not match the trace')
    if optimizer_steps!=len(history)-len(skips):raise ValueError('Inferred skips inconsistent')
    return {'successful_optimizer_steps':optimizer_steps,'skipped_optimizer_steps':len(skips),
            'skipped_visits':skips,'same_schedule_definition_verified':True,'recorded_trace_explained':True}
