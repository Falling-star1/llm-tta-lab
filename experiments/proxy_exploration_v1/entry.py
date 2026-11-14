"""Explicit CPU and GPU stages; run experiments through labctl-generated requests."""
import argparse
import os
from pathlib import Path

from analysis import read, save, summarize
from common import validate_config, experiment_identity, output_lock, bind_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("analyze","preflight","gradients","updates","cell",
                                          "review-export","quality-report"),required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cell")
    parser.add_argument("--geometry-output",type=Path)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--resume-from",type=Path)
    parser.add_argument("--review-csv",type=Path)
    parser.add_argument("--review-key",type=Path)
    parser.add_argument("--reviewer",default="")
    parser.add_argument("--reviewer-type",choices=("human","model_assisted"),default="human")
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    if args.preflight and args.preflight.resolve().is_relative_to(args.output.resolve()):
        parser.error("Preflight receipt must come from a separate completed preflight run")
    if args.resume_from and args.stage!="updates":
        parser.error("--resume-from is supported for a new updates run")
    if args.geometry_output and args.stage not in ("updates","quality-report"):
        parser.error("--geometry-output is read-only input to updates or quality-report")
    config=validate_config(read(args.config))
    if args.stage!="analyze" and args.cell not in {c["name"] for c in config["cells"]}:
        parser.error("A registered --cell is required")
    identity=experiment_identity(config,args.cell,args.stage)
    with output_lock(args.output):
        bind_output(args.output,identity,args.stage)
        if args.stage=="analyze":
            summarize(config,args.output)
            return
        if args.stage in ("review-export","quality-report"):
            from reviews import export_reviews, quality_report
            if args.stage=="review-export":
                export_reviews(config,args.cell,args.output)
            else:
                if not args.geometry_output:
                    parser.error("--geometry-output is required")
                quality_report(config,args.cell,args.geometry_output,args.output,
                               args.review_csv,args.review_key,args.reviewer,args.reviewer_type,args.plots)
            return
        visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
        if not visible.isdecimal():
            raise ValueError("GPU stages require exactly one labctl-assigned GPU")
        if not os.environ.get("LAB_EXPERIMENT_ID"):
            raise ValueError("Start GPU stages through the unified labctl entry")
        if args.stage!="preflight":
            if not args.preflight:
                parser.error("--preflight is required for GPU experiment stages")
            receipt=read(args.preflight)
            if (receipt.get("quality_status")!="PREFLIGHT_COMPLETE"
                    or receipt.get("identity")!=experiment_identity(config,args.cell,"preflight")):
                raise ValueError("Preflight receipt does not match execution")
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        os.environ["NLTK_DATA"]=config["runtime"]["nltk_data"]
        from backend import Backend
        backend=Backend(config,args.cell)
        save(args.output/"model_contract.json",backend.loaded.contract)
        save(args.output/"parameter_layout.json",
             [{"name":n,"shape":list(p.shape),"dtype":str(p.dtype)}
              for n,p in zip(backend.names,backend.parameters)])
        if args.stage=="preflight":
            preflight(backend,args.output,identity)
            return
        from collect import collect
        from updates import run_updates
        geometry_id=experiment_identity(config,args.cell,"gradients")
        geometry_output=args.geometry_output or args.output/"gradients"
        if args.stage in ("gradients","cell"):
            backend.source_replay()
            collect(backend,geometry_output,geometry_id,rank=config["rank"])
        result = None
        if args.stage in ("updates","cell"):
            updates_id=experiment_identity(config,args.cell,"updates")
            if args.resume_from:
                from updates import seed_resume
                seed_resume(args.resume_from,args.output/"updates",updates_id)
            result = run_updates(backend,geometry_output,args.output/"updates",updates_id,geometry_id,
                                 args.resume or bool(args.resume_from))
        quality = "GRADIENTS_COMPLETE_QUALITY_LABELS_SEPARATE"
        if result:
            quality = ("INTERRUPTED" if result.get("status")=="INTERRUPTED" else
                       read(args.output/"updates/metrics.json")["quality_status"])
        save(args.output/"metrics.json",{"quality_status":quality,
                                        "stage":args.stage,"cell":args.cell})


def preflight(backend,output,identity):
    import torch
    from training import ProjectedAdamW, train_sequence, flat
    from geometry import orthonormalize
    backend.source_replay()
    longest=max(backend.split["history"]+backend.split["diagnostic"]+backend.split["adaptation"],
                key=lambda i:len(backend.encoded[i][0]))
    gradient,metadata=backend.gradient(longest)
    if not torch.isfinite(gradient).all() or not float(gradient.norm())>0:
        raise ValueError("Longest input has no usable finite gradient")
    selected=[i for i in backend.split["adaptation"] if backend.adaptation[i]["base_ce"]>3]
    if not selected:
        raise ValueError("No adaptation input passes the registered CE gate")
    weighted_index=max(selected,key=lambda i:backend.adaptation[i]["base_ce"])
    # A full-sized rank-five numerical probe. This basis is discarded and never
    # used as a scientific pool estimate. Coordinate columns ensure full rank.
    a_indices=tuple(i for i,name in enumerate(backend.names) if "lora_A" in name)
    b_coordinates=torch.ones(gradient.numel(),dtype=torch.bool)
    offset=0
    for index,parameter in enumerate(backend.parameters):
        if index in a_indices:
            b_coordinates[offset:offset+parameter.numel()]=False
        offset+=parameter.numel()
    gradient=gradient.clone()
    gradient[~b_coordinates]=0
    if int(b_coordinates.sum())<5 or not float(gradient.norm())>0:
        raise ValueError("Preflight requires five B coordinates and a nonzero B gradient")
    columns=torch.zeros(gradient.numel(),5,dtype=torch.float32)
    columns[:,0]=gradient/gradient.norm()
    magnitudes=gradient.abs().masked_fill(~b_coordinates,float("inf"))
    coordinates=magnitudes.topk(4,largest=False).indices
    columns[coordinates,torch.arange(1,5)]=1
    basis=orthonormalize(columns).to(backend.device)
    before=backend.initial.clone()
    probes={}
    projected_records=None
    try:
        for arm in ("plain","plain_b","project","matched"):
            mode="plain" if arm=="plain_b" else arm
            count,index=4,weighted_index
            backend.reset()
            lr=1e-6 if backend.cell["dataset"]=="gsm8k_5k" else 5e-5
            controller=ProjectedAdamW(backend.parameters,lr=lr,mode=mode,
                                      basis=basis if mode=="project" else None,
                                      frozen_indices=a_indices if arm!="plain" else ())
            # A four-visit numerical probe must start below the known FP16
            # overflow range. Formal 500-visit arms retain their default scaler.
            scaler=torch.amp.GradScaler("cuda",init_scale=1024.) if backend.use_amp else None
            targets=[r["applied_norm"] for r in projected_records] if mode=="matched" else None
            records=train_sequence(controller,lambda i:backend.loss(i,weighted=True),
                [index]*count,Path(output)/"probe"/arm,identity+":"+arm,
                checkpoint_every=count,scaler=scaler,target_norms=targets)
            probes[arm]={"visits":count,"weighted":True,"source_index":index,
                "actual_updates":sum(r["applied_norm"]>0 for r in records),
                "unmatched":sum(not r["matched"] for r in records),
                "max_projection_residual":max((r["projection_residual"] or 0 for r in records)),
                "max_relative_norm_error":max((r.get("relative_norm_error",0) for r in records)),
                "initial_amp_scale":1024. if backend.use_amp else None}
            if probes[arm]["actual_updates"]<2:
                raise ValueError("Preflight did not make enough finite updates: "+mode)
            if probes[arm]["unmatched"] or probes[arm]["max_projection_residual"]>1e-3:
                raise ValueError("Preflight displacement validation failed: "+mode)
            if mode=="project":
                projected_records=records
            backend.verify_frozen()
            del controller
    finally:
        backend.reset()
    if not torch.equal(before,flat(backend.parameters).cpu()):
        raise ValueError("Preflight failed to restore initial weights")
    save(Path(output)/"metrics.json",{"quality_status":"PREFLIGHT_COMPLETE",
          "identity":identity,
          "longest_source_index":longest,"tokens":len(backend.encoded[longest][0]),
          "backward_scale":metadata["backward_scale"],
          "actual_updates":sum(r["actual_updates"] for r in probes.values()),"probes":probes,
          "peak_memory_mib":torch.cuda.max_memory_allocated()/1024**2 if backend.use_amp else None})


if __name__=="__main__":
    main()
