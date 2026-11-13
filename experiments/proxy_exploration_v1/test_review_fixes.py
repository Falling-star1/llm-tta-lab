"""Regression checks for the five issues found in the execution review."""
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

import torch

from analysis import read, save
from common import experiment_identity, sha
from entry import preflight
from geometry import compare_quality
from make_requests import generate
from reviews import export_reviews
from test_pipeline import TinyBackend
import test_requests
from test_six_cells import fixture
from training import ProjectedAdamW


class ReviewFixTests(unittest.TestCase):
    def test_cross_group_cosine_distinguishes_alignment_and_opposition(self):
        for direction, expected in (([1.,0.],1.),([0.,1.],0.),([-1.,0.],-1.)):
            gradients={0:torch.tensor([1.,0.]),1:torch.zeros(2),2:torch.tensor(direction)}
            result=compare_quality([0,1,2],gradients,{0:"good",1:"good",2:"bad"})
            self.assertAlmostEqual(result["between"]["full"]["mean_cosine"],expected)
            self.assertEqual(result["between"]["full"]["valid_pairs"],1)

    def test_actual_zero_step_cannot_match_positive_target(self):
        parameter=torch.nn.Parameter(torch.ones(1))
        controller=ProjectedAdamW([parameter],lr=.01,mode="matched")
        parameter.grad=torch.ones(1)
        row=controller.step(target_norm=1e-9)
        self.assertEqual(row["applied_norm"],0)
        self.assertFalse(row["matched"])
        self.assertEqual(row["match_failure"],"rounded_to_zero")

    def test_recovery_requests_cover_all_six_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=fixture(root)
            args=test_requests.RequestTests().arguments(root,config)
            args.stage="updates"
            indices={k:{"outputs":{}} for k in ("preflight","geometry","resume")}
            for cell in config["cells"]:
                name=cell["name"]
                p,g,r=(root/k/name for k in ("preflight","geometry","resume"))
                save(p/"metrics.json",{"quality_status":"PREFLIGHT_COMPLETE",
                     "identity":experiment_identity(config,name,"preflight")})
                save(g/"complete.json",{"identity":experiment_identity(config,name,"gradients")})
                for file in ("low_basis.pt","low_basis.json","random_basis.pt","random_basis.json"):
                    (g/file).write_bytes(b"fixture")
                save(r/"identity.json",{"identity":experiment_identity(config,name,"updates"),"stage":"updates"})
                for arm in ("low","plain","plain_b","random","matched"):
                    checkpoint=r/arm/"checkpoints/visit-00025.pt"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"committed checkpoint fixture")
                    save(r/arm/"checkpoint.json",{"file":"checkpoints/visit-00025.pt",
                         "sha256":sha(checkpoint)})
                    (r/arm/"predictions.jsonl").write_text('{"source_index":1}\n',encoding="utf-8")
                for key,field,path in (("preflight","output",p),("geometry","geometry",g),("resume","updates",r)):
                    indices[key]["outputs"][name]={field:str(path)}
            for key,value in indices.items():
                path=root/(key+".json")
                save(path,value)
                setattr(args,key+"_index",path)
            # Exercise the real validator for GPU inputs and the release snapshot.
            (args.code_root/"SHA256SUMS").write_text("release fingerprint\n",encoding="utf-8")
            (args.code_root/".git").mkdir()
            (args.code_root/".git/ignored.json").write_text("{}",encoding="utf-8")
            module_path=Path(__file__).resolve().parents[2]/"scripts/labctl.py"
            spec=importlib.util.spec_from_file_location("labctl_gpu_validation",module_path)
            lab=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lab)
            site={"python":sys.executable,"code_roots":[str(root)],"input_roots":[str(root)],
                  "gpu_ids":list(range(8))}
            generate(args)
            index=read(args.output/"index.json")
            self.assertEqual(len(index["outputs"]),6)
            for name,item in index["outputs"].items():
                argv=read(item["request"])["argv"]
                self.assertEqual(argv[argv.index("--resume-from")+1],str(root/"resume"/name))
                inputs=read(item["request"])["inputs"]
                lab.validate(read(item["request"]),site)
                for arm in ("low","plain","plain_b","random","matched"):
                    self.assertEqual(inputs[f"recovery_{arm}_2"]["path"],
                                     str(root/"resume"/name/arm/"predictions.jsonl"))
            args.resume_index=None
            for stage in ("preflight","gradients","cell"):
                args.stage=stage
                args.output=root/(stage+"-requests")
                generate(args)
                for item in read(args.output/"index.json")["outputs"].values():
                    self.assertEqual(lab.validate(read(item["request"]),site)["request"]["gpu_count"],1)

    def test_frozen_a_is_preserved_in_plain_and_matched_updates(self):
        for mode in ("plain","matched"):
            a=torch.nn.Parameter(torch.tensor([.5]))
            b=torch.nn.Parameter(torch.zeros(2))
            controller=ProjectedAdamW([a,b],lr=.01,mode=mode,frozen_indices=(0,))
            for step in range(3):
                a.grad=torch.ones_like(a)
                b.grad=torch.tensor([1.,-1.])
                row=controller.step(target_norm=.005 if mode=="matched" else None)
                self.assertTrue(torch.equal(a.detach(),torch.tensor([.5])))
                self.assertGreater(row["applied_norm"],0)
                self.assertTrue(row["matched"])
            self.assertFalse(torch.equal(b.detach(),torch.zeros(2)))

    def test_blind_positions_differ_by_cell_and_reproduce(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=fixture(root)
            keys=[]
            for cell in config["cells"][:2]:
                output=root/cell["name"]
                export_reviews(config,cell["name"],output)
                keys.append(read(output/"private_key.json")["items"])
            self.assertNotEqual([x["ce_group"] for x in keys[0]],[x["ce_group"] for x in keys[1]])
            export_reviews(config,config["cells"][0]["name"],root/"repeat")
            self.assertEqual(keys[0],read(root/"repeat/private_key.json")["items"])

    def test_preflight_exercises_projected_and_matched_weighted_steps(self):
        torch.set_num_threads(1)
        backend=TinyBackend()
        backend.encoded={i:([1,2],{}) for i in range(524)}
        backend.source_replay=backend.reset
        original_loss=backend.loss
        weighted_calls=[]
        def record_loss(index,weighted=False):
            weighted_calls.append(weighted)
            return original_loss(index,weighted)
        backend.loss=record_loss
        with tempfile.TemporaryDirectory() as temp:
            preflight(backend,Path(temp),"probe")
            result=read(Path(temp)/"metrics.json")
            self.assertEqual(set(result["probes"]),{"plain","plain_b","project","matched"})
            self.assertGreater(result["probes"]["project"]["actual_updates"],0)
            self.assertEqual(result["probes"]["matched"]["unmatched"],0)
            # The initial gradient adds one unweighted call before the probes.
            self.assertEqual(weighted_calls,[False]+[True]*16)
            self.assertTrue(torch.equal(backend.initial,backend.parameters[0].detach()))


if __name__=="__main__":
    unittest.main()
