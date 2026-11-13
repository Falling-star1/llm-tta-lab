import tempfile
import unittest
from pathlib import Path
import torch

from collect import collect
from updates import run_updates, seed_resume
from training import flat, assign
from collect import put_artifact
from analysis import save, read
from common import sha


class TinyBackend:
    """Actual small differentiable model for orchestration and recovery checks."""
    def __init__(self):
        self.parameters=[torch.nn.Parameter(torch.arange(6,dtype=torch.float32)*.01)]
        self.names=["layer.lora_B.default.weight"]
        self.initial=flat(self.parameters).clone()
        self.device=torch.device("cpu")
        self.use_amp=False
        self.cell={"dataset":"dolly_5k","metric":"exact_match"}
        self.split={"history":list(range(256)),"diagnostic":list(range(256,512)),
                    "adaptation":list(range(512,520)),"evaluation":list(range(520,524))}
        self.adaptation={i:{"base_ce":3.1+i/1000} for i in range(524)}
        self._source={i:dict(self.generate(i),score=float(self.generate(i)["predict"]==self.reference(i)))
                      for i in self.split["evaluation"]}

    def features(self,i):
        return torch.randn(6,generator=torch.Generator().manual_seed(i+17))

    def reference(self,i):
        return "one" if i%2 else "zero"

    def reset(self):
        assign(self.parameters,self.initial)
        torch.manual_seed(42)

    def loss(self,i,weighted=False):
        target=float(i%2)*2-1
        loss=(self.parameters[0].dot(self.features(i))-target).square()
        if weighted:
            loss=loss*.1
        return loss,{"selected":True,"base_ce":self.adaptation[i]["base_ce"]}

    def gradient(self,i):
        for p in self.parameters:p.grad=None
        loss,_=self.loss(i)
        loss.backward()
        return self.parameters[0].grad.detach().clone(),{"loss":float(loss.detach()),"backward_scale":1.}

    def source(self):
        return self._source

    def references(self):
        return {i:{"output":self.reference(i)} for i in self.split["evaluation"]}

    def generate(self,i):
        return {"predict":"one" if float(self.parameters[0].detach().dot(self.features(i)))>0 else "zero",
                "generated_tokens":1,"hit_generation_limit":False}

    def scorer(self):
        class Scorer:
            def score(self,prediction,reference):return float(prediction==reference)
        return Scorer()

    def verify_frozen(self):
        pass


class PipelineTests(unittest.TestCase):
    def test_a_b_controls_and_cross_run_resume_keep_a_fixed(self):
        class LoRABackend(TinyBackend):
            def __init__(self):
                super().__init__()
                self.parameters=[torch.nn.Parameter(torch.ones(3)*.1),torch.nn.Parameter(torch.zeros(6))]
                self.names=["layer.lora_A.default.weight","layer.lora_B.default.weight"]
                self.initial=flat(self.parameters).clone()
                self._source={i:dict(self.generate(i),score=float(self.generate(i)["predict"]==self.reference(i)))
                              for i in self.split["evaluation"]}

            def value(self,i):
                a,b=self.parameters
                return (1+a.sum())*b.dot(self.features(i))

            def loss(self,i,weighted=False):
                return (self.value(i)-(float(i%2)*2-1)).square()*(.1 if weighted else 1)

            def generate(self,i):
                if len(self.parameters)==1:
                    return super().generate(i)
                return {"predict":"one" if float(self.value(i).detach())>0 else "zero",
                        "generated_tokens":1,"hit_generation_limit":False}

        backend=LoRABackend()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            basis=torch.zeros(9,5)
            basis[3:8,:]=torch.eye(5)
            geometry=root/"geometry"
            for pool in ("low","random"):
                put_artifact(geometry/(pool+"_basis.pt"),{"identity":"g","basis":basis})
            save(geometry/"complete.json",{"identity":"g","low_basis_sha256":sha(geometry/"low_basis.pt"),
                                          "random_basis_sha256":sha(geometry/"random_basis.pt")})
            run_updates(backend,geometry,root/"first","u","g",stop_after=4)
            seed_resume(root/"first",root/"resumed","u")
            completed=run_updates(backend,geometry,root/"resumed","u","g",resume=True)
            direct=run_updates(backend,geometry,root/"direct","u","g")
            self.assertEqual(completed,direct)
            for arm in ("low","random","plain_b","matched"):
                self.assertEqual(read(root/"resumed"/arm/"complete.json")["a_displacement_norm"],0)
            self.assertGreater(read(root/"resumed/plain/complete.json")["a_displacement_norm"],0)
            self.assertIn("plain_b",completed["low_minus"])

    def test_gradient_collection_and_all_five_update_arms_complete_and_resume(self):
        torch.set_num_threads(1)
        backend=TinyBackend()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            collect(backend,root/"gradients","geometry",rank=5)
            # An explicit interrupted update can continue from its saved cursor.
            first=run_updates(backend,root/"gradients",root/"updates","update","geometry",stop_after=4)
            self.assertEqual(first["status"],"INTERRUPTED")
            completed=run_updates(backend,root/"gradients",root/"updates","update","geometry",resume=True)
            self.assertEqual(set(completed["arms"]),{"low","plain","plain_b","random","matched"})
            self.assertEqual(completed["n"],4)
            self.assertTrue(all(x["n"]==8 for x in completed["visits"].values()))
            repeated=run_updates(backend,root/"gradients",root/"updates","update","geometry",resume=True)
            self.assertEqual(completed,repeated)
            seed_resume(root/"updates",root/"new-run","update")
            recovered=run_updates(backend,root/"gradients",root/"new-run","update","geometry",resume=True)
            self.assertEqual(completed,recovered)


if __name__=="__main__":
    unittest.main()
