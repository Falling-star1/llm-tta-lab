import tempfile
import unittest
from pathlib import Path
import torch

from training import ProjectedAdamW, train_sequence, load_checkpoint, linear_schedule


class TrainingTests(unittest.TestCase):
    def test_unconstrained_step_equals_native_adam_and_projected_step_is_in_space(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        q=torch.nn.Parameter(p.detach().clone())
        native=torch.optim.AdamW([p],lr=.01,weight_decay=0)
        modified=ProjectedAdamW([q],lr=.01,mode="plain")
        p.grad=torch.tensor([2.,-3.]);q.grad=p.grad.clone()
        native.step();modified.step()
        self.assertTrue(torch.equal(p,q))
        projected=ProjectedAdamW([q],lr=.01,mode="project",
                                basis=torch.tensor([[1.],[0.]]))
        before=q.detach().clone();q.grad=torch.tensor([2.,-3.])
        row=projected.step()
        self.assertEqual(float(q[1].detach()),float(before[1]))
        self.assertLess(row["projection_residual"],1e-4)

    def test_matched_step_records_unreachable_zero_direction(self):
        p=torch.nn.Parameter(torch.zeros(2))
        optimizer=ProjectedAdamW([p],lr=.1,mode="matched")
        p.grad=torch.zeros(2)
        row=optimizer.step(target_norm=.1)
        self.assertFalse(row["matched"])
        self.assertEqual(row["applied_norm"],0)
        p.grad=torch.tensor([1.,2.])
        row=optimizer.step(target_norm=.02)
        self.assertAlmostEqual(row["applied_norm"],.02,places=6)
        self.assertTrue(row["matched"])

    def test_real_optimization_disk_resume_equals_uninterrupted(self):
        torch.manual_seed(42)
        def create():
            torch.manual_seed(42)
            p=torch.nn.Parameter(torch.tensor([.2,.4]))
            optimizer=ProjectedAdamW([p],lr=.01,mode="plain")
            def loss(i):
                target=torch.tensor([float(i%3),.1])+torch.randn_like(p)*.01
                return ((p-target)**2).sum()
            return p,optimizer,loss
        with tempfile.TemporaryDirectory() as temp:
            p,opt,loss=create()
            full=train_sequence(opt,loss,list(range(8)),Path(temp)/"full",identity="fixed",checkpoint_every=2)
            q,partial,loss2=create()
            train_sequence(partial,loss2,list(range(8)),Path(temp)/"part",identity="fixed",
                           checkpoint_every=2,stop_after=4)
            r,resumed,loss3=create()
            state=load_checkpoint(Path(temp)/"part/checkpoint.json","fixed")
            continued=train_sequence(resumed,loss3,list(range(8)),Path(temp)/"continued",
                                     identity="fixed",checkpoint_every=2,resume=state)
            self.assertTrue(torch.equal(p,r))
            self.assertEqual(full,continued)
            self.assertEqual(resumed.optimizer.state_dict()["param_groups"],opt.optimizer.state_dict()["param_groups"])
            with self.assertRaises(ValueError):
                load_checkpoint(Path(temp)/"part/checkpoint.json","other")
            import json
            manifest=json.loads((Path(temp)/"part/checkpoint.json").read_text())
            path=Path(temp)/"part"/manifest["file"]
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                load_checkpoint(Path(temp)/"part/checkpoint.json","fixed")


if __name__ == "__main__":
    unittest.main()
