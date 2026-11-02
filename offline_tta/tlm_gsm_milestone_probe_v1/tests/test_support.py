import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from support import probe_indices, reasoning_stub_only, milestone_path


class SupportTests(unittest.TestCase):
    def test_same_probe_indices_for_every_milestone(self):
        a=probe_indices();self.assertEqual(a[:20],list(range(20)))
        self.assertEqual(a,probe_indices());self.assertEqual(len(set(a)),100)

    def test_reasoning_stub_requires_no_actual_answer(self):
        self.assertTrue(reasoning_stub_only("Let's think step by step."))
        self.assertTrue(reasoning_stub_only("### Response:\nLet's think step by step.\n"))
        self.assertFalse(reasoning_stub_only("Let's think step by step. The answer is 4."))
        self.assertFalse(reasoning_stub_only(''))

    def test_registered_checkpoint_paths_only(self):
        root=Path('/original/run')
        self.assertEqual(milestone_path(root,5000),root/'final_adapter/adapter_model.safetensors')
        self.assertEqual(milestone_path(root,1000),root/'trainer/milestones/step-1000/adapter_model.safetensors')
        with self.assertRaises(ValueError):milestone_path(root,200)


if __name__=='__main__':unittest.main()
