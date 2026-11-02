import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coordinator import dispatchable
import coordinator


class CoordinatorTests(unittest.TestCase):
    def jobs(self):
        return [{'key':'llama3b__all','model':'llama3b','mode':'all','state':'WAIT'},
                {'key':'qwen3b__all','model':'qwen3b','mode':'all','state':'WAIT'},
                {'key':'llama3b__both_masked','model':'llama3b','mode':'both_masked','state':'WAIT'}]
    def test_only_baselines_before_success(self):
        self.assertEqual([j['key'] for j in dispatchable(self.jobs())],['llama3b__all','qwen3b__all'])
    def test_masked_job_requires_its_own_completed_baseline(self):
        jobs=self.jobs();jobs[0]['state']='COMPLETE'
        self.assertEqual([j['key'] for j in dispatchable(jobs)],['qwen3b__all','llama3b__both_masked'])
        jobs[0]['state']='FAILED'
        self.assertEqual([j['key'] for j in dispatchable(jobs)],['qwen3b__all'])
    def test_running_or_finished_jobs_not_restarted(self):
        jobs=self.jobs();jobs[0]['state']='ACTIVE';jobs[1]['state']='COMPLETE'
        self.assertEqual(dispatchable(jobs),[])
    def test_duplicate_or_noninteger_gpus_rejected(self):
        for values in [[0,4,7,0],[False,4,7],[0,4,'7']]:
            with self.assertRaises(ValueError):coordinator.validate_safety([{'original':'/old/run'}],values,Path('/new/run'))
    def test_original_output_tree_protected_before_creation(self):
        with self.assertRaises(ValueError):
            coordinator.validate_safety([{'original':'/old/run'}],[0,4,7],Path('/old/run/diagnostic'))
        coordinator.validate_safety([{'original':'/old/run'}],[0,4,7],Path('/new/diagnostic'))
    def test_missing_source_hashes_rejected(self):
        with self.assertRaises(ValueError):coordinator.verify_diagnostic(Path('/missing'),{})


if __name__=='__main__':unittest.main()
