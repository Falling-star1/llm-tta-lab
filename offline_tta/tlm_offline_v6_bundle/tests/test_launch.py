import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class LaunchTests(unittest.TestCase):
    def test_failed_five_can_run_without_restarting_active_gsm8k(self):
        launch=importlib.import_module('launch')
        jobs=launch.jobs_for_gpus('0,1,2,3,4,5',skip=['llama3b__gsm8k_5k'])
        self.assertEqual(len(jobs),5)
        self.assertEqual({job['gpu'] for job in jobs},{0,1,3,4,5})
        self.assertNotIn(('llama3b','gsm8k_5k'),[(j['model'],j['dataset']) for j in jobs])

    def test_unknown_or_all_skipped_tasks_are_rejected(self):
        launch=importlib.import_module('launch')
        with self.assertRaises(ValueError):
            launch.jobs_for_gpus('0,1,2,3,4,5',skip=['llama3b__typo'])
        names=[j['model']+'__'+j['dataset'] for j in launch.jobs_for_gpus('0,1,2,3,4,5')]
        with self.assertRaises(ValueError):
            launch.jobs_for_gpus('0,1,2,3,4,5',skip=names)

    def test_six_jobs_are_distinct_and_use_distinct_physical_gpus(self):
        try:
            launch=importlib.import_module('launch')
        except ImportError:
            self.fail('Parallel launcher is not implemented')
        jobs=launch.jobs_for_gpus('0,1,2,3,4,5')
        self.assertEqual(len(jobs),6)
        self.assertEqual(len({x['gpu'] for x in jobs}),6)
        self.assertEqual(len({(x['model'],x['dataset']) for x in jobs}),6)
        for invalid in ['0,0,1,2,3,4','0,1','0,1,2,3,4,x']:
            with self.assertRaises(ValueError):
                launch.jobs_for_gpus(invalid)


if __name__=='__main__':
    unittest.main()
