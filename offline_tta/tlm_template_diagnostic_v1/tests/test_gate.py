import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnose import require_baseline


class GateTests(unittest.TestCase):
    def good(self):
        return {'status':'COMPLETE','mode':'all','model':'llama3b','steps':200,'original':'/old/run',
                'replay':{'sample_order_exact':True,'learning_rates_exact':True,'step100_adapter_exact':True,
                          'max_loss_difference':0.,'max_base_ce_difference':0.}}

    def test_exact_baseline_required(self):
        require_baseline(self.good(),'llama3b',200,'/old/run')
        for key,value in [('status','FAIL'),('model','qwen3b'),('steps',100),('original','/another/run')]:
            r=self.good();r[key]=value
            with self.assertRaises(ValueError):require_baseline(r,'llama3b',200,'/old/run')

    def test_failed_trajectory_replay_blocks_masked_controls(self):
        for key,value in [('step100_adapter_exact',False),('sample_order_exact',False),
                          ('learning_rates_exact',False),('max_loss_difference',.01)]:
            r=self.good();r['replay'][key]=value
            with self.assertRaises(ValueError):require_baseline(r,'llama3b',200,'/old/run')


if __name__=='__main__':unittest.main()
