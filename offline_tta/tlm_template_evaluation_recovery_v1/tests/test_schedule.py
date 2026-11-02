import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from schedule import rate_at,verify_lr_trace


class ScheduleTests(unittest.TestCase):
    def test_actual_optimizer_skips_explain_lr_trace(self):
        updates=[True,False,True,True,True];steps=0;history=[]
        for update in updates:
            history.append({'lr':rate_at(steps,1.,2,10)})
            steps+=int(update)
        result=verify_lr_trace(history,steps,steps,1.,2,10)
        self.assertEqual(result['skipped_visits'],[2])
        self.assertEqual(result['successful_optimizer_steps'],4)

    def test_unexplained_lr_change_rejected(self):
        with self.assertRaises(ValueError):verify_lr_trace([{'lr':0.},{'lr':.123}],2,2,1.,2,10)

    def test_scheduler_optimizer_counts_must_agree(self):
        with self.assertRaises(ValueError):verify_lr_trace([{'lr':0.},{'lr':.5}],2,1,1.,2,10)

    def test_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):verify_lr_trace([{'lr':math.nan}],1,1,1.,2,10)


if __name__=='__main__':unittest.main()
