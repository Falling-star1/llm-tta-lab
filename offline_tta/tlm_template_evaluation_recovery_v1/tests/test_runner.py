import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner import recognized_lr_rejection


class RunnerTests(unittest.TestCase):
    def test_only_known_lr_rejection_is_recoverable(self):
        error={'type':'ValueError','error':"Original trajectory controls differ: {'learning_rates_exact': False}"}
        self.assertTrue(recognized_lr_rejection(error))
        self.assertFalse(recognized_lr_rejection({'type':'FloatingPointError','error':'Nonfinite adapter'}))
        self.assertFalse(recognized_lr_rejection({'type':'ValueError','error':'Input identity changed'}))
        self.assertFalse(recognized_lr_rejection({}))


if __name__=='__main__':unittest.main()
