"""CPU regression checks for frozen v3, first diagnostic and scaled probes."""
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    here = Path(__file__).resolve().parent
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(here))
    from diagnose_scaled import verify_dependencies
    from probe_support import V3, DIAG
    verify_dependencies()
    with tempfile.TemporaryDirectory(prefix='tlm-scaled-check-') as cache:
        os.environ['HF_HOME'] = cache
        import torch, transformers, peft, accelerate
        if (transformers.__version__, peft.__version__, accelerate.__version__) != ('4.46.1', '0.12.0', '1.0.1'):
            raise ValueError('Use pinned existing libraries')
        print({'torch': torch.__version__, 'transformers': transformers.__version__,
               'peft': peft.__version__, 'accelerate': accelerate.__version__}, flush=True)
        torch.set_num_threads(1)
        suites = [unittest.TestLoader().discover(str(root / 'tests')) for root in (V3, DIAG, here)]
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
