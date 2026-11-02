"""Verify both frozen deliveries and run CPU tests without loading real models."""
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    here = Path(__file__).resolve().parent
    v3 = here.parent / 'tlm_offline_v3_bundle'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(here))
    from diagnose import verify_delivery, V3_CODE_ID
    verify_delivery(here)
    verify_delivery(v3)
    sys.path.insert(0, str(v3))
    from runtime import code_identity
    if code_identity() != V3_CODE_ID:
        raise ValueError('Use the unchanged pinned v3 bundle')
    with tempfile.TemporaryDirectory(prefix='tlm-diagnostic-check-') as cache:
        os.environ['HF_HOME'] = cache
        import torch
        import transformers
        import peft
        import accelerate
        if (transformers.__version__, peft.__version__, accelerate.__version__) != ('4.46.1', '0.12.0', '1.0.1'):
            raise ValueError('Use the existing pinned Transformers/PEFT/Accelerate environment')
        print({'torch': torch.__version__, 'transformers': transformers.__version__,
               'peft': peft.__version__, 'accelerate': accelerate.__version__}, flush=True)
        torch.set_num_threads(1)
        suites = [unittest.TestLoader().discover(str(path)) for path in (v3 / 'tests', here / 'tests')]
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
        return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
