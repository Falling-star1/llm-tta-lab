"""Verify this delivery and run the CPU suite; does not use GPUs or download."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libs',type=Path,help='existing optional compatible library folder')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.dont_write_bytecode=True
    if args.libs:
        sys.path.insert(0,str(args.libs.resolve()))
    with tempfile.TemporaryDirectory(prefix='tlm-cpu-check-') as cache:
        os.environ['HF_HOME']=cache
        import torch,transformers,peft
        torch.set_num_threads(1)
        print({'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},flush=True)
        if transformers.__version__!='4.46.1' or peft.__version__!='0.12.0':
            raise ValueError('Use the existing pinned Transformers/PEFT libraries')
        from launch import verify_delivery
        verify_delivery(Path(__file__).resolve().parent)
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':
    main()
