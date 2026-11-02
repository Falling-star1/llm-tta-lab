from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery import require_canonical_base,EXPECTED_BASE


class RecoveryTests(unittest.TestCase):
    def test_accepts_independently_verified_base(self):
        require_canonical_base({'contract':{'frozen_sha256':EXPECTED_BASE}})
        require_canonical_base({'frozen_sha256':EXPECTED_BASE})

    def test_rejects_historical_noncanonical_base_even_with_pass(self):
        with self.assertRaisesRegex(ValueError,'canonical'):
            require_canonical_base({'status':'PASS','contract':{'frozen_sha256':
                '82148c27a9da9d5f0c72dc886a282e0ce39e302b141077cb6e76c4f7dd2db371'}})

    def test_missing_base_identity_does_not_pass(self):
        with self.assertRaises(ValueError):require_canonical_base({'status':'PASS'})


if __name__=='__main__':unittest.main()
