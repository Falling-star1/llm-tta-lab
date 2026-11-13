import argparse
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from analysis import read, save
from common import experiment_identity
from make_requests import generate
from test_six_cells import fixture


class RequestTests(unittest.TestCase):
    def arguments(self, root, config):
        code=root/"code"
        code.mkdir()
        (code/"entry.py").write_text("pass\n",encoding="utf-8")
        path=root/"config.json"
        save(path,config)
        return argparse.Namespace(stage="analyze",config=path,code_root=code,
            output=root/"requests",lab_root=root/"lab",commit="1"*40,prefix="test-six",
            owner="test",timeout=600,preflight_index=None,geometry_index=None,resume_index=None)

    def test_generated_cpu_request_is_accepted_by_existing_labctl_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=fixture(root)
            args=self.arguments(root,config)
            generate(args)
            request=read(next(p for p in args.output.glob("*.json") if p.name!="index.json"))
            module_path=Path(__file__).resolve().parents[2]/"scripts/labctl.py"
            if not module_path.exists():
                module_path=Path("/data/chuwenjing/lab/bin/labctl.py")
            if not module_path.exists():
                self.skipTest("Existing labctl validator is unavailable in this test environment")
            spec=importlib.util.spec_from_file_location("existing_labctl",module_path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            import sys
            site={"python":sys.executable,"code_roots":[str(root)],"input_roots":[str(root)],
                  "gpu_ids":list(range(8))}
            validated=module.validate(request,site)
            self.assertEqual(validated["request"]["gpu_count"],0)

    def test_gpu_requests_require_matching_preflight_before_any_files_are_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=fixture(root)
            args=self.arguments(root,config)
            args.stage="cell"
            with self.assertRaises(ValueError):
                generate(args)
            self.assertFalse(args.output.exists())


if __name__=="__main__":
    unittest.main()
