import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "examples/team_practice/toy_accuracy.py"
spec = importlib.util.spec_from_file_location("toy_accuracy", MODULE)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


class ToyAccuracyTest(unittest.TestCase):
    def test_fixed_sample_returns_three_of_four(self):
        result = toy.calculate_accuracy(["A", "B", "C", "A"], ["A", "B", "D", "A"])
        self.assertEqual((result["correct"], result["total"], result["accuracy"]), (3, 4, 0.75))

    def test_empty_inputs_raise_value_error(self):
        for predictions, references in [([], []), ([], ["A"]), (["A"], [])]:
            with self.assertRaises(ValueError):
                toy.calculate_accuracy(predictions, references)

    def test_length_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            toy.calculate_accuracy(["A"], ["A", "B"])

    def test_command_line_writes_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            process = subprocess.run(
                [sys.executable, str(MODULE), "--output-dir", str(output_dir)],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual((metrics["correct"], metrics["total"], metrics["accuracy"]), (3, 4, 0.75))
            self.assertEqual(metrics["quality_status"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
