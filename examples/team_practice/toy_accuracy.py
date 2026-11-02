"""Toy accuracy counter for the Issue #6 three-person collaboration practice.

This is a workflow exercise, not a scientific metric. Do not use its output
as evidence for any reproduction or TTA experiment.
"""
import argparse
import json
from pathlib import Path

PREDICTIONS = ["A", "B", "C", "A"]
REFERENCES = ["A", "B", "D", "A"]


def calculate_accuracy(predictions, references):
    """Return correct, total and accuracy for two equally sized answer lists."""
    if not predictions or not references:
        raise ValueError("predictions and references must not be empty")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    total = len(predictions)
    correct = sum(prediction == reference for prediction, reference in zip(predictions, references))
    return {"correct": correct, "total": total, "accuracy": correct / total}


def write_metrics(result, output_dir):
    """Write the toy result and its quality status to output_dir/metrics.json."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    metrics = {
        "correct": result["correct"],
        "total": result["total"],
        "accuracy": result["accuracy"],
        "quality_status": "NOT_APPLICABLE",
        "note": "Toy collaboration exercise; not a scientific metric.",
    }
    path = directory / "metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="directory for metrics.json")
    args = parser.parse_args(argv)
    result = calculate_accuracy(PREDICTIONS, REFERENCES)
    path = write_metrics(result, args.output_dir)
    print(json.dumps(result, indent=2))
    print("wrote " + str(path))


if __name__ == "__main__":
    main()
