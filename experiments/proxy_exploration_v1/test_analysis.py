import unittest

from analysis import quartiles, score_mean, validate_split


class AnalysisTests(unittest.TestCase):
    def test_quartiles_use_model_ce_and_stable_index_ties(self):
        ids = list(range(256))
        ce = {i: float(i // 2) for i in ids}
        groups = quartiles(list(reversed(ids)), ce)
        self.assertEqual(groups[0], list(range(64)))
        self.assertEqual(groups[-1], list(range(192, 256)))
        self.assertEqual(quartiles(ids, {i: 128-ce[i] for i in ids})[0][:4], [254, 255, 252, 253])
        with self.assertRaises(ValueError):
            quartiles(ids, {i: -1.0 for i in ids})

    def test_metric_scale_and_invalid_scores(self):
        self.assertEqual(score_mean([0.0, 1.0]), 50.0)
        self.assertAlmostEqual(score_mean([.3, .4]), 35.0)
        with self.assertRaises(ValueError):
            score_mean([float("nan")])
        with self.assertRaises(ValueError):
            score_mean([2.0])

    def test_evaluation_cannot_overlap_history(self):
        split = {"history":list(range(256)), "evaluation":list(range(256, 556))}
        validate_split(split)
        split["evaluation"][0] = 0
        with self.assertRaises(ValueError):
            validate_split(split)


if __name__ == "__main__":
    unittest.main()
