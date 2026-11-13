import unittest
import torch

from geometry import fit_basis, retention, structure, compare_quality, orthonormalize


class GeometryTests(unittest.TestCase):
    def test_large_parameter_basis_is_orthonormal_in_precise_arithmetic(self):
        torch.set_num_threads(2)
        columns=torch.randn(300000,5,generator=torch.Generator().manual_seed(42))
        basis=orthonormalize(columns)
        gram=basis.double().T@basis.double()
        self.assertTrue(torch.allclose(gram,torch.eye(5,dtype=torch.float64),atol=1e-6,rtol=0))

    def test_basis_matches_exact_known_subspace_and_rejects_rank_shortfall(self):
        gradients = torch.tensor([[3., 0, 0], [2., 0, 0], [0, 1., 0]])
        basis, report = fit_basis(gradients, 2)
        self.assertTrue(torch.allclose(basis.T@basis, torch.eye(2), atol=1e-5))
        self.assertTrue(torch.allclose(basis@basis.T, torch.diag(torch.tensor([1.,1.,0.])), atol=1e-5))
        self.assertEqual(report["rank"], 2)
        with self.assertRaises(ValueError):
            fit_basis(torch.zeros(3, 4), 1)

    def test_retention_separates_zero_and_is_sign_invariant(self):
        basis = torch.tensor([[1.], [0.]])
        self.assertEqual(retention(basis, torch.zeros(2)), None)
        self.assertAlmostEqual(retention(basis, torch.tensor([1.,1.])), 2**-.5, places=6)
        self.assertAlmostEqual(retention(basis, torch.tensor([-1.,-1.])), 2**-.5, places=6)

    def test_structure_excludes_self_pairs_and_zero_vectors(self):
        report = structure(torch.tensor([[1.,0],[-1.,0],[0.,0]]), 1)
        self.assertEqual(report["zero_count"], 1)
        self.assertAlmostEqual(report["mean_cosine"], -1)

    def test_quality_groups_are_reported_with_actual_counts_and_balanced_ids(self):
        ids = list(range(8))
        gradients = {i:torch.tensor([float(i+1), 1.,0]) for i in ids}
        labels = {i:("good" if i<3 else "bad") for i in ids}
        report = compare_quality(ids, gradients, labels, rank=5, seed=42)
        self.assertEqual(report["counts"], {"good":3,"bad":5,"undecided":0})
        self.assertEqual(report["balanced_n"],3)
        self.assertEqual(len(report["balanced_ids"]["bad"]),3)
        self.assertEqual(report, compare_quality(ids,gradients,labels,rank=5,seed=42))


if __name__ == "__main__":
    unittest.main()
