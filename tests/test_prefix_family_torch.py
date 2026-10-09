"""Tests for the Torch prefix-family primitive and its custom backward."""

import unittest
from functools import partial

import numpy as np
import torch

from powersig.torch import compute_prefix_family
from powersig.torch.algorithm import PowerSigTorch
from powersig.torch.static_kernels import linear_kernel, rbf_kernel


CPU = torch.device("cpu")


def _scalar_ref(ps, X, Y):
    return float(ps.compute_gram_matrix(X[None], Y[None], show_progress=False)[0, 0])


class TestPrefixFamilyForward(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def _check(self, ps, Tx, Ty, R, d, *, tol=1e-10):
        rng = np.random.default_rng(0)
        X = torch.tensor(rng.standard_normal((Tx, d)), dtype=torch.float64, device=CPU)
        refs = torch.tensor(rng.standard_normal((R, Ty, d)), dtype=torch.float64, device=CPU)
        out = compute_prefix_family(ps, X, refs, min_prefix_len=2, max_prefix_len=Tx)
        self.assertEqual(tuple(out.shape), (Tx - 1, R))
        for k in range(2, Tx + 1):
            for r in range(R):
                ref_val = _scalar_ref(ps, X[:k], refs[r])
                my_val = float(out[k - 2, r])
                rel = abs(ref_val - my_val) / max(1e-10, abs(ref_val))
                self.assertLess(
                    rel,
                    tol,
                    f"Tx={Tx}, Ty={Ty}, k={k}, r={r}: mine={my_val}, ref={ref_val}, rel_err={rel:.2e}",
                )

    def test_linear_rows_lt_cols(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        self._check(ps, Tx=4, Ty=6, R=3, d=3)

    def test_linear_rows_gt_cols(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        self._check(ps, Tx=7, Ty=4, R=3, d=3)

    def test_linear_square(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        self._check(ps, Tx=5, Ty=5, R=3, d=3)

    def test_rbf_asymmetric(self):
        ps = PowerSigTorch(
            order=5,
            static_kernel=partial(rbf_kernel, bandwidth=0.75),
            device=CPU,
            dtype=torch.float64,
        )
        self._check(ps, Tx=6, Ty=4, R=2, d=2, tol=1e-9)

    def test_single_reference_single_prefix_matches_scalar(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        rng = np.random.default_rng(7)
        X = torch.tensor(rng.standard_normal((5, 3)), dtype=torch.float64, device=CPU)
        Y = torch.tensor(rng.standard_normal((7, 3)), dtype=torch.float64, device=CPU)
        out = compute_prefix_family(ps, X, Y[None], min_prefix_len=5, max_prefix_len=5)
        self.assertEqual(tuple(out.shape), (1, 1))
        ref = _scalar_ref(ps, X, Y)
        self.assertLess(abs(float(out[0, 0]) - ref) / max(1e-10, abs(ref)), 1e-10)

    def test_sub_range(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        rng = np.random.default_rng(2)
        X = torch.tensor(rng.standard_normal((8, 3)), dtype=torch.float64, device=CPU)
        refs = torch.tensor(rng.standard_normal((2, 10, 3)), dtype=torch.float64, device=CPU)
        out = compute_prefix_family(ps, X, refs, min_prefix_len=4, max_prefix_len=7)
        self.assertEqual(tuple(out.shape), (4, 2))
        for k in range(4, 8):
            for r in range(2):
                ref_val = _scalar_ref(ps, X[:k], refs[r])
                self.assertLess(
                    abs(float(out[k - 4, r]) - ref_val) / max(1e-10, abs(ref_val)),
                    1e-10,
                )

    def test_min_prefix_len_below_2_raises(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        X = torch.zeros((4, 2), dtype=torch.float64, device=CPU)
        refs = torch.zeros((1, 4, 2), dtype=torch.float64, device=CPU)
        with self.assertRaises(ValueError):
            compute_prefix_family(ps, X, refs, min_prefix_len=1)

    def test_max_prefix_len_exceeds_T_raises(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        X = torch.zeros((4, 2), dtype=torch.float64, device=CPU)
        refs = torch.zeros((1, 4, 2), dtype=torch.float64, device=CPU)
        with self.assertRaises(ValueError):
            compute_prefix_family(ps, X, refs, max_prefix_len=5)


class TestPrefixFamilyGrad(unittest.TestCase):
    def _naive_prefix_family(self, ps, X, refs, min_prefix_len, max_prefix_len):
        rows = []
        for k in range(min_prefix_len, max_prefix_len + 1):
            row = []
            for r in range(refs.shape[0]):
                row.append(ps.compute_signature_kernel(X[:k], refs[r]))
            rows.append(torch.stack(row))
        return torch.stack(rows)

    def test_grad_wrt_state_path_matches_naive(self):
        ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)
        rng = np.random.default_rng(3)
        X_np = rng.standard_normal((4, 2))
        refs_np = rng.standard_normal((2, 5, 2))
        G = torch.tensor(rng.standard_normal((3, 2)), dtype=torch.float64, device=CPU)

        X_custom = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        refs_custom = torch.tensor(refs_np, dtype=torch.float64, device=CPU, requires_grad=True)
        out_custom = compute_prefix_family(ps, X_custom, refs_custom, min_prefix_len=2, max_prefix_len=4)
        loss_custom = torch.sum(G * out_custom)
        gX_custom, gRefs_custom = torch.autograd.grad(loss_custom, (X_custom, refs_custom))

        X_naive = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        refs_naive = torch.tensor(refs_np, dtype=torch.float64, device=CPU, requires_grad=True)
        out_naive = self._naive_prefix_family(ps, X_naive, refs_naive, 2, 4)
        loss_naive = torch.sum(G * out_naive)
        gX_naive, gRefs_naive = torch.autograd.grad(loss_naive, (X_naive, refs_naive))

        np.testing.assert_allclose(
            gX_custom.detach().cpu().numpy(),
            gX_naive.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            gRefs_custom.detach().cpu().numpy(),
            gRefs_naive.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_grad_wrt_state_path_and_refs_matches_naive_rbf(self):
        ps = PowerSigTorch(
            order=4,
            static_kernel=partial(rbf_kernel, bandwidth=0.8),
            device=CPU,
            dtype=torch.float64,
        )
        rng = np.random.default_rng(4)
        X_np = 0.2 * rng.standard_normal((3, 2))
        refs_np = 0.2 * rng.standard_normal((2, 4, 2))
        G = torch.tensor(rng.standard_normal((2, 2)), dtype=torch.float64, device=CPU)

        X_custom = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        refs_custom = torch.tensor(refs_np, dtype=torch.float64, device=CPU, requires_grad=True)
        out_custom = compute_prefix_family(ps, X_custom, refs_custom, min_prefix_len=2, max_prefix_len=3)
        loss_custom = torch.sum(G * out_custom)
        gX_custom, gRefs_custom = torch.autograd.grad(loss_custom, (X_custom, refs_custom))

        X_naive = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        refs_naive = torch.tensor(refs_np, dtype=torch.float64, device=CPU, requires_grad=True)
        out_naive = self._naive_prefix_family(ps, X_naive, refs_naive, 2, 3)
        loss_naive = torch.sum(G * out_naive)
        gX_naive, gRefs_naive = torch.autograd.grad(loss_naive, (X_naive, refs_naive))

        np.testing.assert_allclose(
            gX_custom.detach().cpu().numpy(),
            gX_naive.detach().cpu().numpy(),
            rtol=2e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            gRefs_custom.detach().cpu().numpy(),
            gRefs_naive.detach().cpu().numpy(),
            rtol=2e-5,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
