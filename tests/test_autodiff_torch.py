"""Tests for the Torch checkpointed autodiff entry points."""

import unittest

import numpy as np
import torch

from powersig.torch import (
    compute_gram_fast_diff,
    compute_prefix_family,
    compute_prefix_family_fast_diff,
    compute_sig_kernel_fast_diff,
)
from powersig.torch.algorithm import PowerSigTorch
from powersig.torch.static_kernels import linear_kernel


CPU = torch.device("cpu")


class TestFastDiffTorch(unittest.TestCase):
    def setUp(self):
        self.ps = PowerSigTorch(order=4, static_kernel=linear_kernel, device=CPU, dtype=torch.float64)

    def test_single_pair_forward_matches_native(self):
        rng = np.random.default_rng(0)
        X = torch.tensor(rng.standard_normal((8, 2)), dtype=torch.float64, device=CPU)
        Y = torch.tensor(rng.standard_normal((7, 2)), dtype=torch.float64, device=CPU)
        native = self.ps.compute_signature_kernel(X, Y)
        fast = compute_sig_kernel_fast_diff(self.ps, X, Y)
        np.testing.assert_allclose(float(native), float(fast), rtol=1e-10, atol=1e-12)

    def test_single_pair_grad_matches_native(self):
        rng = np.random.default_rng(1)
        X_np = rng.standard_normal((6, 2))
        Y_np = rng.standard_normal((5, 2))

        X_fast = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        Y_fast = torch.tensor(Y_np, dtype=torch.float64, device=CPU, requires_grad=True)
        fast = compute_sig_kernel_fast_diff(self.ps, X_fast, Y_fast)
        gX_fast, gY_fast = torch.autograd.grad(fast, (X_fast, Y_fast))

        X_native = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        Y_native = torch.tensor(Y_np, dtype=torch.float64, device=CPU, requires_grad=True)
        native = self.ps.compute_signature_kernel(X_native, Y_native)
        gX_native, gY_native = torch.autograd.grad(native, (X_native, Y_native))

        np.testing.assert_allclose(
            gX_fast.detach().cpu().numpy(),
            gX_native.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            gY_fast.detach().cpu().numpy(),
            gY_native.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_gram_fast_diff_matches_native(self):
        rng = np.random.default_rng(2)
        X = torch.tensor(rng.standard_normal((3, 7, 2)), dtype=torch.float64, device=CPU)
        Y = torch.tensor(rng.standard_normal((2, 6, 2)), dtype=torch.float64, device=CPU)
        native = self.ps.compute_gram_matrix(X, Y, show_progress=False)
        fast = compute_gram_fast_diff(self.ps, X, Y, show_progress=False)
        np.testing.assert_allclose(native.detach().cpu().numpy(), fast.detach().cpu().numpy(), rtol=1e-10, atol=1e-12)

    def test_checkpoint_intervals_give_same_grads(self):
        rng = np.random.default_rng(3)
        X_np = rng.standard_normal((9, 2))
        Y_np = rng.standard_normal((8, 2))

        grads = []
        for ckpt in [1, 2, 3, 5]:
            X = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
            Y = torch.tensor(Y_np, dtype=torch.float64, device=CPU, requires_grad=True)
            out = compute_sig_kernel_fast_diff(self.ps, X, Y, checkpoint_interval=ckpt)
            gX, gY = torch.autograd.grad(out, (X, Y))
            grads.append((gX.detach().cpu().numpy(), gY.detach().cpu().numpy()))

        ref_gX, ref_gY = grads[0]
        for gX, gY in grads[1:]:
            np.testing.assert_allclose(gX, ref_gX, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(gY, ref_gY, rtol=1e-5, atol=1e-8)

    def test_longer_sequence_grad_is_finite(self):
        rng = np.random.default_rng(4)
        X = torch.tensor(rng.standard_normal((96, 2)), dtype=torch.float64, device=CPU, requires_grad=True)
        Y = torch.tensor(rng.standard_normal((80, 2)), dtype=torch.float64, device=CPU, requires_grad=True)
        out = compute_sig_kernel_fast_diff(self.ps, X, Y)
        gX, gY = torch.autograd.grad(out, (X, Y))
        self.assertTrue(torch.isfinite(gX).all())
        self.assertTrue(torch.isfinite(gY).all())

    def test_prefix_family_fast_diff_alias_matches_prefix_family(self):
        rng = np.random.default_rng(5)
        X = torch.tensor(rng.standard_normal((7, 2)), dtype=torch.float64, device=CPU, requires_grad=True)
        refs = torch.tensor(rng.standard_normal((2, 6, 2)), dtype=torch.float64, device=CPU, requires_grad=True)
        G = torch.tensor(rng.standard_normal((6, 2)), dtype=torch.float64, device=CPU)

        out_naive = compute_prefix_family(self.ps, X, refs, min_prefix_len=2, max_prefix_len=7)
        out_fast = compute_prefix_family_fast_diff(self.ps, X, refs, min_prefix_len=2, max_prefix_len=7)
        np.testing.assert_allclose(out_naive.detach().cpu().numpy(), out_fast.detach().cpu().numpy(), rtol=1e-10, atol=1e-12)

        loss_naive = torch.sum(G * out_naive)
        gX_naive, gRefs_naive = torch.autograd.grad(loss_naive, (X, refs), retain_graph=True)
        loss_fast = torch.sum(G * out_fast)
        gX_fast, gRefs_fast = torch.autograd.grad(loss_fast, (X, refs))

        np.testing.assert_allclose(
            gX_fast.detach().cpu().numpy(),
            gX_naive.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            gRefs_fast.detach().cpu().numpy(),
            gRefs_naive.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
