"""CPU-compatible tests for the core Torch signature-kernel implementation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import value_and_grad

from powersig.jax import static_kernels as jax_static_kernels
from powersig.jax.algorithm import PowerSigJax
from powersig.torch import static_kernels as torch_static_kernels
from powersig.torch.algorithm import (
    PowerSigTorch,
    _round_to_power_of_2,
    build_stencil,
    compute_block_size,
    compute_vandermonde_vectors,
    estimate_bytes_per_pair,
    get_available_gpu_memory,
    get_diagonal_range_bool,
)
from powersig.util.grid import get_diagonal_range as get_diagonal_range_py


CPU = torch.device("cpu")
JAX_CPU = jax.devices("cpu")[0]


class TestBuildStencil(unittest.TestCase):
    def setUp(self):
        self.order = 4
        self.dtype = torch.float64

    def test_shape(self):
        stencil = build_stencil(self.order, self.dtype)
        self.assertEqual(stencil.shape, (self.order, self.order))

    def test_first_row_and_column_are_ones(self):
        stencil = build_stencil(self.order, self.dtype)
        np.testing.assert_allclose(stencil[0].cpu().numpy(), np.ones(self.order))
        np.testing.assert_allclose(stencil[:, 0].cpu().numpy(), np.ones(self.order))

    def test_known_values(self):
        stencil = build_stencil(self.order, self.dtype)
        expected = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.5, 1.0 / 3.0],
                [1.0, 0.5, 0.25, 1.0 / 12.0],
                [1.0, 1.0 / 3.0, 1.0 / 12.0, 1.0 / 36.0],
            ]
        )
        np.testing.assert_allclose(stencil.cpu().numpy(), expected, rtol=1e-10)


class TestVandermondeVectors(unittest.TestCase):
    def test_unit_step(self):
        v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, 4, torch.float64)
        np.testing.assert_allclose(v_s.cpu().numpy(), np.ones(4))
        np.testing.assert_allclose(v_t.cpu().numpy(), np.ones(4))

    def test_power_scaling(self):
        v_s, v_t = compute_vandermonde_vectors(0.5, 0.25, 4, torch.float64)
        np.testing.assert_allclose(v_s.cpu().numpy(), [1.0, 0.5, 0.25, 0.125], rtol=1e-10)
        np.testing.assert_allclose(
            v_t.cpu().numpy(), [1.0, 0.25, 0.0625, 0.015625], rtol=1e-10
        )


class TestDiagonalRange(unittest.TestCase):
    def test_boolean_geometry_matches_python_reference(self):
        for rows, cols in [(3, 2), (2, 4), (5, 5)]:
            for d in range(rows + cols - 1):
                expected = get_diagonal_range_py(d, rows, cols)
                self.assertEqual(get_diagonal_range_bool(d, rows, cols), expected)


class TestBlockSizeUtils(unittest.TestCase):
    def test_round_to_power_of_2(self):
        self.assertEqual(_round_to_power_of_2(1), 1)
        self.assertEqual(_round_to_power_of_2(2), 2)
        self.assertEqual(_round_to_power_of_2(3), 4)
        self.assertEqual(_round_to_power_of_2(5), 8)
        self.assertEqual(_round_to_power_of_2(16), 16)
        self.assertEqual(_round_to_power_of_2(17), 32)

    def test_estimate_bytes_per_pair(self):
        bpp = estimate_bytes_per_pair(100, 32, torch.float64)
        self.assertEqual(bpp, 2_636_800)

    def test_compute_block_size_bounded(self):
        bs = compute_block_size(100, 32, torch.float64, CPU, 1000)
        self.assertGreaterEqual(bs, 1)
        self.assertLessEqual(bs, 256)
        self.assertEqual(bs & (bs - 1), 0)

    def test_cpu_memory_budget_is_positive(self):
        self.assertGreater(get_available_gpu_memory(CPU), 0)


class TestGramMatrix(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.ps = PowerSigTorch(order=8, device=CPU, dtype=torch.float64)
        self.ps_jax = PowerSigJax(order=8, device=JAX_CPU, dtype=jnp.float64)
        self.X = torch.tensor(rng.normal(size=(4, 10, 3)), dtype=torch.float64, device=CPU)
        self.Y = torch.tensor(rng.normal(size=(4, 10, 3)), dtype=torch.float64, device=CPU)

    def test_block_size_1_matches_auto(self):
        gram_seq = self.ps.compute_gram_matrix(self.X, self.Y, block_size=1, show_progress=False)
        gram_auto = self.ps.compute_gram_matrix(self.X, self.Y, show_progress=False)
        np.testing.assert_allclose(gram_seq.cpu().numpy(), gram_auto.cpu().numpy(), rtol=1e-10)

    def test_explicit_block_sizes_match(self):
        gram_1 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=1, show_progress=False)
        gram_4 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=4, show_progress=False)
        gram_16 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=16, show_progress=False)
        np.testing.assert_allclose(gram_1.cpu().numpy(), gram_4.cpu().numpy(), rtol=1e-10)
        np.testing.assert_allclose(gram_1.cpu().numpy(), gram_16.cpu().numpy(), rtol=1e-10)

    def test_symmetric(self):
        gram = self.ps.compute_gram_matrix(self.X, self.X, symmetric=True, show_progress=False)
        np.testing.assert_allclose(gram.cpu().numpy(), gram.t().cpu().numpy(), rtol=1e-10)

    def test_symmetric_matches_full(self):
        gram_full = self.ps.compute_gram_matrix(self.X, self.X, symmetric=False, show_progress=False)
        gram_sym = self.ps.compute_gram_matrix(self.X, self.X, symmetric=True, show_progress=False)
        np.testing.assert_allclose(gram_full.cpu().numpy(), gram_sym.cpu().numpy(), rtol=1e-10)

    def test_single_entry_matches_gram(self):
        gram = self.ps.compute_gram_matrix(self.X, self.Y, show_progress=False)
        for i in range(2):
            for j in range(2):
                single = self.ps.compute_signature_kernel(self.X[i], self.Y[j])
                np.testing.assert_allclose(
                    float(gram[i, j]), float(single), rtol=1e-6, err_msg=f"Mismatch at ({i}, {j})"
                )

    def test_call_interface(self):
        gram_method = self.ps.compute_gram_matrix(self.X, self.Y, show_progress=False)
        gram_call = self.ps(self.X, self.Y, show_progress=False)
        np.testing.assert_allclose(gram_method.cpu().numpy(), gram_call.cpu().numpy(), rtol=1e-10)

    def test_call_with_block_size(self):
        gram = self.ps(self.X, self.Y, block_size=2, show_progress=False)
        gram_ref = self.ps(self.X, self.Y, block_size=1, show_progress=False)
        np.testing.assert_allclose(gram.cpu().numpy(), gram_ref.cpu().numpy(), rtol=1e-10)

    def test_boolean_and_minimum_forward_geometry_match(self):
        min_val = self.ps.compute_signature_kernel(self.X[0], self.Y[0])
        bool_val = self.ps.compute_signature_kernel_bool_geometry(self.X[0], self.Y[0])
        np.testing.assert_allclose(float(min_val), float(bool_val), rtol=1e-10, atol=1e-12)

    def test_chunked_matches_forward(self):
        direct = self.ps.compute_signature_kernel(self.X[0], self.Y[0])
        chunked = self.ps.compute_signature_kernel_chunked(self.X[0], self.Y[0])
        np.testing.assert_allclose(float(direct), float(chunked), rtol=1e-10, atol=1e-12)

    def test_asymmetric_lengths_transpose_symmetry_small(self):
        rng = np.random.default_rng(7)
        X = torch.tensor(rng.normal(size=(1, 4, 2)), dtype=torch.float64, device=CPU)
        Y = torch.tensor(rng.normal(size=(1, 3, 2)), dtype=torch.float64, device=CPU)
        xy = self.ps.compute_gram_matrix(X, Y, show_progress=False)
        yx = self.ps.compute_gram_matrix(Y, X, show_progress=False)
        np.testing.assert_allclose(xy.cpu().numpy(), yx.t().cpu().numpy(), rtol=1e-10, atol=1e-12)

    def test_asymmetric_lengths_transpose_symmetry_longer(self):
        rng = np.random.default_rng(17)
        X = torch.tensor(0.1 * rng.normal(size=(1, 80, 2)), dtype=torch.float64, device=CPU)
        Y = torch.tensor(0.1 * rng.normal(size=(1, 70, 2)), dtype=torch.float64, device=CPU)
        xy = self.ps.compute_gram_matrix(X, Y, show_progress=False)
        yx = self.ps.compute_gram_matrix(Y, X, show_progress=False)
        np.testing.assert_allclose(xy.cpu().numpy(), yx.t().cpu().numpy(), rtol=1e-10, atol=1e-12)

    def test_linear_kernel_matches_jax_reference(self):
        X_np = np.asarray(self.X[:2].cpu().numpy())
        Y_np = np.asarray(self.Y[:2].cpu().numpy())
        torch_gram = self.ps.compute_gram_matrix(X_np, Y_np, show_progress=False).cpu().numpy()
        jax_gram = np.asarray(
            self.ps_jax.compute_gram_matrix(
                jnp.asarray(X_np, dtype=jnp.float64),
                jnp.asarray(Y_np, dtype=jnp.float64),
            )
        )
        np.testing.assert_allclose(torch_gram, jax_gram, rtol=1e-9, atol=1e-11)

    def test_rbf_kernel_matches_jax_reference(self):
        rng = np.random.default_rng(99)
        X_np = rng.normal(size=(2, 8, 2))
        Y_np = rng.normal(size=(2, 7, 2))

        torch_ps = PowerSigTorch(
            order=6, static_kernel=torch_static_kernels.rbf_kernel, device=CPU, dtype=torch.float64
        )
        jax_ps = PowerSigJax(
            order=6, static_kernel=jax_static_kernels.rbf_kernel, device=JAX_CPU, dtype=jnp.float64
        )

        torch_gram = torch_ps.compute_gram_matrix(X_np, Y_np, show_progress=False).cpu().numpy()
        jax_gram = np.asarray(
            jax_ps.compute_gram_matrix(
                jnp.asarray(X_np, dtype=jnp.float64),
                jnp.asarray(Y_np, dtype=jnp.float64),
            )
        )
        np.testing.assert_allclose(torch_gram, jax_gram, rtol=1e-8, atol=1e-10)


class TestAutograd(unittest.TestCase):
    def test_linear_gradient_matches_jax(self):
        rng = np.random.default_rng(123)
        X_np = rng.normal(size=(8, 2))
        Y_np = rng.normal(size=(7, 2))

        torch_ps = PowerSigTorch(order=4, device=CPU, dtype=torch.float64)
        X_t = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        Y_t = torch.tensor(Y_np, dtype=torch.float64, device=CPU, requires_grad=True)
        torch_val = torch_ps.compute_signature_kernel(X_t, Y_t)
        torch_grad_x, torch_grad_y = torch.autograd.grad(torch_val, (X_t, Y_t))

        jax_ps = PowerSigJax(order=4, device=JAX_CPU, dtype=jnp.float64)

        def jax_fn(x, y):
            return jax_ps.compute_signature_kernel(x, y)

        jax_val, (jax_grad_x, jax_grad_y) = value_and_grad(jax_fn, argnums=(0, 1))(
            jnp.asarray(X_np, dtype=jnp.float64), jnp.asarray(Y_np, dtype=jnp.float64)
        )

        np.testing.assert_allclose(float(torch_val.detach().cpu()), float(jax_val), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            torch_grad_x.detach().cpu().numpy(),
            np.asarray(jax_grad_x),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            torch_grad_y.detach().cpu().numpy(),
            np.asarray(jax_grad_y),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_rbf_gradient_matches_jax(self):
        rng = np.random.default_rng(321)
        X_np = 0.2 * rng.normal(size=(6, 2))
        Y_np = 0.2 * rng.normal(size=(5, 2))

        torch_ps = PowerSigTorch(
            order=4, static_kernel=torch_static_kernels.rbf_kernel, device=CPU, dtype=torch.float64
        )
        X_t = torch.tensor(X_np, dtype=torch.float64, device=CPU, requires_grad=True)
        Y_t = torch.tensor(Y_np, dtype=torch.float64, device=CPU, requires_grad=True)
        torch_val = torch_ps.compute_signature_kernel(X_t, Y_t)
        torch_grad_x, torch_grad_y = torch.autograd.grad(torch_val, (X_t, Y_t))

        jax_ps = PowerSigJax(
            order=4, static_kernel=jax_static_kernels.rbf_kernel, device=JAX_CPU, dtype=jnp.float64
        )

        def jax_fn(x, y):
            return jax_ps.compute_signature_kernel(x, y)

        jax_val, (jax_grad_x, jax_grad_y) = value_and_grad(jax_fn, argnums=(0, 1))(
            jnp.asarray(X_np, dtype=jnp.float64), jnp.asarray(Y_np, dtype=jnp.float64)
        )

        np.testing.assert_allclose(float(torch_val.detach().cpu()), float(jax_val), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            torch_grad_x.detach().cpu().numpy(),
            np.asarray(jax_grad_x),
            rtol=2e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            torch_grad_y.detach().cpu().numpy(),
            np.asarray(jax_grad_y),
            rtol=2e-5,
            atol=1e-8,
        )


@unittest.skipUnless(torch.cuda.is_available(), "compile_forward only applies on CUDA")
class TestCompiledForward(unittest.TestCase):
    """compile_forward=True must be as correct as it is fast.

    It routes the sweep through torch.compile(mode="reduce-overhead"), which
    replays a CUDA graph writing into a fixed output buffer. The hazard is that
    a result a caller is still holding gets overwritten by their next call.
    """

    N_PTS, ORDER = 65, 8

    def setUp(self):
        self.dev = torch.device("cuda")
        self.eager = PowerSigTorch(order=self.ORDER, device=self.dev,
                                   dtype=torch.float64, compile_forward=False)
        self.compiled = PowerSigTorch(order=self.ORDER, device=self.dev,
                                      dtype=torch.float64, compile_forward=True)
        rng = np.random.default_rng(0)
        self.pairs = [
            (torch.tensor(0.1 * rng.normal(size=(self.N_PTS, 2)).cumsum(0), device=self.dev),
             torch.tensor(0.1 * rng.normal(size=(self.N_PTS, 2)).cumsum(0), device=self.dev))
            for _ in range(6)
        ]

    def test_compiled_matches_eager(self):
        for i, (X, Y) in enumerate(self.pairs):
            np.testing.assert_allclose(
                float(self.compiled.compute_signature_kernel(X, Y)),
                float(self.eager.compute_signature_kernel(X, Y)),
                rtol=1e-9, atol=1e-11, err_msg=f"pair {i}",
            )

    def test_held_results_survive_later_calls(self):
        """Regression: results used to alias the CUDA-graph output buffer, so
        collecting them in a list yielded the last value repeated."""
        held = [self.compiled.compute_signature_kernel(X, Y) for X, Y in self.pairs[:3]]
        before = [float(h) for h in held]
        for X, Y in self.pairs[3:]:
            self.compiled.compute_signature_kernel(X, Y)
        after = [float(h) for h in held]
        self.assertEqual(before, after, "held results were overwritten by later calls")

        expected = [float(self.eager.compute_signature_kernel(X, Y))
                    for X, Y in self.pairs[:3]]
        np.testing.assert_allclose(after, expected, rtol=1e-9, atol=1e-11)

    def test_gram_matrix_matches_eager(self):
        rng = np.random.default_rng(1)
        X = torch.tensor(0.1 * rng.normal(size=(5, self.N_PTS, 2)).cumsum(1), device=self.dev)
        np.testing.assert_allclose(
            self.compiled.compute_gram_matrix(X, show_progress=False).cpu().numpy(),
            self.eager.compute_gram_matrix(X, show_progress=False).cpu().numpy(),
            rtol=1e-9, atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
