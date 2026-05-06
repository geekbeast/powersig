"""CPU-compatible tests for the core JAX signature kernel implementation.

These tests are designed to run without GPU, torch, cupy, ksig, or sigkernel
dependencies, making them suitable for CI on free-tier GitHub Actions runners.
"""

import unittest
from math import ceil, sqrt

import jax
import jax.numpy as jnp
import numpy as np

from powersig.jax.algorithm import (
    PowerSigJax,
    batch_ADM_for_diagonal,
    build_stencil,
    compute_block_size,
    compute_vandermonde_vectors,
    estimate_bytes_per_pair,
    get_available_gpu_memory,
    _round_to_power_of_2,
)
from powersig.util.grid import get_diagonal_range


# ---------------------------------------------------------------------------
# Stencil construction
# ---------------------------------------------------------------------------
class TestBuildStencil(unittest.TestCase):
    def setUp(self):
        self.order = 4
        self.dtype = jnp.float64

    def test_shape(self):
        stencil = build_stencil(self.order, self.dtype)
        self.assertEqual(stencil.shape, (self.order, self.order))

    def test_first_row_and_column_are_ones(self):
        stencil = build_stencil(self.order, self.dtype)
        np.testing.assert_allclose(np.array(stencil[0, :]), np.ones(self.order))
        np.testing.assert_allclose(np.array(stencil[:, 0]), np.ones(self.order))

    def test_known_values(self):
        stencil = build_stencil(self.order, self.dtype)
        expected = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 1 / 3],
            [1.0, 0.5, 0.25, 1 / 12],
            [1.0, 1 / 3, 1 / 12, 1 / 36],
        ])
        np.testing.assert_allclose(np.array(stencil), expected, rtol=1e-10)

    def test_symmetry(self):
        stencil = build_stencil(self.order, self.dtype)
        np.testing.assert_allclose(
            np.array(stencil), np.array(stencil.T), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Vandermonde vectors
# ---------------------------------------------------------------------------
class TestVandermondeVectors(unittest.TestCase):
    def test_unit_step(self):
        v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, 4, jnp.float64)
        np.testing.assert_allclose(np.array(v_s), np.ones(4))
        np.testing.assert_allclose(np.array(v_t), np.ones(4))

    def test_power_scaling(self):
        v_s, v_t = compute_vandermonde_vectors(0.5, 0.25, 4, jnp.float64)
        np.testing.assert_allclose(
            np.array(v_s), [1.0, 0.5, 0.25, 0.125], rtol=1e-10
        )
        np.testing.assert_allclose(
            np.array(v_t), [1.0, 0.25, 0.0625, 0.015625], rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Diagonal grid geometry
# ---------------------------------------------------------------------------
class TestDiagonalRange(unittest.TestCase):
    def test_square_grid(self):
        # 3x3 grid: diagonals 0,1,2
        s, t, dlen = get_diagonal_range(0, 3, 3)
        self.assertEqual((s, t, dlen), (0, 0, 1))

        s, t, dlen = get_diagonal_range(1, 3, 3)
        self.assertEqual((s, t, dlen), (1, 0, 2))

        s, t, dlen = get_diagonal_range(2, 3, 3)
        self.assertEqual((s, t, dlen), (2, 0, 3))

    def test_rectangular_grid(self):
        # 2 rows, 4 cols
        s, t, dlen = get_diagonal_range(0, 2, 4)
        self.assertEqual((s, t, dlen), (0, 0, 1))

        s, t, dlen = get_diagonal_range(3, 2, 4)
        self.assertEqual((s, t, dlen), (3, 0, 2))

        s, t, dlen = get_diagonal_range(4, 2, 4)
        self.assertEqual((s, t, dlen), (3, 1, 1))


# ---------------------------------------------------------------------------
# Block size auto-tuning utilities
# ---------------------------------------------------------------------------
class TestBlockSizeUtils(unittest.TestCase):
    def test_round_to_power_of_2(self):
        self.assertEqual(_round_to_power_of_2(1), 1)
        self.assertEqual(_round_to_power_of_2(2), 2)
        self.assertEqual(_round_to_power_of_2(3), 4)
        self.assertEqual(_round_to_power_of_2(5), 8)
        self.assertEqual(_round_to_power_of_2(16), 16)
        self.assertEqual(_round_to_power_of_2(17), 32)

    def test_estimate_bytes_per_pair(self):
        bpp = estimate_bytes_per_pair(100, 32, jnp.float64)
        # 8 * 100 * (7*32 + 3*32^2) = 8 * 100 * 3296 = 2_636_800
        self.assertEqual(bpp, 2_636_800)

    def test_compute_block_size_bounded(self):
        device = jax.devices("cpu")[0]
        bs = compute_block_size(100, 32, jnp.float64, device, 1000)
        self.assertGreaterEqual(bs, 1)
        self.assertLessEqual(bs, 256)
        # Must be a power of 2
        self.assertEqual(bs & (bs - 1), 0)

    def test_compute_block_size_clamped_to_total(self):
        device = jax.devices("cpu")[0]
        bs = compute_block_size(1, 4, jnp.float64, device, 3)
        self.assertLessEqual(bs, 4)  # rounded power of 2 of min(computed, 3)


# ---------------------------------------------------------------------------
# Gram matrix computation (end-to-end)
# ---------------------------------------------------------------------------
class TestGramMatrix(unittest.TestCase):
    def setUp(self):
        self.ps = PowerSigJax(order=8, device=jax.devices("cpu")[0])
        key = jax.random.PRNGKey(42)
        self.X = jax.random.normal(key, (4, 10, 3))
        self.Y = jax.random.normal(jax.random.PRNGKey(99), (4, 10, 3))

    def test_block_size_1_matches_auto(self):
        gram_seq = self.ps.compute_gram_matrix(self.X, self.Y, block_size=1)
        gram_auto = self.ps.compute_gram_matrix(self.X, self.Y)
        np.testing.assert_allclose(
            np.array(gram_seq), np.array(gram_auto), rtol=1e-10
        )

    def test_explicit_block_sizes_match(self):
        gram_1 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=1)
        gram_4 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=4)
        gram_16 = self.ps.compute_gram_matrix(self.X, self.Y, block_size=16)
        np.testing.assert_allclose(np.array(gram_1), np.array(gram_4), rtol=1e-10)
        np.testing.assert_allclose(np.array(gram_1), np.array(gram_16), rtol=1e-10)

    def test_symmetric(self):
        gram = self.ps.compute_gram_matrix(self.X, self.X, symmetric=True)
        np.testing.assert_allclose(
            np.array(gram), np.array(gram.T), rtol=1e-10
        )

    def test_symmetric_matches_full(self):
        gram_full = self.ps.compute_gram_matrix(self.X, self.X, symmetric=False)
        gram_sym = self.ps.compute_gram_matrix(self.X, self.X, symmetric=True)
        np.testing.assert_allclose(
            np.array(gram_full), np.array(gram_sym), rtol=1e-10
        )

    def test_diagonal_positive(self):
        """Signature kernel of a path with itself should be positive."""
        gram = self.ps.compute_gram_matrix(self.X, self.X, symmetric=True)
        diag = np.diag(np.array(gram))
        self.assertTrue(np.all(diag > 0), f"Diagonal has non-positive entries: {diag}")

    def test_single_entry_matches_gram(self):
        """compute_signature_kernel should match the corresponding Gram entry."""
        gram = self.ps.compute_gram_matrix(self.X, self.Y)
        for i in range(min(2, self.X.shape[0])):
            for j in range(min(2, self.Y.shape[0])):
                single = self.ps.compute_signature_kernel(self.X[i], self.Y[j])
                np.testing.assert_allclose(
                    float(gram[i, j]), float(single), rtol=1e-6,
                    err_msg=f"Mismatch at ({i},{j})"
                )

    def test_call_interface(self):
        """__call__ should produce the same result as compute_gram_matrix."""
        gram_method = self.ps.compute_gram_matrix(self.X, self.Y)
        gram_call = self.ps(self.X, self.Y)
        np.testing.assert_allclose(
            np.array(gram_method), np.array(gram_call), rtol=1e-10
        )

    def test_call_with_block_size(self):
        gram = self.ps(self.X, self.Y, block_size=2)
        gram_ref = self.ps(self.X, self.Y, block_size=1)
        np.testing.assert_allclose(
            np.array(gram), np.array(gram_ref), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Batch ADM
# ---------------------------------------------------------------------------
class TestBatchADM(unittest.TestCase):
    def test_2x2(self):
        rho = jnp.array([0.5, 0.7], dtype=jnp.float64)
        S = jnp.array([[10.0, 30.0], [100.0, 300.0]], dtype=jnp.float64)
        T = jnp.array([[10.0, 20.0], [100.0, 200.0]], dtype=jnp.float64)
        stencil = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
        U_buf = jnp.empty((2, 2, 2), dtype=jnp.float64)

        result = batch_ADM_for_diagonal(rho, U_buf, S, T, stencil)
        self.assertEqual(result.shape, (2, 2, 2))
        # Verify not all zeros (computation happened)
        self.assertFalse(jnp.allclose(result[:2], jnp.zeros_like(result[:2])))


if __name__ == "__main__":
    unittest.main()
