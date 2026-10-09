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
from powersig.jax.algorithm import get_diagonal_range as jax_get_diagonal_range
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
        # 2 rows, 4 cols. s_start is the largest s on the anti-diagonal and
        # t_start the smallest t, so both stay inside the grid once the sweep
        # runs off the bottom edge.
        s, t, dlen = get_diagonal_range(0, 2, 4)
        self.assertEqual((s, t, dlen), (0, 0, 1))

        s, t, dlen = get_diagonal_range(3, 2, 4)
        self.assertEqual((s, t, dlen), (1, 2, 2))

        s, t, dlen = get_diagonal_range(4, 2, 4)
        self.assertEqual((s, t, dlen), (1, 3, 1))

    def test_tall_rectangular_grid(self):
        # 3 rows, 2 cols -- the transpose of the wide case above.
        expected = [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 2),
            (2, 1, 1),
        ]
        self.assertEqual([get_diagonal_range(d, 3, 2) for d in range(4)], expected)

    def test_matches_brute_force_geometry(self):
        # Ground truth: enumerate the cells on each anti-diagonal directly.
        for rows in range(1, 7):
            for cols in range(1, 7):
                for d in range(rows + cols - 1):
                    cells = [
                        (s, t)
                        for s in range(rows)
                        for t in range(cols)
                        if s + t == d
                    ]
                    expected = (
                        max(s for s, _ in cells),
                        min(t for _, t in cells),
                        len(cells),
                    )
                    self.assertEqual(
                        get_diagonal_range(d, rows, cols),
                        expected,
                        msg=f"d={d} rows={rows} cols={cols}",
                    )


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


# ---------------------------------------------------------------------------
# Instance reuse across the jitted entry points
# ---------------------------------------------------------------------------
class TestInstanceReuse(unittest.TestCase):
    def test_signature_kernel_then_gram_matrix(self):
        """compute_signature_kernel is jitted; it must not leave a tracer on self.

        Regression test: assigning to self.exponents inside the jitted method
        used to poison the instance, so a later compute_gram_matrix raised
        InvalidInputException on a leaked JitTracer.
        """
        ps = PowerSigJax(order=8)
        path = jnp.asarray(np.linspace(0.0, 1.0, 17).reshape(17, 1))
        first = float(ps.compute_signature_kernel(path, path))

        batch = jnp.asarray(np.linspace(0.0, 1.0, 17).reshape(1, 17, 1))
        gram = np.asarray(ps.compute_gram_matrix(batch, batch))

        self.assertEqual(gram.shape, (1, 1))
        np.testing.assert_allclose(gram[0, 0], first, rtol=1e-10, atol=1e-12)

    def test_asymmetric_path_lengths_match_equal_length_reference(self):
        """The kernel depends on the paths, not on how finely they are sampled.

        Upsampling a piecewise-linear path along its own segments leaves the path
        unchanged, so a 6-vs-5 point pair must agree with the same two paths
        re-gridded to a common length. This fails if the diagonal sweep keys its
        geometry off `cols` when rows != cols.
        """

        def upsample(P, factor):
            out = []
            for i in range(len(P) - 1):
                for k in range(factor):
                    out.append(P[i] + (P[i + 1] - P[i]) * k / factor)
            out.append(P[-1])
            return np.array(out)

        rng = np.random.default_rng(123)
        X = 0.2 * rng.normal(size=(6, 2))   # 5 segments
        Y = 0.2 * rng.normal(size=(5, 2))   # 4 segments
        # lcm(5, 4) = 20 -> both become 21 points tracing the identical paths.
        Xr, Yr = upsample(X, 4), upsample(Y, 5)
        self.assertEqual(Xr.shape, Yr.shape)

        asymmetric = float(
            PowerSigJax(order=16).compute_signature_kernel(jnp.asarray(X), jnp.asarray(Y))
        )
        reference = float(
            PowerSigJax(order=16).compute_signature_kernel(jnp.asarray(Xr), jnp.asarray(Yr))
        )
        np.testing.assert_allclose(asymmetric, reference, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Cost of the corrected sweep geometry
# ---------------------------------------------------------------------------
class TestGeometryCost(unittest.TestCase):
    """The asymmetric-geometry fix must stay free.

    The sweep runs this arithmetic once per anti-diagonal inside the hot loop, so
    a correctness fix that reached for jnp.where, a Python-level branch, or a
    helper call could pay for itself in the inner loop. Measured on the compiled
    module, the shipped correction costs exactly the same as the original
    expression it replaced.

    Both are compiled in-process with the same JAX build and compared to each
    other rather than to a recorded number, so the assertion does not drift as
    JAX changes its cost model.
    """

    ROWS, COLS, N_DIAGONALS = 128, 96, 256

    @staticmethod
    def _original_geometry(d, rows, cols):
        # The expression this fix replaced: correct only when rows == cols, but
        # it is the cheapness baseline the corrected version has to match.
        t_start = (d < cols) * 0 + (d >= cols) * (d - cols + 1)
        s_start = (d < cols) * d + (d >= cols) * (cols - 1)
        return s_start, t_start, jnp.minimum(rows - t_start, s_start + 1)

    @staticmethod
    def _corrected_geometry(d, rows, cols):
        # Must stay in step with powersig/jax/algorithm.py::compute_diagonal and
        # powersig/util/grid.py; test_corrected_geometry_matches_shared_helper
        # fails if it drifts.
        s_start = (d < rows) * d + (d >= rows) * (rows - 1)
        t_start = (d < rows) * 0 + (d >= rows) * (d - rows + 1)
        return s_start, t_start, jnp.minimum(s_start + 1, cols - t_start)

    def _compiled_cost(self, fn):
        ds = jnp.arange(self.N_DIAGONALS, dtype=jnp.int32)
        compiled = jax.jit(
            lambda d: jax.vmap(lambda x: fn(x, self.ROWS, self.COLS))(d)
        ).lower(ds).compile()
        analysis = compiled.cost_analysis()
        if isinstance(analysis, list):
            analysis = analysis[0]
        return analysis.get("flops"), analysis.get("bytes accessed")

    def test_corrected_geometry_matches_shared_helper(self):
        """Guards the copy above against drifting from the shipped geometry.

        Checks it against both implementations the sweep can reach: the python
        helper in powersig/util/grid.py (used by the Torch and CuPy backends)
        and the jitted one in powersig/jax/algorithm.py that the JAX sweep
        calls. They are written differently, so this also pins them to each
        other.
        """
        for rows in range(1, 7):
            for cols in range(1, 7):
                for d in range(rows + cols - 1):
                    where = f"d={d} rows={rows} cols={cols}"
                    expected = get_diagonal_range(d, rows, cols)
                    self.assertEqual(
                        tuple(int(v) for v in self._corrected_geometry(d, rows, cols)),
                        expected,
                        msg=where,
                    )
                    self.assertEqual(
                        tuple(int(v) for v in jax_get_diagonal_range(d, rows, cols)),
                        expected,
                        msg=f"{where} (jitted JAX helper)",
                    )

    def test_corrected_geometry_costs_no_more_than_original(self):
        original_flops, original_bytes = self._compiled_cost(self._original_geometry)
        corrected_flops, corrected_bytes = self._compiled_cost(self._corrected_geometry)

        self.assertIsNotNone(original_flops)
        self.assertLessEqual(
            corrected_flops,
            original_flops,
            msg=(
                f"corrected geometry costs {corrected_flops} flops vs "
                f"{original_flops} for the expression it replaced"
            ),
        )
        self.assertLessEqual(corrected_bytes, original_bytes)

    def test_shipped_jax_geometry_costs_no_more_than_original(self):
        """The same bound on the helper the JAX sweep actually calls.

        The test above measures a local copy, which cannot catch a regression in
        powersig/jax/algorithm.py. This measures the shipped helper directly.
        """
        original_flops, original_bytes = self._compiled_cost(self._original_geometry)
        shipped_flops, shipped_bytes = self._compiled_cost(
            lambda d, rows, cols: jax_get_diagonal_range(d, rows, cols)
        )

        self.assertIsNotNone(original_flops)
        self.assertLessEqual(
            shipped_flops,
            original_flops,
            msg=(
                f"shipped JAX geometry costs {shipped_flops} flops vs "
                f"{original_flops} for the expression it replaced"
            ),
        )
        self.assertLessEqual(shipped_bytes, original_bytes)


if __name__ == "__main__":
    unittest.main()
