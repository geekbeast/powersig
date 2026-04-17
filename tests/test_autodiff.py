"""Tests for the custom autodiff implementation.

Three layers:
1. Forward match: custom_vjp forward matches the original forward.
2. Gradient match: custom VJP gradients match native JAX autodiff on short paths.
3. Numerical gradient checks: jax.test_util.check_grads in float64.
4. Structural: symmetry, zero-gradient sanity, endpoint scatter correctness.
"""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, value_and_grad

from powersig.jax.algorithm import PowerSigJax, compute_vandermonde_vectors
from powersig.jax import static_kernels
from powersig.jax.autodiff import (
    _map_diagonal_entry_bwd,
    _map_diagonal_entry_fwd,
    _forward_sweep,
    _forward_sweep_with_checkpoints,
    _make_sig_kernel_diff,
    _compute_checkpoint_interval,
    compute_sig_kernel_fast_diff,
    compute_gram_fast_diff,
)


CPU = jax.devices("cpu")[0]


def _make_ps(order=8, kernel=static_kernels.linear_kernel):
    return PowerSigJax(order=order, static_kernel=kernel, device=CPU,
                       dtype=jnp.float64)


def _random_paths(key, batch=1, length=10, dim=3):
    """Generate random paths for testing."""
    return jax.random.normal(key, (batch, length, dim), dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Layer 0: Local adjoint unit tests
# ---------------------------------------------------------------------------
class TestLocalAdjoint(unittest.TestCase):
    """Test _map_diagonal_entry_bwd against JAX autodiff of the forward tile."""

    def setUp(self):
        self.ps = _make_ps(order=8)
        self.key = jax.random.PRNGKey(0)

    def test_local_adjoint_matches_jax_autodiff(self):
        """The hand-coded local adjoint must match jax.vjp of the forward tile."""
        P = self.ps.order
        key = self.key
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        rho = jax.random.normal(k1, (), dtype=jnp.float64) * 0.5
        s = jax.random.normal(k2, (P,), dtype=jnp.float64)
        t = jax.random.normal(k3, (P,), dtype=jnp.float64)
        bar_s_next = jax.random.normal(k4, (P,), dtype=jnp.float64)
        bar_t_next = jax.random.normal(k5, (P,), dtype=jnp.float64)

        psi_s = self.ps.psi_s
        psi_t = self.ps.psi_t
        exponents = self.ps.exponents

        # Use JAX autodiff as ground truth
        def fwd_fn(rho_val, s_val, t_val):
            r = rho_val ** exponents
            from jax.scipy.linalg import toeplitz as tp
            T_mat = psi_t * tp(t_val, s_val)
            s_next = (r @ jnp.triu(T_mat, 1)) + ((psi_s @ t_val) * r)
            t_next = ((psi_s @ s_val) * r) + (jnp.tril(T_mat, -1) @ r)
            return s_next, t_next

        primals, vjp_fn = jax.vjp(fwd_fn, rho, s, t)
        bar_rho_ref, bar_s_ref, bar_t_ref = vjp_fn((bar_s_next, bar_t_next))

        # Custom adjoint
        bar_s_custom, bar_t_custom, bar_rho_custom = _map_diagonal_entry_bwd(
            rho, s, t, psi_s, psi_t, exponents, bar_s_next, bar_t_next)

        np.testing.assert_allclose(float(bar_rho_custom), float(bar_rho_ref),
                                   rtol=1e-10, atol=1e-12,
                                   err_msg="bar_rho mismatch")
        np.testing.assert_allclose(np.array(bar_s_custom), np.array(bar_s_ref),
                                   rtol=1e-10, atol=1e-12,
                                   err_msg="bar_s mismatch")
        np.testing.assert_allclose(np.array(bar_t_custom), np.array(bar_t_ref),
                                   rtol=1e-10, atol=1e-12,
                                   err_msg="bar_t mismatch")

    def test_local_adjoint_various_rho(self):
        """Test adjoint at rho = 0, small, and large values."""
        P = self.ps.order
        psi_s, psi_t, exponents = self.ps.psi_s, self.ps.psi_t, self.ps.exponents

        for rho_val in [0.0, 1e-10, 0.1, 0.5, 1.0, 2.0]:
            rho = jnp.float64(rho_val)
            key = jax.random.PRNGKey(int(rho_val * 1000) + 1)
            k1, k2, k3, k4 = jax.random.split(key, 4)
            s = jax.random.normal(k1, (P,), dtype=jnp.float64)
            t = jax.random.normal(k2, (P,), dtype=jnp.float64)
            bar_s_next = jax.random.normal(k3, (P,), dtype=jnp.float64)
            bar_t_next = jax.random.normal(k4, (P,), dtype=jnp.float64)

            def fwd_fn(rho_v, s_v, t_v):
                r = rho_v ** exponents
                from jax.scipy.linalg import toeplitz as tp
                T_mat = psi_t * tp(t_v, s_v)
                s_n = (r @ jnp.triu(T_mat, 1)) + ((psi_s @ t_v) * r)
                t_n = ((psi_s @ s_v) * r) + (jnp.tril(T_mat, -1) @ r)
                return s_n, t_n

            _, vjp_fn = jax.vjp(fwd_fn, rho, s, t)
            bar_rho_ref, bar_s_ref, bar_t_ref = vjp_fn((bar_s_next, bar_t_next))

            bar_s_c, bar_t_c, bar_rho_c = _map_diagonal_entry_bwd(
                rho, s, t, psi_s, psi_t, exponents, bar_s_next, bar_t_next)

            np.testing.assert_allclose(
                float(bar_rho_c), float(bar_rho_ref), rtol=1e-8, atol=1e-12,
                err_msg=f"bar_rho mismatch at rho={rho_val}")
            np.testing.assert_allclose(
                np.array(bar_s_c), np.array(bar_s_ref), rtol=1e-8, atol=1e-12,
                err_msg=f"bar_s mismatch at rho={rho_val}")
            np.testing.assert_allclose(
                np.array(bar_t_c), np.array(bar_t_ref), rtol=1e-8, atol=1e-12,
                err_msg=f"bar_t mismatch at rho={rho_val}")


# ---------------------------------------------------------------------------
# Layer 1: Forward output match
# ---------------------------------------------------------------------------
class TestForwardMatch(unittest.TestCase):
    """custom_vjp forward must produce the same scalar as the original."""

    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def _check_forward(self, order, length_x, length_y, dim, kernel):
        ps = _make_ps(order=order, kernel=kernel)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (length_x, dim), dtype=jnp.float64)
        Y_j = jax.random.normal(k2, (length_y, dim), dtype=jnp.float64)

        # Use _forward_sweep as reference to avoid JIT side-effect issues
        v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, ps.order, jnp.float64)
        ref = _forward_sweep(X_i, Y_j, v_s, v_t, ps.psi_s, ps.psi_t,
                             ps.ic, ps.exponents, ps.static_kernel)
        custom = compute_sig_kernel_fast_diff(ps, X_i, Y_j)

        np.testing.assert_allclose(
            float(ref), float(custom), rtol=1e-10,
            err_msg=f"Forward mismatch: order={order}, "
                    f"len=({length_x},{length_y}), dim={dim}")

    def test_linear_small(self):
        self._check_forward(4, 8, 8, 2, static_kernels.linear_kernel)

    def test_linear_medium(self):
        self._check_forward(8, 16, 16, 4, static_kernels.linear_kernel)

    def test_linear_asymmetric(self):
        self._check_forward(8, 10, 16, 3, static_kernels.linear_kernel)

    def test_rbf_small(self):
        self._check_forward(4, 8, 8, 2, static_kernels.rbf_kernel)

    def test_rbf_medium(self):
        self._check_forward(8, 16, 16, 4, static_kernels.rbf_kernel)

    def test_forward_with_checkpoints_matches(self):
        """_forward_sweep_with_checkpoints must match _forward_sweep."""
        ps = _make_ps(order=8)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (12, 3), dtype=jnp.float64)
        Y_j = jax.random.normal(k2, (12, 3), dtype=jnp.float64)

        v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, ps.order, jnp.float64)
        ref = _forward_sweep(X_i, Y_j, v_s, v_t, ps.psi_s, ps.psi_t,
                             ps.ic, ps.exponents, ps.static_kernel)

        for ckpt_interval in [1, 3, 5, 11, 20]:
            out, _, _ = _forward_sweep_with_checkpoints(
                X_i, Y_j, v_s, v_t, ps.psi_s, ps.psi_t,
                ps.ic, ps.exponents, ps.static_kernel, ckpt_interval)
            np.testing.assert_allclose(
                float(ref), float(out), rtol=1e-12,
                err_msg=f"Checkpoint interval {ckpt_interval}")


# ---------------------------------------------------------------------------
# Layer 2: Gradient match against native JAX autodiff
# ---------------------------------------------------------------------------
class TestGradientMatch(unittest.TestCase):
    """Custom VJP gradients must match native JAX autodiff on short paths."""

    def setUp(self):
        self.key = jax.random.PRNGKey(123)

    def _check_grads(self, order, length, dim, kernel):
        ps = _make_ps(order=order, kernel=kernel)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (length, dim), dtype=jnp.float64)
        Y_j = jax.random.normal(k2, (length, dim), dtype=jnp.float64)

        # Reference: native JAX autodiff through the original forward sweep
        v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, ps.order, jnp.float64)

        def ref_fn(Xi, Yj):
            return _forward_sweep(Xi, Yj, v_s, v_t, ps.psi_s, ps.psi_t,
                                  ps.ic, ps.exponents, ps.static_kernel)

        ref_val, (ref_gX, ref_gY) = value_and_grad(ref_fn, argnums=(0, 1))(
            X_i, Y_j)

        # Custom VJP
        def custom_fn(Xi, Yj):
            return compute_sig_kernel_fast_diff(ps, Xi, Yj)

        custom_val, (custom_gX, custom_gY) = value_and_grad(
            custom_fn, argnums=(0, 1))(X_i, Y_j)

        np.testing.assert_allclose(float(ref_val), float(custom_val), rtol=1e-10,
                                   err_msg="Forward value mismatch")
        np.testing.assert_allclose(np.array(custom_gX), np.array(ref_gX),
                                   rtol=1e-6, atol=1e-10,
                                   err_msg="grad_X mismatch")
        np.testing.assert_allclose(np.array(custom_gY), np.array(ref_gY),
                                   rtol=1e-6, atol=1e-10,
                                   err_msg="grad_Y mismatch")

    def test_linear_8x8_dim2_order4(self):
        self._check_grads(4, 8, 2, static_kernels.linear_kernel)

    def test_linear_8x8_dim4_order8(self):
        self._check_grads(8, 8, 4, static_kernels.linear_kernel)

    def test_linear_16x16_dim2_order4(self):
        self._check_grads(4, 16, 2, static_kernels.linear_kernel)

    def test_rbf_8x8_dim2_order4(self):
        self._check_grads(4, 8, 2, static_kernels.rbf_kernel)

    def test_rbf_8x8_dim4_order8(self):
        self._check_grads(8, 8, 4, static_kernels.rbf_kernel)


# ---------------------------------------------------------------------------
# Layer 3: Numerical gradient checks (finite differences)
# ---------------------------------------------------------------------------
class TestNumericalGradients(unittest.TestCase):
    """Use finite differences to validate gradients in float64."""

    def setUp(self):
        self.key = jax.random.PRNGKey(999)

    def _check_numerical(self, order, length, dim, kernel, eps=1e-6):
        ps = _make_ps(order=order, kernel=kernel)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (length, dim), dtype=jnp.float64) * 0.5
        Y_j = jax.random.normal(k2, (length, dim), dtype=jnp.float64) * 0.5

        def fn_x(Xi):
            return compute_sig_kernel_fast_diff(ps, Xi, Y_j)

        def fn_y(Yj):
            return compute_sig_kernel_fast_diff(ps, X_i, Yj)

        # Check grad_X
        analytic_gX = grad(fn_x)(X_i)
        for idx in [(0, 0), (1, 0), (length - 1, dim - 1)]:
            e = jnp.zeros_like(X_i).at[idx].set(eps)
            fd = (fn_x(X_i + e) - fn_x(X_i - e)) / (2 * eps)
            np.testing.assert_allclose(
                float(analytic_gX[idx]), float(fd), rtol=1e-4, atol=1e-8,
                err_msg=f"grad_X numerical mismatch at {idx}")

        # Check grad_Y
        analytic_gY = grad(fn_y)(Y_j)
        for idx in [(0, 0), (1, 0), (length - 1, dim - 1)]:
            e = jnp.zeros_like(Y_j).at[idx].set(eps)
            fd = (fn_y(Y_j + e) - fn_y(Y_j - e)) / (2 * eps)
            np.testing.assert_allclose(
                float(analytic_gY[idx]), float(fd), rtol=1e-4, atol=1e-8,
                err_msg=f"grad_Y numerical mismatch at {idx}")

    def test_linear_numerical(self):
        self._check_numerical(4, 8, 2, static_kernels.linear_kernel)

    def test_rbf_numerical(self):
        self._check_numerical(4, 8, 2, static_kernels.rbf_kernel)


# ---------------------------------------------------------------------------
# Layer 4: Structural tests
# ---------------------------------------------------------------------------
class TestStructural(unittest.TestCase):
    """Symmetry, zero-gradient, and consistency tests."""

    def setUp(self):
        self.key = jax.random.PRNGKey(77)
        self.ps = _make_ps(order=8)

    def test_self_kernel_symmetry(self):
        """K(X, X) Gram matrix should be symmetric."""
        X = _random_paths(self.key, batch=3, length=10, dim=3)
        gram = compute_gram_fast_diff(self.ps, X, X, symmetric=True)
        np.testing.assert_allclose(
            np.array(gram), np.array(gram.T), rtol=1e-10)

    def test_gram_forward_matches_original(self):
        """Gram matrix from diff path matches original."""
        k1, k2 = jax.random.split(self.key)
        X = _random_paths(k1, batch=3, length=10, dim=3)
        Y = _random_paths(k2, batch=3, length=10, dim=3)

        ref = self.ps.compute_gram_matrix(X, Y)
        diff = compute_gram_fast_diff(self.ps, X, Y)
        np.testing.assert_allclose(np.array(ref), np.array(diff), rtol=1e-8)

    def test_checkpoint_intervals_give_same_grads(self):
        """Different checkpoint intervals should produce identical gradients."""
        ps = _make_ps(order=4)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (8, 2), dtype=jnp.float64)
        Y_j = jax.random.normal(k2, (8, 2), dtype=jnp.float64)

        def fn(Xi, Yj, ckpt):
            return compute_sig_kernel_fast_diff(ps, Xi, Yj,
                                                checkpoint_interval=ckpt)

        grads = {}
        for ckpt in [1, 2, 3, 5, 13]:
            g = grad(lambda Xi, Yj: fn(Xi, Yj, ckpt), argnums=(0, 1))(
                X_i, Y_j)
            grads[ckpt] = g

        ref_gX, ref_gY = grads[1]
        for ckpt, (gX, gY) in grads.items():
            np.testing.assert_allclose(
                np.array(gX), np.array(ref_gX), rtol=1e-10,
                err_msg=f"grad_X differs at checkpoint_interval={ckpt}")
            np.testing.assert_allclose(
                np.array(gY), np.array(ref_gY), rtol=1e-10,
                err_msg=f"grad_Y differs at checkpoint_interval={ckpt}")

    def test_diagonal_positive_diff(self):
        """Self-kernel diagonal should be positive."""
        X = _random_paths(self.key, batch=3, length=10, dim=3)
        gram = compute_gram_fast_diff(self.ps, X, X, symmetric=True)
        diag = np.diag(np.array(gram))
        self.assertTrue(np.all(diag > 0),
                        f"Non-positive diagonal: {diag}")

    def test_constant_path_zero_gradient(self):
        """A constant path has zero increments, so rho = 0 everywhere.
        The kernel value should be 1 (from initial conditions) and
        gradients should be zero because the output doesn't depend on
        the path values when all increments are zero."""
        X_i = jnp.ones((8, 2), dtype=jnp.float64) * 3.0
        Y_j = jnp.ones((8, 2), dtype=jnp.float64) * 5.0
        ps = _make_ps(order=4)

        val, (gX, gY) = value_and_grad(
            lambda Xi, Yj: compute_sig_kernel_fast_diff(ps, Xi, Yj),
            argnums=(0, 1))(X_i, Y_j)

        # With constant paths, all rho = 0, kernel should be 1.0
        np.testing.assert_allclose(float(val), 1.0, rtol=1e-10)
        # Gradients should be zero
        np.testing.assert_allclose(np.array(gX), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.array(gY), 0.0, atol=1e-10)

    def test_float32_instance_with_float32_paths(self):
        """A float32 PowerSigJax instance should work with float32 paths."""
        ps32 = PowerSigJax(order=4, device=CPU, dtype=jnp.float32)
        k1, k2 = jax.random.split(self.key)
        X_i = jax.random.normal(k1, (8, 2), dtype=jnp.float32)
        Y_j = jax.random.normal(k2, (8, 2), dtype=jnp.float32)

        val, (gX, gY) = value_and_grad(
            lambda Xi, Yj: compute_sig_kernel_fast_diff(ps32, Xi, Yj),
            argnums=(0, 1))(X_i, Y_j)

        self.assertEqual(gX.dtype, jnp.float32)
        self.assertEqual(gY.dtype, jnp.float32)
        self.assertTrue(jnp.isfinite(val))

    def test_mixed_dtype_instances_in_same_process(self):
        """Two PowerSigJax instances with different dtypes should not
        interfere via JIT cache."""
        ps64 = _make_ps(order=4)  # float64
        ps32 = PowerSigJax(order=4, device=CPU, dtype=jnp.float32)

        k1, k2 = jax.random.split(self.key)
        X64 = jax.random.normal(k1, (8, 2), dtype=jnp.float64)
        Y64 = jax.random.normal(k2, (8, 2), dtype=jnp.float64)
        X32 = X64.astype(jnp.float32)
        Y32 = Y64.astype(jnp.float32)

        # Run float64 first
        val64, (gX64, gY64) = value_and_grad(
            lambda Xi, Yj: compute_sig_kernel_fast_diff(ps64, Xi, Yj),
            argnums=(0, 1))(X64, Y64)

        # Then float32 — this must not crash from dtype mismatch
        val32, (gX32, gY32) = value_and_grad(
            lambda Xi, Yj: compute_sig_kernel_fast_diff(ps32, Xi, Yj),
            argnums=(0, 1))(X32, Y32)

        self.assertEqual(gX64.dtype, jnp.float64)
        self.assertEqual(gX32.dtype, jnp.float32)
        # Values should be close (float32 has less precision)
        np.testing.assert_allclose(float(val64), float(val32), rtol=1e-4)


# ---------------------------------------------------------------------------
# Layer 5: Performance regression
# ---------------------------------------------------------------------------
class TestPerformanceRegression(unittest.TestCase):
    """Ensure the custom backward stays within a bounded ratio of the forward.

    The checkpointed replay design targets backward ~= 2x forward, so
    grad (which includes one forward + one backward) should be ~3x forward.
    We set a generous 5x ceiling to avoid flaky failures from scheduling jitter
    while still catching algorithmic regressions (e.g., accidental O(T^2) replay).

    Performance context (GPU, order=8, dim=4, linear kernel):
    ┌─────────┬────────────┬────────────┬─────────┐
    │  Length  │ Native AD  │ Custom VJP │  Ratio  │
    ├─────────┼────────────┼────────────┼─────────┤
    │      32 │     5.1 ms │     7.1 ms │  1.37x  │
    │     512 │    88.2 ms │   108.9 ms │  1.23x  │
    │    2048 │   427.1 ms │   495.4 ms │  1.16x  │
    │    3072 │       OOM  │   957.4 ms │    --   │
    │    8192 │       OOM  │  4097.3 ms │    --   │
    └─────────┴────────────┴────────────┴─────────┘

    The custom VJP trades ~1.2-1.4x runtime overhead for O(sqrt(T)) memory
    instead of O(T). Native autodiff OOMs at ~3072 (order=8) or ~2048
    (order=16); the custom VJP handles 8192+.
    """

    MAX_GRAD_TO_FWD_RATIO = 5.0
    N_WARMUP = 3
    N_TRIALS = 5

    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def _measure_ratio(self, order, length, dim, kernel):
        import time

        ps = _make_ps(order=order, kernel=kernel)
        k1, k2 = jax.random.split(self.key)
        X = jax.random.normal(k1, (length, dim), dtype=jnp.float64)
        Y = jax.random.normal(k2, (length, dim), dtype=jnp.float64)

        fwd_fn = lambda Xi, Yj: compute_sig_kernel_fast_diff(ps, Xi, Yj)
        grad_fn = grad(fwd_fn, argnums=(0, 1))

        # Warmup
        for _ in range(self.N_WARMUP):
            out = fwd_fn(X, Y)
            jax.block_until_ready(out)
            gX, gY = grad_fn(X, Y)
            jax.block_until_ready(gX)

        # Forward timing
        fwd_times = []
        for _ in range(self.N_TRIALS):
            t0 = time.perf_counter()
            out = fwd_fn(X, Y)
            jax.block_until_ready(out)
            fwd_times.append(time.perf_counter() - t0)

        # Grad timing (forward + backward)
        grad_times = []
        for _ in range(self.N_TRIALS):
            t0 = time.perf_counter()
            gX, gY = grad_fn(X, Y)
            jax.block_until_ready(gX)
            jax.block_until_ready(gY)
            grad_times.append(time.perf_counter() - t0)

        fwd_median = np.median(fwd_times)
        grad_median = np.median(grad_times)
        ratio = grad_median / fwd_median
        return ratio

    def test_linear_ratio_bounded(self):
        ratio = self._measure_ratio(8, 32, 4, static_kernels.linear_kernel)
        self.assertLess(
            ratio, self.MAX_GRAD_TO_FWD_RATIO,
            f"Linear grad/fwd ratio {ratio:.2f}x exceeds "
            f"{self.MAX_GRAD_TO_FWD_RATIO}x ceiling")

    def test_rbf_ratio_bounded(self):
        ratio = self._measure_ratio(8, 32, 4, static_kernels.rbf_kernel)
        self.assertLess(
            ratio, self.MAX_GRAD_TO_FWD_RATIO,
            f"RBF grad/fwd ratio {ratio:.2f}x exceeds "
            f"{self.MAX_GRAD_TO_FWD_RATIO}x ceiling")

    def test_longer_path_ratio_bounded(self):
        """Ensure ratio doesn't degrade at longer sequences."""
        ratio = self._measure_ratio(8, 128, 4, static_kernels.linear_kernel)
        self.assertLess(
            ratio, self.MAX_GRAD_TO_FWD_RATIO,
            f"Long-path grad/fwd ratio {ratio:.2f}x exceeds "
            f"{self.MAX_GRAD_TO_FWD_RATIO}x ceiling")

    def test_ratio_does_not_degrade_with_length(self):
        """The grad/fwd ratio should stay roughly constant (not grow with T).
        If it grows, it indicates the backward has super-linear complexity."""
        ratio_short = self._measure_ratio(4, 16, 2, static_kernels.linear_kernel)
        ratio_long = self._measure_ratio(4, 128, 2, static_kernels.linear_kernel)
        # Allow 50% degradation for overhead noise at short lengths
        self.assertLess(
            ratio_long, ratio_short * 1.5,
            f"Ratio degraded from {ratio_short:.2f}x (len=16) to "
            f"{ratio_long:.2f}x (len=128), suggesting super-linear backward")


if __name__ == "__main__":
    unittest.main()
