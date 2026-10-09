"""Dataset-free FP32 regressions for inactive diagonal lanes."""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import toeplitz
import numpy as np
import pytest

from powersig.jax.algorithm import PowerSigJax
from powersig.jax.diagonal import diagonal_tile_inputs
from powersig.jax.static_kernels import linear_kernel, rbf_kernel


@pytest.fixture(autouse=True)
def strict_float32():
    old_x64 = jax.config.jax_enable_x64
    old_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", old_x64)
    jax.config.update("jax_default_matmul_precision", old_precision)


def make_ps(kernel=linear_kernel):
    return PowerSigJax(order=9, dtype=jnp.float32,
                       device=jax.devices()[0], static_kernel=kernel)


def overflow_paths():
    # A single large bank increment, followed by constant knots. All REAL
    # tile states and derivatives are finite. The old fixed-width sweep
    # repeatedly reuses this increment in nonexistent tiles and overflows.
    x = jnp.linspace(0, 1, 64, dtype=jnp.float32)[:, None]
    y = jnp.full_like(x, 256).at[0].set(0)
    return x, y


def dense_reference(ps, x, y):
    """Row-major scan over only real tiles; no diagonal padding or masking."""
    cols = len(y) - 1
    bottom = jnp.tile(ps.ic, (cols, 1))

    def row(bottom_edges, i):
        def tile(left_edge, args):
            j, bottom_edge = args
            rho = ps.static_kernel(x[i + 1], x[i], y[j + 1], y[j])
            r = rho ** ps.exponents
            matrix = ps.psi_t * toeplitz(bottom_edge, left_edge)
            right = r @ jnp.triu(matrix, 1) + (ps.psi_s @ bottom_edge) * r
            top = (ps.psi_s @ left_edge) * r + jnp.tril(matrix, -1) @ r
            return right, top
        right, top_edges = jax.lax.scan(tile, ps.ic,
                                       (jnp.arange(cols), bottom_edges))
        return top_edges, right

    _, rights = jax.lax.scan(row, bottom, jnp.arange(len(x) - 1))
    return jnp.sum(rights[-1])


def assert_close(actual, expected, cancellation=False):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert got.dtype == jnp.float32
        assert bool(jnp.isfinite(got).all())
        assert bool(jnp.isfinite(want).all())
        # The stress case subtracts O(1e11) terms into near-zero coordinates.
        # Allow eight ULPs of the largest component ONLY for that case; keep
        # strict elementwise tolerances for ordinary linear/RBF paths.
        atol = 2e-5
        if cancellation:
            atol = max(atol, 8 * float(np.spacing(np.max(np.abs(np.asarray(want))))))
        np.testing.assert_allclose(got, want, rtol=3e-4, atol=atol)


@pytest.mark.parametrize("chunked", [False, True])
def test_unused_lanes_cannot_overflow_value_or_native_gradient(chunked):
    ps = make_ps()
    x, y = overflow_paths()
    forward = (ps.compute_signature_kernel_chunked if chunked
               else ps.compute_signature_kernel)
    got = jax.jit(jax.value_and_grad(forward, argnums=(0, 1)))(x, y)
    want = jax.jit(jax.value_and_grad(lambda a, b: dense_reference(ps, a, b),
                                    argnums=(0, 1)))(x, y)
    assert_close(got, want, cancellation=True)


@pytest.mark.parametrize("lengths", [(2, 2), (2, 7), (7, 2), (5, 9), (9, 5)])
@pytest.mark.parametrize("kernel", [linear_kernel, rbf_kernel])
@pytest.mark.parametrize("chunked", [False, True])
def test_rectangular_values_and_gradients(lengths, kernel, chunked):
    ps = make_ps(kernel)
    rng = np.random.default_rng(123)
    x = jnp.asarray(rng.normal(size=(lengths[0], 2)).astype(np.float32) * np.float32(.15))
    y = jnp.asarray(rng.normal(size=(lengths[1], 2)).astype(np.float32) * np.float32(.15))
    forward = (ps.compute_signature_kernel_chunked if chunked
               else ps.compute_signature_kernel)
    got = jax.jit(jax.value_and_grad(forward, argnums=(0, 1)))(x, y)
    want = jax.jit(jax.value_and_grad(lambda a, b: dense_reference(ps, a, b),
                                    argnums=(0, 1)))(x, y)
    assert_close(got, want)
    np.testing.assert_allclose(forward(x, y), forward(y, x), rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("kernel", [linear_kernel, rbf_kernel])
def test_poisoned_unused_inputs_are_masked_before_arithmetic(kernel):
    ps = make_ps(kernel)
    x = jnp.full((4, 2), jnp.float32(1e30))
    s = jnp.full((3, 9), jnp.inf, dtype=jnp.float32)
    t = jnp.full((3, 9), jnp.nan, dtype=jnp.float32)
    left, bottom, rho, xi, yi = diagonal_tile_inputs(
        jnp.int32(2), jnp.int32(0), jnp.int32(0), jnp.int32(1), True,
        x, -x, s, t, ps.ic, ps.static_kernel)
    np.testing.assert_array_equal(left, jnp.zeros_like(left))
    np.testing.assert_array_equal(bottom, jnp.zeros_like(bottom))
    assert float(rho) == 0
    assert 0 <= int(xi) < 3 and 0 <= int(yi) < 3


def test_mask_does_not_hide_overflow_in_a_real_tile():
    ps = make_ps()
    x = jnp.array([[0.], [1e10]], dtype=jnp.float32)
    assert not bool(jnp.isfinite(ps.compute_signature_kernel(x, x)))
