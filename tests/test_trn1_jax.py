"""CPU parity in CI. Set POWERSIG_TEST_TRN1=1 to run the candidate on a NeuronCore."""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powersig.jax.algorithm import PowerSigJax
from powersig.jax.static_kernels import linear_kernel, rbf_kernel, rel_l2_kernel
from powersig.jax.trn1 import PowerSigTrn1, _diagonal_blocks


CPU = jax.devices("cpu")[0]
DEVICE = jax.devices("neuron")[0] if os.getenv("POWERSIG_TEST_TRN1") == "1" else CPU


@pytest.mark.parametrize("kernel", [linear_kernel, partial(rbf_kernel, bandwidth=0.75), rel_l2_kernel])
@pytest.mark.parametrize("lengths", [(2, 2), (2, 5), (5, 3), (4, 6), (4, 4)])
def test_gram_and_path_gradients(kernel, lengths):
    rng = np.random.default_rng(104)
    x, y = (rng.normal(scale=0.4, size=(count, length, 3)).astype(np.float32)
            for count, length in zip((2, 3), lengths))
    x[0, 0] = 0
    y[1, -1] = y[1, -2]
    weights = rng.normal(size=(2, 3)).astype(np.float32)
    reference = PowerSigJax(order=5, static_kernel=kernel, device=CPU, dtype=jnp.float32)
    reference_gram = jax.vmap(jax.vmap(reference.compute_signature_kernel, in_axes=(None, 0)),
                              in_axes=(0, None))
    candidate = PowerSigTrn1(order=5, static_kernel=kernel, device=DEVICE)

    def evaluate(fn, device):
        cotangent = jax.device_put(weights, device)

        def objective(a, b):
            gram = fn(a, b)
            return (gram * cotangent).sum(), gram

        (_, gram), (dx, dy) = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
            jax.device_put(x, device), jax.device_put(y, device))
        return gram, dx, dy

    expected = evaluate(reference_gram, CPU)
    for got, want in zip(evaluate(candidate, DEVICE), expected):
        assert got.device == DEVICE
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(got, want, atol=3e-5, rtol=5e-5)
    np.testing.assert_allclose(candidate.compute_signature_kernel(x[0], y[0]), expected[0][0, 0],
                               atol=3e-5, rtol=5e-5)


def test_self_gram_and_order_one():
    ps = PowerSigTrn1(order=1, device=DEVICE)
    paths = np.ones((2, 4, 1), dtype=np.float32)
    np.testing.assert_array_equal(ps(paths), np.ones((2, 2)))
    assert ps(paths[:0], paths).shape == (0, 2)
    with pytest.raises(ValueError, match="two samples"):
        ps(paths[:, :1])


def test_sweep_visits_each_cell_once_without_padding():
    rows, cols = 46, 93
    ids = np.arange(rows * cols)
    grid = jax.device_put(ids[:, None], CPU)
    blocks = list(_diagonal_blocks(grid, rows, cols))
    assert len(blocks) == rows + cols - 1
    seen = []
    for d, (first_row, block) in enumerate(blocks):
        cells = np.asarray(block[:, 0])
        i, j = np.divmod(cells, cols)
        np.testing.assert_array_equal(i, np.arange(first_row, first_row + len(block)))
        np.testing.assert_array_equal(i + j, d)
        seen.extend(cells)
    np.testing.assert_array_equal(np.sort(seen), ids)
