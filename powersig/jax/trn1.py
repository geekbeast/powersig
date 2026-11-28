"""Float32 JAX signature kernels for AWS Trainium1, with static per-diagonal shapes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from powersig.jax.static_kernels import _unit_vector, linear_kernel, rel_l2_kernel
from powersig.util.grid import get_diagonal_range


def _matmul(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _increments(path):
    # A constant difference matrix keeps differencing and its transpose in a
    # matmul, including when the Neuron compiler fuses the following dot.
    length = path.shape[1]
    difference = np.eye(length - 1, length, k=1, dtype=np.float32)
    difference -= np.eye(length - 1, length, dtype=np.float32)
    return _matmul(difference, path)


def _cell_grid(x, y, kernel):
    """Return (row, col, x batch, y batch) with one scalar per real cell."""
    if kernel is rel_l2_kernel:
        x, y = _unit_vector(x), _unit_vector(y)
    if kernel is linear_kernel or kernel is rel_l2_kernel:
        return jnp.einsum("bid,mjd->ijbm", _increments(x), _increments(y),
                          precision=jax.lax.Precision.HIGHEST)

    # Evaluate the supplied four-endpoint kernel, including RBF with a bound
    # bandwidth, on the Cartesian product of both paths' real segments.
    along_y = jax.vmap(kernel, in_axes=(None, None, 0, 0))
    cells = jax.vmap(along_y, in_axes=(0, 0, None, None))
    dim = x.shape[-1]
    grid = cells(x[:, 1:].reshape(-1, dim), x[:, :-1].reshape(-1, dim),
                 y[:, 1:].reshape(-1, dim), y[:, :-1].reshape(-1, dim))
    return grid.reshape(x.shape[0], x.shape[1] - 1, y.shape[0], y.shape[1] - 1).transpose(1, 3, 0, 2)


def _tile_matrices(order):
    """Host constants for transferring a polynomial edge across a unit cell.

    An incoming monomial of degree m times rho**k has coefficient
    m! / (k! (m+k)!). It contributes to degree k on the opposite edge
    and, for m > 0, to degree m+k on the same edge. The constant corner
    is shared. Keep only terms with m+k < order, as in PowerSigJax.
    """
    total_degrees, powers = np.tril_indices(order)
    degrees = total_degrees - powers
    terms = len(degrees)
    same = np.zeros((order, terms), dtype=np.float32)
    opposite = np.zeros_like(same)
    for term, (degree, power) in enumerate(zip(degrees, powers)):
        coefficient = 1.0
        for k in range(1, power + 1):
            coefficient /= k * (degree + k)
        opposite[power, term] = coefficient
        if degree:
            same[degree + power, term] = coefficient
    identity = np.eye(order, dtype=np.float32)
    return (identity[degrees], identity[powers], np.concatenate((same, opposite), axis=1),
            np.concatenate((opposite, same), axis=1))


def _diagonal_blocks(grid, rows, cols):
    """Pack cells once, in diagonal order, so the reverse pass concatenates their gradients."""
    indices = []
    first_rows = []
    for d in range(rows + cols - 1):
        last, _, width = get_diagonal_range(d, rows, cols)
        first = last - width + 1
        row = np.arange(first, last + 1, dtype=np.int32)
        indices.append(row * cols + (d - row))
        first_rows.append(first)
    blocks = jax.lax.split(grid[np.concatenate(indices)], tuple(index.size for index in indices))
    return zip(first_rows, blocks)


@partial(jax.jit, static_argnames=("order", "kernel"))
def _gram(x, y, order, kernel):
    rows, cols = x.shape[1] - 1, y.shape[1] - 1
    pairs = x.shape[0] * y.shape[0]
    grid = _cell_grid(x, y, kernel).reshape(rows * cols, pairs)
    edge_basis, power_basis, right_matrix, top_matrix = _tile_matrices(order)

    boundary = jnp.broadcast_to(jnp.asarray(np.eye(order, 1, dtype=np.float32)), (order, pairs))
    right = top = boundary
    previous_first = 0
    for first, block in _diagonal_blocks(grid, rows, cols):
        lanes = block.shape[0] * pairs
        rho = block.reshape(1, lanes)
        powers = [jnp.ones_like(rho)]
        for _ in range(1, order):
            powers.append(powers[-1] * rho)
        weights = _matmul(power_basis, jnp.concatenate(powers, axis=0))

        # In ascending row order, predecessors are the same row (left)
        # and the row before it (bottom). Only the outer boundary needs e0.
        start = (first - previous_first) * pairs
        stop = start + lanes
        left = jnp.concatenate((right, boundary), axis=1)[:, start:stop]
        bottom = jnp.concatenate((boundary, top), axis=1)[:, start:stop]
        terms = jnp.concatenate((_matmul(edge_basis, left) * weights,
                                 _matmul(edge_basis, bottom) * weights), axis=0)
        right, top = _matmul(right_matrix, terms), _matmul(top_matrix, terms)
        previous_first = first
    return right.sum(axis=0).reshape(x.shape[0], y.shape[0])


class PowerSigTrn1:
    """JAX estimator for a single NeuronCore. Pass a CPU device for a reference run.

    Both inputs are evaluated in float32. Each batch is compiled as a block
    of all path pairs, with path lengths, batch sizes and order fixed at trace time.
    """

    def __init__(self, order=32, static_kernel=linear_kernel, device=None):
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        self.order = order
        self.static_kernel = static_kernel
        self.device = jax.devices("neuron")[0] if device is None else device

    def _array(self, value):
        # A surrounding JAX transformation controls placement of its tracers.
        # Re-pinning them inserts a Sharding custom call into the Neuron HLO.
        if isinstance(value, jax.core.Tracer):
            return jnp.asarray(value, dtype=jnp.float32)
        return jnp.asarray(value, dtype=jnp.float32, device=self.device)

    def compute_gram_matrix(self, X, Y=None):
        """Kernel values for batches shaped (batch, path length, feature dimension)."""
        x = self._array(X)
        y = x if Y is None else self._array(Y)
        if x.ndim != 3 or y.ndim != 3 or x.shape[-1] != y.shape[-1] or x.shape[-1] == 0:
            raise ValueError("expected path batches with a common, nonempty feature dimension")
        if min(x.shape[1], y.shape[1]) < 2:
            raise ValueError("each path must have at least two samples")
        if x.shape[0] == 0 or y.shape[0] == 0:
            return jnp.empty((x.shape[0], y.shape[0]), dtype=x.dtype, device=self.device)
        return _gram(x, y, self.order, self.static_kernel)

    __call__ = compute_gram_matrix

    def compute_signature_kernel(self, X, Y):
        """Kernel value for two individual paths shaped (path length, feature dimension)."""
        return self.compute_gram_matrix(self._array(X)[None], self._array(Y)[None])[0, 0]
