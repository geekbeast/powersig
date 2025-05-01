from typing import Tuple

import jax.numpy as jnp
from jax import jit

def double_length(array: jnp.ndarray) -> jnp.ndarray:
    new_array = jnp.zeros([array.shape[0] * 2, array.shape[1] * 2], dtype=jnp.float64)
    new_array = new_array.at[:array.shape[0], :array.shape[1]].set(array)
    return new_array

def resize(array: jnp.ndarray, new_shape) -> jnp.ndarray:
    new_array = jnp.zeros(new_shape, dtype=jnp.float64)
    new_array = new_array.at[:array.shape[0], :array.shape[1]].set(array)
    return new_array

@jit
def jax_compute_derivative(X):
    diff = (X[1:,:] - X[:-1,:])
    if X.shape[0] == 1:
        return diff
    else:
        return diff * (X.shape[0] - 1)

@jit
def jax_compute_dot_prod(X, Y):
    assert X.shape == Y.shape, "X and Y must have the same shape."
    return jnp.sum(X * Y)

@jit
def jax_compute_dot_prod_batch(X, Y):
    return jnp.sum(X * Y, axis=len(X.shape) - 1)

@jit
def jax_compute_derivative_batch(X, dt=None) -> jnp.ndarray:
    diff = (X[:, 1:, :] - X[:, :-1, :])
    if dt is not None:
        diff = diff / dt
    elif X.shape[1] != 1:
        diff = diff * (X.shape[1] - 1)

    return diff 