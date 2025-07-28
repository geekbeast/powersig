import jax
import jax.numpy as jnp

@jax.jit
def linear_kernel(x2: jnp.ndarray, x1: jnp.ndarray, y2: jnp.ndarray, y1: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x2-x1, y2-y1, precision = jax.lax.Precision.HIGHEST)
                                                                                       

@jax.jit
def rbf_kernel(x2: jnp.ndarray, x1: jnp.ndarray, y2: jnp.ndarray, y1: jnp.ndarray, bandwidth : float = 1.0) -> jnp.ndarray:
    kx2y2 = rbf_fn(x2 - y2, bandwidth)
    kx1y1 = rbf_fn(x1 - y1, bandwidth)
    kx2y1 = rbf_fn(x2 - y1, bandwidth)
    kx1y2 = rbf_fn(x1 - y2, bandwidth)
    return (kx2y2 - kx1y2) - (kx2y1 - kx1y1)

@jax.jit
def rbf_fn(diff: jnp.ndarray, bandwidth : float = 1.0) -> jnp.ndarray:
    sq_dist = jnp.einsum('...i,...i->', diff, diff, precision = jax.lax.Precision.HIGHEST)
    return jnp.exp(-sq_dist / (2 * (bandwidth ** 2)))
