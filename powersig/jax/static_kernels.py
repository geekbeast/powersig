import jax
import jax.numpy as jnp

@jax.jit
def linear_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x, y, precision = jax.lax.Precision.HIGHEST)

@jax.jit
def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, bandwidth : float = 1.0) -> jnp.ndarray:
    diff = x - y
    sq_dist = jnp.einsum('...i,...i->', diff, diff, precision = jax.lax.Precision.HIGHEST)
    return jnp.exp(-sq_dist / (2 * (bandwidth ** 2)))

