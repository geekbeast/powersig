"""Safe inputs for fixed-width JAX diagonal sweeps.

Inactive SIMD lanes are not signature-kernel tiles. Mask before arithmetic,
not just at the output: an unused overflowing state can otherwise produce
0 * Inf in a reverse pass and contaminate real path gradients.
"""
import jax.numpy as jnp


def masked_path_points(X, Y, x_index, y_index, active):
    """Finite dummy inputs for inactive lanes, without changing active tiles."""
    return (
        jnp.where(active, X[x_index + 1], 0),
        jnp.where(active, X[x_index], 0),
        jnp.where(active, Y[y_index + 1], 0),
        jnp.where(active, Y[y_index], 0),
    )


def diagonal_tile_inputs(index, s_start, t_start, length, before_wrap,
                         X, Y, S, T, ic, static_kernel):
    """Gather one tile with in-bounds indices and zeroed inactive operands."""
    active = index < length
    s_index = jnp.clip(index - before_wrap, 0, S.shape[0] - 1)
    t_index = jnp.clip(index + (1 - before_wrap), 0, T.shape[0] - 1)
    s = jnp.where(t_start + index == 0, ic, S[s_index])
    t = jnp.where(s_start - index == 0, ic, T[t_index])
    s, t = jnp.where(active, s, 0), jnp.where(active, t, 0)
    x_index = jnp.clip(s_start - index, 0, X.shape[0] - 2)
    y_index = jnp.clip(t_start + index, 0, Y.shape[0] - 2)
    rho = static_kernel(*masked_path_points(X, Y, x_index, y_index, active))
    rho = jnp.where(active, rho, 0)
    return s, t, rho, x_index, y_index
