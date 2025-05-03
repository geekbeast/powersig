import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from math import factorial
from typing import Optional, Tuple

# Import and use JAX configuration before any JAX imports
from .jax_config import configure_jax
configure_jax()
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from .util.grid import get_diagonal_range
from .util.jax_series import jax_compute_dot_prod_batch


@jit
def batch_ADM_for_diagonal(
    rho: jnp.ndarray, U_buf: jnp.ndarray, S_buf: jnp.ndarray, T_buf: jnp.ndarray, stencil: jnp.ndarray
) -> jnp.ndarray:
    """
    Use ADM to compute the truncated power series representation for each tile on the diagonal with refinement determined by the shape of stencil.
    Args:
        rho: Array of shape (batch_size,) containing the rho values
        U_buf: Pre-allocated buffer for U matrices of shape (max_batch_size, n, n)
        S_buf: Buffer with coefficients for diagonals 0...n-1
        T_buf: Buffer with coefficients for diagonals 0...-(n-1)
        stencil: Array of shape (n, n) containing the initial condition
    """
    # length of current diagonal is batch_size and determined by rho
    batch_size = rho.shape[0]
    n = stencil.shape[0]
    
    # Initialize U with stencil directly in buffer (no intermediate allocation)
    U_buf = U_buf.at[:batch_size, :, :].set(stencil)
    
    # Reshape rho for broadcasting
    rho = rho.reshape(batch_size, 1)
    
    # Loop over exponents
    for exponent in range(n):
        # Compute rho^exponent
        rho_power = jnp.power(rho, exponent)
        
        # Update rows using broadcasting - compute directly in buffer to avoid views
        row_update = U_buf[:batch_size, exponent, exponent+1:] * S_buf[:batch_size, 1:S_buf.shape[1]-exponent] * rho_power
        U_buf = U_buf.at[:batch_size, exponent, exponent+1:].set(row_update)
        
        # Update columns using broadcasting - compute directly in buffer to avoid views
        col_update = U_buf[:batch_size, exponent:, exponent] * T_buf[:batch_size, :T_buf.shape[1]-exponent] * rho_power
        U_buf = U_buf.at[:batch_size, exponent:, exponent].set(col_update)
    
    return U_buf

@partial(jit, static_argnums=(0, 1, 2, 3))
def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype=jnp.float64
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Vandermonde vectors efficiently."""
    powers = jnp.arange(n, dtype=dtype)
    # Direct power calculation is more efficient for n <= 64
    v_s = jnp.power(ds, powers)
    v_t = jnp.power(dt, powers)
    return v_s, v_t


@partial(jit, static_argnums=(0,1))
def build_stencil(
    order: int = 32, dtype=jnp.float64
) -> jnp.ndarray:
    """Build stencil matrix with optimized implementation."""
    stencil = jnp.ones([order, order], dtype=dtype)

    # Fill in the rest of the matrix with 1/(i*j) in a single vectorized operation
    i_indices = jnp.arange(1, order, dtype=dtype).reshape(-1, 1)
    j_indices = jnp.arange(1, order, dtype=dtype).reshape(1, -1)
    
    # More numerically stable division
    stencil = stencil.at[1:, 1:].set(1.0 / (i_indices * j_indices))

    # Process diagonals using a more vectorized approach where possible
    for k in range(-(order - 1), order):
        if k >= 0:
            i_indices = jnp.arange(order - k)
            j_indices = i_indices + k
        else:
            j_indices = jnp.arange(order + k)
            i_indices = j_indices - k
            
        diag_values = stencil[i_indices, j_indices]
        diag_values = jnp.cumprod(diag_values)
        stencil = stencil.at[i_indices, j_indices].set(diag_values)

    return stencil

@partial(jit, static_argnums=(5, 6))
def batch_compute_boundaries(
    U: jnp.ndarray,
    S_buf: jnp.ndarray,
    T_buf: jnp.ndarray,
    v_s: jnp.ndarray,
    v_t: jnp.ndarray,
    skip_first: bool = False,
    skip_last: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the boundary tensor power series for a given diagonal.

    Args:
        U: Array of shape (batch_size, n, n) containing the power series coefficients
        S_buf: Pre-allocated buffer for S of shape (max_batch_size, n)
        T_buf: Pre-allocated buffer for T of shape (max_batch_size, n)
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        skip_first: Whether to skip propagating the rightmost boundary of the first tile in the diagonal
        skip_last: Whether to skip propagating the topmost boundary of the last tile of the diagonal
    """
    batch_size = U.shape[0]
    n = U.shape[1]
    
    U_vs = jnp.matmul(U, v_s)
    vt_U = jnp.matmul(v_t, U)
    
    if skip_first and skip_last:
        # Shrinking
        S_buf = S_buf.at[:batch_size-1].set(vt_U[:-1])
        T_buf = T_buf.at[:batch_size-1].set(U_vs[1:])

    elif not skip_first and not skip_last:
        # Growing
        S_buf = S_buf.at[1:batch_size+1].set(vt_U)
        T_buf = T_buf.at[:batch_size].set(U_vs)
        S_buf = S_buf.at[0,0].set(1.0)
        T_buf = T_buf.at[batch_size,0].set(1.0)
        S_buf = S_buf.at[0,1:].set(0.0)
        T_buf = T_buf.at[batch_size,1:].set(0.0)
    else:
        # Staying the same size
        if skip_first:
            T_buf = T_buf.at[1:batch_size].set(U_vs[1:])
            T_buf = T_buf.at[0,0].set(1.0)
            T_buf = T_buf.at[0,1:].set(0.0)
        else:
            T_buf = T_buf.at[:batch_size].set(U_vs)
        
        if skip_last:
            S_buf = S_buf.at[:batch_size-1].set(vt_U[:-1])
            S_buf = S_buf.at[batch_size-1,0].set(1.0)
            S_buf = S_buf.at[batch_size-1,1:].set(0.0)
        else:
            S_buf = S_buf.at[:batch_size].set(vt_U)

    return S_buf, T_buf

        
        


@partial(jit, static_argnums=(3,))
def batch_compute_gram_entry(
    dX_i: jnp.ndarray, dY_j: jnp.ndarray, scales: Optional[jnp.ndarray] = None, order: int = 32
) -> jnp.ndarray:
    """
    Compute the gram matrix entry using a batched approach.
    
    Args:
        dX_i: First time series derivatives
        dY_j: Second time series derivatives
        scales: Optional scaling factors
        order: Order of the polynomial approximation
        
    Returns:
        Gram matrix entry (scalar)
    """
    # Preprocessing
    dX_i = jnp.flip(dX_i, axis=0)
    longest_diagonal = max(dX_i.shape[0], dY_j.shape[0])
    
    # Build stencil once
    stencil = build_stencil(order)

    # Initialize buffers once with proper shapes
    u_buf = jnp.zeros(
        [longest_diagonal, stencil.shape[0], stencil.shape[1]], dtype=dX_i.dtype
    )
    S_buf = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype)
    T_buf = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype)
    
    # Initialize first elements with single operations
    S_buf = S_buf.at[:1, 0].set(1.0)
    T_buf = T_buf.at[:1, 0].set(1.0)

    # Generate Vandermonde vectors with high precision
    ds = 1.0 / dX_i.shape[0]
    dt = 1.0 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, dX_i.dtype)

    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    def loop_body(d):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        dX_L = dX_i.shape[0] - (s_start + 1)
        
        # Compute dot products efficiently
        rho = jax_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen].reshape(-1, 1, dX_i.shape[1]),
            dY_j[t_start : (t_start + dlen)].reshape(-1, 1, dY_j.shape[1]),
        )

        # Process with ADM diagonal computation
        u_buf = batch_ADM_for_diagonal(rho, u_buf, S_buf, T_buf, stencil)

        # Compute boundaries with branching optimizations
        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]
        S_buf, T_buf = batch_compute_boundaries(
            u_buf[:dlen], S_buf, T_buf, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        )

    jax.lax.fori_loop(0, diagonal_count, lambda d: loop_body)
    # Process each diagonal
    # for d in range(diagonal_count):
        # s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        # dX_L = dX_i.shape[0] - (s_start + 1)
        
        # # Compute dot products efficiently
        # rho = jax_compute_dot_prod_batch(
        #     dX_i[dX_L : dX_L + dlen].reshape(-1, 1, dX_i.shape[1]),
        #     dY_j[t_start : (t_start + dlen)].reshape(-1, 1, dY_j.shape[1]),
        # )

        # # Process with ADM diagonal computation
        # u_buf = batch_ADM_for_diagonal(rho, u_buf, S_buf, T_buf, stencil)

        # # Compute boundaries with branching optimizations
        # skip_first = (s_start + 1) >= dX_i.shape[0]
        # skip_last = (t_start + dlen) >= dY_j.shape[0]
        # S_buf, T_buf = batch_compute_boundaries(
        #     u_buf[:dlen], S_buf, T_buf, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        # )

    # Final result is always in the first element since final diagonal length is always 1
    return jnp.einsum('i,bij,j->', v_t, u_buf[:1], v_s) 