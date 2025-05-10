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
from jax.scipy.linalg import toeplitz
from functools import partial

from .util.jax_series import jax_compute_dot_prod_batch


DIAGONAL_CHUNK_SIZE = 32

# class PowerSigJax:
#     def __init__(self, order: int = 32, device: jax.Device = jax.devices()[0], dtype: jnp.dtype = jnp.float64):
#         self.order = order
#         self.device = device
#         self.dtype = dtype
#         # Generate Vandermonde vectors with high precision
#         self.ds = 1.0 / dX_i.shape[0]
#         self.dt = 1.0 / dY_j.shape[0]
#         self.v_s, self.v_t = compute_vandermonde_vectors(self.ds, self.dt, self.order, dX_i.dtype)

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

@partial(jit, static_argnums=(2, 3))
def compute_vandermonde_vectors_jit(
    v_ds: jnp.ndarray, v_dt: jnp.ndarray, n: int, dtype=jnp.float64
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Vandermonde vectors efficiently."""
    powers = jnp.arange(n, dtype=dtype)
    # Direct power calculation is more efficient for n <= 64
    v_s = jnp.power(v_ds[0], powers)
    v_t = jnp.power(v_dt[0], powers)
    return v_s, v_t

def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype=jnp.float64
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Vandermonde vectors by wrapping the JIT version."""
    # Convert to JAX arrays outside the compiled function
    v_ds = jnp.array([ds], dtype=dtype)
    v_dt = jnp.array([dt], dtype=dtype)
    # Explicitly use jit.jit here to control compilation
    return compute_vandermonde_vectors_jit(v_ds, v_dt, n, dtype)

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

@partial(jit, static_argnums=(1,2))
def build_stencil_s(v_s: jnp.ndarray, order: int = 32, dtype=jnp.float64) -> jnp.ndarray:
    """
    Build stencil matrix and multiply each row by v_s.
    
    Args:
        v_s: Vandermonde vector for s direction of shape (order,)
        order: Order of the polynomial approximation
        dtype: Data type of the tensor
        
    Returns:
        Stencil matrix with rows multiplied by v_s
    """
    # First build the standard stencil
    stencil = build_stencil(order=order, dtype=dtype)
    
    # Multiply each row by v_s (broadcasting automatically)
    # v_s has shape (order,) and will broadcast across each row
    return stencil * v_s

@partial(jit, static_argnums=(1,2))
def build_stencil_t(v_t: jnp.ndarray, order: int = 32, dtype=jnp.float64) -> jnp.ndarray:
    """
    Build stencil matrix and multiply each column by v_t.
    
    Args:
        v_t: Vandermonde vector for t direction of shape (order,)
        order: Order of the polynomial approximation
        dtype: Data type of the tensor
        
    Returns:
        Stencil matrix with columns multiplied by v_t
    """
    # First build the standard stencil
    stencil = build_stencil(order=order, dtype=dtype)
    
    # Multiply each column by v_t
    # Reshape v_t to allow broadcasting across columns (order, 1)
    return stencil * jnp.reshape(v_t, (-1, 1))

@jit
def compute_boundary(psi_s: jnp.ndarray, psi_t: jnp.ndarray, S: jnp.ndarray, T: jnp.ndarray, rho: jnp.ndarray):
    """
    Compute the boundary tensor power series for a fixed-size chunk.
    
    Args:
        psi_s: Fixed-size chunk from larger preallocated U buffer
        psi_t: Fixed-size chunk from larger preallocated U buffer
        S: Tensor of shape (batch_size, n) containing coefficients for upper diagonals
        T: Tensor of shape (batch_size, n) containing coefficients for main and lower diagonals
        rho: Tensor of shape (batch_size,) containing the rho values
        offset: Offset in the larger buffer
    """
    assert psi_s.shape[0] == psi_t.shape[0], f"psi_s and psi_t must have the same batch size, but got {psi_s.shape[0]} and {psi_t.shape[0]}"
    assert S.shape[1] == psi_s.shape[1], f"S must have the same number of elements as psi_s and psi_t have columns {S.shape[0]} and {psi_s.shape[1]}"
    assert T.shape[1] == psi_s.shape[0], f"T must have the same number of elements as psi_s and psi_t have rows {T.shape[0]} and {psi_s.shape[0]}"

    n = psi_s.shape[0]
    batch_size = rho.shape[0]
    # print(f"rho.shape = {rho.shape}")
    # print(f"S.shape = {S.shape}")
    # print(f"T.shape = {T.shape}")
    # Create the Toeplitz matrix U using vmap with the full T and S
    U = vmap(lambda c, r: toeplitz(c, r))(T, S)
    
    # Use direct broadcasting for element-wise multiplication
    # JAX will automatically broadcast psi_s and psi_t [n, n] to match U [batch_size, n, n]
    U_s = U * psi_s  # Broadcasting happens automatically
    U_t = U * psi_t  # Broadcasting happens automatically
    
    # Fix JAX syntax for rho_powers and add the broadcast dimension
    # Shape goes from (batch_size, n) to (batch_size, n, 1) for broadcasting
    rho_powers = jnp.power(jnp.reshape(rho, (batch_size, 1)), jnp.arange(n, dtype=rho.dtype))
    # rho_powers = jnp.power(rho, jnp.arange(n, dtype=rho.dtype))
    # rho_powers = jnp.reshape(rho_powers, (batch_size, n, 1))
    
    # Process all powers for each batch efficiently using JAX functional updates
    # We'll use a loop over exponents but vectorize the batch operations
    for exponent in range(n):
        # print(f"rho_powers.shape = {rho_powers.shape}")
        # print(f"rho_powers[:, exponent].shape = {rho_powers[:, exponent].shape}")
     
        # print(f"U_s[:, exponent:, exponent].shape = {U_s[:, exponent:, exponent].shape}")
        # print(f"U_t[:, exponent:, exponent].shape = {U_t[:, exponent:, exponent].shape}")
        # print(f"U_s[:, exponent, exponent+1:].shape = {U_s[:, exponent, exponent+1:].shape}")
        # print(f"U_t[:, exponent, exponent+1:].shape = {U_t[:, exponent, exponent+1:].shape}")
        U_s = U_s.at[:,exponent, exponent+1:].set(U_s[:, exponent, exponent+1:] * rho_powers[:,exponent:exponent+1])
        U_t = U_t.at[:,exponent, exponent+1:].set(U_t[:, exponent, exponent+1:] * rho_powers[:,exponent:exponent+1])
        
        U_s = U_s.at[:,exponent:, exponent].set(U_s[:, exponent:, exponent] * rho_powers[:,exponent:exponent+1])
        U_t = U_t.at[:,exponent:, exponent].set(U_t[:, exponent:, exponent] * rho_powers[:,exponent:exponent+1])
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    S, T = jnp.sum(U_t, axis=1), jnp.sum(U_s, axis=2)

    # print("Results from compute_boundary:")
    # print(f"S.shape = {S.shape}")
    # print(f"T.shape = {T.shape}")
    return S, T

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

@jit
def get_diagonal_range(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    # d, s_start, t_start are 0 based indexes while rows/cols are shapes.
    t_start = jnp.where(d<cols, 0, d-cols +1)
    s_start = jnp.where(d<cols, d, cols - 1)
    dlen = jnp.minimum(rows - t_start, s_start + 1)
    # if d < cols:
    #     # if d < cols, then we haven't hit the right edge of the grid
    #     t_start = 0
    #     s_start = d
    # else:
    #     # if d >= cols then we have the right edge and wrapped around the corner
    #     t_start = d - cols + 1  # diag index - cols + 1
    #     s_start = cols - 1
    # return s_start, t_start, min(rows - t_start, s_start + 1)
    return s_start, t_start, dlen

@jit
def pad_if_needed(arr, target_size, current_size):
    return jax.lax.cond(
        current_size == target_size,
        lambda x: x,  # True branch: return as is
        lambda x: jnp.pad(x, ((0, target_size - current_size), (0, 0)), mode='constant'),  # False branch: pad
        arr
    )

@partial(jit, static_argnums=(2,))
def compute_gram_entry(dX_i: jnp.ndarray, dY_j: jnp.ndarray, order: int = 32) -> jnp.ndarray:
    """
    Compute the gram matrix entry using a batched approach.
    
    Args:
        dX_i: First time series derivatives (padded)
        dY_j: Second time series derivatives (padded)
        order: Order of the polynomial approximation
        
    Returns:
        Gram matrix entry (scalar)
    """
    # Reverse dX_i in place along axis=0
    dX_i = jnp.flip(dX_i, axis=0)
    
    # Calculate values we need before padding
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1
    longest_diagonal = max(dX_i.shape[0], dY_j.shape[0])
    
    # Generate Vandermonde vectors with high precision
    ds = 1.0 / dX_i.shape[0]
    dt = 1.0 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, dX_i.dtype)

    # Create the stencil matrices with Vandermonde scaling
    psi_s = build_stencil_s(v_s, order, dX_i.dtype)
    psi_t = build_stencil_t(v_t, order, dY_j.dtype)

    # Initialize buffers with proper shapes
    S_buf = jnp.zeros([longest_diagonal+1, order], dtype=dX_i.dtype)
    T_buf = jnp.zeros([longest_diagonal+1, order], dtype=dX_i.dtype)

    # Initialize first elements with 1.0
    S_buf = S_buf.at[:, 0].set(1.0)
    T_buf = T_buf.at[:, 0].set(1.0)

    def compute_diagonal(d, carry):
        S_buf, T_buf = carry
        cols = dY_j.shape[0]
        rows = dX_i.shape[0]
        # s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        t_start = jnp.where(d<cols, 0, d-cols +1)
        s_start = jnp.where(d<cols, d, cols - 1)
        dlen = jnp.minimum(rows - t_start, s_start + 1)
        dX_L = dX_i.shape[0] - (s_start + 1)
        # rho = jax_compute_dot_prod_batch(
        #     dX_i[(dX_L):(dX_L + dlen)],
        #     dY_j[(t_start):(t_start + dlen)]
        # )

        # # Update boundaries with the computed values
        # S_result, T_result = compute_boundary(psi_s, psi_t, S_buf[:dlen,:], T_buf[:dlen,:], rho)
        
        # for chunk_index in range(0, d, DIAGONAL_CHUNK_SIZE):
        def process_chunk(index, carry):
            chunk_index = index * DIAGONAL_CHUNK_SIZE
            S_buf, T_buf = carry
            valid_size = jnp.minimum(DIAGONAL_CHUNK_SIZE, dlen - chunk_index)
            mask = jnp.arange(DIAGONAL_CHUNK_SIZE) < valid_size
            x_indices = jnp.arange(dX_L+chunk_index, dX_L+chunk_index + DIAGONAL_CHUNK_SIZE)
            y_indices = jnp.arange(t_start+chunk_index, t_start+chunk_index + DIAGONAL_CHUNK_SIZE)

            x_indices = jnp.where(mask, x_indices, jnp.zeros_like(x_indices))
            y_indices = jnp.where(mask, y_indices, jnp.zeros_like(y_indices))

            dX_chunk = jnp.take(dX_i, x_indices, axis=0)
            dY_chunk = jnp.take(dY_j, y_indices, axis=0)

            # # dX_chunk = jax.lax.cond(
            # #         valid_size == DIAGONAL_CHUNK_SIZE,
            #         lambda x: jax.lax.dynamic_slice(x, (dX_L + chunk_index, 0), (valid_size, x.shape[1])),  # True branch: return as is
            #         lambda x: jnp.pad(jax.lax.dynamic_slice(x, (dX_L + chunk_index, 0), (valid_size, x.shape[1])), ((0, DIAGONAL_CHUNK_SIZE - valid_size), (0, 0)), mode='constant'),  # False branch: pad
            #         dX_i
            # )

            # dY_chunk = jax.lax.cond(
            #         valid_size == DIAGONAL_CHUNK_SIZE,
            #         lambda x: jax.lax.dynamic_slice(x, (t_start + chunk_index, 0), (DIAGONAL_CHUNK_SIZE, x.shape[1])),  # True branch: return as is
            #         lambda x: jnp.pad(jax.lax.dynamic_slice(x, (t_start + chunk_index, 0), (valid_size, x.shape[1])), ((0, DIAGONAL_CHUNK_SIZE - valid_size), (0, 0)), mode='constant'),  # False branch: pad
            #         dY_j
            # )
            
            S_batch = pad_if_needed(jax.lax.dynamic_slice(S_buf, (t_start, 0), (valid_size, order)), valid_size, valid_size)
            T_batch = pad_if_needed(jax.lax.dynamic_slice(T_buf, (t_start, 0), (valid_size, order)), valid_size, valid_size)
        
            # Compute dot products for valid entries
            rho = jax_compute_dot_prod_batch(dX_chunk * mask, dY_chunk * mask)
            # Process valid entries with compute_boundary
            S_result, T_result = compute_boundary(psi_s, psi_t, S_batch, T_batch, rho)
            S_buf = jax.lax.dynamic_update_slice(S_buf, S_result, (t_start+1, 0))
            T_buf = jax.lax.dynamic_update_slice(T_buf, T_result, (t_start, 0))
        
            return S_buf, T_buf
        
        return jax.lax.fori_loop(0, (dlen + DIAGONAL_CHUNK_SIZE - 1) // DIAGONAL_CHUNK_SIZE, process_chunk, (S_buf, T_buf))
        # if d == diagonal_count - 1:
        #     return jax.lax.dynamic_index_in_dim(S_buf, t_start+1, axis=0) @ v_s
    S, _ = jax.lax.fori_loop(0, diagonal_count, compute_diagonal, (S_buf, T_buf))
    return S[dY_j.shape[0]] @ v_s
      


def process_diagonal(
    v_s: jnp.ndarray,
    v_t: jnp.ndarray,
    psi_s: jnp.ndarray,
    psi_t: jnp.ndarray,
    dX_i: jnp.ndarray,
    dY_j: jnp.ndarray,
    S_buf: jnp.ndarray,
    T_buf: jnp.ndarray,
    diagonal_count: int,
    d: int,
    dX_L: int,
    t_start: int,
    dlen: int,
    skip_first: bool,
    skip_last: bool,
) -> jnp.ndarray:
    rho = jax_compute_dot_prod_batch(
        dX_i[(dX_L):(dX_L + dlen)],
        dY_j[(t_start):(t_start + dlen)]
    )

    # Update boundaries with the computed values
    S_result, T_result = compute_boundary(psi_s, psi_t, S_buf[:dlen,:], T_buf[:dlen,:], rho)
    print(f"(full)S_result = {S_result}")
    print(f"(full)T_result = {T_result}")

    # if d == diagonal_count - 1:
        # print(f"v_s = {v_s}")
        # print(f"S_buf = {S_buf}")
        # print(f"result = {jnp.matmul(S_result[0], v_s)}")
        # return S_result[0] @ v_s

    if skip_first and skip_last:
        # Shrinking
        S_buf = S_buf.at[:dlen-1].set(S_result[:-1])
        T_buf = T_buf.at[:dlen-1].set(T_result[1:])
    elif not skip_first and not skip_last:
        # Growing
        S_buf = S_buf.at[1:dlen+1].set(S_result)
        T_buf = T_buf.at[:dlen].set(T_result)
    elif skip_first and not skip_last:
        # Staying the same size
        S_buf = S_buf.at[:dlen].set(S_result)
        T_buf = T_buf.at[:dlen-1].set(T_result[1:])
    else:
        # Staying the same size
        S_buf = S_buf.at[1:dlen].set(S_result[:-1])
        T_buf = T_buf.at[:dlen].set(T_result)

    return S_buf, T_buf

def process_diagonal_chunks(
    v_s: jnp.ndarray,
    v_t: jnp.ndarray,
    psi_s: jnp.ndarray,
    psi_t: jnp.ndarray,
    dX_i: jnp.ndarray,
    dY_j: jnp.ndarray,
    S_buf: jnp.ndarray,
    T_buf: jnp.ndarray,
    diagonal_count: int,
    d: int,
    dX_L: int,
    t_start: int,
    dlen: int,
    skip_first: bool,
    skip_last: bool,
) -> jnp.ndarray:
    # Process each chunk of the diagonal
    for offset in range(0, dlen, DIAGONAL_CHUNK_SIZE):
        first_chunk = offset == 0 
        last_chunk  = offset + DIAGONAL_CHUNK_SIZE >= dlen
        chunk_size = min(DIAGONAL_CHUNK_SIZE, dlen - offset)
        # Compute dot products for the current chunk
        rho = jax_compute_dot_prod_batch(
            dX_i[(dX_L + offset):(dX_L + offset + chunk_size)],
            dY_j[(t_start + offset):(t_start + offset + chunk_size)]
        )

        # Update boundaries with the computed values
        S_batch = S_buf[offset:offset+chunk_size]
        T_batch = T_buf[offset:offset+chunk_size]

        print(f"S_batch.shape = {S_batch.shape}")
        print(f"T_batch.shape = {T_batch.shape}")
        print(f"rho.shape = {rho.shape}")

        S_result, T_result = compute_boundary(psi_s, psi_t, S_batch, T_batch, rho)
        print(f"(chunks)S_result = {S_result}")
        print(f"(chunks)T_result = {T_result}")
        # if d == diagonal_count - 1 and last_chunk: 
        #     print(f"v_s = {v_s}")
        #     print(f"S_result = {S_result}")
        #     print(f"S_buf = {S_buf}")
        #     print(f"result = {jnp.matmul(S_result[0], v_s)}")
        #     return S_result[0] @ v_s

        if skip_first and skip_last:
            # Shrinking
            # S starts at 0 and stops at 1 before the last element
            # T starts at 1 and stops at the last element
            # All chunks after the first need to be shifted down by 1
            if first_chunk:
                T_buf = T_buf.at[offset:offset+chunk_size-1].set(T_result[1:])
            else:
                T_buf = T_buf.at[offset-1:offset+chunk_size-1].set(T_result)

            if last_chunk:
                S_buf = S_buf.at[offset:offset+chunk_size-1].set(S_result[:-1])
            else:
                S_buf = S_buf.at[offset:offset+chunk_size].set(S_result)
        elif not skip_first and not skip_last:
            # Growing
            # S is shifted by 1 if it's the first chunk, since it will just take
            # initial bottom boundary for that tile.
            S_buf = S_buf.at[offset+1:offset+chunk_size+1].set(S_result)
            # T is just straight assignment
            T_buf = T_buf.at[offset:offset+chunk_size].set(T_result)
        elif skip_first and not skip_last:
            # Staying the same size
            S_buf = S_buf.at[offset:offset+chunk_size].set(S_result)
            # This one is simpler since we skip first and just keep writing chunk size for T
            # while S operation stays the same
            if first_chunk:
                T_buf = T_buf.at[offset:offset+chunk_size-1].set(T_result[1:])
            else:
                # Since first chunk was shifted by 1, we need to shift T_buf by 1 for future updates
                T_buf = T_buf.at[offset-1:offset+chunk_size-1].set(T_result)
        else:
            # Staying the same size
            if first_chunk and not last_chunk:
                S_buf = S_buf.at[1+offset:offset+chunk_size+1].set(S_result)
            elif first_chunk and last_chunk:
                # S_result will be chunk_size elements, but we are skipping last, which lines up with offset
                S_buf = S_buf.at[1+offset:offset+chunk_size].set(S_result[:-1])
            else: # first_chunk is false and last_chunk is false
                S_buf = S_buf.at[1+offset:offset+chunk_size+1].set(S_result[:-1])
            # T is just straight assignement
            T_buf = T_buf.at[offset:offset+chunk_size].set(T_result)

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
    stencil = build_stencil(order, dX_i.dtype)

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
    
    # Process each diagonal
    for d in range(diagonal_count):
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

    # Final result is always in the first element since final diagonal length is always 1
    return jnp.einsum('i,bij,j->', v_t, u_buf[:1], v_s) 
