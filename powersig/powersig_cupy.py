import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional, Tuple

# Import CuPy
import cupy as cp
from .util.grid import get_diagonal_range
from .util.cupy_series import cupy_compute_dot_prod_batch


def compute_boundary(
    psi_s: cp.ndarray,
    psi_t: cp.ndarray,
    S: cp.ndarray,
    T: cp.ndarray,
    rho
):
    """
    Compute the boundary tensor power series for a fixed-size chunk.
    
    Args:
        U_s: Fixed-size chunk from larger preallocated U buffer
        U_t: Fixed-size chunk from larger preallocated U buffer
        S: Tensor of shape (batch_size, n) containing coefficients for upper diagonals
        T: Tensor of shape (batch_size, n) containing coefficients for main and lower diagonals
        rho: Tensor of shape (batch_size,) containing the rho values
        offset: Offset in the larger buffer
    """

    n = psi_s.shape[0]
    U_s = psi_s.copy()
    U_t = psi_t.copy()

    for exponent in range(n):
        rho_power = rho ** exponent
        s = S[1:S.shape[0]-exponent]  
        t = T[:T.shape[0]-exponent]
        U_s[exponent, exponent+1:] *= s
        U_s[exponent, exponent+1:] *= rho_power
        U_s[exponent:, exponent] *= t
        U_s[exponent:, exponent] *= rho_power
    
        U_t[exponent, exponent+1:] *= s
        U_t[exponent, exponent+1:] *= rho_power
        U_t[exponent:, exponent] *= t
        U_t[exponent:, exponent] *= rho_power

    return U_t.sum(axis=0), U_s.sum(axis=1)

def batch_ADM_for_diagonal(
    rho: cp.ndarray, U_buf: cp.ndarray, S_buf: cp.ndarray, T_buf: cp.ndarray, stencil: cp.ndarray
) -> cp.ndarray:
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
    U_buf[:batch_size, :, :] = stencil
    
    # Reshape rho for broadcasting
    rho = rho.reshape(batch_size, 1)
    
    # Loop over exponents
    for exponent in range(n):
        # Compute rho^exponent
        # rho_power = cp.power(rho, exponent)
        
        # Update rows using broadcasting
        U_buf[:batch_size, exponent, exponent+1:] *= S_buf[:batch_size, 1:S_buf.shape[1]-exponent] * (rho ** exponent)
        
        
        # Update columns using broadcasting
        U_buf[:batch_size, exponent:, exponent] *= T_buf[:batch_size, :T_buf.shape[1]-exponent] * (rho ** exponent)
       
    
    return U_buf


def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype=cp.float64
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Compute Vandermonde vectors efficiently."""
    powers = cp.arange(n, dtype=dtype)
    # Direct power calculation is more efficient for n <= 64
    v_s = cp.power(ds, powers)
    v_t = cp.power(dt, powers)
    return v_s, v_t

def build_stencil_s(v_s: cp.ndarray, order: int, dtype=cp.float64) -> cp.ndarray:
    """Build stencil matrix with optimized implementation."""
    stencil = build_stencil(order, dtype)
    return stencil * v_s

def build_stencil_t(v_t: cp.ndarray, order: int, dtype=cp.float64) -> cp.ndarray:
    """Build stencil matrix with optimized implementation."""
    stencil = build_stencil(order, dtype)
    return stencil * v_t.reshape(-1, 1)

def build_stencil(
    order: int = 32, dtype=cp.float64
) -> cp.ndarray:
    """Build stencil matrix with optimized implementation."""
    stencil = cp.ones([order, order], dtype=dtype)

    # Fill in the rest of the matrix with 1/(i*j) in a single vectorized operation
    i_indices = cp.arange(1, order, dtype=dtype).reshape(-1, 1)
    j_indices = cp.arange(1, order, dtype=dtype).reshape(1, -1)
    
    # More numerically stable division
    stencil[1:, 1:] = 1.0 / (i_indices * j_indices)

    # Process diagonals using a more vectorized approach where possible
    for k in range(-(order - 1), order):
        if k >= 0:
            i_indices = cp.arange(order - k)
            j_indices = i_indices + k
        else:
            j_indices = cp.arange(order + k)
            i_indices = j_indices - k
            
        diag_values = stencil[i_indices, j_indices]
        diag_values = cp.cumprod(diag_values)
        stencil[i_indices, j_indices] = diag_values

    return stencil

def batch_compute_boundaries(
    U: cp.ndarray,
    S_buf: cp.ndarray,
    T_buf: cp.ndarray,
    v_s: cp.ndarray,
    v_t: cp.ndarray,
    skip_first: bool = False,
    skip_last: bool = False,
) -> Tuple[cp.ndarray, cp.ndarray]:
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
        
    # U_vs = cp.matmul(U, v_s)
    # vt_U = cp.matmul(v_t, U)
    
    if skip_first and skip_last:
        # Shrinking
        next_dlen = batch_size - 1
        cp.matmul(v_t,U[:-1],out=cp.expand_dims(S_buf[:next_dlen], 1))
        cp.matmul(U[1:],v_s,out=cp.expand_dims(T_buf[:next_dlen], 2))
        # S_buf[:batch_size-1] = vt_U[:-1]
        # T_buf[:batch_size-1] = U_vs[1:]

    elif not skip_first and not skip_last:
        # Growing
        next_dlen = batch_size + 1
        # print(f"U.shape = {U.shape}")
        # print(f"U_vs.shape = {(U @ v_s).shape}")
        # print(f"S_buf.shape = {cp.expand_dims(S_buf,1).shape}")
        cp.matmul(v_t,U,out=cp.expand_dims(S_buf[1:next_dlen], 1))
        cp.matmul(U,v_s,out=cp.expand_dims(T_buf[:next_dlen-1], 2))
        # S_buf[1:batch_size+1] = vt_U
        # T_buf[:batch_size] = U_vs
    elif skip_first and not skip_last:
        next_dlen = batch_size
        cp.matmul(v_t,U,out=cp.expand_dims(S_buf[:next_dlen], 1))
        cp.matmul(U[1:],v_s,out=cp.expand_dims(T_buf[:next_dlen-1], 2))
        # S_buf[:,:] = vt_U
        # T_buf[:-1,:] = U_vs[1:]
    else:
        next_dlen = batch_size
        cp.matmul(v_t,U[:-1],out=cp.expand_dims(S_buf[1:next_dlen], 1))
        cp.matmul(U,v_s,out=cp.expand_dims(T_buf[:next_dlen], 2))
        # S_buf[1:,:] = vt_U[:-1]
        # T_buf[:, :] = U_vs
        

    return S_buf, T_buf


def batch_compute_gram_entry(
    dX_i: cp.ndarray, dY_j: cp.ndarray, scales: Optional[cp.ndarray] = None, order: int = 32
) -> cp.ndarray:
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
    dX_i = cp.flip(dX_i, axis=0)
    longest_diagonal = min(dX_i.shape[0], dY_j.shape[0])
    
    # Build stencil once
    stencil = build_stencil(order)

    # Initialize buffers once with proper shapes
    u_buf = cp.zeros(
        [longest_diagonal, stencil.shape[0], stencil.shape[1]], dtype=dX_i.dtype
    )
    S_buf = cp.zeros([longest_diagonal, order], dtype=dX_i.dtype)
    T_buf = cp.zeros([longest_diagonal, order], dtype=dX_i.dtype)
    
    # Initialize first elements with single operations
    S_buf[:, 0] = 1.0
    T_buf[:, 0] = 1.0

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
        rho = cupy_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen].reshape(-1, 1, dX_i.shape[1]),
            dY_j[t_start : (t_start + dlen)].reshape(-1, 1, dY_j.shape[1]),
        )

        # Process with ADM diagonal computation
        u_buf = batch_ADM_for_diagonal(rho, u_buf, S_buf, T_buf, stencil)

        if d == diagonal_count - 1:
            return cp.einsum('i,ij,j->', v_t, u_buf[0], v_s)
        
        # Compute boundaries with branching optimizations
        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]
        
        
        S_buf, T_buf = batch_compute_boundaries(
            u_buf[:dlen], S_buf, T_buf, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        )

