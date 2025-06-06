from math import ceil, sqrt
from typing import Optional, Tuple

# Import and use JAX configuration before any JAX imports
from powersig.jax.jax_config import configure_jax
configure_jax()
import jax
# jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import toeplitz
from functools import partial

from powersig.jax.jax_series import jax_compute_derivative, jax_compute_derivative_batch


class PowerSigJax:
    def __init__(self, order: int = 32, device: Optional[jax.Device] = None):
        # Select device - prefer CUDA if available, otherwise use CPU
        self.order = order
        if device is None:
            devices = jax.devices()
            cuda_devices = [d for d in devices if d.platform == 'gpu']
            self.device = cuda_devices[0] if cuda_devices else devices[0]
        else:
            self.device = device
        # self.exponents = jnp.arange(self.order)
        self.exponents = build_increasing_matrix(self.order, dtype=jnp.int8, device=self.device)
    
    def __call__(self, X, Y = None, symmetric: bool = False) -> jnp.ndarray:
        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)
        
        if Y is None:
            Y = X
        elif not isinstance(Y, jnp.ndarray):
            Y = jnp.array(Y)
            
        return self.compute_gram_matrix(X, Y, symmetric)
    
    @partial(jit, static_argnums=(0,3))
    def compute_signature_kernel(self, X: jnp.ndarray, Y: jnp.ndarray, device=None) -> jnp.ndarray:
        """
        Compute the signature kernel between two sets of time series. 
        Args:
            X: JAX array of shape (length, dim) representing the first set of time series
            Y: JAX array of shape (length, dim) representing the second set of time series
            symmetric: If True, computes the kernel matrix for the combined set of X and Y. Default is False.
            
        Returns:
            A float representing the signature kernel between X and Y

        """
        dX = jax_compute_derivative(X.squeeze(0))
        dY = jax_compute_derivative(Y.squeeze(0))
        # Ensure exponents are on the same device as input
        exponents = jax.device_put(self.exponents, device)
         # Calculate values we need before padding
        diagonal_count = dX.shape[0] + dY.shape[0] - 1
        longest_diagonal = min(dX.shape[0], dY.shape[0])
        ic = jnp.zeros([ self.order], dtype=dX.dtype, device=device).at[0].set(1)
        # Generate Vandermonde vectors with high precision
        ds = 1.0 / dX.shape[0]
        dt = 1.0 / dY.shape[0]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dX.dtype, device)

        # Create the stencil matrices with Vandermonde scaling
        psi_s = build_stencil_s(v_s, self.order, dX.dtype, device)
        psi_t = build_stencil_t(v_t, self.order, dY.dtype, device) 

        # psi_s = build_stencil(self.order, dX.dtype, device)
        # psi_t = psi_s
        indices = jnp.arange(longest_diagonal,dtype=jnp.int32,device=device)
        return compute_gram_entry(dX, dY, v_s, v_t, psi_s, psi_t, diagonal_count, longest_diagonal, ic, indices, exponents, order=self.order)

    @partial(jit, static_argnums=(0,3))
    def compute_signature_kernel_chunked(self, X: jnp.ndarray, Y: jnp.ndarray, device=None) -> jnp.ndarray:
        """
        Compute the signature kernel between two sets of time series. This is not jitted
        """
        dX = jax_compute_derivative(X.squeeze(0))
        dY = jax_compute_derivative(Y.squeeze(0))
        # Ensure exponents are on the same device as input
        exponents = jax.device_put(self.exponents, device)
         # Calculate values we need before padding
        diagonal_count = dX.shape[0] + dY.shape[0] - 1
        longest_diagonal = min(dX.shape[0], dY.shape[0])
        ic = jnp.zeros([ self.order], dtype=dX.dtype, device=device).at[0].set(1)
        # Generate Vandermonde vectors with high precision
        ds = 1.0 / dX.shape[0]
        dt = 1.0 / dY.shape[0]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dX.dtype, device)

        # Create the stencil matrices with Vandermonde scaling
        psi_s = build_stencil_s(v_s, self.order, dX.dtype, device)
        psi_t = build_stencil_t(v_t, self.order, dY.dtype, device)

        # psi_s = build_stencil(self.order, dX.dtype, device)
        # psi_t = psi_s  

        indices = jnp.arange(longest_diagonal,dtype=jnp.int32,device=device)
        diagonal_batch_size = ceil(sqrt(longest_diagonal))
        return chunked_compute_gram_entry(dX, dY, v_s, v_t, psi_s, psi_t, diagonal_count, diagonal_batch_size, longest_diagonal, ic, indices, exponents, order=self.order)

    # TODO: Think about jitting this
    def compute_gram_matrix(self, X: jnp.ndarray, Y: jnp.ndarray, symmetric: bool = False) -> jnp.ndarray:
        """
        Compute the Gram matrix between two sets of time series.
        Args:
            X: JAX array of shape (batch_size,length, dim) representing the first set of time series
            Y: JAX array of shape (batch_size, length, dim) representing the second set of time series
            symmetric: If True, computes the kernel matrix for the combined set of X and Y. Default is False.

        Returns:
            A JAX array of shape (batch_size, batch_size) containing the Gram matrix between X and Y
        """
        gram_matrix = jnp.zeros([X.shape[0], Y.shape[0]], dtype=X.dtype, device=X.device)
        
        # Compute the derivatives for the entire batch at once
        dX = jax_compute_derivative_batch(X)
        dY = jax_compute_derivative_batch(Y)
        
        # These will stay the same for the entire batch
        ds = 1.0 / dX.shape[1]
        dt = 1.0 / dY.shape[1]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dtype=jnp.float64,device=X.device)
        psi_s = build_stencil_s(v_s, order=self.order, dtype=jnp.float64, device=X.device)
        psi_t = build_stencil_t(v_t, order=self.order, dtype=jnp.float64, device=X.device)

        # psi_s = build_stencil(self.order, dX.dtype, X.device)
        # psi_t = psi_s
        ic = jnp.zeros([self.order], dtype=X.dtype, device=X.device).at[0].set(1)

        longest_diagonal = min(dX.shape[1], dY.shape[1])
        diagonal_count = dX.shape[1] + dY.shape[1] - 1
        indices = jnp.arange(longest_diagonal,dtype=jnp.int32,device=X.device)
        diagonal_batch_size = ceil(sqrt(longest_diagonal))

        # Ensure exponents are on the same device as input
        exponents = jax.device_put(self.exponents, X.device)
        

        # If the longest diagonal is greater than the JIT threshold, use the chunked approach.
        # The chunked approach wastes less work and is more memory efficient for large diagonals,
        # while avoiding recompilation
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if longest_diagonal <= JIT_BOUNDARY_THRESHOLD:
                    gram_matrix = gram_matrix.at[i,j].set(compute_gram_entry(dX[i], dY[j], v_s, v_t, psi_s, psi_t, diagonal_count,longest_diagonal, ic, indices, exponents, order=self.order, device=X.device))
                else:
                    gram_matrix = gram_matrix.at[i,j].set(chunked_compute_gram_entry(dX[i], dY[j], v_s, v_t, psi_s, psi_t, diagonal_count, diagonal_batch_size, longest_diagonal, ic, indices, exponents, order=self.order, device=X.device))

        return gram_matrix


DIAGONAL_CHUNK_SIZE = 1024
JIT_BOUNDARY_THRESHOLD = 64

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

@partial(jit, static_argnums=(2, 3,4))
def compute_vandermonde_vectors_jit(
    v_ds: jnp.ndarray, v_dt: jnp.ndarray, n: int, dtype=jnp.float64, device = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Vandermonde vectors efficiently."""
    powers = jnp.arange(n, dtype=dtype, device=device)
    # Direct power calculation is more efficient for n <= 64
    v_s = jnp.power(v_ds[0], powers)
    v_t = jnp.power(v_dt[0], powers)
    return v_s, v_t

def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype=jnp.float64, device=None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Vandermonde vectors by wrapping the JIT version."""
    # Convert to JAX arrays outside the compiled function
    v_ds = jnp.array([ds], dtype=dtype, device=device)
    v_dt = jnp.array([dt], dtype=dtype, device=device)
    # Explicitly use jit.jit here to control compilation
    return compute_vandermonde_vectors_jit(v_ds, v_dt, n, dtype, device)

@partial(jit, static_argnums=(0,1,2))
def build_stencil(
    order: int = 32, dtype=jnp.float64, device=None
) -> jnp.ndarray:
    """Build stencil matrix with optimized implementation."""
    stencil = jnp.ones([order, order], dtype=dtype,device=device)

    # Fill in the rest of the matrix with 1/(i*j) in a single vectorized operation
    i_indices = jnp.arange(1, order, dtype=dtype,device=device).reshape(-1, 1)
    j_indices = jnp.arange(1, order, dtype=dtype,device=device).reshape(1, -1)
    
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

@partial(jit, static_argnums=(1,2,3))
def build_stencil_s(v_s: jnp.ndarray, order: int = 32, dtype=jnp.float64, device=None) -> jnp.ndarray:
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
    stencil = build_stencil(order=order, dtype=dtype,device=device)
    
    # Multiply each row by v_s (broadcasting automatically)
    # v_s has shape (order,) and will broadcast across each row
    return stencil * v_s

@partial(jit, static_argnums=(1,2,3))
def build_stencil_t(v_t: jnp.ndarray, order: int = 32, dtype=jnp.float64, device=None) -> jnp.ndarray:
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
    stencil = build_stencil(order=order, dtype=dtype,device=device)
    
    # Multiply each column by v_t
    # Reshape v_t to allow broadcasting across columns (order, 1)
    return stencil * jnp.reshape(v_t, (-1, 1))

def build_rho_powers(order: int = 32, device=None):
    rps = jnp.zeros([order, order], dtype=jnp.float64, device=device)
    return jax.fori_loop(0, order, lambda i, x: x.at[i,i:].set(i).at[i:,i].set(i), rps)


@jit
def compute_boundary(psi_s: jnp.ndarray, psi_t: jnp.ndarray, exponents: jnp.ndarray, S: jnp.ndarray, T: jnp.ndarray, rho: jnp.ndarray):
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
    R = rho ** exponents
    U = toeplitz(T, S) * R
    # Use direct broadcasting for element-wise multiplication
    # JAX will automatically broadcast psi_s and psi_t [n, n] to match U [batch_size, n, n]
    U_s = U * psi_s # Broadcasting happens automatically
    U_t = U * psi_t # Broadcasting happens automatically
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    S, T = jnp.sum(U_t, axis=0), jnp.sum(U_s, axis=1)

    # print("Results from compute_boundary:")
    # print(f"S.shape = {S.shape}")
    # print(f"T.shape = {T.shape}")
    return S, T


@jit
def compute_boundary_vmap(psi_s: jnp.ndarray, psi_t: jnp.ndarray, exponents: jnp.ndarray, S: jnp.ndarray, T: jnp.ndarray, rho: jnp.ndarray):
    """
    Deprecated.Compute the boundary tensor power series for a fixed-size chunk. Use vmap, but it is better to use a single fused kernel.
    
    Args:
        psi_s: Fixed-size chunk from larger preallocated U buffer
        psi_t: Fixed-size chunk from larger preallocated U buffer
        S: Tensor of shape (batch_size, n) containing coefficients for upper diagonals
        T: Tensor of shape (batch_size, n) containing coefficients for main and lower diagonals
        rho: Tensor of shape (batch_size,) containing the rho values
        offset: Offset in the larger buffer
    """
    U = toeplitz(T, S)
    U_s = U * psi_s  
    U_t = U * psi_t  
    
    def process_row(r):
        def process_column(c):
            p = jnp.minimum(r, c)
            return U_s[r,c]* (rho ** p), U_t[r,c]* (rho ** p)
        return vmap(process_column, in_axes=(0))(exponents)
    
    U_s, U_t = vmap(process_row, in_axes=(0))(exponents)
    
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    S, T = jnp.sum(U_t, axis=0), jnp.sum(U_s, axis=1)

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
def map_diagonal_entry(rho,psi_s, psi_t,exponents, s, t):
    """
    Compute the boundary powers series for a given diagonal tile. This version is more highly parallelizable, but 
    will likely suffer from numerical instability for rho << 1 or large polynomial orders.
    
    Args:
        dx: s-axis time series derivative vector
        dy: t-axis time series derivative vector
        psi_s: First time series vector
        psi_t: Second time series vector
        exponents: The exponents to use for representing repeated ADM iterations for the given truncation order.
        s: Coefficients for bottom boundary
        t: Coefficients for left boundary
    """

    R = rho ** exponents
    U = toeplitz(t, s) * R
    U_s = U * psi_s 
    U_t = U * psi_t 
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    s, t = jnp.sum(U_t, axis=0), jnp.sum(U_s, axis=1)

    # print("Results from compute_boundary:")
    # print(f"s.shape = {s.shape}")
    # print(f"t.shape = {t.shape}")
    return s, t

@jit
def stable_diagonal_entry(rho: jnp.ndarray,v_s: jnp.ndarray, v_t: jnp.ndarray, stencil: jnp.ndarray, exponents: jnp.ndarray, s_coeff: jnp.ndarray, t_coeff: jnp.ndarray):
    """
    Compute the diagonal entry for a given time series derivative pair. Not as highly parallelizable as the other version, but
    in theory this should be more numerically stable rho > 1 and larger polynomial orders.
    Args:
        dx: s-axis time series derivative vector
        dy: t-axis time series derivative vector
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        exponents: The exponents to use for representing repeated ADM iterations for the given truncation order.
        s_coeff: Coefficients for upper diagonals
        t_coeff: Coefficients for main and lower diagonals
        
    Returns:
        Tuple of (row sum, column sum) for the boundary computation
    """
    R = stencil *(rho ** exponents)
    U = toeplitz(t_coeff, s_coeff) * R
    return v_t @ U, U @ v_s

@partial(jit, static_argnums=(6,7,11,12))
def compute_gram_entry(
    dX_i: jnp.ndarray, 
    dY_j: jnp.ndarray, 
    v_s: jnp.ndarray, 
    v_t: jnp.ndarray, 
    psi_s: jnp.ndarray, 
    psi_t: jnp.ndarray, 
    diagonal_count: int,
    longest_diagonal: int,
    ic: jnp.ndarray,
    indices: jnp.ndarray, 
    exponents: jnp.ndarray, 
    order: int = 32,
    device=None) -> jnp.ndarray:
    """
    Compute the gram matrix entry using a batched approach.
    
    Args:
        dX_i: First time series derivatives 
        dY_j: Second time series derivatives 
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        psi_s: First time series power series coefficients
        psi_t: Second time series power series coefficients
        diagonal_count: Number of diagonals to compute
        longest_diagonal: Longest diagonal to compute
        order: Order of the polynomial approximation
        device: Device to use for the computation
    Returns:
        Gram matrix entry (scalar)
    """

    # Initialize buffers with proper shapes
    S_buf = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype,device=device)
    T_buf = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype,device=device)

    # Initialize first elements with 1.0
    S_buf = S_buf.at[:, 0].set(1.0)
    T_buf = T_buf.at[:, 0].set(1.0)

    cols = dY_j.shape[0]
    rows = dX_i.shape[0]

    def compute_diagonal(d, carry):
        S_buf, T_buf = carry
        # s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        t_start = (d<cols)*0 + (d>=cols)*(d-cols +1)
        s_start = (d<cols)*d + (d>=cols)*(cols - 1)
        dlen = jnp.minimum(rows - t_start, s_start + 1)
        is_before_wrap = d < dX_i.shape[0]
        # dX_L = dX_i.shape[0] - (s_start + 1)

        # print(f"dX_i.shape = {dX_i.shape}")
        # print(f"dY_j.shape = {dY_j.shape}")
        # rho = jax_compute_dot_prod_batch(jnp.take(dX_i, s_start-indices, axis=0, fill_value=0), jnp.take(dY_j, t_start+indices, axis=0, fill_value=0))
        # print(f"dX_i.shape = {jnp.take(dX_i, s_start-indices, axis=0, fill_value=0).shape}")
        # print(f"dY_j.shape = {jnp.take(dY_j, t_start+indices, axis=0, fill_value=0).shape}")
        # rho = jnp.einsum('ij,ij->i', jnp.take(dX_i, s_start-indices, axis=0, fill_value=0), jnp.take(dY_j, t_start+indices, axis=0, fill_value=0),
        #                      precision=jax.lax.Precision.HIGHEST)

        def next_diagonal_entry(diagonal_index, S, T):
            # Combine the first two where statements into a single mask
            s_index = diagonal_index - is_before_wrap
            t_index = diagonal_index + (1 - is_before_wrap)

            # Avoid branching
            s = ((t_start + diagonal_index == 0) * ic) + ((t_start + diagonal_index != 0) * S[s_index])
            t = ((s_start - diagonal_index == 0) * ic) + ((s_start - diagonal_index != 0) * T[t_index])
            dX_idx = (s_start - diagonal_index) * ((s_start - diagonal_index) < dX_i.shape[0])
            dY_idx = (t_start + diagonal_index) * ((t_start + diagonal_index) < dY_j.shape[0])
            rho = jnp.dot(dX_i[dX_idx], dY_j[dY_idx], precision = jax.lax.Precision.HIGHEST)
            # jax.debug.print("""
            #     d = {},
            #     diagonal_index {}:
            #     s_start = {}
            #     t_start = {}
            #     dlen = {}
            #     is_before_wrap = {}
            #     s_index = {}
            #     t_index = {}
            #     s = {}
            #     t = {}
            #     """, d, diagonal_index, s_start, t_start, dlen, is_before_wrap, s_index, t_index, s, t)
            return map_diagonal_entry(rho, psi_s, psi_t, exponents, s, t)
            # return stable_diagonal_entry(rho, v_s, v_t, psi_s, exponents, s, t)

        
        S_next,T_next = vmap(next_diagonal_entry, in_axes=(0,None,None))(indices, S_buf, T_buf)
        
        return S_next, T_next
     
    S_buf, T_buf = jax.lax.fori_loop(0, diagonal_count, compute_diagonal, (S_buf, T_buf))
    
    return S_buf[0] @ v_s

@jit
def map_entry(
    diagonal_index,
    rho: jnp.ndarray,
    psi_s: jnp.ndarray,
    psi_t: jnp.ndarray,
    exponents: jnp.ndarray,
    s_start: int,
    t_start: int,
    ic: jnp.ndarray,
    S: jnp.ndarray,
    T: jnp.ndarray,
    is_before_wrap: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Combine the first two where statements into a single mask

    s_index = diagonal_index - is_before_wrap
    t_index = diagonal_index + (1 - is_before_wrap)

    # Select ic or existing coefficients.
    s = jnp.where(t_start + diagonal_index == 0, ic, S[s_index])
    t = jnp.where(s_start - diagonal_index == 0, ic, T[t_index])

    # jax.debug.print("""
    #     d = {},
    #     diagonal_index {}:
    #     s_start = {}
    #     t_start = {}
    #     dlen = {}
    #     is_before_wrap = {}
    #     s_index = {}
    #     t_index = {}
    #     s = {}
    #     t = {}
    #     """, d, diagonal_index, s_start, t_start, dlen, is_before_wrap, s_index, t_index, s, t)

    return map_diagonal_entry(rho, psi_s, psi_t, exponents, s, t)


@jit      
def process_chunk(
    chunk_indices: jnp.ndarray,
    rho: jnp.ndarray,
    psi_s: jnp.ndarray,
    psi_t: jnp.ndarray,
    exponents: jnp.ndarray,
    s_start: int,
    t_start: int,
    ic: jnp.ndarray,
    S_chunk: jnp.ndarray,
    T_chunk: jnp.ndarray,
    is_before_wrap: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    chunk_length = chunk_indices.shape[0]
    return vmap(
        map_entry,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
    )(
        chunk_indices,
        rho,
        psi_s,
        psi_t,
        exponents,
        s_start,
        t_start,
        ic,
        S_chunk,
        T_chunk,
        is_before_wrap,
    )

@partial(jit, static_argnums=(6,7,8,12,13))
def chunked_compute_gram_entry(
    dX_i: jnp.ndarray, 
    dY_j: jnp.ndarray, 
    v_s: jnp.ndarray, 
    v_t: jnp.ndarray, 
    psi_s: jnp.ndarray, 
    psi_t: jnp.ndarray,
    diagonal_count: int,
    diagonal_batch_size: int,
    longest_diagonal: int,
    ic: jnp.ndarray,
    indices: jnp.ndarray, 
    exponents: jnp.ndarray, 
    order: int = 32,
    device=None) -> jnp.ndarray:
    """
    Compute the gram matrix entry using a batched approach.
    
    Args:
        dX_i: First time series derivatives (padded)
        dY_j: Second time series derivatives (padded)
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        psi_s: First time series power series coefficients
        psi_t: Second time series power series coefficients
        diagonal_count: Number of diagonals to compute
        diagonal_batch_size: Size of the batch to process
        longest_diagonal: Longest diagonal to compute
        ic: Initial conditions
        indices: Indices to compute
        exponents: Exponents to use for the ADM iteration
        order: Order of the polynomial approximation
        
    Returns:
        Gram matrix entry (scalar)
    """

    # Initialize buffers with proper shapes
    S = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype,device=device)
    T = jnp.zeros([longest_diagonal, order], dtype=dX_i.dtype,device=device)

    # Initialize first elements with 1.0
    S = S.at[:, 0].set(1.0)
    T = T.at[:, 0].set(1.0)

    cols = dY_j.shape[0]
    rows = dX_i.shape[0]
    
    # print(f"dX_i.shape = {dX_i.shape}")
    # print(f"dY_j.shape = {dY_j.shape}")
    # print(f"diagonal_count = {diagonal_count}")
    # print(f"longest_diagonal = {longest_diagonal}")
    # print(f"diagonal_batch_size = {diagonal_batch_size}")

    # We will process the diagonals in fixed size batches. If number of diagonals is less than diagonal_batch size this
    # will essentially be the normal for i loop that does everything in 
    for d in range(0,diagonal_count, diagonal_batch_size):
        # print(f"d = {d} to {d+diagonal_batch_size}")
        # The reason we slice indices is avoid recompilation for all diagonal sizes.
        # s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        # of steps for this unrolled piece of the loop
        # This is the highest diagonal we will get to for this unrolled piece of the loop. 
        max_diag = min(diagonal_count, d + diagonal_batch_size)
        
        # This length of the longest diagonal length we will get to for this unrolled piece of the loop.        
        if (d+1) >= longest_diagonal and max_diag <= max(rows,cols):
            batch_longest_diag = longest_diagonal
        elif max_diag < longest_diagonal:
            batch_longest_diag = max_diag
        else:
            batch_longest_diag =  longest_diagonal - ((d + 1) - max(rows,cols))
            

        # print(f"batch_longest_diag = {batch_longest_diag}")
        # print(f"max_diag = {max_diag}")

        # Select the necessary indices for vmap [0, DIAGONAL_LIMIT)
        diagonal_indices = indices[:batch_longest_diag]
        # print(f"diagonal_indices.shape = {diagonal_indices.shape}")
        # print(f"batch_longest_diag = {batch_longest_diag}")
        def next_diagonal(diagonal_index,carry):
            # jax.debug.print("========================= START OF BATCH {} =========================\n", d)
            t_start = (diagonal_index<cols)*0 + (diagonal_index>=cols)*(diagonal_index-cols +1)
            s_start = (diagonal_index<cols)*diagonal_index + (diagonal_index>=cols)*(cols - 1)
            
            is_before_wrap = diagonal_index < dX_i.shape[0]
            # rho = jax_compute_dot_prod_batch(jnp.take(dX_i, s_start-diagonal_indices, axis=0, fill_value=0), jnp.take(dY_j, t_start+diagonal_indices, axis=0, fill_value=0))
            # rho = jnp.einsum('ij,ij->i', jnp.take(dX_i, s_start-diagonal_indices, axis=0, fill_value=0), jnp.take(dY_j, t_start+diagonal_indices, axis=0, fill_value=0),
            #                  precision=jax.lax.Precision.HIGHEST)
            def next_diagonal_entry(index_in_diagonal, S, T):
                # Combine the first two where statements into a single mask
                s_index = index_in_diagonal - is_before_wrap
                t_index = index_in_diagonal + (1 - is_before_wrap)

                # Avoid branching
                s = ((t_start + index_in_diagonal == 0) * ic) + ((t_start + index_in_diagonal != 0) * S[s_index])
                t = ((s_start - index_in_diagonal == 0) * ic) + ((s_start - index_in_diagonal != 0) * T[t_index])
                dX_idx = (s_start - index_in_diagonal) * ((s_start - index_in_diagonal) < dX_i.shape[0])
                dY_idx = (t_start + index_in_diagonal) * ((t_start + index_in_diagonal) < dY_j.shape[0])
                rho = jnp.dot(dX_i[dX_idx], dY_j[dY_idx], precision = jax.lax.Precision.HIGHEST)
                # jax.debug.print("""
                #     d = {},
                #     diagonal_index {}:
                #     s_start = {}
                #     t_start = {}
                #     dlen = {}
                #     is_before_wrap = {}
                #     s_index = {}
                #     t_index = {}
                #     s = {}
                #     t = {}
                #     """, d, diagonal_index, s_start, t_start, dlen, is_before_wrap, s_index, t_index, s, t)
                return map_diagonal_entry(rho, psi_s, psi_t, exponents, s, t)
                # return stable_diagonal_entry(rho, v_s, v_t, psi_s, exponents, s, t)
            
            # We will use our reduced diagonal indices to compute the next diagonal.
            S_current, T_current = carry
            S_next, T_next = vmap(next_diagonal_entry,in_axes=(0,None,None))(diagonal_indices, S_current, T_current)
        
            S_current = jax.lax.dynamic_update_slice(S_current, S_next, (0, 0))
            T_current = jax.lax.dynamic_update_slice(T_current, T_next, (0, 0))
            # jax.debug.print("S_current = {}\nT_current = {}", S_current, T_current)
            # jax.debug.print("========================= END OF BATCH {}=========================\n", d)
            return S_current, T_current
        
        S, T = jax.lax.fori_loop(d, max_diag, next_diagonal, (S, T))


    return S[0] @ v_s

@partial(jit, static_argnums=(0,1,2))
def build_increasing_matrix(n: int, dtype=jnp.int8, device=None) -> jnp.ndarray:
    """
    Build an n x n matrix where each value is the maximum of its row and column indices.
    For example, for n=4:
    [[0, 0, 0, 0],
     [0, 1, 1, 1],
     [0, 1, 2, 2],
     [0, 1, 2, 3]]
    
    Args:
        n: Size of the matrix
        dtype: Data type of the matrix
        
    Returns:
        Matrix of shape (n, n) with the specified pattern
    """
    # Create row and column indices
    rows = jnp.arange(n, dtype=dtype,device=device)[:, None]  # Shape: (n, 1)
    cols = jnp.arange(n, dtype=dtype,device=device)[None, :]  # Shape: (1, n)
    
    # Take maximum of row and column indices
    matrix = jnp.minimum(rows, cols)
    
    return matrix

def estimate_required_order(X: jnp.ndarray, Y: jnp.ndarray, eps=1e-8,max_order=64) -> Tuple[int, float, float, float]:
    """
    Estimate the required polynomialorder for the PowerSig signature kernel.
    """
    dX = jax_compute_derivative_batch(X)
    dY = jax_compute_derivative_batch(Y)
    def compute_sample_dx(dX_i):
        def compute_sample_dy(dY_j):
            def compute_rho_dx(dx):
                return jnp.max( (dx * dY_j).sum(axis=1))
            return jnp.max(vmap(compute_rho_dx, in_axes=(0))(dX_i))
        return jnp.max(vmap(compute_sample_dy, in_axes=(0))(dY))
    max_rho = jnp.max(vmap(compute_sample_dx, in_axes=(0))(dX))
    print(f"max_rho = {max_rho}")
    ds = 1.0 / dX.shape[0]
    dt = 1.0 / dY.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, max_order, dX.dtype, dX.device)
    # print(f"v_s = {v_s}")
    # print(f"v_t = {v_t}")
    # Create the stencil matrices with Vandermonde scaling
    psi_s = build_stencil_s(v_s, max_order, dX.dtype, dX.device)
    psi_t = build_stencil_t(v_t, max_order, dY.dtype, dX.device) 
    exponents = build_increasing_matrix(max_order, jnp.int8, dX.device)
    rho_powers = max_rho ** exponents
    U_s = psi_s * rho_powers
    U_t = psi_t * rho_powers
    
    min_s_error = jnp.inf
    min_t_error = jnp.inf
    for i in range(max_order - 1):
        U_s = U_s.at[:i,:i].set(0)
        U_t = U_t.at[:i,:i].set(0)
        # print(f"U_s = {U_s}")
        # print(f"U_t = {U_t}")
        s,t = jnp.sum(U_t), jnp.sum(U_t)
        
        if s < eps and t < eps:
            print(f"s = {s}, t = {t}")
            return i,max_rho,s,t
        min_s_error = s
        min_t_error = t
    return max_order,max_rho,min_s_error,min_t_error
