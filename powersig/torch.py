import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from math import ceil, factorial, sqrt
from typing import Optional, Tuple
import torch
import torch._dynamo

from .util.grid import get_diagonal_range

from .util.series import torch_compute_derivative, torch_compute_derivative_batch, torch_compute_dot_prod_batch

torch._dynamo.config.capture_scalar_outputs = True
DIAGONAL_CHUNK_SIZE = 16

class PowerSigTorch:
    def __init__(self, order: int = 32, device: Optional[torch.device] = None):
        # Select device - prefer CUDA if available, otherwise use CPU
        self.order = order
        if device is None:
            devices = torch.cuda.device_count()
            self.device = torch.device("cuda:1" if devices == 2 else "cuda" if devices >0 else "cpu")
        else:
            self.device = device
        # self.exponents = jnp.arange(self.order)
        self.exponents = build_increasing_matrix(self.order, dtype=torch.int8, device=self.device)
    
    @torch.compile(mode="max-autotune", fullgraph=True)
    def compute_signature_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the signature kernel between two sets of time series. 
        Args:
            X: torch tensor of shape (length, dim) representing the first set of time series
            Y: torch tensor of shape (length, dim) representing the second set of time series
            symmetric: If True, computes the kernel matrix for the combined set of X and Y. Default is False.
            
        Returns:
            A float representing the signature kernel between X and Y

        """
        dX = torch_compute_derivative(X.squeeze(0))
        dY = torch_compute_derivative(Y.squeeze(0))
         # Calculate values we need before padding
        diagonal_count = dX.shape[0] + dY.shape[0] - 1
        longest_diagonal = min(dX.shape[0], dY.shape[0])
        indices = torch.arange(longest_diagonal)
        ic = torch.zeros([ self.order], dtype=dX.dtype).at[0].set(1)
        diagonal_batch_size = ceil(sqrt(longest_diagonal))
        # Generate Vandermonde vectors with high precision
        ds = 1.0 / dX.shape[0]
        dt = 1.0 / dY.shape[0]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dX.dtype, dX.device)

        # Create the stencil matrices with Vandermonde scaling
        psi_s = build_stencil_s(v_s, self.order, dX.dtype)
        psi_t = build_stencil_t(v_t, self.order, dY.dtype)        
        exponents = self.exponents.to(dX.device)
        return compute_gram_entry(dX, dY, v_s, v_t, psi_s, psi_t, diagonal_count, diagonal_batch_size, longest_diagonal, ic, indices, exponents, order=self.order)
    # @torch.compile(mode="max-autotune", fullgraph=True)
    # def chunked_compute_signature_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute the signature kernel between two sets of time series.
    #     """
    #     # Generate the stencil and Vandermonde vectors
    #     ds = torch.tensor([1 / dX_i.shape[0]], dtype=dX_i.dtype, device=dX_i.device)
    #     dt = torch.tensor([1 / dY_j.shape[0]], dtype=dY_j.dtype, device=dY_j.device)
    #     torch.compiler.cudagraph_mark_step_begin()
    #     v_s, v_t = compute_vandermonde_vectors(ds, dt, order)
    #     psi_s = build_stencil_s(v_s, order, dX_i.device, dX_i.dtype)
    #     psi_t = build_stencil_t(v_t, order, dY_j.device, dY_j.dtype)

    #     diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1
    #     return compute_gram_entry(dX, dY, v_s, v_t, psi_s, psi_t, diagonal_count, longest_diagonal, ic, indices, self.exponents, order=self.order)
    
    # TODO: Think about jitting this
    def compute_gram_matrix(self, X: torch.Tensor, Y: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
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
        
        # These will stay the same for the entire batch
        ds = 1.0 / X.shape[1]
        dt = 1.0 / Y.shape[1]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dtype=jnp.float64)
        psi_s = build_stencil_s(v_s, order=self.order, dtype=X.dtype)
        psi_t = build_stencil_t(v_t, order=self.order, dtype=X.dtype)
        ic = torch.zeros([self.order], dtype=X.dtype).at[0].set(1)
        longest_diagonal = min(X.shape[1], Y.shape[1])
        diagonal_count = X.shape[1] + Y.shape[1] - 1
        indices = torch.arange(longest_diagonal)
        

        dX = torch_compute_derivative_batch(X)
        dY = torch_compute_derivative_batch(Y)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                gram_matrix[i,j] = compute_gram_entry(dX, dY, v_s, v_t, psi_s, psi_t, diagonal_count, longest_diagonal, ic, indices, self.exponents)

        return gram_matrix

@torch.compile(mode="max-autotune", fullgraph=True)
def batch_ADM_for_diagonal(
    rho: torch.Tensor,
    U_buf: torch.Tensor,
    S: torch.Tensor,
    T: torch.Tensor,
    stencil: torch.Tensor
) -> torch.Tensor:
    """
    Use ADM to compute the truncated power series representation for each tile on the diagonal with refinement determined by the shape of stencil.
    Args:
        rho: Tensor of shape (batch_size,) containing the rho values
        U_buf: Pre-allocated buffer for U matrices of shape (max_batch_size, n, n)
        S: Tensor of shape (batch_size, n) containing coefficients for diagonals 0...n-1
        T: Tensor of shape (batch_size, n) containing coefficients for diagonals 0...-(n-1)
        stencil: Tensor of shape (n, n) containing the initial condition
    """
    # length of current diagonal is batch_size and determined by rho
    batch_size = rho.shape[0]
    n = stencil.shape[0]
    U = U_buf[:batch_size, :, :]
    U[:] = stencil
    rho = rho.view(batch_size,1)
    rho_powers = rho.view(batch_size,1) ** torch.arange(n, device=rho.device, dtype=rho.dtype)
    # for exponent in range(n):
    #     U[:, exponent, exponent+1:] *= S[:, 1:S.shape[1]-exponent] * (rho ** exponent)
    #     U[:, exponent:, exponent] *= T[:, :T.shape[1]-exponent] * (rho ** exponent)

    # Iterate over all diagonals from -(n-1) (bottom-left diagonal) to (n-1) (top-right diagonal)
    for k in range(-(n - 1), n):
        # multiply_diagonal(U, k, S, T, vandermonde_full)

        # Calculate the length of the diagonal
        diag_length = n - abs(k)

        # Get the view of the diagonal for all matrices in the batch
        diagonal_view = torch.diagonal(U, offset=k, dim1=1, dim2=2)

        # Take the appropriate slice of the full Vandermonde matrix
        rho_diag = rho_powers[:, :diag_length]

        # Get the coefficient and reshape for broadcasting
        if k > 0:
            # Use S for upper diagonals (k > 0)
            # Map k to index in S (1 to n-1)
            # coefficients = S[:, k].view(batch_size, 1)
            diagonal_view.mul_(S[:,k].view(batch_size, 1))
        else:
            # Use T for main and lower diagonals (k <= 0)
            # Map k to index in T (0 to n-1)
            # coefficients = T[:, -k].view(batch_size, 1)
            diagonal_view.mul_(T[:, -k].view(batch_size, 1))

        # In-place multiplication: diagonal * coefficient * vandermonde_slice
        diagonal_view.mul_(rho_diag)

    return U


@torch.compile(mode="max-autotune", fullgraph=True)
def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    powers = torch.arange(n, device=device, dtype=dtype)
    v_s = ds**powers
    v_t = dt**powers
    return v_s, v_t


# @torch.compile(mode="max-autotune", fullgraph=True)
@torch.compile()
def build_stencil(
    order: int = 32, device: torch.device=torch.device("cpu"), dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    stencil = torch.ones([order, order], dtype=dtype, device=device)

    # Fill in the rest of the matrix with 1/(i*j)
    i_indices = torch.arange(1, order, device=device).reshape(-1, 1)
    j_indices = torch.arange(1, order, device=device).reshape(1, -1)

    stencil[1:, 1:] /= i_indices
    stencil[1:, 1:] /= j_indices

    # Replace each diagonal with its cumulative product
    for k in range(-(order - 1), order):
        diag = torch.diagonal(stencil, offset=k)
        diag[:] = torch.cumprod(diag, dim=0)

    return stencil

# @torch.compile(mode="max-autotune", fullgraph=True,disable=False)
@torch.compile()
def build_stencil_t(v_t: torch.Tensor, order: int = 32, device: torch.device = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Build stencil matrix and multiply each row by v_t in place.
    
    Args:
        v_t: Vandermonde vector for t direction of shape (order,)
        order: Order of the polynomial approximation
        device: Device to create tensor on
        dtype: Data type of the tensor
        
    Returns:
        Stencil matrix with columns multiplied by v_t
    """
    # First build the standard stencil
    stencil = build_stencil(order=order, device=device, dtype=dtype)
    
    # Multiply each column by v_t in place
    # Since v_t has the same length as the columns, we can use broadcasting
    # by indexing with None/newaxis along the row dimension
    stencil.mul_(v_t.view(-1, 1))
    
    return stencil

# @torch.compile(mode="max-autotune", fullgraph=True,disable=False)
@torch.compile()
def build_stencil_s(v_s: torch.Tensor, order: int = 32, device: torch.device = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Build stencil matrix and multiply each column by v_s in place.
    
    Args:
        v_s: Vandermonde vector for s direction of shape (order,)
        order: Order of the polynomial approximation
        device: Device to create tensor on
        dtype: Data type of the tensor
        
    Returns:
        Stencil matrix with rows multiplied by v_s
    """
    # First build the standard stencil
    stencil = build_stencil(order=order, device=device, dtype=dtype)
    
    # Multiply each row by v_s in place
    stencil.mul_(v_s)
    
    return stencil

@torch.compile(mode="max-autotune", fullgraph=True)
def batch_compute_boundaries(
    U: torch.Tensor,
    S_buf: torch.Tensor,
    T_buf: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    skip_first: bool = False,
    skip_last: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the boundary tensor power series for a given diagonal.

    Args:
        U: Tensor of shape (batch_size, n, n) containing the power series coefficients
        S_buf: Pre-allocated buffer for S of shape (max_batch_size, n)
        T_buf: Pre-allocated buffer for T of shape (max_batch_size, n)
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        skip_first: Whether to skip propagating the rightmost boundary of the first tile in the diagonal
        skip_last: Whether to skip propagating the topmost boundary of the last tile of the diagonal
    """

    # Diagonal will always grow until it reaches top (skip_last) or right (skip_first) of grid
    if skip_first and skip_last:
        # Shrinking
        next_dlen = U.shape[0] - 1
        # T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        # S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]

        # Skip first, don't propagate coefficients right
        torch.matmul(U[1:, :, :], v_s, out=T)
        # Skip last, don't propagate coefficients up
        torch.matmul(v_t, U[:-1, :, :], out=S) 

    elif not skip_first and not skip_last:
        # Growing
        next_dlen = U.shape[0] + 1
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]

        # Top tile already has initial left boundary, tiles below propagate top boundary
        torch.matmul(U, v_s, out=T[:-1, :])

        # Bottom tile already has initial bottom boundary, tiles above propagate right boundary
        torch.matmul(v_t, U, out=S[1:, :])
    elif skip_first and not skip_last:
        # Staying the same size
        next_dlen = U.shape[0]
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]

        # Bottom tile not propagating right boundary, but top tile receives initial left boundary
        torch.matmul(v_t, U, out=S)
        torch.matmul(U[1:, :, :], v_s, out=T[:-1, :])
    else:
        # Staying the same size
        next_dlen = U.shape[0]
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]
        # Top tile not propagating top boundary, but bottom tile receives initial bottom boundary
        torch.matmul(v_t, U[:-1, :, :], out=S[1:, :])
        torch.matmul(U, v_s, out=T)                

    return S, T


# @torch.compile(mode="max-autotune", fullgraph=True,disable=False)
@torch.compile(dynamic=True)
def compute_boundary(
    psi_s: torch.Tensor,
    psi_t: torch.Tensor,
    S: torch.Tensor,
    T: torch.Tensor,
    rho: torch.Tensor,
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
    # assert psi_s.shape[0] == psi_t.shape[0], f"psi_s and psi_t must have the same batch size, but got {psi_s.shape[0]} and {psi_t.shape[0]}"
    # assert S.shape[1] == psi_s.shape[1], f"S must have the same number of elements as psi_s and psi_t have columns {S.shape[0]} and {psi_s.shape[1]}"
    # assert T.shape[1] == psi_s.shape[0], f"T must have the same number of elements as psi_s and psi_t have rows {T.shape[0]} and {psi_s.shape[0]}"

    n = psi_s.shape[0]
    batch_size = rho.shape[0]
    U_s = psi_s.repeat(batch_size, 1, 1)
    U_t = psi_t.repeat(batch_size, 1, 1)

    # rho_powers = rho.view(batch_size,1) ** torch.arange(n, device=rho.device, dtype=rho.dtype)
    # Initialize U_s and U_t from batch_size tilings of psi_s and psi_t
    # Use repeat for actual memory allocation since we'll modify these tensors in-place

    rho = rho.view(batch_size,1)

    for exponent in range(n):
        rho_power = rho ** exponent
        s = S[:, 1:S.shape[1]-exponent]  
        t = T[:, :T.shape[1]-exponent]
        U_s[:, exponent, exponent+1:] *= s
        U_s[:, exponent, exponent+1:] *= rho_power
        U_s[:, exponent:, exponent] *= t
        U_s[:, exponent:, exponent] *= rho_power
    
        U_t[:, exponent, exponent+1:] *= s
        U_t[:, exponent, exponent+1:] *= rho_power
        U_t[:, exponent:, exponent] *= t
        U_t[:, exponent:, exponent] *= rho_power

    # Iterate over all diagonals from -(n-1) (bottom-left diagonal) to (n-1) (top-right diagonal)
    # for k in range(-(n - 1), 1):
    #     diag_index = -k
    #     diag_length = n - diag_index
    #     diagonals_of_U_s = torch.diagonal(U_s, offset=k, dim1=1, dim2=2)
    #     diagonals_of_U_s.mul_(T[:, diag_index].view(batch_size,1))
    #     diagonals_of_U_s.mul_(rho_powers[:,:diag_length])

    #     diagonals_of_U_t = torch.diagonal(U_t, offset=k, dim1=1, dim2=2)
    #     diagonals_of_U_t.mul_(T[:, diag_index].view(batch_size,1))
    #     diagonals_of_U_t.mul_(rho_powers[:,:diag_length])

    # for k in range(1, n):
    #     diag_index = k
    #     diag_length = n - diag_index
    #     diagonals_of_U_s = torch.diagonal(U_s, offset=k, dim1=1, dim2=2)
    #     diagonals_of_U_s.mul_(S[:, diag_index].view(batch_size,1))
    #     diagonals_of_U_s.mul_(rho_powers[:,:diag_length])n

    #     diagonals_of_U_t = torch.diagonal(U_t, offset=k, dim1=1, dim2=2)
    #     diagonals_of_U_t.mul_(S[:, diag_index].view(batch_size,1))
    #     diagonals_of_U_t.mul_(rho_powers[:,:diag_length])

    # sum cols, sum rows
    return U_t.sum(dim=1), U_s.sum(dim=2)

    
def compute_boundary_inplace(psi_s: torch.Tensor, psi_t: torch.Tensor, exponents: torch.Tensor, S: torch.Tensor, T: torch.Tensor, rho: torch.Tensor):
    """
    Compute the boundary tensor power series for a fixed-size chunk.
    
    Args:
        psi_s: Fixed-size chunk from larger preallocated U buffer
        psi_t: Fixed-size chunk from larger preallocated U buffer
        S: Tensor of shape (n) containing coefficients for upper diagonals
        T: Tensor of shape (n) containing coefficients for main and lower diagonals
        rho: Tensor of shape (batch_size,) containing the rho values
        offset: Offset in the larger buffer
    """
    U = rho ** exponents

    def toeplitz(index):
        torch.diagonal(U, offset=index, dim1=1, dim2=2)[:,index:].mul_( (index == 0) * S[index] + (index != 0) * T[index])
        torch.diagonal(U, offset=index, dim1=1, dim2=2)[:,index:].mul_(S[index])
        U[:,index:] = T[:T.shape[0]-index]
    
    torch.vmap(toeplitz,out_dims=None)(exponents[-1]) 
    
    # Use direct broadcasting for element-wise multiplication
    # JAX will automatically broadcast psi_s and psi_t [n, n] to match U [batch_size, n, n]
    U_s = U * psi_s # Broadcasting happens automatically
    U_t = U * psi_t # Broadcasting happens automatically
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    S = torch.sum(U_t, axis=0, out=S)
    T = torch.sum(U_s, axis=1, out=T)

    return S, T

def stable_compute_boundaries(rho:torch.Tensor, psi_s: torch.Tensor, psi_t: torch.Tensor, exponents: torch.Tensor, s: torch.Tensor, t: torch.Tensor):
    """
    Compute the boundary tensor power series for a fixed-size chunk.
    
    Args:
        psi_s: Fixed-size chunk from larger preallocated U buffer
        psi_t: Fixed-size chunk from larger preallocated U buffer
        S: Tensor of shape (n) containing coefficients for upper diagonals
        T: Tensor of shape (n) containing coefficients for main and lower diagonals
        rho: Tensor of shape (batch_size,) containing the rho values
        offset: Offset in the larger buffer
    """
    # U = rho ** exponents

    def outer(i,tv,exprow):
        def inner(j,sv,exp):
            r = rho ** (((i <= j) * i ) + ((i>j) * j))
            return r * ((tv * (j<=i)) + (sv * (j>i) ))
        return torch.vmap(inner,)(exponents[-1], s, exprow)
    
    U = torch.vmap(outer)(exponents[-1], t, exponents) 
    
    # Use direct broadcasting for element-wise multiplication
    # JAX will automatically broadcast psi_s and psi_t [n, n] to match U [batch_size, n, n]
    U_s = U * psi_s # Broadcasting happens automatically
    U_t = U * psi_t # Broadcasting happens automatically
    
    # Sum all rows of U_s and all columns of U_t within each batch and store directly in S and T
    S = torch.sum(U_t, axis=0)
    T = torch.sum(U_s, axis=1)

    return S, T

def map_diagonal_entry(dX_i, dY_j, psi_s, psi_t,exponents, s_coeff, t_coeff, s_start: int, t_start: int, diagonal_index: int):
    # Compute dot products for valid entries
    rho = torch.dot(dX_i[s_start - diagonal_index,], dY_j[t_start+ diagonal_index,])

    # Process valid entries with compute_boundary
    s, t = compute_boundary(psi_s, psi_t, exponents, s_coeff, t_coeff, rho)


@torch.compile(mode="max-autotune", fullgraph=True,dynamic=True)
def compute_gram_entry_vmap(
    dX_i: torch.Tensor,
    dY_j: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    psi_s: torch.Tensor,
    psi_t: torch.Tensor,  
    diagonal_count: int,
    longest_diagonal: int,
    ic: torch.Tensor,
    indices: torch.Tensor,
    exponents: torch.Tensor,
    order: int = 32,
) -> torch.Tensor:
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
        ic: Initial condition for the power series
        indices: Indices of the diagonals to compute
        exponents: Exponents of the power series
        order: Order of the polynomial approximation
        
    Returns:
        Gram matrix entry (scalar)
    """
    # Initialize buffers with proper shapes
    S_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)
    T_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)

    # Initialize first elements with 1.0
    S_buf[:, 0] = 1.0
    T_buf[:, 0] = 1.0

    for d in range(diagonal_count):
        rows = dX_i.shape[0]
        cols = dY_j.shape[0]
        t_start = (d<cols)*0 + (d>=cols)*(d-cols +1)
        s_start = (d<cols)*d + (d>=cols)*(cols - 1)
        dlen = min(rows - t_start, s_start + 1)
        
        def next_diagonal_entry(diagonal_index):  
            # Combine the first two where statements into a single mask
            is_before_wrap = d < dX_i.shape[0]
            s_index = diagonal_index - is_before_wrap
            t_index = diagonal_index + (1 - is_before_wrap)
            
            # Avoid branching


            # s = ((t_start + diagonal_index == 0).cuda() * ic) + ((t_start + diagonal_index != 0).cuda() * S_buf[s_index])
            # t = ((s_start - diagonal_index == 0).cuda() * ic) + ((s_start - diagonal_index != 0).cuda() * T_buf[t_index])
                   # Use vectorized operations instead of control flow
            s_mask = (t_start + diagonal_index == 0).to(dtype=dX_i.dtype, device=dX_i.device)
            t_mask = (s_start - diagonal_index == 0).to(dtype=dX_i.dtype, device=dX_i.device)
            
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
            s = s_mask * ic + (1 - s_mask) * S_buf[s_index]
            t = t_mask * ic + (1 - t_mask) * T_buf[t_index]
            map_diagonal_entry(dX_i, dY_j, psi_s, psi_t, exponents, s, t, s_start, t_start, diagonal_index)
        
        torch.vmap(next_diagonal_entry, out_dims=None)(indices[:d+1])
    
    return S_buf[0] @ v_s


        

# @torch.compile(mode="max-autotune-no-cudagraphs",dynamic=True)
# @torch.compile(dynamic=True)
@torch.compile(mode="max-autotune", fullgraph=True,dynamic=True)
def compute_gram_entry(
    dX_i: torch.Tensor,
    dY_j: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    psi_s: torch.Tensor,
    psi_t: torch.Tensor,
    diagonal_count: int,
    diagonal_batch_size: int,
    longest_diagonal: int,
    ic: torch.Tensor,
    indices: torch.Tensor,
    exponents: torch.Tensor,
    order: int = 32,
) -> torch.Tensor:
    # Initial tile
    S_buf = torch.zeros(
        [longest_diagonal+1, order],
        dtype=dX_i.dtype,
        device=dX_i.device,
    )

    T_buf = torch.zeros(
        [longest_diagonal, order],
        dtype=dX_i.dtype,
        device=dX_i.device,
    )

    S_buf[:, 0] = 1
    T_buf[:, 0] = 1

    cols = dY_j.shape[0]
    rows = dX_i.shape[0]

    for d in range(0,diagonal_count, diagonal_batch_size):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]
        max_diag = min(diagonal_count, d + diagonal_batch_size)

        # This length of the longest diagonal length we will get to for this unrolled piece of the loop.        
        if (d+1) >= longest_diagonal and max_diag <= max(rows,cols):
            batch_longest_diag = longest_diagonal
        elif max_diag < longest_diagonal:
            batch_longest_diag = max_diag
        else:
            batch_longest_diag =  longest_diagonal - ((d + 1) - max(rows,cols))

        diagonal_indices = indices[:batch_longest_diag]
        

        print(f"d = {d}")
        print(f"max_diag = {max_diag}")
        print(f"diagonal_count = {diagonal_count}")
        # Process each chunk of the diagonal    
        for diagonal_index in range(d,max_diag):
            t_start = (diagonal_index<cols)*0 + (diagonal_index>=cols)*(diagonal_index-cols +1)
            s_start = (diagonal_index<cols)*diagonal_index + (diagonal_index>=cols)*(cols - 1)
            is_before_wrap = diagonal_index < dX_i.shape[0]
            
            # rho = jax_compute_dot_prod_batch(jnp.take(dX_i, s_start-diagonal_indices, axis=0, fill_value=0), jnp.take(dY_j, t_start+diagonal_indices, axis=0, fill_value=0))
            def next_diagonal_entry(index_in_diagonal, dx,dy,S, T):
                # Combine the first two where statements into a single mask
                s_index = index_in_diagonal - is_before_wrap
                t_index = index_in_diagonal + (1 - is_before_wrap)

                # Avoid branching
                # rho = torch.dot(dX_i[s_start-index_in_diagonal], dY_j[t_start+index_in_diagonal, :])
                s = ((t_start + index_in_diagonal == 0) * ic) + ((t_start + index_in_diagonal != 0) * S)
                t = ((s_start - index_in_diagonal == 0) * ic) + ((s_start - index_in_diagonal != 0) * T)
                
                # Use torch.dot directly instead of creating intermediate arrays
                # dX_idx = (s_start - index_in_diagonal) * ((s_start - index_in_diagonal) < dX_i.shape[0])
                # dY_idx = (t_start + index_in_diagonal) * ((t_start + index_in_diagonal) < dY_j.shape[0])
                rho = torch.dot(dx, dy)
                
                return stable_compute_boundaries(rho, psi_s, psi_t, exponents, s, t)
            
            S_next, T_next = torch.vmap(next_diagonal_entry, in_dims=(0,0,0,0,0))(
                diagonal_indices, 
                dX_i[(s_start - diagonal_indices) * ((s_start - diagonal_indices) < dX_i.shape[0])],
                dY_j[(t_start + diagonal_indices) * ((t_start + diagonal_indices) < dY_j.shape[0])],
                S_buf[:diagonal_indices.shape[0]], 
                T_buf[:batch_longest_diag])
            
            print(f"S_next = {S_next}")
            print(f"T_next = {T_next}")
            print(f"is_before_wrap = {is_before_wrap}")
            print(f"diagonal_index = {diagonal_index}")
            S_buf[is_before_wrap:S_next.shape[0]+is_before_wrap], T_buf[:T_next.shape[0]] = S_next, T_next[(1-is_before_wrap):]
        
        print(f"S_buf = {S_buf}")
        print(f"T_buf = {T_buf}")
        return S_buf[0] @ v_s

@torch.compile(mode="max-autotune", fullgraph=True,dynamic=True)
def batch_compute_gram_entry_psi(
    dX_i: torch.Tensor,
    dY_j: torch.Tensor,
    order: int = 32,
) -> torch.Tensor:
    # Preprocessing
    dX_i[:] = dX_i.flip(0)
    longest_diagonal = min(dX_i.shape[0], dY_j.shape[0])
    
    # Initial tile
    S_buf = torch.zeros([longest_diagonal+1, order], dtype=dX_i.dtype, device=dX_i.device)
    T_buf = torch.zeros([longest_diagonal+1, order], dtype=dX_i.dtype, device=dX_i.device)
    S_buf[:, 0] = 1
    T_buf[:, 0] = 1

    

    # Generate the stencil and Vandermonde vectors
    ds = torch.tensor([1 / dX_i.shape[0]], dtype=dX_i.dtype, device=dX_i.device)
    dt = torch.tensor([1 / dY_j.shape[0]], dtype=dY_j.dtype, device=dY_j.device)
    torch.compiler.cudagraph_mark_step_begin()
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order)
    psi_s = build_stencil_s(v_s, order, dX_i.device, dX_i.dtype)
    psi_t = build_stencil_t(v_t, order, dY_j.device, dY_j.dtype)
    
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        S = S_buf[t_start:t_start+dlen, :]
        T = T_buf[t_start:t_start+dlen, :]

        dX_L = dX_i.shape[0] - (s_start + 1)

        rho = torch_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen],
            dY_j[t_start : (t_start + dlen)],
        )

        S_next, T_next = compute_boundary(psi_s, psi_t, S, T, rho)
        
        S_buf[t_start+1:t_start+dlen+1,:] = S_next
        T_buf[t_start:t_start+dlen,:] = T_next
    
        if dlen == 1 and d == diagonal_count - 1:
            return v_t @ S_buf[t_start+1]        


def batch_compute_gram_entry(
    dX_i: torch.Tensor,
    dY_j: torch.Tensor,
    order: int = 32,
) -> torch.Tensor:
    # Preprocessing
    dX_i[:] = dX_i.flip(0)
    longest_diagonal = min(dX_i.shape[0], dY_j.shape[0])
    torch.compiler.cudagraph_mark_step_begin()
    stencil = build_stencil(order, dX_i.device, dX_i.dtype)
    # Initial tile
    u_buf = torch.empty(
        [longest_diagonal, stencil.shape[0], stencil.shape[1]],
        dtype=dX_i.dtype,
        device=dX_i.device,
    )
    S_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)
    T_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)
    S_buf[:, 0] = 1
    T_buf[:, 0] = 1

    u = u_buf[:1, :, :]
    S = S_buf[:1, :]
    T = T_buf[:1, :]


    # Generate the stencil and Vandermonde vectors
    ds = torch.tensor([1 / dX_i.shape[0]], dtype=dX_i.dtype, device=dX_i.device)
    dt = torch.tensor([1 / dY_j.shape[0]], dtype=dY_j.dtype, device=dY_j.device)
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order)
    
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        dX_L = dX_i.shape[0] - (s_start + 1)
        # print(f"dX_L = {dX_L}")
        # print(f"s_start = {s_start}")
        rho = torch_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen],
            dY_j[t_start : (t_start + dlen)],
        )

        u = batch_ADM_for_diagonal(rho, u_buf, S, T, stencil)        

        if d == diagonal_count - 1:
            return torch.einsum("i,ij,j->", v_t, u[0], v_s)
        
        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]

        # old_S, old_T = S, T
        S, T = batch_compute_boundaries(
            u, S_buf, T_buf, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        )
        # del old_S, old_T

    # return torch.matmul(torch.matmul(v_t, u), v_s).item()
    # return torch.einsum("i,bij,j->", v_t, u, v_s)


@torch.compile(mode="max-autotune", fullgraph=True)
def build_increasing_matrix(n: int, dtype=torch.int8, device=None) -> torch.Tensor:
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
    rows = torch.arange(n, dtype=dtype, device=device)[:, None]  # Shape: (n, 1)
    cols = torch.arange(n, dtype=dtype, device=device)[None, :]  # Shape: (1, n)
    
    # Take maximum of row and column indices
    matrix = torch.minimum(rows, cols)
    
    return matrix