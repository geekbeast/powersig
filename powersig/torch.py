import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from math import factorial
from typing import Optional, Tuple
import torch
import torch._dynamo

from .util.grid import get_diagonal_range

from .util.series import torch_compute_dot_prod_batch

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(mode="max-autotune", fullgraph=True,disable=True)
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
    # rho = rho.view(batch_size,1)
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


@torch.compile(mode="max-autotune", fullgraph=True,disable=True)
def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    powers = torch.arange(n, device=device, dtype=dtype)
    v_s = ds**powers
    v_t = dt**powers
    return v_s, v_t


@torch.compile(mode="max-autotune", fullgraph=True,disable=True)
def build_stencil(
    order: int = 32, device: torch.device = None, dtype: torch.dtype = torch.float64
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


@torch.compile(mode="max-autotune", fullgraph=True,disable=True)
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
        torch.matmul(
            U[1:, :, :], v_s, out=T
        )  # Skip first, don't propagate coefficients right
        torch.matmul(
            v_t, U[:-1, :, :], out=S
        )  # Skip last, don't propagate coefficients up

    elif not skip_first and not skip_last:
        # Growing
        next_dlen = U.shape[0] + 1
        # T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        # S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]

        # Top tile receives initial left boundary, tiles below propagate top boundary
        torch.matmul(U, v_s, out=T[:-1, :])
        T[-1, 0] = 1
        T[-1, 1:] = 0

        # Bottom tile receives initial bottom boundary, tiles above propagate right boundary
        torch.matmul(v_t, U, out=S[1:, :])
        S[0, 0] = 1
        S[0, 1:] = 0
    elif skip_first and not skip_last:
        # Staying the same size
        next_dlen = U.shape[0]
        # T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        # S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]
        # Bottom tile not propagating right boundary, but top tile receives initial left boundary
        torch.matmul(v_t, U, out=S)
        torch.matmul(U[1:, :, :], v_s, out=T[:-1, :])
        T[-1, 0] = 1
        T[-1, 1:] = 0            # Top boundaries are all propagating
    else:
        # Staying the same size
        next_dlen = U.shape[0]
        # T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        # S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        T = T_buf[:next_dlen, :]
        S = S_buf[:next_dlen, :]
        # Top tile not propagating top boundary, but bottom tile receives initial bottom boundary
        torch.matmul(v_t, U[:-1, :, :], out=S[1:, :])
        S[0, 0] = 1
        S[0, 1:] = 0
        torch.matmul(U, v_s, out=T)                

    return S, T


bcb = torch.compile(batch_compute_boundaries)
cdp = torch.compile(torch_compute_dot_prod_batch)
adm = torch.compile(batch_ADM_for_diagonal)

def batch_compute_gram_entry(
    dX_i: torch.Tensor,
    dY_j: torch.Tensor,
    order: int = 32,
) -> torch.Tensor:
    # Preprocessing
    dX_i[:] = dX_i.flip(0)
    longest_diagonal = min(dX_i.shape[0], dY_j.shape[0])
    stencil = build_stencil(order, dX_i.device, dX_i.dtype)
    # Initial tile
    u_buf = torch.empty(
        [longest_diagonal, stencil.shape[0], stencil.shape[1]],
        dtype=dX_i.dtype,
        device=dX_i.device,
    )
    S_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)
    T_buf = torch.zeros([longest_diagonal, order], dtype=dX_i.dtype, device=dX_i.device)
    u = u_buf[:1, :, :]
    S = S_buf[:1, :]
    T = T_buf[:1, :]
    S[0, 0] = 1
    T[0, 0] = 1

    # Generate the stencil and Vandermonde vectors

    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, u.dtype, u.device)

    prev_u = None
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        dX_L = dX_i.shape[0] - (s_start + 1)
        # print(f"dX_L = {dX_L}")
        # print(f"s_start = {s_start}")
        rho = cdp(
            dX_i[dX_L : dX_L + dlen].unsqueeze(1),
            dY_j[t_start : (t_start + dlen)].unsqueeze(1),
        )

        # prev_u = u
        u = adm(rho, u_buf, S, T, stencil)
        # del prev_u

        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]
        # old_S, old_T = S, T
        S, T = bcb(
            u, S_buf, T_buf, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        )
        # del old_S, old_T

    # return torch.matmul(torch.matmul(v_t, u), v_s).item()
    return torch.einsum("i,bij,j->", v_t, u, v_s)
