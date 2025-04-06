import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from math import factorial
from typing import Optional, Tuple

import torch

from .util.grid import get_diagonal_range

from .util.series import torch_compute_dot_prod_batch


def multiply_diagonal(
    U: torch.Tensor,
    k: int,
    S: torch.Tensor,
    T: torch.Tensor,
    vandermonde_full: torch.Tensor,
) -> None:
    """
    Does an inplace multiply ofa diagonal of U by its corresponding coefficient and Vandermonde slice.

    Args:
        U: Tensor of shape (batch_size, n, n) containing the matrices
        k: Diagonal offset (positive for upper diagonals, negative for lower)
        S: Tensor of shape (batch_size, n) containing coefficients for diagonals 0...n-1
        T: Tensor of shape (batch_size, n) containing coefficients for diagonals 0...-(n-1)
        vandermonde_full: Tensor of shape (batch_size, n) containing the Vandermonde vectors
    """
    batch_size = U.shape[0]
    n = U.shape[1]

    # Calculate the length of the diagonal
    diag_length = n - abs(k)

    # Get the view of the diagonal for all matrices in the batch
    diagonal_view = torch.diagonal(U, offset=k, dim1=1, dim2=2)

    # Take the appropriate slice of the full Vandermonde matrix
    vandermonde_slice = vandermonde_full[:, :diag_length]

    # These two cases happen before you either hit top or right of grid
    # If you didn't skip first and S.shape[0] < batch_size then use left initial boundary
    # If you didn't skip last and T.shape[0] < batch_size then use bottom initial boundary

    # These two cases happen after you either hit top or right of grid
    # If you skipped first and S.shape[0] < batch_size then use left initial boundary on last
    # If you skipped last and T.shape[0] < batch_size then use bottom initial boundary
    # If you skipped first and last and S.shape[0] = batch_size and T.shape[0] = batch_size

    # Get the coefficient and reshape for broadcasting
    if k > 0:
        # Use S for upper diagonals (k > 0)
        # Map k to index in S (1 to n-1)
        coefficients = S[:, k].view(batch_size, 1)
    else:
        # Use T for main and lower diagonals (k <= 0)
        # Map k to index in T (0 to n-1)
        coefficients = T[:, -k].view(batch_size, 1)

    # In-place multiplication: diagonal * coefficient * vandermonde_slice
    diagonal_view.mul_(coefficients * vandermonde_slice)


def batch_ADM_for_diagonal(
    rho: torch.Tensor, S: torch.Tensor, T: torch.Tensor, stencil: torch.Tensor
) -> torch.Tensor:
    """
    Use ADM to compute the truncated power series representation for each tile on the diagonal with refinement determined by the shape of stencil.
    Args:
        rho: Tensor of shape (batch_size,) containing the rho values
        S: Tensor of shape (batch_size, n-1) containing coefficients for diagonals 0...n-1
        T: Tensor of shape (batch_size, n) containing coefficients for diagonals 0...-(n-1)
        stencil: Tensor of shape (n, n) containing the initial condition
    """
    batch_size = rho.shape[
        0
    ]  # length of current diagonal is batch_size and determined by rho
    n = stencil.shape[0]
    U = stencil.unsqueeze(0).repeat(batch_size, 1, 1)

    # Create Vandermonde vectors for the longest diagonal (main diagonal) once
    powers = torch.arange(n, device=rho.device).view(1, -1)
    vandermonde_full = rho.view(batch_size, 1) ** powers  # shape: (batch_size, n)

    # Iterate over all diagonals from -(n-1) (bottom-left diagonal) to (n-1) (top-right diagonal)
    for k in range(-(n - 1), n):
        multiply_diagonal(U, k, S, T, vandermonde_full)

    return U


@torch.compile()
def compute_vandermonde_vectors(
    ds: float, dt: float, n: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    powers = torch.arange(n, device=device, dtype=dtype)
    v_s = ds**powers
    v_t = dt**powers
    return v_s, v_t


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


# @torch.compile()
def batch_compute_boundaries(
    U: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    skip_first: bool = False,
    skip_last: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the boundary tensor power series for a given diagonal.

    Args:
        U: Tensor of shape (batch_size, n, n) containing the power series coefficients
        v_s: Vandermonde vector for s direction
        v_t: Vandermonde vector for t direction
        skip_first: Whether to skip propagating the rightmost boundary of the first tile in the diagonal
        skip_last: Whether to skip propagating the topmost boundary of the last tile of the diagonal
    """

    # Diagonal will always grow until it reaches top (skip_last) or right (skip_first) of grid
    if skip_first and skip_last:
        # Shrinking
        next_dlen = U.shape[0] - 1
        T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        torch.matmul(
            U[1:, :, :], v_s, out=T
        )  # Skip first, don't propagate coefficients right
        torch.matmul(
            v_t, U[:-1, :, :], out=S
        )  # Skip last, don't propagate coefficients up

    elif not skip_first and not skip_last:
        # Growing
        next_dlen = U.shape[0] + 1
        T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)

        # Top tile receives initial left boundary, tiles below propagate top boundary
        torch.matmul(U, v_s, out=T[:-1, :])
        T[-1, 0] = 1
        T[-1, 1:] = 0

        # Bottom tile receives initial bottom boundary, tiles above propagate right boundary
        torch.matmul(v_t, U, out=S[1:, :])
        S[0, 0] = 1
        S[0, 1:] = 0

    elif skip_first or skip_last:
        # Staying the same size
        next_dlen = U.shape[0]
        T = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)
        S = torch.empty((next_dlen, U.shape[1]), dtype=U.dtype, device=U.device)

        if skip_first:
            # Top tile, not propagating top boundary, but bottom tile receives initial bottom boundary
            torch.matmul(U[1:, :, :], v_s, out=T[1:, :])
            T[0, 0] = 1
            T[0, 1:] = 0
        else:
            # Top boundaries are all propagating
            torch.matmul(U, v_s, out=T)

        if skip_last:
            # Bottom tile, not propagating right boundary, but top tile receives initial left boundary
            torch.matmul(v_t, U[:-1, :, :], out=S[:-1, :])
            S[-1, 0] = 1
            S[-1, 1:] = 0
        else:
            # Bottom boundaries are all propagating
            torch.matmul(v_t, U, out=S)

    return S, T


def batch_compute_gram_entry(
    dX_i: torch.Tensor, dY_j: torch.Tensor, scales: torch.Tensor, order: int = 32
) -> float:
    dX_i[:] = dX_i.flip(0)
    # Initial tile
    u = torch.zeros([1, order, order], dtype=dX_i.dtype, device=dX_i.device)
    S = torch.zeros([1, order], dtype=dX_i.dtype, device=dX_i.device)
    T = torch.zeros([1, order], dtype=dX_i.dtype, device=dX_i.device)
    S[0, 0] = 1
    T[0, 0] = 1
    stencil = build_stencil(order, u.device, u.dtype)
    prev_u = None

    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, u.dtype, u.device)

    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])

        dX_L = dX_i.shape[0] - (s_start + 1)
        # print(f"dX_L = {dX_L}")
        # print(f"s_start = {s_start}")
        rho = torch_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen].unsqueeze(1),
            dY_j[t_start : (t_start + dlen)].unsqueeze(1),
        )

        prev_u = u
        u = batch_ADM_for_diagonal(rho, S, T, stencil)
        del prev_u

        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen + 1) >= dY_j.shape[0]
        old_S, old_T = S, T
        S, T = batch_compute_boundaries(
            u, v_s, v_t, skip_first=skip_first, skip_last=skip_last
        )
        del old_S, old_T

    return torch.mm(v_t, torch.mm(u, v_s)).item()
