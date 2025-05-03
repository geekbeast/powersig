import time
import cupy as cp
from numba import cuda
import numpy as np

from .util.cupy_series import cupy_compute_dot_prod_batch
from .util.grid import get_diagonal_range
from .powersig_cupy import build_stencil, compute_vandermonde_vectors
from numba.cuda.cudadrv.devicearray import DeviceNDArray

def cuda_compute_gram_entry(
    dX_i: cp.ndarray, dY_j: cp.ndarray, order: int
) -> np.ndarray:
    """
    Host function to process two input tensors using the CUDA kernel.

    Args:
        dX: Input tensor of shape [N, d]
        dY: Input tensor of shape [M, d]

    Returns:
        Processed output grid
    """
    N, d_x = dX_i.shape
    M, d_y = dY_j.shape
    assert d_x == d_y, "Input tensors must have same second dimension"

    min_NM = min(N, M)

    # Build stencil once
    stencil = build_stencil(order)

    # Flip dX_i along the first dimension (time dimension)
    dX_i = cp.flip(dX_i, axis=0)

    # S = cuda.device_array((min_NM, order), dtype=dX_i.dtype)
    # T = cuda.device_array((min_NM, order), dtype=dX_i.dtype)
    S = cp.zeros((min_NM, order), dtype=dX_i.dtype)
    T = cp.zeros((min_NM, order), dtype=dX_i.dtype)
    S_out = cp.zeros((min_NM, order), dtype=dX_i.dtype)
    T_out = cp.zeros((min_NM, order), dtype=dX_i.dtype)
    S[0, 0] = 1.0
    T[0, 0] = 1.0

    # Generate Vandermonde vectors with high precision
    ds = 1.0 / dX_i.shape[0]
    dt = 1.0 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, dX_i.dtype)

    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    # Process each diagonal
    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dY_j.shape[0], dX_i.shape[0])

        dX_L = dX_i.shape[0] - (s_start + 1)

        # Compute dot products efficiently
        rho = cupy_compute_dot_prod_batch(
            dX_i[dX_L : dX_L + dlen],
            dY_j[t_start : (t_start + dlen)],
        )

        # Compute boundaries with branching optimizations
        skip_first = (s_start + 1) >= dX_i.shape[0]
        skip_last = (t_start + dlen) >= dY_j.shape[0]

        # Calculate grid and block dimensions
        threadsperblock = (32, 32)
        blockspergrid = (min_NM,)

        # Launch kernel
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            v_s, v_t, stencil, rho, S, T, S_out, T_out, skip_first, skip_last
        )
        
        # Swap the buffers for the next diagonal.
        S_out, S = S, S_out
        T_out, T = T, T_out

    # Return the result from T[0]
    return T[0,0]


MAX_ORDER = 8


@cuda.jit
def cuda_compute_next_boundary_inplace(
    v_s, v_t, stencil, rho, S_in, T_in, S_out, T_out, skip_first: bool, skip_last: bool
) -> None:
    """
    CUDA kernel for processing two input tensors with a scaling matrix.
    Implements a parallel wavefront pattern for computing double integrals.

    Args:
        v_s: Input tensor of shape [order]
        v_t: Input tensor of shape [order]
        stencil: Input tensor of shape [order, order]
        S_in: Input tensor of shape [diagonal_length, order]
        T_in: Input tensor of shape [diagonal_length, order]
        S_out: Output tensor of shape [diagonal_length, order]
        T_out: Output tensor of shape [diagonal_length, order]
        skip_first: Whether to skip propagating the rightmost boundary of the first tile in the diagonal
        skip_last: Whether to skip propagating the topmost boundary of the last tile of the diagonal
    """

    # Get thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    tid = tx * 32 + ty
    dlen = rho.shape[0]
    order = stencil.shape[0]

    # Declare shared memory arrays for S_current and T_current
    rho_shared = cuda.shared.array(shape=(MAX_ORDER,), dtype=cp.float64)
    stencil_shared = cuda.shared.array(shape=(MAX_ORDER, MAX_ORDER), dtype=cp.float64)
    S = cuda.shared.array(shape=(MAX_ORDER,), dtype=cp.float64)
    T = cuda.shared.array(shape=(MAX_ORDER,), dtype=cp.float64)

    if tid < order:
        # Initialize rho_shared to be the power of rho[bx] for every order.
        rho_shared[tid] = rho[bx] ** tid
        # Initialize shared memory arrays by copying from global memory
        S[tid] = S_in[bx, tid]
        T[tid] = T_in[bx, tid]

    cuda.syncthreads()

    # ADM coefficient matrix computation
    if tx < order and ty < order:
        diagonal_index = ty - tx
        if diagonal_index > 0:
            stencil_shared[tx, ty] = (
                stencil[tx, ty] * S[diagonal_index] * rho_shared[min(tx, ty)]
            )
        else:
            stencil_shared[tx, ty] = (
                stencil[tx, ty] * T[-diagonal_index] * rho_shared[min(tx, ty)]
            )

    cuda.syncthreads()

    # if tx < order and ty < order:
    #     stencil_debug[bx, tx, ty] = stencil_shared[tx, ty]
    # stencil_shared[tx, ty] = stencil[tx, ty] * rho_shared[min(tx,ty)]
    
    # Reset S and T to 0, so that we can use them for the new boundaries
    if tid < order:
        S[tid] = 0.0
        T[tid] = 0.0
    
    cuda.syncthreads()
    
    # Computation of next boundary coefficients, currently only handles orders up to 32
    # TODO: Handle orders up to 64
    
        # Calculate the values for this thread
    val_S = stencil_shared[tx, ty] * v_t[tx] if tx < order and ty< order else 0
    val_T = stencil_shared[ty, tx] * v_s[tx] if tx < order and ty <order else 0
    
    # Calculate active threads mask based on order
    # active_mask = (1 << order) - 1
    # active_mask = active_mask if order < 32 else 0xffffffff
    active_mask = 0xffffffff
    
    # Perform shuffle reduction only within active threads
    tmp_S = cuda.shfl_down_sync(active_mask, val_S, 16)
    tmp_T = cuda.shfl_down_sync(active_mask, val_T, 16)
    if tx < 16:
        val_S += tmp_S
        val_T += tmp_T

    tmp_S = cuda.shfl_down_sync(active_mask, val_S, 8)
    tmp_T = cuda.shfl_down_sync(active_mask, val_T, 8)
    if tx < 8:
        val_S += tmp_S
        val_T += tmp_T

    tmp_S = cuda.shfl_down_sync(active_mask, val_S, 4)
    tmp_T = cuda.shfl_down_sync(active_mask, val_T, 4)
    if tx < 4:
        val_S += tmp_S
        val_T += tmp_T

    tmp_S = cuda.shfl_down_sync(active_mask, val_S, 2)
    tmp_T = cuda.shfl_down_sync(active_mask, val_T, 2)
    if tx < 2:
        val_S += tmp_S
        val_T += tmp_T

    tmp_S = cuda.shfl_down_sync(active_mask, val_S, 1)
    tmp_T = cuda.shfl_down_sync(active_mask, val_T, 1)
    if tx < 1:
        val_S += tmp_S
        val_T += tmp_T

    if tx == 0:
        S[ty] += val_S
        T[ty] += val_T
    
    cuda.syncthreads()

    # if tid < order:
    #     S_debug[bx, tid] = S[tid]
    #     T_debug[bx, tid] = T[tid]
    # This is the last tile
    if skip_first and skip_last and dlen == 1 and tx == 0:
        if ty == 0:
            T[tx] = 0.0
        cuda.syncthreads()
        cuda.atomic.add(T, tx, S[ty] * v_s[ty])
        cuda.syncthreads()
        if ty == 0:
            T_out[0, tx] = T[tx]
        return

    # Write data out to global memory
    if tid < order:  # Ensure that we don't try to write out of bounds
        if skip_first and skip_last:
            # Shrinking
            if bx > 0:
                # Bottom tile of diagonal is not propagating to right boundary
                S_out[bx - 1, tid] = S[tid]
            if (
                bx < dlen - 2
            ):  # -2 because -1 is length of next diagonal and -2 is the index of the last tile in that diagonal
                T_out[bx, tid] = T[tid]

        elif not skip_first and not skip_last:
            # Growing
            if bx > 0:
                S_out[bx + 1, tid] = S[tid]
            else:
                if tid == 0:
                    S_out[bx, tid] = 1  # bx = 0
                else:
                    S_out[bx, tid] = 0  # bx = 0

            if bx < dlen - 1:
                T_out[bx, tid] = T[tid]
            else:
                if tid == 0:
                    T_out[bx, tid] = 1  # bx = dlen-1
                else:
                    T_out[bx, tid] = 0  # bx = dlen-1
        elif skip_first and not skip_last:
            # Staying the same size
            S_out[bx, tid] = S[tid]
            if bx > 0:
                T_out[bx - 1, tid] = T[tid]
            else:
                if tid == 0:
                    T_out[dlen - 1, tid] = 1
                else:
                    T_out[dlen - 1, tid] = 0
        else:
            # Staying the same size
            if bx < dlen - 1:
                S_out[bx + 1, tid] = S[tid]
            else:
                if tid == 0:
                    S_out[0, tid] = 1
                else:
                    S_out[0, tid] = 0

            T_out[bx, tid] = T[tid]

    return


if __name__ == "__main__":
    # Example usage with large random tensors
    dX = np.random.random((1 << 14, 2)).astype(np.float64)
    dY = np.random.random((1 << 14, 2)).astype(np.float64)

    start = time.time()
    rounds = 100
    for i in range(rounds):
        result = cuda_compute_gram_entry(dX, dY, 8)

    print(
        f"Processed {dX.shape[0]}x{dY.shape[0]} grid in {(time.time() - start) / rounds} seconds"
    )
