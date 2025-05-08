import time
from typing import Optional
import cupy as cp
from numba import cuda, float64
import numpy as np

from .util.cupy_series import cupy_compute_dot_prod_batch
from .util.grid import get_diagonal_range
from .powersig_cupy import (
    build_stencil,
    build_stencil_s,
    build_stencil_t,
    compute_vandermonde_vectors,
)
from numba.cuda.cudadrv.devicearray import DeviceNDArray

# def get_blocks_per_grid(order: int, dtype)


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
    psi_s = build_stencil_s(v_s, order, dX_i.dtype)
    psi_t = build_stencil_t(v_t, order, dX_i.dtype)
    
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
            v_s, v_t, psi_s, psi_t, rho, S, T, S_out, T_out, skip_first, skip_last
        )

        # Swap the buffers for the next diagonal.
        S_out, S = S, S_out
        T_out, T = T, T_out

    # Return the result from T[0]
    return T[0, 0]


MAX_ORDER = 8


@cuda.jit
def powersig_cooperative(
    v_s,
    v_t,
    psi_s: float64[:, :],
    psi_t: float64[:, :],
    S_buf: float64[:, :],
    T_buf: float64[:, :],
    dX_i: float64[:,:],
    dY_j: float64[:,:],
    result: float64[:],
    cp1_s: float64[:, :],
    cp2_t: float64[:, :],
) -> None:
    """
    CUDA kernel for computing a single signature kernel value.
    Returns a single scalar through the result array.

    Args:
        v_s: Input tensor of shape [order]
        v_t: Input tensor of shape [order]
        psi_s: Input tensor of shape [order, order]
        psi_t: Input tensor of shape [order, order]
        rho: Input tensor of shape [diagonal_length]

        result: Output tensor of shape [1] to hold the scalar result
        cp1_s: Optional tensor for intermediate results
        cp2_t: Optional tensor for intermediate results
    """
    # Get grid synchronization object
    grid = cuda.cg.this_grid()
    
    # Get thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y
    batch_size = cuda.gridDim.x
    tid = tx * 32 + ty
    order = psi_s.shape[0]
    
    longest_diagonal = min(dX_i.shape[0], dY_j.shape[0])
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    # First we setup the registers for matrices shared across all blocks.
    # Should be able to proceed to issue additional loads since no computing is going on
    if tx < order and ty < order:
        val_psi_s = psi_s[ty, tx]
        val_psi_t = psi_t[tx, ty]
    else:
        val_psi_s = 0.0
        val_psi_t = 0.0

    # Declare shared memory arrays for S_current and T_current and issue data loads
    rho = cuda.shared.array(shape=(batch_size,1), dtype=cp.float64)
    S = cuda.shared.array(shape=(batch_size,MAX_ORDER), dtype=cp.float64)
    T = cuda.shared.array(shape=(batch_size,MAX_ORDER), dtype=cp.float64)
    
    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        # We need to determine if this block is active for the current length of the diaganal.
        # A block is active if the length of the current diagonal index is greater than or equal the block index.
        if bx <= d:
            # block bx is responsible for bx + i*batch_size with i in [0, (d+ batch_size -1)//batch_size]
            for addr in range(bx, d+1, batch_size):
                shmem_addr = (addr-bx)//batch_size
                # Compute the rho value for the current diagonal.
                
                dX_L = dX_i.shape[0] - (s_start + 1)
                # Read the S and T buffers for this diagonal element. We queue these loads up as we won't need them until after rho is computed.
                if ty < order:
                    S[shmem_addr, ty] = S_buf[addr, ty]
                    T[shmem_addr, ty] = T_buf[addr, ty]
                
                # Initialize dot product value for this thread
                rho_dot = 0.0
                
                # Compute dot product using all threads in the block
                # Each thread loads and multiplies a portion of the arrays
                
                
                # Process elements in strided fashion - no bounds checking for performance (each thread is responsible for a portion of the array)
                for i in range(tid, dX_i.shape[1], threads_per_block):
                    # Directly access and multiply elements - assume inputs are properly sized
                    rho_dot += dX_i[dX_L + addr,i] * dY_j[t_start+addr,i]
                
                # Use warp shuffle to reduce within each warp
                active_mask = 0xFFFFFFFF
                for i in [16, 8, 4, 2, 1]:
                    other_val = cuda.shfl_down_sync(active_mask, rho_dot, i)
                    if (tx < i):  # Only threads with lower indices update their values
                        rho_dot += other_val
                
                # First thread in each warp has the warp's partial sum
                if tx == 0:
                    # Store in shared memory
                    cuda.atomic.add(rho, shmem_addr, rho_dot)
                
                # Make sure all warps have updated shared memory
                cuda.syncthreads()
                
                # First thread reads the final result
                if tx == 0 and ty < order:
                    # Now shared_rho[shmem_addr] has the complete dot product 
                    rho_val = rho[shmem_addr]
                else:
                    rho_val = 1000

                # Broadcast to all threads in the warp (much more efficient!)
                rho_val = cuda.shfl_sync(0xFFFFFFFF, rho_val, 0)
                
                # Now we can compute the boundary values.
                # ADM coefficient matrix computation
                if ty < order:
                    # rho_power = rho_val ** min(tx,ty)
                    if tx < ty:
                        diagonal_index = ty - tx
                        r = rho_val**tx

                        val_psi_s *= T[d, diagonal_index] * r
                        val_psi_t *= S[d, diagonal_index] * r
                        # cp1_s[bx, tx, ty] = S[diagonal_index]
                        # cp2_t[bx, ty, tx] = S[diagonal_index]
                    elif tx == ty:
                        r = rho_val**tx
                        val_psi_s *= T[d, diagonal_index] * r
                        val_psi_t *= S[d, diagonal_index] * r
                        # cp1_s[bx, tx, ty] = T[diagonal_index]
                        # cp2_t[bx, ty, tx] = T[diagonal_index]
                    else:
                        diagonal_index = tx - ty
                        r = rho_val**ty

                        val_psi_s *= S[d, diagonal_index] * r
                        val_psi_t *= T[d, diagonal_index] * r
                        # cp1_s[bx, tx, ty] = T[diagonal_index]
                        # cp2_t[bx, ty, tx] = T[diagonal_index]

                    if tx < order and ty < order:
                        cp1_s[addr, ty, tx] = val_psi_s 
                        cp2_t[addr, tx, ty] = val_psi_t 

                    active_mask = 0xFFFFFFFF
                    # Perform shuffle reduction only within active threads
                    for i in [16, 8, 4, 2, 1]:
                        tmp_S = cuda.shfl_down_sync(active_mask, val_psi_s, i)
                        tmp_T = cuda.shfl_down_sync(active_mask, val_psi_t, i)
                        if tx < i:
                            val_psi_s += tmp_S
                            val_psi_t += tmp_T

                    # At the end the first thread in each warp has the reduced value
                    if ty < order and tx == 0:
                        S_buf[addr, ty] = val_psi_t
                        T_buf[addr, ty] = val_psi_s
        else:
            break


@cuda.jit
def cuda_compute_next_boundary_inplace(
    v_s,
    v_t,
    psi_s: float64[:, :],
    psi_t: float64[:, :],
    rho: float64[:],
    S_in: float64[:, :],
    T_in: float64[:, :],
    S_out: float64[:, :],
    T_out: float64[:, :],
    skip_first: bool,
    skip_last: bool,
    cp1_s: float64[:, :],
    cp2_t: float64[:, :],
) -> None:
    """
    CUDA kernel for processing two input tensors with a scaling matrix.
    Implements a parallel wavefront pattern for computing double integrals.

    Args:
        dlen: Length of the diagonal
        v_s: Input tensor of shape [order]
        v_t: Input tensor of shape [order]
        psi_s: Input tensor of shape [order, order]
        psi_t: Input tensor of shape [order, order]
        rho: Input tensor of shape [diagonal_length]
        S_in: Input tensor of shape [diagonal_length, order]
        T_in: Input tensor of shape [diagonal_length, order]
        S_out: Output tensor of shape [diagonal_length, order]
        T_out: Output tensor of shape [diagonal_length, order]
        skip_first: Whether to skip propagating the rightmost boundary of the first tile in the diagonal
        skip_last: Whether to skip propagating the topmost boundary of the last tile of the diagonal
        cp1_s: Optional tensor of shape [diagonal_length, order, order]
        cp2_t: Optional tensor of shape [diagonal_length, order, order]
    """
    total_blocks = cuda.gridDim.x
    batch_size = cuda.blockDim.x
    # Get thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    tid = tx * 32 + ty
    order = psi_s.shape[0]

    # First we setup the registers for matrices shared across all blocks.
    # Should be able to proceed to issue additional loads since no computing is going on
    if tx < order and ty < order:
        val_psi_s = psi_s[ty, tx]
        val_psi_t = psi_t[tx, ty]
    else:
        val_psi_s = 0.0
        val_psi_t = 0.0

    # Declare shared memory arrays for S_current and T_current and issue data loads
    S = cuda.shared.array(shape=(MAX_ORDER,), dtype=cp.float64)
    T = cuda.shared.array(shape=(MAX_ORDER,), dtype=cp.float64)

    
    # Initialize shared memory arrays by copying from global memory
    if tid < order:
        S[tid] = S_in[bx, tid]
        T[tid] = T_in[bx, tid]
    
    cuda.syncthreads()
    # Thread 0 loads the value
    if tx == 0 and ty < order:  # First thread in each warp under the order 
        rho_val = rho[bx]
    else:
        rho_val = 1000

    # Broadcast to all threads in the warp (much more efficient!)
    rho_val = cuda.shfl_sync(0xFFFFFFFF, rho_val, 0)

    # ADM coefficient matrix computation
    if tx < order and ty < order:
        # rho_power = rho_val ** min(tx,ty)
        if tx < ty:
            diagonal_index = ty - tx
            r = rho_val**tx

            val_psi_s *= T[diagonal_index] * r
            val_psi_t *= S[diagonal_index] * r
            # cp1_s[bx, tx, ty] = S[diagonal_index]
            # cp2_t[bx, ty, tx] = S[diagonal_index]
        elif tx == ty:
            r = rho_val**tx
            val_psi_s *= T[diagonal_index] * r
            val_psi_t *= T[diagonal_index] * r
            # cp1_s[bx, tx, ty] = T[diagonal_index]
            # cp2_t[bx, ty, tx] = T[diagonal_index]
        else:
            diagonal_index = tx - ty
            r = rho_val**ty

            val_psi_s *= S[diagonal_index] * r
            val_psi_t *= T[diagonal_index] * r
            # cp1_s[bx, tx, ty] = T[diagonal_index]
            # cp2_t[bx, ty, tx] = T[diagonal_index]
    else:
        val_psi_s = 0.0
        val_psi_t = 0.0

    if tx < order and ty < order:
        cp1_s[bx, ty, tx] = val_psi_s 
        cp2_t[bx, tx, ty] = val_psi_t 

    active_mask = 0xFFFFFFFF
    # Perform shuffle reduction only within active threads
    for i in [16, 8, 4, 2, 1]:
        tmp_S = cuda.shfl_down_sync(active_mask, val_psi_s, i)
        tmp_T = cuda.shfl_down_sync(active_mask, val_psi_t, i)
        if tx < i:
            val_psi_s += tmp_S
            val_psi_t += tmp_T

    # At the end the first thread in each warp has the reduced value
    if ty < order and tx == 0:
        T_out[bx, ty] = val_psi_s
        S_out[bx, ty] = val_psi_t

    # cuda.syncthreads()

    # if tid < order:
    #     S_debug[bx, tid] = S[tid]
    #     T_debug[bx, tid] = T[tid]
    # This is the last tile
    # if skip_first and skip_last and dlen == 1 and tx == 0:
    #     if ty == 0:
    #         T[tx] = 0.0
    #     cuda.syncthreads()
    #     cuda.atomic.add(T, tx, S[ty] * v_s[ty])
    #     cuda.syncthreads()
    #     if ty == 0:
    #         T_out[0, tx] = T[tx]
    #     return

    # # Write data out to global memory
    # if tid < order:  # Ensure that we don't try to write out of bounds
    #     if skip_first and skip_last:
    #         # Shrinking
    #         if bx > 0:
    #             # Bottom tile of diagonal is not propagating to right boundary
    #             S_out[bx - 1, tid] = S[tid]
    #         if (
    #             bx < dlen - 2
    #         ):  # -2 because -1 is length of next diagonal and -2 is the index of the last tile in that diagonal
    #             T_out[bx, tid] = T[tid]

    #     elif not skip_first and not skip_last:
    #         # Growing
    #         if bx > 0:
    #             S_out[bx + 1, tid] = S[tid]
    #         else:
    #             if tid == 0:
    #                 S_out[bx, tid] = 1  # bx = 0
    #             else:
    #                 S_out[bx, tid] = 0  # bx = 0

    #         if bx < dlen - 1:
    #             T_out[bx, tid] = T[tid]
    #         else:
    #             if tid == 0:
    #                 T_out[bx, tid] = 1  # bx = dlen-1
    #             else:
    #                 T_out[bx, tid] = 0  # bx = dlen-1
    #     elif skip_first and not skip_last:
    #         # Staying the same size
    #         S_out[bx, tid] = S[tid]
    #         if bx > 0:
    #             T_out[bx - 1, tid] = T[tid]
    #         else:
    #             if tid == 0:
    #                 T_out[dlen - 1, tid] = 1
    #             else:
    #                 T_out[dlen - 1, tid] = 0
    #     else:
    #         # Staying the same size
    #         if bx < dlen - 1:
    #             S_out[bx + 1, tid] = S[tid]
    #         else:
    #             if tid == 0:
    #                 S_out[0, tid] = 1
    #             else:
    #                 S_out[0, tid] = 0

    #         T_out[bx, tid] = T[tid]

    # return


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
