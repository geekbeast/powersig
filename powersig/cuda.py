import os
import time
from typing import Optional, Tuple
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
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


# def get_blocks_per_grid(order: int, dtype)

def cuda_compute_gram_entry_cooperative(
    dX_i: cp.ndarray, dY_j: cp.ndarray, order: int
) -> float:
    """
    Host function to process two input tensors using the CUDA kernel.

    Args:
        dX: Input tensor of shape [N, d]
        dY: Input tensor of shape [M, d]

    """

    N, d_x = dX_i.shape
    M, d_y = dY_j.shape
    assert d_x == d_y, "Input tensors must have same second dimension"

    min_NM = min(N, M)

    # Flip dX_i along the first dimension (time dimension)
    dX_i = cp.flip(dX_i, axis=0)

    result = cp.zeros((1,), dtype=dX_i.dtype)
    S = cp.zeros((min_NM+1, order), dtype=dX_i.dtype)
    T = cp.zeros((min_NM+1, order), dtype=dX_i.dtype)
    cp1_s = cp.zeros((256, order, order), dtype=dX_i.dtype)
    cp2_t = cp.zeros((256, order, order), dtype=dX_i.dtype)
    S_next = cp.zeros((min_NM+1, order), dtype=dX_i.dtype)
    T_next = cp.zeros((min_NM+1, order), dtype=dX_i.dtype)
    S[:, 0] = 1.0
    T[:, 0] = 1.0
    S_next[:, 0] = 1.0
    T_next[:, 0] = 1.0

    # Generate Vandermonde vectors with high precision
    ds = 1.0 / dX_i.shape[0]
    dt = 1.0 / dY_j.shape[0]
    v_s, v_t = compute_vandermonde_vectors(ds, dt, order, dX_i.dtype)
    psi_s = build_stencil_s(v_s, order, dX_i.dtype)
    psi_t = build_stencil_t(v_t, order, dX_i.dtype)

     # Calculate grid and block dimensions
    threadsperblock = (32, 32)
    blockspergrid = (min(64, min_NM),)

    # Launch kernel
    powersig_cooperative[blockspergrid, threadsperblock](
        v_s, v_t, psi_s, psi_t, S, S_next, T, T_next, dX_i, dY_j, result
    )
    cuda.synchronize()
    print("Result: ", result[0])
    return result[0]

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


@cuda.jit(max_registers=64)
def powersig_cooperative(
    v_s,
    v_t,
    psi_s: float64[:, :],
    psi_t: float64[:, :],
    S: float64[:, :],
    S_next: float64[:, :],
    T: float64[:, :],
    T_next: float64[:, :],
    dX_i: float64[:,:],
    dY_j: float64[:,:],
    result: float64[:],
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
    # if tx == 0 and ty == 0:
    #     print("Launched kernel!")
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
        s_psi_s = psi_s[ty, tx]
        s_psi_t = psi_t[tx, ty]
    else:
        val_psi_s = 0.0
        val_psi_t = 0.0

    # Declare shared memory arrays for S_current and T_current and issue data loads
    rho = cuda.shared.array(shape=(1,), dtype=cp.float64)
    sk = cuda.shared.array(shape=(1,),dtype=cp.float64)
    # if tx == 0 and ty == 0:
    #     print("Starting diagonal processing.")
    # Only the first block is active at the start.
    active = bx == 0
    # if tx == 0 and ty == 0:
    #     print("Block", bx, "active = ", 1 if active else 0)
    for d in range(diagonal_count):
        # This block will never be active under the current chunking scheme
        if bx >= longest_diagonal:
            grid.sync()
            continue

        s_start, t_start, dlen = cuda_get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        dX_L = dX_i.shape[0] - (s_start + 1)
        # if tx == 0 and ty == 0:
        #     print("Block", bx, "processing diagonal", d, "with length", dlen)

        # We will only start doing work if the block index is less than the length of the diagonal.
        # Each block will stride through the diagonal in batch_size steps. The first thing to do is 
        # calculate the rho for the block. After that we will have to wait on the necessary number of
        # grid syncs, before loading the S and T values from global memory.

        for addr in range(bx, dlen, batch_size):
            val_psi_s = s_psi_s
            val_psi_t = s_psi_t
            # Compute the rho value for the current diagonal.
            
            # Initialize dot product value for this thread
            rho_val = 0.0
            
            # Compute dot product using all threads in the block
            # Each thread loads and multiplies a portion of the arrays
            # Process elements in strided fashion - no bounds checking for performance (each thread is responsible for a portion of the array)
            for i in range(tid, dX_i.shape[1], threads_per_block):
                # Directly access and multiply elements - assume inputs are properly sized
                rho_val += dX_i[dX_L + addr,i] * dY_j[t_start+addr,i]
            
            # Use warp shuffle to reduce within each warp
            
            for i in [16, 8, 4, 2, 1]:
                other_val = cuda.shfl_down_sync(0xFFFFFFFF, rho_val, i)
                if (tx < i):  # Only threads with lower indices update their values
                    rho_val += other_val
            
            # First thread in each warp has the warp's partial sum
            if tx == 0:
                # Store in shared memory
                cuda.atomic.add(rho, 0, rho_val)

            # Make sure all warps have updated shared memory
            cuda.syncthreads()
            # First we need to compute the rho value for the current diagonal. Inactive blocks will proceed up the grid until find a rho they can process.
            # Active blocks will proceed up the grid until they find a rho they can process.

            # First thread reads the final result
            if tx == 0 and ty < order:
                # Now rho[0] has the complete dot product for the current diagonal entry.
                rho_val = rho[0]
            else:
                rho_val = 1000
            # Broadcast to all threads in the warp (much more efficient as it avoids bank conflicts!)
            rho_val = cuda.shfl_sync(0xFFFFFFFF, rho_val, 0)
            
            diagonal_index = abs(tx-ty)

            # If this is the first stride of a newly active block, we need to wait on the other blocks 
            # to write out S and T values required for this diagonal before proceeding.     
            if addr == bx and not active:
                for i in range(d):
                    # if tx == 0 and ty == 0:
                    #     print("Block", bx, "marking epoch ", i ,"as completed.")
                    grid.sync()
                    # if tx == 0 and ty == 0:
                    #     print("Block", bx, "marked epoch ", i ,"as completed.")
                    active = True
            # else:
                # if tx == 0 and ty == 0: 
                #     print("Block", bx, "is executing.")
            S_val = S[addr, diagonal_index]
            T_val = T[addr, diagonal_index]

            # ADM coefficient matrix computation
            if tx < order and ty < order:
                # rho_power = rho_val ** min(tx,ty)
                if tx < ty:
                    r = rho_val**tx
                    val_psi_s *= T_val * r
                    val_psi_t *= S_val * r
                    # cp1_s[bx, tx, ty] = S[diagonal_index]
                    # cp2_t[bx, ty, tx] = S[diagonal_index]
                elif tx == ty:
                    r = rho_val**tx
                    val_psi_s *= T_val * r
                    val_psi_t *= T_val * r
                    # cp1_s[bx, tx, ty] = T[diagonal_index]
                    # cp2_t[bx, ty, tx] = T[diagonal_index]
                else:
                    r = rho_val**ty
                    val_psi_s *= S_val * r
                    val_psi_t *= T_val * r
                    # cp1_s[bx, tx, ty] = T[diagonal_index]
                    # cp2_t[bx, ty, tx] = T[diagonal_index]

            # Perform shuffle reduction only within active threads
            for i in [16, 8, 4, 2, 1]:
                tmp_S = cuda.shfl_down_sync(0xFFFFFFFF, val_psi_s, i)
                tmp_T = cuda.shfl_down_sync(0xFFFFFFFF, val_psi_t, i)
                if tx < i:
                    val_psi_s += tmp_S
                    val_psi_t += tmp_T

            # Write out results directly to global memory.
            
            # if tx == 0 and ty == 0:
                # print("Block", bx, "writing out results for diagonal", d, "and addr", addr)
            if dlen == 1 and d == diagonal_count - 1:
                # if tx == 0 and ty == 0:
                    # print("Block", bx, "writing out signature kernel result")
                if ty < order and tx == 0:
                    # We are done just need to write out the signature kernel value.
                    val_psi_t *= v_s[ty]
                    cuda.atomic.add(sk,0,val_psi_t)
                    # print("Contributed", val_psi_t, "to signature kernel result")
                cuda.syncthreads()
                if tx == 0 and ty == 0:
                    result[0] = sk[0]
                    # local = sk[0]
                    # print("Kernel computed result: ", local)
            else:
                # if tx == 0 and ty == 0:
                #     print("Writing out boundaries for diagonal ", d, "and addr ", addr)
                if ty < order and tx == 0:
                    S_next[addr+t_start+1, ty] = val_psi_t
                    T_next[addr+t_start, ty] = val_psi_s
        
        # Always swap the buffers, because next diagonal you would be expecting to read from the swapped buffer.
        S, S_next = S_next, S
        T, T_next = T_next, T

        # Only blocks that were active will call grid sync here to signify starting new epoch. 
        # Newly inactive threads will also call grid sync here, even if they don't process and diagonal entries.
        # Never active threads proceed and start computing rho for their first active entry
        if active:
            # if tx == 0 and ty == 0:
            #     print("Active block", bx, "marking epoch ", d, "as completed.")
            grid.sync()
            # if tx == 0 and ty == 0:
            #     print("Active block", bx, "marked epoch ", d, "as completed.")
        

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


@cuda.jit(device=True, inline=True)
def cuda_get_diagonal_range(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    if d < cols:
        # if d < cols, then we haven't hit the right edge of the grid
        t_start = 0
        s_start = d
    else:
        # if d >= cols then we have the right edge and wrapped around the corner
        t_start = d - cols + 1  # diag index - cols + 1
        s_start = cols - 1

    return s_start, t_start, min(rows - t_start, s_start + 1)


if __name__ == "__main__":
    # Example usage with large random tensors
    dX = np.random.random((1 << 14, 2)).astype(np.float64)
    dY = np.random.random((1 << 14, 2)).astype(np.float64)

    start = time.time()
    rounds = 100
    for i in range(rounds):
        result = cuda_compute_gram_entry_cooperative(dX, dY, 8)

    print(
        f"Processed {dX.shape[0]}x{dY.shape[0]} grid in {(time.time() - start) / rounds} seconds"
    )
