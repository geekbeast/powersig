import time
from typing import Tuple

import numpy as np
import torch
from numba import cuda
import math

from util.cuda import print_shared_matrix, get_number_threads


@cuda.jit(device=True)
def get_point(num_points, idx):
    if num_points == 1:
        return 0
    return idx / (num_points - 1)


@cuda.jit(device=True)
def cuda_get_step_length(d: int, rows:int, cols:int) -> Tuple[int, int, int]:
    if d < cols:
        # if d < cols, then we haven't hit the right edge of the grid
        t_start = 0
        s_start = d
    else:
        # if d >= cols then we have the right edge and wrapped around the corner
        t_start = d - cols + 1  # diag index - cols + 1
        s_start = cols - 1

    return s_start, t_start, min(rows - t_start, s_start + 1)

def get_step_length(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    # d, s_start, t_start are 0 based indexes while rows/cols are shapes.
    if d < cols:
        # if d < cols, then we haven't hit the right edge of the grid
        t_start = 0
        s_start = d
    else:
        # if d >= cols then we have the right edge and wrapped around the corner
        t_start = d - cols + 1  # diag index - cols + 1
        s_start = cols - 1

    return s_start, t_start, min(rows - t_start, s_start + 1)

@cuda.jit
def build_scaling_matrix(scaling_matrix: cuda.device_array, order: int = 32):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    scaling_matrix[ty, tx] = 1.0 / ((ty + 1) * (tx + 1))
    cuda.syncthreads()

@cuda.jit
def compute_rho_diagonal(dX : cuda.device_array, dY : cuda.device_array, rho : cuda.device_array):
    """
    CUDA kernel for computing the rho diagonal for the current step. Computes the dot product 
    starting at s_start, t_start and progressing up current_length steps from the bottom of the diagonal.

    :param dX: Input tensor of shape [R, d]
    :param dY: Input tensor of shape [R, d]
    :param rho: Output tensor of shape [R]

    This function expects `dX` and `dY` to be sliced to be of the same shape with the same leading dimension as `rho`.

    """
    _, d = dX.shape
    step = cuda.blockDim.x * cuda.blockDim.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    # We may not use all 32 entries if there aren't 32 threads for dimensions.
    shared_dp = cuda.shared.array(shape=(32,), dtype=np.float64)
    
    if tx == 0:
        if ty == 0:
            rho[bx] = 0.0
        shared_dp[ty] = 0.0
    
    cuda.syncthreads()

    # Using a local sum doesn't work since you end up with tx x ty registers that then need to be accumulated.
    local_sum = 0.0
    for i in range(0, d, step):
        d_idx = i + tx + 32*ty 
        if d_idx < d:
            local_sum += dX[bx, d_idx] * dY[bx, d_idx]

    # Add in the local_sum using a warp aggregate
    cuda.atomic.add(shared_dp, (ty), local_sum)

    cuda.syncthreads()
    
    if tx == 0:    
        cuda.atomic.add(rho, (bx), shared_dp[ty])

@cuda.jit
def compute_sigkernel_diagonal(N, M, step, global_scaling_matrix, rho_diagonal, input_diagonal, output_diagonal):
    """
    CUDA kernel for processing two input tensors with a scaling matrix.

    Args:
        dX: Input tensor of shape [N, d]
        dY: Input tensor of shape [M, d]
        output_grid: Output tensor of shape [2 * min(N,M), 32, 32]
    """
    # Get thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x

    # Shared memory for the 32x32 scaling matrix and the rotating buffer.
    scaling_matrix = cuda.shared.array(shape=(32, 32), dtype=np.float64)
    u = cuda.shared.array(shape=(32, 32), dtype=np.float64)
    u_n = cuda.shared.array(shape=(2, 32, 32), dtype=np.float64)
    u_n_partial = cuda.shared.array(shape=(32,), dtype=np.float64)

    # Only one thread per block needs to copy rho to shared memory
    # if tx == 0 and ty == 0:
    # This stores rho in a local register.
    rho = rho_diagonal[bx]

    # Copy the scaling matrix to shared memory from global memory
    scaling_matrix[tx, ty] = global_scaling_matrix[tx, ty]
   
    # u_0 = current boundary conditions computed in last step
    u[tx, ty] = input_diagonal[bx, tx, ty]
    u_n[0, tx, ty] = input_diagonal[bx, tx, ty]
    
    # Need to compute propagation to other blocks
    # Only process if thread is within valid range for this step    
    s_start, t_start, step_length = cuda_get_step_length(step, N, M)
    _, _, next_step_length = cuda_get_step_length(step + 1, N, M)
    
    # We need to compute the s, t indices for the current block
    # assuming the 0th block corresponds to the diagonal starting at s_start, t_start
    s_idx = s_start - bx
    t_idx = t_start + bx
    
    s = get_point(N, s_idx)
    t = get_point(M, t_idx)
    
    cuda.syncthreads()
    
    # Scale for integration
    for i in range(32):
        # u_n = rho * double integral of u_{n-1} with correct limits

        u_n_current_index = i % 2
        u_n_next_index = (i + 1) % 2

        if tx == 0 or ty == 0:
            u_n[u_n_next_index, tx, ty] = 0.0

        cuda.syncthreads()

        # Truncate and only shift and scale necessary elements
        if tx < 31 and ty < 31:
            s_tx = (s ** (tx + 1))
            # s_ty = (s ** (ty + 1))
            # t_tx = (t ** (tx + 1))
            t_ty = (t ** (ty + 1))

            # Compute indefinite integral
            scaled_val_ty = rho * u_n[u_n_current_index, tx, ty] * scaling_matrix[tx, ty]
            scaled_val_tx = rho * u_n[u_n_current_index, ty, tx] * scaling_matrix[ty, tx]
            u_n[u_n_next_index, tx + 1, ty + 1] = scaled_val_ty
            next_s_coeff = scaled_val_ty * t_ty # Scaled coefficient of s

            # Apply limits of integration using atomic operations for accumulation.
            cuda.atomic.add(u_n, (u_n_next_index, 0, ty + 1), next_s_coeff) # Warp aggregate accumulating local scaled_val
            cuda.atomic.add(u_n, (u_n_next_index, ty + 1, 0), scaled_val_tx * s_tx) # Warp aggregate accumulating local scaled_val * t_coeff
            cuda.atomic.add(u_n_partial, (ty+1), next_s_coeff * t_ty) # Warp aggregate accumulating local scaled_s * t_coeff

        cuda.syncthreads()
        if ty == 0 and tx > 0:
            cuda.atomic.add(u_n, (u_n_next_index, 0, 0), u_n_partial[tx])
            print_shared_matrix(u_n[u_n_next_index,:,:])

        # # u = u + u_n, each thread accumulates its own value so no need to sync        
        u[tx, ty] += u_n[u_n_next_index, tx, ty]
           
    input_diagonal[bx, tx, ty] = u[tx, ty]
    
    # # If the length of the next diagonal is the same or shorter than the current diagonal
    # # then all blocks propagate to the right. Otherwise, skip propagating the first one and reduce first dimension by 1.
    
    # # If the length of the next diagonal is longer than the current diagonal
    # # then all blocks propagate up. Otherwise, skip propagating the last one, but do not adjust first dimension.
    
    # bottom_offset = 0
    # top_limit = step_length

    # if step_length > next_step_length:
    #     bottom_offset = 1  # skip propagating first bottom
    #     top_limit = step_length - 1  # skip propagating last top
    # elif step_length == next_step_length:
    #     bottom_offset = 0  # propagate first bottom.
    #     top_limit = step_length - 1  # skip propagating last top

    # # Next steps.
    # s_right = get_point(N, s_idx + 1)
    # t_above = get_point(M, t_idx + 1)
    # s_coeff = (s_right ** tx)
    # t_coeff = (t_above ** ty)
    
    # # Only propagate to the right if we are not at the right limit. Should result in warp aggregate for each chunk of 32 threads.
    # if bx > bottom_offset and tx > 0:
    #     resolved_entry = input_diagonal[bx - bottom_offset, tx, ty] * t_coeff
    #     # Propagate the right boundary condition resulting in a function of s
    #     cuda.atomic.add(output_diagonal, (bx, 0, ty),resolved_entry)
    #     # Propagate the initial condition
    #     cuda.atomic.add(output_diagonal, (bx, 0, 0),resolved_entry * s_coeff)
    
    # # Only propagate to the top if we are not at the top limit
    # if bx < top_limit and ty > 0:
    #     # Propagate to the top boundary condition resulting in a function of t. 
    #     # Reverse tx and ty, so that we get warp aggregate for each chunk of 32 threads.
    #     cuda.atomic.add(output_diagonal, (bx, ty, 0),input_diagonal[bx, ty, tx] * s_coeff)


def launch_scaling_matrix(order: int = 32):
    assert order <= 32, "Order must be less than or equal to 32"
    threadsperblock = (order, order)
    blockspergrid = (1,)
    scaling_matrix = cuda.device_array((order, order), dtype=np.float64)
    build_scaling_matrix[blockspergrid, threadsperblock](scaling_matrix, order)
    return scaling_matrix

# Host function to set up and launch kernel
def compute_signature(dX, dY) -> float:
    """
    Host function to process two input tensors using the CUDA kernel.

    Args:
        dX: Input tensor of shape [N, d]
        dY: Input tensor of shape [M, d]

    Returns:
        Processed output grid
    """
    # cuda.select_device(1)
    N, d_x = dX.shape
    M, d_y = dY.shape

    assert d_x == d_y, "Input tensors must have same second dimension"

    dX = cuda.to_device(dX)
    dY = cuda.to_device(dY)

    min_NM = min(N, M)
    
    total_diagonals = N + M - 1

    # Function to compute result length for each step

    # Total steps
    total_diagonals = N + M - 1    

    # Calculate grid and block dimensions
    threadsperblock = (32, 32)
    rho_threadsperblock = (32, get_number_threads(dX.shape[1]))
    start = time.time()
    
    scaling_matrix = launch_scaling_matrix()
    
    s_start, t_start, current_length = get_step_length(0, N, M)
    s_next, t_next, next_length = get_step_length(1, N, M)

    # Initialize input and output diagonals on device
    input_diagonal = cuda.device_array((min_NM, 32, 32), dtype=np.float64)
    output_diagonal = cuda.device_array((min_NM, 32, 32), dtype=np.float64)    
    rho_diagonal = cuda.device_array((min_NM,), dtype=np.float64)

    for i in range(total_diagonals):        
        print(f"Processing {i}th diagonal with s_start={s_start}, t_start={t_start}, current_length={current_length}, next_length={next_length}")
        rho_blockspergrid = (current_length)
        blockspergrid = (current_length, 32,32)

        print(f"Rho blocks per grid: {rho_blockspergrid}")
        print(f"Rho threads per block: {rho_threadsperblock}")
        print(f"Blocks per grid: {blockspergrid}")
        print(f"Threads per block: {threadsperblock}")

        dstart = time.time()
        # Compute the rho diagonal for the current step
        compute_rho_diagonal[blockspergrid, rho_threadsperblock](
            dX[s_start:(s_start+current_length), :], 
            dY[t_start:(t_start+current_length), :], 
            rho_diagonal[:current_length]
            )
        cuda.synchronize()
        print(f"Compute rho diagonal in {(time.time() - dstart)} seconds: {rho_diagonal.copy_to_host()}")

        print(f"Using blockspergrid={blockspergrid}")
        print(f"Using threadsperblock={threadsperblock}")
        dstart = time.time()
        # Compute the sigkernel diagonal
        compute_sigkernel_diagonal[blockspergrid,threadsperblock](
            N,
            M,
            i,
            scaling_matrix,
            rho_diagonal[:current_length],
            input_diagonal[:current_length,:,:],
            output_diagonal[:next_length,:,:]
        )
        
        # The output diagonal becomes the input diagonal for the next step
        s_start, t_start, current_length = s_next, t_next, next_length
        s_next, t_next, next_length = get_step_length(i + 2, N, M)
        print(f"Kernel launched for {i}th diagonal in {(time.time() - dstart)} seconds")
        cuda.synchronize()
        print(f"Kernel execution finished for diagonal {i}  in {(time.time() - dstart)} seconds")

        print(f"Input diagonal: {input_diagonal[:current_length,:,:].copy_to_host()}")
        print(f"Output diagonal: {output_diagonal[:next_length,:,:].copy_to_host()}")

        input_diagonal, output_diagonal = output_diagonal, input_diagonal

    print(f"Processed {dX.shape[0]}x{dY.shape[0]} grid in {(time.time() - start)} seconds")
    

    return output_diagonal.sum()


# Example usage
if __name__ == "__main__":
    # dX = np.random.random((1 << 2, 2)).astype(np.float64)
    # dY = np.random.random((1 << 2, 2)).astype(np.float64)
    dX = np.asarray([[2,4,6,8]]).astype(np.float64)
    dY = np.asarray([[2,4,6,8]]).astype(np.float64)
    

    # Process entire grid

    result = compute_signature(dX, dY)

