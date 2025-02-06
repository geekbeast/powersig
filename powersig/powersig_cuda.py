import time
from typing import Tuple

import numpy as np
from numba import cuda
import math


@cuda.jit(device=True)
def get_point(num_points, idx):
    if num_points == 1:
        return 0
    return idx / (num_points - 1)


@cuda.jit(device=True)
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
    return scaling_matrix

@cuda.jit
def compute_rho_diagonal(dX : cuda.device_array, dY : cuda.device_array, rho : cuda.device_array, s_start: int, t_start: int, current_length: int):
    """
    CUDA kernel for computing the rho diagonal for the current step. Computes the dot product 
    starting at s_start, t_start and going up to current_length.
    """
    _, d = dX.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    s_idx = s_start - bx
    t_idx = t_start + bx
    tid = tx * 32 + ty
    
    for i in range(0, d, 1024):
        if tid + i < d:
            cuda.atomic.add(rho, (bx), dX[s_idx, tid + i] * dY[t_idx, tid + i])

    
@cuda.jit
def tensor_processing_kernel(dX, dY, global_scaling_matrix, rho_diagonal, input_diagonal, output_diagonal):
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
    tid = tx * 32 + ty

    # Shared memory for the 32x32 scaling matrix and the rotating buffer.
    scaling_matrix = cuda.shared.array(shape=(32, 32), dtype=np.float64)
    shared_u_n = cuda.shared.array(shape=(2, 32, 32), dtype=np.float64)

    # Only on thread per block needs to copy rho to shared memory
    if tx == 0 and ty == 0:
        rho = rho_diagonal[bx]

    # Copy the scaling matrix to shared memory from global memory
    scaling_matrix[tx, ty] = global_scaling_matrix[tx, ty]

    # Synchronize threads after initializing scaling matrix
    cuda.syncthreads()

    # Calculate dimensions
    N = dX.shape[0]
    M = dY.shape[0]
        
    # Main processing loop
    
    # u_0 = current boundary conditions computed in last step
    shared_u_n[0, tx, ty] = input_diagonal[bx, tx, ty]
    
    s = get_point(N, s_idx)
    t = get_point(M, t_idx)
    
    cuda.syncthreads()
    
    # Scale for integration
    for i in range(31):
        # u_n = rho * double integral of u_{n-1} with correct limits

        u_n_current_index = i % 2
        u_n_next_index = (i + 1) % 2

        # Truncate and only shift and scale necessary elements
        if tx < 31 and ty < 31:
            s_coeff = (s ** (tx + 1))
            t_coeff = (t ** (ty + 1))

            # Compute indefinite integral
            scaled_val = shared_u_n[u_n_current_index, tx, ty] * scaling_matrix[tx, ty]
            shared_u_n[u_n_next_index, tx + 1, ty + 1] = scaled_val
            scaled_s = scaled_val * s_coeff
            # Apply limits of integration using atomic operations for accumulation.
            # cuda.atomic.sub(shared_u_n, (u_n_next_index, 0, ty + 1), scaled)
            # cuda.atomic.sub(shared_u_n, (u_n_next_index, tx + 1, 0), scaled_val * t_coeff)
            # cuda.atomic.add(shared_u_n, (u_n_next_index, 0, 0), scaled_s * t_coeff)

            cuda.syncthreads()

        # u = u + u_n, each thread accumulates its own value so no need to sync
        input_diagonal[bx, tx, ty] += rho[bx] * shared_u_n[u_n_next_index, tx, ty]
        cuda.syncthreads()
    
    # Need to compute propagation to other blocks
    # Only process if thread is within valid range for this step
    s_start, t_start, step_length = get_step_length(step, N, M)
    next_s_start, next_t_start, next_step_length = get_step_length(step, N, M)

    # We need to compute the s, t indices for the current block
    # assuming the 0th block corresponds to the diagonal starting at s_start, t_start
    s_idx = s_start - bx
    t_idx = t_start + bx
    
    # If the length of the next diagonal is the same or shorter than the current diagonal
    # then all blocks propagate to the right. Otherwise, skip propagating the first one and reduce first dimension by 1.
    
    # If the length of the next diagonal is longer than the current diagonal
    # then all blocks propagate up. Otherwise, skip propagating the last one, but do not adjust first dimension.
    
    bottom_offset = 0
    top_limit = step_length

    if step_length > next_step_length:
        bottom_offset = 1  # skip propagating first bottom
        top_limit = step_length - 1  # skip propagating last top
    elif step_length == next_step_length:
        bottom_offset = 0  # propagate first bottom.
        top_limit = step_length - 1  # skip propagating last top
    
    # Next steps.
    s_right = get_point(N, s_idx + 1)
    t_above = get_point(M, t_idx + 1)
    s_coeff = (s_right ** tx)
    t_coeff = (t_above ** ty)
    
    # Only propagate to the right if we are not at the right limit
    if bx > bottom_offset and tx > 0:
        # Propagate the right boundary condition resulting in a function of s
        cuda.atomic.add(output_diagonal, (bx, 0, ty),input_diagonal[bx - bottom_offset, tx, ty] * t_coeff)
        # Propagate the initial condition
        cuda.atomic.add(output_diagonal, (bx, 0, 0),input_diagonal[bx, tx, ty] * s_coeff * t_coeff)
    
    # Only propagate to the top if we are not at the top limit
    if bx < top_limit and ty > 0:
        # Propagate to the top boundary condition resulting in a function of t
        cuda.atomic.add(output_diagonal, (bx, tx, 0),input_diagonal[bx, tx, ty] * s_coeff)

    # Synchronize all threads before next step
    cuda.syncthreads()

def launch_scaling_matrix(order: int = 32):
    assert order <= 32, "Order must be less than or equal to 32"
    threadsperblock = (order, order)
    blockspergrid = (1,)
    return build_scaling_matrix[blockspergrid, threadsperblock](order)

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

    N, d_x = dX.shape
    M, d_y = dY.shape

    assert d_x == d_y, "Input tensors must have same second dimension"

    min_NM = min(N, M)
    
    total_diagonals = N + M - 1

    # Function to compute result length for each step

    # Total steps
    total_diagonals = N + M - 1    

    # Calculate grid and block dimensions
    threadsperblock = (32, 32)
    
    start = time.time()
    
    scaling_matrix = launch_scaling_matrix()
    
    s_start, t_start, current_length = get_step_length(0, N, M)
    s_next, t_next, next_length = get_step_length(1, N, M)

    # Initialize input and output diagonals on device
    input_diagonal = cuda.device_array((min_NM, 32, 32), dtype=np.float64)
    output_diagonal = cuda.device_array((min_NM, 32, 32), dtype=np.float64)    
    rho_diagonal = cuda.device_array((min_NM), dtype=np.float64)

    for i in range(total_diagonals):        
        dstart = time.time()
        blockspergrid = (current_length,)
        
        # Compute the rho diagonal for the current step
        compute_rho_diagonal[blockspergrid, threadsperblock](dX, dY, rho_diagonal, s_start, t_start, current_length)
        
        # Launch kernel
        tensor_processing_kernel[blockspergrid, threadsperblock](
            rho_diagonal,            
            scaling_matrix,
            input_diagonal[current_length,:,:],
            output_diagonal[next_length,:,:]
        )

        # The output diagonal becomes the input diagonal for the next step
        input_diagonal = output_diagonal
        s_start, t_start, current_length = s_next, t_next, next_length
        s_next, t_next, next_length = get_step_length(i + 1, N, M)
        print(f"Processed {i}th diagonal in {(time.time() - dstart)} seconds")

    print(f"Processed {dX.shape[0]}x{dY.shape[0]} grid in {(time.time() - start)} seconds")
    print(f"Compute rho diagonal in {(time.time() - dstart)} seconds")

    return output_diagonal.sum()


# Example usage
if __name__ == "__main__":
    dX = np.random.random((1 << 2, 2)).astype(np.float64)
    dY = np.random.random((1 << 2, 2)).astype(np.float64)

    # Process entire grid

    result = process_tensors(dX, dY)

