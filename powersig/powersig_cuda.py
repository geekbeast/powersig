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
def tensor_processing_kernel(dX, dY, output_grid):
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

    # Shared memory for the 32x32 scaling matrix
    scaling_matrix = cuda.shared.array(shape=(32, 32), dtype=np.float64)
    shared_u_n = cuda.shared.array(shape=(2, 32, 32), dtype=np.float64)
    shared_rho = cuda.shared.array(shape=(1,), dtype=np.float64)

    if cuda.blockIdx.x == 0:
        # Initialize scaling matrix (only need to do this once per kernel launch)
        # Only let first block initialize the matrix
        scaling_matrix[ty, tx] = 1.0 / ((ty + 1) * (tx + 1))

    # Synchronize threads after initializing scaling matrix
    cuda.syncthreads()

    # Calculate dimensions
    N = dX.shape[0]
    M = dY.shape[0]
    d = dX.shape[1]  # Feature dimension
    min_NM = min(N, M)

    # Function to compute result length for each step

    # Total steps
    total_steps = N + M - 1

    # Main processing loop
    for step in range(total_steps):
        # Determine current and next state indices
        current_offset = (step % 2) * min_NM
        next_offset = ((step + 1) % 2) * min_NM

        # Only process if thread is within valid range for this step
        s_start, t_start, step_length = get_step_length(step, N, M)

        if bx < step_length:
            # We need to scale and multiply by rho
            s_idx = s_start - bx
            t_idx = t_start + bx
            rho = 0.0

            for i in range(0, d, 1024):
                rho += dX[s_idx, tid + i] * dY[t_idx, tid]

            shared_rho[0] = rho
            # u_0 = current boundary conditions computed in last step
            shared_u_n[0, tx, ty] = output_grid[current_offset + bx, tx, ty]

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

                    # Apply limits of integration
                    shared_u_n[u_n_next_index, 0, ty + 1] -= scaled_val * s_coeff
                    shared_u_n[u_n_next_index, tx + 1, 0] -= scaled_val * t_coeff
                    shared_u_n[u_n_next_index, 0, 0] += scaled_val * s_coeff * t_coeff

                    cuda.syncthreads()

                # u = u + u_n
                output_grid[current_offset + bx, tx, ty] += rho*shared_u_n[u_n_next_index, tx, ty]
                cuda.syncthreads()

            # Need to compute propagation to other blocks

            next_s_start, next_t_start, next_step_length = get_step_length(step, N, M)
            # If the length of the next diagonal is the same or shorter than the current diagonal
            # then all blocks propagate to the right. Otherwise, skip propagating the first one and reduce first dimension by 1.

            # If the length of the next diagonal is longer than the current diagonal
            # then all blocks propagate up. Otherwise, skip propagating the last one, but do not adjust first dimension.

            bottom_offset = 0
            top_limit = step_length
            if step_length > next_step_length:
                bottom_offset =  1 # skip propagating first bottom
                top_limit = step_length - 1 # skip propagating last top
            elif step_length == next_step_length:
                bottom_offset = 0 # propagate first bottom.
                top_limit = step_length - 1 # skip propagating last top


            if s_idx < (N-1):
                s = get_point(N, s_idx + 1)
                if bx > bottom_offset and tx > 0:
                    s_coeff = (s ** tx)
                    # We're using shared_u_n buffer here for a different purpose.
                    shared_u_n[0,0,ty] += output_grid[current_offset + bx, tx, ty] * s_coeff
                    cuda.syncthreads()
                    output_grid[next_offset + bx - bottom_offset, 0, ty] = output_grid[current_offset + bx, tx, ty] * s_coeff

            if t_idx < (M-1):
                t = get_point(M, t_idx + 1)
                if bx < top_limit and ty > 0:
                    t_coeff = (t ** (ty + 1))
                    output_grid[next_offset + bx, tx, 0] = output_grid[current_offset + bx , tx, ty] * t_coeff

            if tx == 0 and ty == 0:
                # For the (0,0) position, use atomic add to safely combine contributions
                cuda.atomic.add(output_grid, (next_offset + bx, 0, 0),
                                output_grid[current_offset + bx, 0, 0])


        # Synchronize all threads before next step
        cuda.syncthreads()


# Host function to set up and launch kernel
def process_tensors(dX, dY):
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

    # Initialize output grid on device
    output_grid = cuda.device_array((2 * min_NM, 32, 32), dtype=np.float64)

    # Calculate grid and block dimensions
    threadsperblock = (32, 32)
    blockspergrid = (min_NM,)

    # Launch kernel
    tensor_processing_kernel[blockspergrid, threadsperblock](
        cuda.to_device(dX),
        cuda.to_device(dY),
        output_grid
    )

    return output_grid.copy_to_host()


# Example usage
if __name__ == "__main__":
    dX = np.random.random((1 << 14, 2)).astype(np.float64)
    dY = np.random.random((1 << 14, 2)).astype(np.float64)

    # Process entire grid
    start = time.time()
    result = process_tensors(dX, dY)
    print(f"Processed {dX.shape[0]}x{dY.shape[0]} grid in {time.time() - start} seconds")
