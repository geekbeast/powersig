from math import ceil
from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import jit


@jit
def _normalize_kernel_matrix_jitted(K: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled version of kernel matrix normalization.
    """
    # Extract diagonal elements
    diag = jnp.diag(K)
    
    # Add small epsilon to avoid division by zero for near-zero diagonal elements
    epsilon = 1e-10
    diag_safe = diag + epsilon
    
    # Compute sqrt(K_ii * K_jj) for all i, j
    # This creates a matrix where element (i,j) is sqrt(K_ii * K_jj)
    diag_sqrt = jnp.sqrt(diag_safe)
    normalization_matrix = jnp.outer(diag_sqrt, diag_sqrt)
    
    # Normalize the kernel matrix
    K_normalized = K / normalization_matrix
    
    return K_normalized


def normalize_kernel_matrix(K: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize a kernel matrix K such that K'_ij = K_ij / sqrt(K_ii * K_jj).
    
    This normalization ensures that diagonal elements become 1 and off-diagonal
    elements represent cosine similarities between the original vectors.
    
    Args:
        K: JAX array of shape (n, n) representing a kernel matrix
        
    Returns:
        K_normalized: JAX array of shape (n, n) with normalized kernel matrix,
                     or NaN if input contains NaN/inf values
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    
    # Check for NaN or inf values in input
    if jnp.any(jnp.isnan(K)) or jnp.any(jnp.isinf(K)):
        # Return a matrix filled with NaN
        return jnp.full_like(K, jnp.nan)
    
    return _normalize_kernel_matrix_jitted(K)

def unity_transform(
    delta_X: jnp.array, delta_y: Optional[jnp.array] = None, n_roots:Optional[int]=None, epsilon=0.001
) -> Union[tuple[jnp.array, int], tuple[jnp.array, jnp.array, int]]:
    """
    Transform scaled data to roots of unity using arccos transformation.

    Args:
        scaled_dataset (jnp.array): Input dataset with values in [-1, 1]
        delta_y (jnp.array, optional): Regression target of shape (num_samples, dimensions) or None
        epsilon (float): Parameter for the transformation, defaults to 0.001

    Returns:
        tuple: (transformed_dataset, n_roots) or (transformed_dataset, transformed_y, n_roots) where:
            - transformed_dataset: Complex-valued tensor with roots of unity
            - transformed_y: Complex-valued tensor with roots of unity for delta_y
            - n_roots: Number of roots of unity used (2 * ceil(pi/(2*epsilon)))
    """
    # Calculate number of roots of unity, if not provided
    if n_roots is None:
        n_roots = 2 * int(ceil(jnp.pi / (2 * epsilon)))
    
    if delta_y is None:
        return unity_clamp_and_map(delta_X, n_roots), n_roots
    else:
        return unity_clamp_and_map(delta_X, n_roots), unity_clamp_and_map(delta_y, n_roots), n_roots

def unity_clamp_and_map(input: jnp.array, n_roots: int) -> jnp.array:
    # Clamp values to [-1, 1] to avoid NaN from acos
    clipped_dataset = jnp.clip(input, -1.0, 1.0)
    
    # Compute arccos and multiply by n_roots/(2*pi)
    arccos_data = jnp.acos(clipped_dataset) * (n_roots / (2 * jnp.pi))
    
    # Round to nearest integer
    indices = jnp.round(arccos_data)
    
    # Compute differences in the original data space
    differences = jnp.cos(2 * jnp.pi * indices / n_roots) - clipped_dataset
    
    # Print absolute values of differences
    abs_differences = jnp.abs(differences)
    print(f"Absolute differences from nearest root of unity:")
    print(f"  - Max difference: {jnp.max(abs_differences):.6f}")
    print(f"  - Mean difference: {jnp.mean(abs_differences):.6f}")
    print(f"  - Min difference: {jnp.min(abs_differences):.6f}")
    
    # Convert to integer indices for roots of unity
    # Map [0, n_roots/2] to roots of unity    
    # Warn if any indices are outside the expected range [0, n_roots/2]
    if jnp.any(indices < 0) or jnp.any(indices > n_roots // 2):
        min_idx = jnp.min(indices).item()
        max_idx = jnp.max(indices).item()
        print(f"Warning: Indices outside expected range [0, {n_roots // 2}]: [{min_idx}, {max_idx}]")
        print("This shouldn't happen with properly clamped arccos values")
    
    # Compute roots of unity: exp(2πi * k / n_roots) for k = 0, 1, ..., n_roots/2
    angles = 2 * jnp.pi * indices / n_roots
    
    # Create complex-valued tensor with complex128 dtype
    angles_64 = angles.astype(jnp.float64)
    real_part = jnp.cos(angles_64)
    imag_part = jnp.sin(angles_64)
    transformed_dataset = jax.lax.complex(real_part, imag_part)
    
    return transformed_dataset

def scale_and_shift(delta_X: jnp.array, delta_y: Optional[jnp.array] = None, scaling_type="channel", scale: Union[jnp.ndarray, float, None] = None) -> Union[Tuple[jnp.array,jnp.array,jnp.array], Tuple[jnp.array,jnp.array,jnp.array,jnp.array]]:
    """
    Scale and shift each sample's dimension so all values are between -1 and 1.

    Args:
        dataset (torch.Tensor): Input dataset of shape (num_samples, num_timesteps, dimensions)
        scaling_type (str): Type of scaling to apply. Options:
            - "channel": Scale each sample and dimension independently (default, current behavior)
            - "global": Scale using global min/max across all channels for each sample
        scale (torch.Tensor, optional): Custom scale parameter for custom scaling type.
            Must be >= global scale to ensure all data maps to [-1, 1].

    Returns:
        tuple: (scaled_dataset, shift, scale) where:
            - scaled_dataset: Tensor with values in [-1, 1]
            - shift: Tensor of shifts applied (num_samples, dimensions)
            - scale: Tensor of scaling factors (num_samples, dimensions)
    """
    # Calculate bounds across time dimension (dim=1) for each sample and dimension
    # Shape: (num_samples, dimensions)
    lower_bounds, _ = jnp.min(delta_X, axis=1)  # Extract values, ignore indices
    upper_bounds, _ = jnp.max(delta_X, axis=1)  # Extract values, ignore indices
    
    # If delta_y is provided, we have to include it in the bounds calculation
    if delta_y is not None:
        lower_bounds = jnp.minimum(lower_bounds, delta_y)
        upper_bounds = jnp.maximum(upper_bounds, delta_y)

    if scaling_type in ["global"]:
        lower_bounds = jnp.min(lower_bounds)
    # Compute roots of unity: exp(2πi * k / n_roots) for k = 0, 1, ..., n_roots/2t is greater than or equal tothe computed global scale or all computed channel scales    
    if scale is not None and jnp.all(scales <= scale):
        if isinstance(scale, jnp.ndarray):
            if scales.shape == scale.shape:
                scales = scale
            elif scale.shape == (1,) or scale.shape == ():
                scales = scales.at[:].set(scale)
        elif isinstance(scale, float):
            scales = scales.at[:].set(scale)
        else:
            raise ValueError("scale parameter must be a jnp.ndarray or a float")
    
    # Reshape for broadcasting: (num_samples, 1, dimensions)
    shifts_expanded = jnp.expand_dims(shifts, axis=1)
    scales_expanded = jnp.expand_dims(scales, axis=1) if len(scales.shape) >= 1 else scales

    # Apply transformation: (x - shift) / scale
    # Broadcasting will handle the time dimension automatically
    delta_X_scaled = (delta_X - shifts_expanded) / scales_expanded

    if delta_y is None:
        return delta_X_scaled, shifts, scales
    else:
        delta_y_scaled = (delta_y - shifts_expanded) / scales_expanded
        return delta_X_scaled, delta_y_scaled, shifts, scales


def chebychev_transformation(
    delta_X: jnp.array, delta_y: Optional[jnp.array] = None, n:Optional[int]=None, epsilon=0.00001
) -> Union[tuple[jnp.array, int], tuple[jnp.array, jnp.array, int]]:
    """
    Maps a dataset onto the minimum number of Chebychev nodes that can be used to approximate the dataset.  

    Args:
        delta_X (jnp.array): Input dataset of shape (num_samples, num_timesteps, dimensions)
        delta_y (jnp.array, optional): Regression target of shape (num_samples, dimensions) or None
        epsilon (float): Parameter for the transformation, defaults to 0.00001

    Returns:
        tuple: (nodes, n) or (nodes, nodes_y, n) where:
            - nodes: JAX array of shape (num_samples, num_timesteps, dimensions) with Chebychev nodes
            - nodes_y: JAX array of shape (num_samples, dimensions) with Chebychev nodes
            - n: Number of Chebychev nodes used
    """

    # Starting number of chebychev nodes
    if n is None:
        n = ceil(jnp.pi / (2*epsilon))
    
    if delta_y is None:
        return chebychev_clamp_and_map(delta_X, n), n
    else:
        return chebychev_clamp_and_map(delta_X, n), chebychev_clamp_and_map(delta_y, n), n

def chebychev_clamp_and_map(input: jnp.array, n: int) -> jnp.array:
    clipped_dataset = jnp.clip(input, -1.0, 1.0)

    original_phase = jnp.acos(clipped_dataset) * (n / jnp.pi)
    indices = jnp.round(original_phase)
    nodes = jnp.cos(jnp.pi*(indices/n))
    diff =  clipped_dataset - nodes
    diff_norm = jnp.linalg.norm(diff, ord='fro')
    max_error = jnp.max(diff)
    print(f"Total error: {diff_norm }")
    print(f"Average error: {diff_norm / (input.shape[0] * input.shape[1] * input.shape[2])}")
    print(f"Max error: {max_error}")
    print(f"Min error: {jnp.min(diff)}")

    return nodes


def jax_compute_differences(
    X: jnp.ndarray, y: jnp.ndarray = None
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute differences between consecutive elements in a JAX array along the second dimension.

    This function calculates the first-order differences (deltas) between adjacent elements
    in the input array X along the second dimension (axis=1). Optionally, it can also
    compute the difference between a target array y and the last element of X.

    Args:
        X (jnp.ndarray): Input array of shape (batch_size, sequence_length, features).
                          Must have at least 2 elements along the second dimension.
        y (jnp.ndarray, optional): Target array of shape (batch_size, features).
                                   If provided, the function returns both deltaX and deltaY.
                                   If None, only deltaX is returned.

    Returns:
        jnp.ndarray or tuple:
            - If y is None: Returns deltaX array of shape (batch_size, sequence_length-1, features)
            - If y is provided: Returns tuple (deltaX, deltaY) where:
                * deltaX: array of shape (batch_size, sequence_length-1, features)
                * deltaY: array of shape (batch_size, features)

    Note:
        The function requires X to have at least 2 elements along the second dimension
        to compute differences. The output deltaX will have one fewer element along
        the second dimension compared to the input X.

    Example:
        >>> X = jnp.array([[[1, 2], [3, 4], [5, 6]]])  # shape: (1, 3, 2)
        >>> deltaX = jax_compute_differences(X)
        >>> print(deltaX)  # array([[[2, 2], [2, 2]]])  # shape: (1, 2, 2)
        >>>
        >>> y = jnp.array([[7, 8]])  # shape: (1, 2)
        >>> deltaX, deltaY = jax_compute_differences(X, y)
        >>> print(deltaY)  # array([[2, 2]])  # shape: (1, 2)
    """
    deltaX = X[:, 1:, :] - X[:, :-1, :]
    if y is not None:
        deltaY = y - X[:, -1, :]
        return deltaX, deltaY
    else:
        return deltaX


def variation_normalizer(
    X_train: jnp.ndarray, 
    y: jnp.ndarray, 
    scaling_type: str = "global"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalize variations in the data by computing differences and scaling them.
    
    Args:
        X_train (jnp.ndarray): Input array of shape (num_samples, num_timesteps, dimensions)
        y (jnp.ndarray): Target array of shape (num_samples, dimensions)
        scaling_type (str): Type of scaling to apply. Options:
            - "global": Use global maximum across all samples and dimensions (default)
            - "channel": Use channel-wise maximum for each sample
            
    Returns:
        tuple: (X_train / scales, y / scales) - Normalized input and target arrays
    """
    # Compute differences using the JAX function
    delta_X, delta_y = jax_compute_differences(X_train, y)
    
    # Take absolute values of all differences
    abs_diffs_X = jnp.abs(delta_X)
    abs_diffs_Y = jnp.abs(delta_y)
    
    if scaling_type == "global":
        # Global normalization: use maximum across all samples and dimensions
        scales = max(jnp.max(abs_diffs_X), jnp.max(abs_diffs_Y))
    elif scaling_type == "channel":
        # Channel normalization: use maximum for each sample and dimension
        scales = jnp.maximum(jnp.max(abs_diffs_X, axis=1), abs_diffs_Y)
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}. Must be 'global' or 'channel'")
    
    # Return normalized arrays
    return X_train / scales, y / scales