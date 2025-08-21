from typing import Optional, Tuple, Union

import torch

import powersig.util.fbm_utils


def fractional_brownian_motion(
    n_steps, n_paths=1, device=None, dim: int = 1, hurst: float = 0.5, t: float = 1.0
) -> Tuple[torch.tensor, float]:
    """
    Generate multi-dimensional fractional Brownian motion paths using the fbm package.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        device (device): Device to use for the resulting tensor
        dim (int): Dimension of the fractional Brownian motion (default: 1)
        hurst (float): Hurst parameter, must be in (0,1). H=0.5 gives standard Brownian motion.
        t (float): Total time period (default: 1.0)

    Returns:
        torch.Tensor: Fractional Brownian motion paths of shape (n_paths, n_steps + 1, dim)
        float: Time step size (dt)
    """
    dt = t / n_steps

    return (
        torch.tensor(
            powersig.util.fbm_utils.fractional_brownian_motion(
                n_steps, n_paths, dim=dim, hurst=hurst, t=t
            )[0]
        ).to(device),
        dt,
    )


def torch_compute_differences(
    X: torch.Tensor, y: torch.Tensor = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute differences between consecutive elements in a tensor along the second dimension.

    This function calculates the first-order differences (deltas) between adjacent elements
    in the input tensor X along the second dimension (axis=1). Optionally, it can also
    compute the difference between a target tensor y and the last element of X.

    Args:
        X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
                          Must have at least 2 elements along the second dimension.
        y (torch.Tensor, optional): Target tensor of shape (batch_size, features).
                                   If provided, the function returns both deltaX and deltaY.
                                   If None, only deltaX is returned.

    Returns:
        torch.Tensor or tuple:
            - If y is None: Returns deltaX tensor of shape (batch_size, sequence_length-1, features)
            - If y is provided: Returns tuple (deltaX, deltaY) where:
                * deltaX: tensor of shape (batch_size, sequence_length-1, features)
                * deltaY: tensor of shape (batch_size, features)

    Note:
        The function requires X to have at least 2 elements along the second dimension
        to compute differences. The output deltaX will have one fewer element along
        the second dimension compared to the input X.

    Example:
        >>> X = torch.tensor([[[1, 2], [3, 4], [5, 6]]])  # shape: (1, 3, 2)
        >>> deltaX = torch_compute_differences(X)
        >>> print(deltaX)  # tensor([[[2, 2], [2, 2]]])  # shape: (1, 2, 2)
        >>>
        >>> y = torch.tensor([[7, 8]])  # shape: (1, 2)
        >>> deltaX, deltaY = torch_compute_differences(X, y)
        >>> print(deltaY)  # tensor([[2, 2]])  # shape: (1, 2)
    """
    deltaX = X[:, 1:, :] - X[:, :-1, :]
    if y is not None:
        deltaY = y - X[:, -1, :]
        return deltaX, deltaY
    else:
        return deltaX


def unity_transform(
    delta_x: torch.Tensor, delta_y: Optional[torch.Tensor] = None, epsilon=0.001
) -> tuple[torch.Tensor, int]:
    """
    Transform scaled data to roots of unity using arccos transformation.

    Args:
        scaled_dataset (torch.Tensor): Input dataset with values in [-1, 1]
        epsilon (float): Parameter for the transformation, defaults to 0.001

    Returns:
        tuple: (transformed_dataset, n_roots) where:
            - transformed_dataset: Complex-valued tensor with roots of unity
            - n_roots: Number of roots of unity used (2 * ceil(pi/(2*epsilon)))
    """
    # Calculate number of roots of unity
    n_roots = 2 * int(torch.ceil(torch.pi / (2 * epsilon)))

    if delta_y is None:
        return unity_clamp_and_map(delta_x, n_roots), n_roots
    else:
        return (
            unity_clamp_and_map(delta_x, n_roots),
            unity_clamp_and_map(delta_y, n_roots),
            n_roots,
        )


def unity_clamp_and_map(input: torch.Tensor, n_roots: int) -> torch.Tensor:
    # Clamp values to [-1, 1] to avoid NaN from acos
    clipped_dataset = torch.clamp(input, -1.0, 1.0)

    # Compute arccos and multiply by n_roots/(2*pi)
    arccos_data = torch.acos(clipped_dataset) * (n_roots / (2 * torch.pi))

    # Round to nearest integer
    indices = torch.round(arccos_data)

    # Compute differences in the original data space
    differences = torch.cos(2 * torch.pi * indices / n_roots) - clipped_dataset

    # Print absolute values of differences
    abs_differences = torch.abs(differences)
    print(f"Absolute differences from nearest root of unity:")
    print(f"  - Max difference: {torch.max(abs_differences):.6f}")
    print(f"  - Mean difference: {torch.mean(abs_differences):.6f}")
    print(f"  - Min difference: {torch.min(abs_differences):.6f}")

    # Convert to integer indices for roots of unity
    # Map [0, n_roots/2] to roots of unity
    # Warn if any indices are outside the expected range [0, n_roots/2]
    if torch.any(indices < 0) or torch.any(indices > n_roots // 2):
        min_idx = torch.min(indices).item()
        max_idx = torch.max(indices).item()
        print(
            f"Warning: Indices outside expected range [0, {n_roots // 2}]: [{min_idx}, {max_idx}]"
        )
        print("This shouldn't happen with properly clamped arccos values")

    # Compute roots of unity: exp(2πi * k / n_roots) for k = 0, 1, ..., n_roots/2
    angles = 2 * torch.pi * (indices / n_roots)

    # Create complex-valued tensor with complex128 dtype
    angles_64 = angles.to(torch.float64)
    real_part = torch.cos(angles_64)
    imag_part = torch.sin(angles_64)
    transformed_dataset = torch.complex(real_part, imag_part)

    return transformed_dataset, n_roots


def scale_and_shift(
    delta_X: torch.Tensor,
    delta_y: Optional[torch.Tensor] = None,
    scaling_type="channel",
    scale: Union[float, torch.Tensor, None] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Scale and shift each sample's dimension so all values are between -1 and 1. If deltaY is provided, the shift and scale are computed using

    Args:
        X (torch.Tensor): Input dataset of shape (num_samples, num_timesteps, dimensions)
        y (torch.Tensor): Regression target of shape (num_samples, dimensions) or None
        scaling_type (str): Type of scaling to apply. Options:
            - "channel": Scale each sample and dimension independently (default, current behavior)
            - "global": Scale using global min/max across all channels for each sample
        scale (torch.Tensor, optional): Custom scale parameter for custom scaling type.
            Must be >= global scale to ensure all data maps to [-1, 1].

    Returns:
        tuple: (scaled_delta_X, shift, scale) or (scaled_delta_X, scaled_delta_y, shift, scale) where:
            - scaled_delta_X: Tensor with values in [-1, 1] (could be narrower if scale is provided)
            - scaled_delta_y: Tensor with values in [-1, 1] (could be narrower if scale is provided)
            - shift: Tensor of shifts applied (num_samples, dimensions)
            - scale: Tensor of scaling factors (num_samples, dimensions) or () if global scaling is used
    """
    # Calculate bounds across time dimension (dim=1) for each sample and dimension
    # Shape: (num_samples, dimensions)
    lower_bounds, _ = torch.min(delta_X, dim=1)  # Extract values, ignore indices
    upper_bounds, _ = torch.max(delta_X, dim=1)  # Extract values, ignore indices

    # If delta_y is provided, we have to include it in the bounds calculation
    if delta_y is not None:
        lower_bounds = torch.minimum(lower_bounds, delta_y)
        upper_bounds = torch.maximum(upper_bounds, delta_y)

    if scaling_type in ["global"]:
        lower_bounds = lower_bounds.min()
        upper_bounds = upper_bounds.max()

    # Calculate shift and scale
    shifts = (lower_bounds + upper_bounds) / 2.0
    scales = (upper_bounds - lower_bounds) / 2.0

    # Use scale parameter if provided. It must be a float or tensor that is greater than or equal tothe computed global scale or all computed channel scales
    if scale is not None and torch.all(scales <= scale):
        if isinstance(scale, torch.Tensor):
            if scales.shape == scale.shape:
                scales = scale
            elif scale.shape == (1,) or scale.shape == ():
                scales[:] = scale[0]
        elif isinstance(scale, float):
            scales[:] = scale[0]
        else:
            raise ValueError("scale parameter must be a tensor or a float")

    # Reshape for broadcasting: (num_samples, 1, dimensions)
    shifts_expanded = shifts.unsqueeze(1)
    scales_expanded = scales.unsqueeze(1) if len(scales.shape) >= 1 else scales

    # Apply transformation: (x - shift) / scale
    # Broadcasting will handle the time dimension automatically
    delta_X_scaled = (delta_X - shifts_expanded) / scales_expanded

    if delta_y is None:
        return delta_X_scaled, shifts, scales
    else:
        delta_y_scaled = (delta_y - shifts_expanded) / scales_expanded
        return delta_X_scaled, delta_y_scaled, shifts, scales


def chebychev_transformation(
    delta_X: torch.Tensor, delta_y: Optional[torch.Tensor] = None, epsilon=0.00001
) -> Union[tuple[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Maps a dataset onto the minimum number of Chebychev nodes that can be used to approximate the dataset.

    Args:
        delta_X (torch.Tensor): Input dataset of shape (num_samples, num_timesteps, dimensions)
        epsilon (float): Parameter for the transformation, defaults to 0.00001

    Returns:
        tuple: (nodes, n) or (nodes, nodes_y, n) where:
            - nodes: JAX array of shape (num_samples, num_timesteps, dimensions) with Chebychev nodes
            - nodes_y: JAX array of shape (num_samples, dimensions) with Chebychev nodes
            - n: Number of Chebychev nodes used
    """

    # Starting number of chebychev nodes
    n = np.ceil(np.pi / (2 * epsilon))

    if delta_y is None:
        return chebychev_clamp_and_map(delta_X, n), n
    else:
        return (
            chebychev_clamp_and_map(delta_X, n),
            chebychev_clamp_and_map(delta_y, n),
            n,
        )


def chebychev_clamp_and_map(input: torch.Tensor, n: int) -> torch.Tensor:
    clipped_dataset = torch.clamp(input, -1.0, 1.0)

    original_phase = torch.acos(clipped_dataset) * (n / torch.pi)
    indices = torch.round(original_phase)
    nodes = torch.cos(torch.pi * (indices / n))
    diff = clipped_dataset - nodes
    diff_norm = torch.norm(diff, "fro")
    max_error = torch.max(diff)
    print(f"Total error: {diff_norm }")
    print(
        f"Average error: {diff_norm / (input.shape[0] * input.shape[1] * input.shape[2])}"
    )
    print(f"Max error: {max_error}")
    print(f"Min error: {torch.min(diff)}")

    return nodes
