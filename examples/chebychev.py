from typing import Tuple
import torch
import numpy as np
import pandas as pd
import jax.numpy as jnp
import cupy as cp
from powersig.util.fbm_utils import fractional_brownian_motion
from powersig.jax.algorithm import PowerSigJax
from powersig.jax import static_kernels
import ksig
from ksig.kernels import SignaturePDEKernel
from ksig.static.kernels import LinearKernel, RBFKernel
import cupy as cp

from powersig.util.normalization import normalize_kernel_matrix

def generate_multivariate_timeseries(num_samples, num_timesteps, dimensions, hurst=0.5):
    """
    Generate multivariate time series data using fractional Brownian motion.

    Args:
        num_samples (int): Number of samples to generate
        num_timesteps (int): Number of time steps per sample
        dimensions (int): Number of dimensions per sample
        hurst (float): Hurst parameter for FBM (0 < hurst < 1)

    Returns:
        torch.Tensor: Tensor of shape (num_samples, num_timesteps, dimensions)
    """
    # Use powersig's FBM utility to generate all samples at once
    # Note: powersig returns (n_paths, n_steps + 1, dim) so we need to adjust
    fbm_paths, dt = fractional_brownian_motion(
        n_steps=num_timesteps - 1,  # Adjust because powersig returns n_steps + 1
        n_paths=num_samples,
        dim=dimensions,
        hurst=hurst,
        cuda=False,  # Use CPU for compatibility
    )

    # Convert to torch tensor and ensure correct shape
    dataset = torch.tensor(fbm_paths, dtype=torch.float64)

    # If we need exactly num_timesteps, we can truncate or interpolate
    # For now, return as is (powersig gives n_steps + 1)
    return dataset


def scale_and_shift(dataset: torch.Tensor, scaling_type="channel", scale=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    Scale and shift each sample's dimension so all values are between -1 and 1.

    Args:
        dataset (torch.Tensor): Input dataset of shape (num_samples, num_timesteps, dimensions)
        scaling_type (str): Type of scaling to apply. Options:
            - "channel": Scale each sample and dimension independently (default, current behavior)
            - "global": Scale using global min/max across all channels for each sample
            - "custom": Use provided scale parameter for all channels after shift logic
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
    lower_bounds, _ = torch.min(dataset, dim=1)  # Extract values, ignore indices
    upper_bounds, _ = torch.max(dataset, dim=1)  # Extract values, ignore indices
    if scaling_type in ["global","custom"]:
        if scale is None:
            raise ValueError("scale parameter must be provided for custom scaling type")
        lower_bounds = lower_bounds.min()
        upper_bounds = upper_bounds.max()
    
    # Calculate shift and scale
    shifts = (lower_bounds + upper_bounds) / 2.0
    scales = (upper_bounds - lower_bounds) / 2.0
    
    if scaling_type == "custom" and scales <= scale:
        if isinstance(scale, torch.Tensor):
            scales = scale
        else:
            scales = torch.tensor( scale, dtype=dataset.dtype, device=dataset.device)
    
    # Reshape for broadcasting: (num_samples, 1, dimensions)
    shifts_expanded = shifts.unsqueeze(1)
    scales_expanded = scales.unsqueeze(1)

    # Apply transformation: (x - shift) / scale
    # Broadcasting will handle the time dimension automatically
    scaled_dataset = (dataset - shifts_expanded) / scales_expanded

    return scaled_dataset, shifts, scales


def chebychev_transformation(
    scaled_dataset: torch.Tensor, epsilon=0.00001
) -> tuple[torch.Tensor, int]:
    """
    Stub function for the second transformation.

    Args:
        epsilon (float): Parameter for the transformation, defaults to 0.1

    Returns:
        torch.Tensor: Placeholder return
    """

    # Starting number of chebychev nodes
    n = np.ceil(np.pi / (2*epsilon))

    # Clip values to [-1, 1] to avoid NaN from acos
    clipped_dataset = torch.clamp(scaled_dataset, -1.0, 1.0)

    original_phase = torch.acos(clipped_dataset) * (n / torch.pi)

    nearest = torch.round(original_phase)
    nodes = torch.cos(nearest* torch.pi/n)
    diff =  clipped_dataset - nodes
    diff_norm = torch.norm(diff, "fro")
    max_error = torch.max(diff)
    print(f"Total error: {diff_norm }")
    print(f"Average error: {diff_norm / (scaled_dataset.shape[0] * scaled_dataset.shape[1] * scaled_dataset.shape[2])}")
    print(f"Max error: {max_error}")
    print(f"Min error: {torch.min(diff)}")
    return nodes, n
    # Nearest chebychev indexes, we want to find the nearest chebychev node to the error
    # We round because below .5 error doubles until above .5 where it shrinks.
    # This means we will go through increase/decrease until we snap to a chebychev node.

    # while True:
    #     nearest = torch.round(original_phase)
    #     diff = original_phase - torch.cos(nearest* torch.pi/n)
    #     diff_norm = torch.norm(diff, "fro")
    #     max_error = torch.max(diff)
    #     print(f"Total error: {diff_norm }")
    #     print(f"Average error: {diff_norm / (scaled_dataset.shape[0] * scaled_dataset.shape[1] * scaled_dataset.shape[2])}")
    #     print(f"Max error: {max_error}")
    #     print(f"Min error: {torch.min(diff)}")

    #     if max_error > epsilon:
    #         n *= 2
    #         original_phase *= 2.0
    #     else:
    #         return torch.cos(nearest * torch.pi / n), n


def unity_transform(
    scaled_dataset: torch.Tensor, epsilon=0.001
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
    n_roots = 2 * int(np.ceil(np.pi / (2 * epsilon)))
    
    # Clip values to [-1, 1] to avoid NaN from acos
    clipped_dataset = torch.clamp(scaled_dataset, -1.0, 1.0)
    
    # Compute arccos and multiply by n_roots/(2*pi)
    arccos_data = torch.acos(clipped_dataset) * (n_roots / (2 * np.pi))
    
    # Round to nearest integer
    rounded_values = torch.round(arccos_data)
    
    # Compute differences in the original data space
    differences = torch.cos(2 * np.pi * rounded_values / n_roots) - clipped_dataset
    
    # Print absolute values of differences
    abs_differences = torch.abs(differences)
    print(f"Absolute differences from nearest root of unity:")
    print(f"  - Max difference: {torch.max(abs_differences):.6f}")
    print(f"  - Mean difference: {torch.mean(abs_differences):.6f}")
    print(f"  - Min difference: {torch.min(abs_differences):.6f}")
    
    # Convert to integer indices for roots of unity
    # Map [0, n_roots/2] to roots of unity
    indices = rounded_values.long()
    
    # Warn if any indices are outside the expected range [0, n_roots/2]
    if torch.any(indices < 0) or torch.any(indices > n_roots // 2):
        min_idx = torch.min(indices).item()
        max_idx = torch.max(indices).item()
        print(f"Warning: Indices outside expected range [0, {n_roots // 2}]: [{min_idx}, {max_idx}]")
        print("This shouldn't happen with properly clamped arccos values")
    
    # Compute roots of unity: exp(2πi * k / n_roots) for k = 0, 1, ..., n_roots/2
    k_values = indices.float()
    angles = 2 * np.pi * k_values / n_roots
    
    # Create complex-valued tensor with complex128 dtype
    angles_64 = angles.to(torch.float64)
    real_part = torch.cos(angles_64)
    imag_part = torch.sin(angles_64)
    transformed_dataset = torch.complex(real_part, imag_part)
    
    return transformed_dataset, n_roots


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    print("Generating multivariate time series data...")
    data = generate_multivariate_timeseries(
        num_samples=5, num_timesteps=500, dimensions=3
    )
    print(f"Generated data shape: {data.shape}")
    print(f"Note: powersig returns n_steps + 1 time steps")
    print(f"Data range: [{torch.min(data):.4f}, {torch.max(data):.4f}]")

    # Compute differences (derivatives) of the time series
    print("\nComputing differences (derivatives)...")
    differences = data[:, 1:, :] - data[:, :-1, :]
    print(f"Differences shape: {differences.shape}")
    print(f"Differences range: [{torch.min(differences):.4f}, {torch.max(differences):.4f}]")

    # Apply scaling and shifting to the differences
    print("\nApplying scaling and shifting to differences...")
    scaled_data, shift_values, scale_values = scale_and_shift(differences)
    print(f"Scaled differences shape: {scaled_data.shape}")
    print(
        f"Scaled differences range: [{torch.min(scaled_data):.4f}, {torch.max(scaled_data):.4f}]"
    )
    print(f"Shift values shape: {shift_values.shape}")
    print(f"Scale values shape: {scale_values.shape}")

    # Test the Chebyshev transformation on the scaled differences
    print("\nTesting Chebyshev transformation on differences...")
    result, n_nodes = chebychev_transformation(scaled_data, epsilon=0.0001)
    print(f"Chebyshev transformation result shape: {result.shape}")
    print(f"Number of Chebyshev nodes used: {n_nodes}")
    print(f"Chebyshev data range: [{torch.min(result):.4f}, {torch.max(result):.4f}]")

    # Test the Unity transformation on the scaled differences
    print("\nTesting Unity transformation on differences...")
    unity_result, n_roots = unity_transform(scaled_data, epsilon=0.1)
    print(f"Unity transformation result shape: {unity_result.shape}")
    print(f"Number of Unity roots used: {n_roots}")
    print(f"Unity data magnitude range: [{torch.min(torch.abs(unity_result)):.4f}, {torch.max(torch.abs(unity_result)):.4f}]")

    # Generate cumulative sum datasets starting with zero
    print("\nGenerating cumulative sum datasets...")
    
    # Original data cumulative sum (starting with zero)
    original_cumsum = torch.cumsum(data, dim=1)
    original_cumsum = torch.cat([torch.zeros(original_cumsum.shape[0], 1, original_cumsum.shape[2], dtype=original_cumsum.dtype), original_cumsum], dim=1)
    
    # Scaled data cumulative sum (starting with zero)
    scaled_cumsum = torch.cumsum(scaled_data, dim=1)
    scaled_cumsum = torch.cat([torch.zeros(scaled_cumsum.shape[0], 1, scaled_cumsum.shape[2], dtype=scaled_cumsum.dtype), scaled_cumsum], dim=1)
    
    # Chebyshev data cumulative sum (starting with zero)
    chebychev_cumsum = torch.cumsum(result, dim=1)
    chebychev_cumsum = torch.cat([torch.zeros(chebychev_cumsum.shape[0], 1, chebychev_cumsum.shape[2], dtype=chebychev_cumsum.dtype), chebychev_cumsum], dim=1)
    
    # Unity data cumulative sum (starting with zero)
    unity_cumsum = torch.cumsum(unity_result, dim=1)
    unity_cumsum = torch.cat([torch.zeros(unity_cumsum.shape[0], 1, unity_cumsum.shape[2], dtype=unity_cumsum.dtype), unity_cumsum], dim=1)
    
    print(f"Original cumsum shape: {original_cumsum.shape}")
    print(f"Scaled cumsum shape: {scaled_cumsum.shape}")
    print(f"Chebyshev cumsum shape: {chebychev_cumsum.shape}")
    print(f"Unity cumsum shape: {unity_cumsum.shape}")
    
    # Convert to JAX arrays for PowerSigJax
    print("\nConverting to JAX arrays for PowerSigJax...")
    import jax.numpy as jnp
    original_jax = jnp.array(original_cumsum.numpy())
    scaled_jax = jnp.array(scaled_cumsum.numpy())
    chebychev_jax = jnp.array(chebychev_cumsum.numpy())
    unity_jax = jnp.array(unity_cumsum.numpy())
    
    # Compute PowerSigJax gram matrices
    print("\nComputing PowerSigJax gram matrices...")
    from powersig.jax.algorithm import PowerSigJax
    from powersig.jax import static_kernels

    print("Unity JAX dtype: ", unity_jax.dtype)
    
    # Initialize PowerSigJax with order 9 and dtype
    powersig_linear = PowerSigJax(order=9, static_kernel=static_kernels.linear_kernel, dtype=original_jax.dtype)
    powersig_rbf = PowerSigJax(order=9, static_kernel=static_kernels.rbf_kernel, dtype=original_jax.dtype)
    powersig_unity = PowerSigJax(order=9, static_kernel=static_kernels.linear_kernel, dtype=unity_jax.dtype)
    powersig_unity_rbf = PowerSigJax(order=9, static_kernel=static_kernels.rbf_kernel, dtype=unity_jax.dtype)
    # Compute gram matrices for each dataset
    print("Computing PowerSigJax linear kernel...")
    ps_linear_original = powersig_linear(original_jax)
    ps_linear_scaled = powersig_linear(scaled_jax)
    ps_linear_chebychev = powersig_linear(chebychev_jax)
    ps_linear_unity = powersig_unity(unity_jax)
    
    print("Computing PowerSigJax RBF kernel...")
    ps_rbf_original = powersig_rbf(original_jax)
    ps_rbf_scaled = powersig_rbf(scaled_jax)
    ps_rbf_chebychev = powersig_rbf(chebychev_jax)
    ps_rbf_unity = powersig_unity_rbf(unity_jax)
    
    # Print parts of PowerSig gram matrices
    print("\n" + "="*60)
    print("POWERSIG GRAM MATRICES (showing first 3x3 submatrices)")
    print("="*60)
    
    print("\nPowerSig Linear Kernel - Original Data:")
    print("Shape:", np.array(ps_linear_original).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_linear_original)[:3, :3])
    
    print("\nPowerSig Linear Kernel - Scaled Data:")
    print("Shape:", np.array(ps_linear_scaled).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_linear_scaled)[:3, :3])
    
    print("\nPowerSig Linear Kernel - Chebyshev Data:")
    print("Shape:", np.array(ps_linear_chebychev).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_linear_chebychev)[:3, :3])
    
    print("\nPowerSig RBF Kernel - Original Data:")
    print("Shape:", np.array(ps_rbf_original).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_rbf_original)[:3, :3])
    
    print("\nPowerSig RBF Kernel - Scaled Data:")
    print("Shape:", np.array(ps_rbf_scaled).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_rbf_scaled)[:3, :3])
    
    print("\nPowerSig RBF Kernel - Chebyshev Data:")
    print("Shape:", np.array(ps_rbf_chebychev).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_rbf_chebychev)[:3, :3])
    
    print("\nPowerSig Linear Kernel - Unity Data:")
    print("Shape:", np.array(ps_linear_unity).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_linear_unity)[:3, :3])
    
    print("\nPowerSig RBF Kernel - Unity Data:")
    print("Shape:", np.array(ps_rbf_unity).shape)
    print("First 3x3 submatrix:")
    print(np.array(ps_rbf_unity)[:3, :3])
    
    # Compute KSigPDE gram matrices
    print("\nComputing KSigPDE gram matrices...")
    
    # Initialize KSig kernels
    ksig_linear = SignaturePDEKernel(normalize=False, static_kernel=LinearKernel())
    ksig_rbf = SignaturePDEKernel(normalize=False, static_kernel=RBFKernel())
    
    # Convert to cupy for KSig (KSig requires cupy arrays)
    original_np = cp.array(original_cumsum.numpy())
    scaled_np = cp.array(scaled_cumsum.numpy())
    chebychev_np = cp.array(chebychev_cumsum.numpy())
    unity_np = cp.array(unity_cumsum.numpy())
    
    print("Computing KSig linear kernel...")
    ks_linear_original = ksig_linear(original_np, original_np)
    ks_linear_scaled = ksig_linear(scaled_np, scaled_np)
    ks_linear_chebychev = ksig_linear(chebychev_np, chebychev_np)
    ks_linear_unity = ksig_linear(unity_np, unity_np)
    
    print("Computing KSig RBF kernel...")
    ks_rbf_original = ksig_rbf(original_np, original_np)
    ks_rbf_scaled = ksig_rbf(scaled_np, scaled_np)
    ks_rbf_chebychev = ksig_rbf(chebychev_np, chebychev_np)
    ks_rbf_unity = ksig_rbf(unity_np, unity_np)
    
    # Print parts of KSig gram matrices
    print("\n" + "="*60)
    print("KSIG GRAM MATRICES (showing first 3x3 submatrices)")
    print("="*60)
    
    print("\nKSig Linear Kernel - Original Data:")
    print("Shape:", cp.asnumpy(ks_linear_original).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_linear_original)[:3, :3])
    
    print("\nKSig Linear Kernel - Scaled Data:")
    print("Shape:", cp.asnumpy(ks_linear_scaled).shape)
    print("First 3x3 submatrix:")
    print("\nKSig Linear Kernel - Chebyshev Data:")
    print("Shape:", cp.asnumpy(ks_linear_chebychev).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_linear_chebychev)[:3, :3])
    
    print("\nKSig RBF Kernel - Original Data:")
    print("Shape:", cp.asnumpy(ks_rbf_original).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_rbf_original)[:3, :3])
    
    print("\nKSig RBF Kernel - Scaled Data:")
    print("Shape:", cp.asnumpy(ks_rbf_scaled).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_rbf_scaled)[:3, :3])
    
    print("\nKSig RBF Kernel - Chebyshev Data:")
    print("Shape:", cp.asnumpy(ks_rbf_chebychev).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_rbf_chebychev)[:3, :3])
    
    print("\nKSig Linear Kernel - Unity Data:")
    print("Shape:", cp.asnumpy(ks_linear_unity).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_linear_unity)[:3, :3])
    
    print("\nKSig RBF Kernel - Unity Data:")
    print("Shape:", cp.asnumpy(ks_rbf_unity).shape)
    print("First 3x3 submatrix:")
    print(cp.asnumpy(ks_rbf_unity)[:3, :3])
    
    # Compute condition numbers
    print("\nComputing condition numbers...")
    
    def compute_condition_number(matrix):
        """Compute condition number of a matrix"""
        try:
            # Convert CuPy arrays to numpy for condition number computation
            if hasattr(matrix, '__class__') and 'cupy' in str(matrix.__class__):
                matrix_np = cp.asnumpy(matrix)
            else:
                matrix_np = matrix
            
            # Check if matrix contains NaN values
            if np.any(np.isnan(matrix_np)):
                return np.nan
            
            return np.linalg.cond(matrix_np)
        except Exception as e:
            print(f"Error computing condition number: {e}")
            return np.nan
    
    # PowerSig condition numbers - Scaled and Unscaled
    ps_linear_orig_cn = compute_condition_number(np.array(ps_linear_original))
    
    # For PowerSig, create both normalized and unnormalized versions
    ps_linear_scaled_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_linear_scaled)))
    ps_linear_scaled_unnormalized_cn = compute_condition_number(np.array(ps_linear_scaled))
    
    ps_linear_chebychev_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_linear_chebychev)))
    ps_linear_chebychev_unnormalized_cn = compute_condition_number(np.array(ps_linear_chebychev))
    
    ps_linear_unity_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_linear_unity)))
    ps_linear_unity_unnormalized_cn = compute_condition_number(np.array(ps_linear_unity))
    
    ps_rbf_orig_cn = compute_condition_number(np.array(ps_rbf_original))
    
    ps_rbf_scaled_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_rbf_scaled)))
    ps_rbf_scaled_unnormalized_cn = compute_condition_number(np.array(ps_rbf_scaled))
    
    ps_rbf_chebychev_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_rbf_chebychev)))
    ps_rbf_chebychev_unnormalized_cn = compute_condition_number(np.array(ps_rbf_chebychev))
    
    ps_rbf_unity_cn = compute_condition_number(normalize_kernel_matrix(np.array(ps_rbf_unity)))
    ps_rbf_unity_unnormalized_cn = compute_condition_number(np.array(ps_rbf_unity))
    
    # Debug: print the actual values
    print("\nPowerSig Condition Numbers:")
    print(f"Linear Scaled - Normalized: {ps_linear_scaled_cn:.2e}, Unnormalized: {ps_linear_scaled_unnormalized_cn:.2e}")
    print(f"Linear Chebyshev - Normalized: {ps_linear_chebychev_cn:.2e}, Unnormalized: {ps_linear_chebychev_unnormalized_cn:.2e}")
    print(f"Linear Unity - Normalized: {ps_linear_unity_cn:.2e}, Unnormalized: {ps_linear_unity_unnormalized_cn:.2e}")
    print(f"RBF Scaled - Normalized: {ps_rbf_scaled_cn:.2e}, Unnormalized: {ps_rbf_scaled_unnormalized_cn:.2e}")
    print(f"RBF Chebyshev - Normalized: {ps_rbf_chebychev_cn:.2e}, Unnormalized: {ps_rbf_chebychev_unnormalized_cn:.2e}")
    print(f"RBF Unity - Normalized: {ps_rbf_unity_cn:.2e}, Unnormalized: {ps_rbf_unity_unnormalized_cn:.2e}")
    
    # KSig condition numbers - Scaled and Unscaled
    ks_linear_orig_cn = compute_condition_number(ks_linear_original)
    
    # For KSig, create both normalized and unnormalized versions
    ks_linear_scaled_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_linear_scaled)))
    ks_linear_scaled_unnormalized_cn = compute_condition_number(ks_linear_scaled)
    
    ks_linear_chebychev_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_linear_chebychev)))
    ks_linear_chebychev_unnormalized_cn = compute_condition_number(ks_linear_chebychev)
    
    ks_linear_unity_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_linear_unity)))
    ks_linear_unity_unnormalized_cn = compute_condition_number(ks_linear_unity)
    
    ks_rbf_orig_cn = compute_condition_number(ks_rbf_original)
    
    ks_rbf_scaled_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_rbf_scaled)))
    ks_rbf_scaled_unnormalized_cn = compute_condition_number(ks_rbf_scaled)
    
    ks_rbf_chebychev_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_rbf_chebychev)))
    ks_rbf_chebychev_unnormalized_cn = compute_condition_number(ks_rbf_chebychev)
    
    ks_rbf_unity_cn = compute_condition_number(normalize_kernel_matrix(cp.asnumpy(ks_rbf_unity)))
    ks_rbf_unity_unnormalized_cn = compute_condition_number(ks_rbf_unity)
    
    # Debug: print the actual values
    print("\nKSig Condition Numbers:")
    print(f"Linear Scaled - Normalized: {ks_linear_scaled_cn:.2e}, Unnormalized: {ks_linear_scaled_unnormalized_cn:.2e}")
    print(f"Linear Chebyshev - Normalized: {ks_linear_chebychev_cn:.2e}, Unnormalized: {ks_linear_chebychev_unnormalized_cn:.2e}")
    print(f"Linear Unity - Normalized: {ks_linear_unity_cn:.2e}, Unnormalized: {ks_linear_unity_unnormalized_cn:.2e}")
    print(f"RBF Scaled - Normalized: {ks_rbf_scaled_cn:.2e}, Unnormalized: {ks_rbf_scaled_unnormalized_cn:.2e}")
    print(f"RBF Chebyshev - Normalized: {ks_rbf_chebychev_cn:.2e}, Unnormalized: {ks_rbf_chebychev_unnormalized_cn:.2e}")
    print(f"RBF Unity - Normalized: {ks_rbf_unity_cn:.2e}, Unnormalized: {ks_rbf_unity_unnormalized_cn:.2e}")
    
    # Print results table
    # Save results to CSV
    print("\nSaving results to chebychev.csv...")
    
    # Create results data with normalized vs unnormalized distinction
    results_data = []
    
    # PowerSig results
    results_data.extend([
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Original', 'Normalized': False, 'Condition_Number': ps_linear_orig_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Original', 'Normalized': False, 'Condition_Number': ps_rbf_orig_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Scaled', 'Normalized': True, 'Condition_Number': ps_linear_scaled_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Scaled', 'Normalized': True, 'Condition_Number': ps_rbf_scaled_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Scaled', 'Normalized': False, 'Condition_Number': ps_linear_scaled_unnormalized_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Scaled', 'Normalized': False, 'Condition_Number': ps_rbf_scaled_unnormalized_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Chebyshev', 'Normalized': True, 'Condition_Number': ps_linear_chebychev_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Chebyshev', 'Normalized': True, 'Condition_Number': ps_rbf_chebychev_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Chebyshev', 'Normalized': False, 'Condition_Number': ps_linear_chebychev_unnormalized_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Chebyshev', 'Normalized': False, 'Condition_Number': ps_rbf_chebychev_unnormalized_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Unity', 'Normalized': True, 'Condition_Number': ps_linear_unity_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Unity', 'Normalized': True, 'Condition_Number': ps_rbf_unity_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'Linear', 'Data': 'Unity', 'Normalized': False, 'Condition_Number': ps_linear_unity_unnormalized_cn},
        {'Algorithm': 'PowerSig', 'Library': 'PowerSig', 'Kernel': 'RBF', 'Data': 'Unity', 'Normalized': False, 'Condition_Number': ps_rbf_unity_unnormalized_cn}
    ])
    
    # KSig results
    results_data.extend([
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Original', 'Normalized': False, 'Condition_Number': ks_linear_orig_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Original', 'Normalized': False, 'Condition_Number': ks_rbf_orig_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Scaled', 'Normalized': True, 'Condition_Number': ks_linear_scaled_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Scaled', 'Normalized': True, 'Condition_Number': ks_rbf_scaled_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Scaled', 'Normalized': False, 'Condition_Number': ks_linear_scaled_unnormalized_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Scaled', 'Normalized': False, 'Condition_Number': ks_rbf_scaled_unnormalized_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Chebyshev', 'Normalized': True, 'Condition_Number': ks_linear_chebychev_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Chebyshev', 'Normalized': True, 'Condition_Number': ks_rbf_chebychev_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Chebyshev', 'Normalized': False, 'Condition_Number': ks_linear_chebychev_unnormalized_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Chebyshev', 'Normalized': False, 'Condition_Number': ks_rbf_chebychev_unnormalized_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Unity', 'Normalized': True, 'Condition_Number': ks_linear_unity_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Unity', 'Normalized': True, 'Condition_Number': ks_rbf_unity_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'Linear', 'Data': 'Unity', 'Normalized': False, 'Condition_Number': ks_linear_unity_unnormalized_cn},
        {'Algorithm': 'KSigPDE', 'Library': 'KSig', 'Kernel': 'RBF', 'Data': 'Unity', 'Normalized': False, 'Condition_Number': ks_rbf_unity_unnormalized_cn}
    ])
    
    df = pd.DataFrame(results_data)
    df.to_csv('chebychev.csv', index=False)
    print("Results saved to chebychev.csv")
