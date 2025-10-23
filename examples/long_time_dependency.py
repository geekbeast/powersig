"""
Eigenworms Regression using SVR with Custom Kernels

This script implements Eigenworms regression using Support Vector Regression (SVR)
with custom signature kernels: KSigPDE, KSig RFSF-TRP, and PowerSigJax.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from examples.large_window import build_chebychev_dataset, build_chebychev_from_integer_recurrence, build_dataset, build_integer_recurrence, build_unity_from_integer_recurrence
from powersig.util.normalization import normalize_kernel_matrix
import jax
import numpy as np
from numpy.linalg import cond
import jax.numpy as jnp
import cupy as cp
import torch
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

import time
import logging
from typing import Tuple, Dict, Any, Optional, Literal
import os
import pickle
from enum import Enum

# Import PowerSig modules
from powersig.jax.algorithm import PowerSigJax
from powersig.jax import static_kernels

# Import KSig modules
import ksig
from ksig.kernels import SignaturePDEKernel, SignatureKernel
from ksig.static.kernels import LinearKernel, RBFKernel

# MLP functions not available for regression

# Dataset type enum
class DatasetType(Enum):
    """Enum for valid dataset types."""
    CHEBYCHEV = "chebychev"
    INTEGER_RECURRENCE = "integer_recurrence"
    UNITY_FROM_IR = "unity_from_ir"
    CHEBYCHEV_FROM_IR = "chebychev_from_ir"

# Subsampling method enum
class SubsampleMethod(Enum):
    """Enum for valid subsampling methods."""
    EQUALLY_SPACED = "equally_spaced"
    SLIDING_WINDOW = "sliding_window"
    NONE = "none"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import cuML SVR, fallback to sklearn if not available
try:
    from cuml.svm import SVR as cuMLSVR
    CUML_AVAILABLE = True

except ImportError:
    CUML_AVAILABLE = False

# Constants for quick experiments
MAX_TIMESTEPS = 2600  # Limit number of timesteps for faster experiments max is 17984
# Subsample method: "equally_spaced", "sliding_window", or "none"
SUBSAMPLE_METHOD = SubsampleMethod.NONE  
# Cache directory for gram matrices
CACHE_DIR = "gram_matrix_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Kernel type constants
KERNEL_LINEAR = "linear"
KERNEL_RBF = "rbf"
KernelType = Literal["linear", "rbf"]

# Kernel names and control set
KERNEL_NAMES = {
    "KSigPDE": "KSigPDE",
    "KSig RFSF-TRP": "KSig RFSF-TRP", 
    "PowerSigJax": "PowerSigJax",
    "KNN_DTW": "KNN_DTW"
}

# Set of kernels to run (modify this to control which kernels execute)
KERNELS_TO_RUN = {
    "KSigPDE",
    "KSig RFSF-TRP", 
    "PowerSigJax",
    "cuML_Baseline",
    "KNN_DTW"
}

# C parameter grid for GridSearchCV
C_GRID = [10 ** (i-1) for i in range(6)]  # [0.1, 1, 10, 100, 1000, 10000]


def preserve_complex_or_float64(array):
    """
    Preserve complex128 types, convert other types to float64.
    
    Args:
        array: Input array that may be complex128 or other numeric types
        
    Returns:
        Array with complex128 preserved or converted to float64
    """
    if array is None:
        return None
    
    # Ensure it's a numpy array first
    array = ensure_numpy_array(array)
    
    # If it's already complex128, preserve it
    if array.dtype == np.complex128:
        return array
    
    # Otherwise convert to float64
    return array.astype(np.float64)


def generate_cache_filename(kernel_name: str, X_train_shape: Tuple[int, ...], X_test_shape: Tuple[int, ...], 
                          kernel_type: Optional[KernelType] = None, seed: Optional[int] = None, **kwargs) -> str:
    """
    Generate a cache filename based on kernel name, data shapes, parameters, and seed.
    
    Args:
        kernel_name: Name of the kernel
        X_train_shape: Shape of training data
        X_test_shape: Shape of test data
        kernel_type: Type of kernel (linear or rbf), optional
        seed: Random seed for reproducibility, optional
        **kwargs: Additional parameters for the kernel
        
    Returns:
        Cache filename
    """
    # Build plaintext filename components
    filename_parts = [kernel_name]
    
    if kernel_type is not None:
        filename_parts.append(kernel_type)
    
    # Add shape information
    train_shape_str = f"{X_train_shape[0]}x{X_train_shape[1]}x{X_train_shape[2]}" if len(X_train_shape) == 3 else f"{X_train_shape[0]}x{X_train_shape[1]}"
    test_shape_str = f"{X_test_shape[0]}x{X_test_shape[1]}x{X_test_shape[2]}" if len(X_test_shape) == 3 else f"{X_test_shape[0]}x{X_test_shape[1]}"
    filename_parts.extend([train_shape_str, test_shape_str])
    
    # Add MAX_TIMESTEPS
    filename_parts.append(f"t{MAX_TIMESTEPS}")
    
    # Add seed if provided
    if seed is not None:
        filename_parts.append(f"seed{seed}")
    
    # Add additional parameters
    for key, value in sorted(kwargs.items()):
        filename_parts.append(f"{key}{value}")
    
    # Create filename - ensure all parts are strings
    filename = "_".join(str(part) for part in filename_parts) + ".pkl"
    return os.path.join(CACHE_DIR, filename)


def ensure_numpy_array(array):
    """
    Ensure an array is a numpy array, converting from cupy if necessary.
    
    Args:
        array: Input array (numpy or cupy)
        
    Returns:
        numpy.ndarray: The array as a numpy array
    """
    if hasattr(array, 'get'):  # cupy array
        return array.get()
    else:
        return array


def validate_gram_matrices(train_gram: np.ndarray, test_gram: np.ndarray, 
                          X_train: np.ndarray, X_test: np.ndarray) -> bool:
    """
    Validate that loaded gram matrices have correct dimensions.
    
    Args:
        train_gram: Training gram matrix
        test_gram: Test gram matrix
        X_train: Training data
        X_test: Test data
        
    Returns:
        True if dimensions match, False otherwise
    """
    expected_train_shape = (X_train.shape[0], X_train.shape[0])
    expected_test_shape = (X_test.shape[0], X_train.shape[0])
    
    if train_gram.shape != expected_train_shape:
        logger.warning(f"Train gram matrix shape mismatch: expected {expected_train_shape}, got {train_gram.shape}")
        return False
    
    if test_gram.shape != expected_test_shape:
        logger.warning(f"Test gram matrix shape mismatch: expected {expected_test_shape}, got {test_gram.shape}")
        return False
    
    return True


def build_regression_dataset(dataset_type: DatasetType, history_length: int = 22, num_samples: int = 50, 
                           num_timesteps: int = MAX_TIMESTEPS, dimensions: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a regression dataset using different dataset generation methods.
    
    Args:
        dataset_type: Type of dataset to generate (DatasetType enum)
        history_length: Length of each time series
        num_samples: Number of samples
        num_timesteps: Number of timestamps to use
        dimensions: Number of dimensions
        
    Returns:
        Tuple of (X, y) where X is the time series data and y are the targets
    """
    logger.info(f"Building regression dataset with {num_samples} samples using {dataset_type}...")
    
    # Use the appropriate dataset generation function based on dataset_type
    from examples.large_window import build_dataset, build_chebychev_dataset, build_integer_recurrence, build_unity_from_integer_recurrence, build_chebychev_from_integer_recurrence
    
    if dataset_type == DatasetType.CHEBYCHEV:
        X, y = build_chebychev_dataset(num_samples=num_samples, num_timestamps=num_timesteps, dimensions=dimensions)
    elif dataset_type == DatasetType.INTEGER_RECURRENCE:
        X, y = build_integer_recurrence(p=history_length, num_samples=num_samples, num_timesteps=num_timesteps, dimensions=dimensions)
        X = X /20.0
        y = y /20.0
        # max_X = 1.0*torch.max(torch.abs(X))
        # X =X / max_X
        # y = y / max_X
    elif dataset_type == DatasetType.UNITY_FROM_IR:
        X, y = build_unity_from_integer_recurrence(p=history_length, num_samples=num_samples, num_timesteps=num_timesteps, dimensions=dimensions)
    elif dataset_type == DatasetType.CHEBYCHEV_FROM_IR:
        X, y = build_chebychev_from_integer_recurrence(p=history_length, num_samples=num_samples, num_timesteps=num_timesteps, dimensions=dimensions)
        max_X = 1.0*torch.max(torch.abs(X))
        X =X / max_X
        y = y / max_X
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be one of: {[dt.value for dt in DatasetType]}")
    
    logger.info(f"Dataset built successfully!")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"X.dtype: {X.dtype}")
    logger.info(f"y.dtype: {y.dtype}")
    
    # Defensively convert to float64 for min, mean, and std computations
    if torch.is_complex(y):
        y_fp64 = y.to(torch.complex128)
        # For complex numbers, compute statistics on real and imaginary parts separately
        y_real = y_fp64.real
        y_imag = y_fp64.imag
        logger.info(f"Target real range: [{y_real.min():.4f}, {y_real.max():.4f}]")
        logger.info(f"Target imag range: [{y_imag.min():.4f}, {y_imag.max():.4f}]")
        logger.info(f"Target real mean: {y_real.mean():.4f}, std: {y_real.std():.4f}")
        logger.info(f"Target imag mean: {y_imag.mean():.4f}, std: {y_imag.std():.4f}")
        # Also log magnitude statistics
        y_magnitude = torch.abs(y_fp64)
        logger.info(f"Target magnitude range: [{y_magnitude.min():.4f}, {y_magnitude.max():.4f}]")
        logger.info(f"Target magnitude mean: {y_magnitude.mean():.4f}, std: {y_magnitude.std():.4f}")
        return X.to(torch.complex128), y.to(torch.complex128)
    else:
        y_fp64 = y.to(torch.float64)
        logger.info(f"Target range: [{y_fp64.min():.4f}, {y_fp64.max():.4f}]")
        logger.info(f"Target mean: {y_fp64.mean():.4f}, std: {y_fp64.std():.4f}")
        return X.to(torch.float64), y.to(torch.float64)


def plot_regression_samples(X: np.ndarray, y: np.ndarray = None, num_samples: int = 5, seed: Optional[int] = None):
    """
    Plot regression samples.
    
    Args:
        X: Dataset with shape (samples, timesteps, dimensions) - can be numpy array or torch tensor
        y: Regression targets with shape (samples, dimensions) - optional, can be numpy array or torch tensor
        num_samples: Number of samples to plot
        seed: Random seed for filename identification
    """
    logger.info(f"Plotting regression samples...")
    
    # Convert torch tensors to numpy arrays if needed
    if hasattr(X, 'cpu'):  # Check if it's a torch tensor
        X = X.cpu().numpy()
        logger.info("Converted X from torch tensor to numpy array")
    
    if y is not None and hasattr(y, 'cpu'):  # Check if y is a torch tensor
        y = y.cpu().numpy()
        logger.info("Converted y from torch tensor to numpy array")
    
    # Create time axis
    timesteps = X.shape[1]
    time_axis = np.linspace(0, timesteps - 1, timesteps)
    num_dimensions = X.shape[2]
    
    # TEMPORARY: Limit timesteps to 1290-2000 range
    start_idx = 1290
    end_idx = min(2000, timesteps)
    if timesteps > end_idx:
        logger.info(f"Temporarily limiting plot to timesteps {start_idx}-{end_idx} (original: 0-{timesteps-1})")
        time_axis = time_axis[start_idx:end_idx]
        X = X[:, start_idx:end_idx, :]
    
    # Create seed suffix for filenames
    seed_suffix = f"_seed_{seed}" if seed is not None else ""
    
    # Check if data is complex
    is_complex = np.iscomplexobj(X)
    logger.info(f"Data is complex: {is_complex}")
    
    # Plot first few samples
    for sample_idx in range(min(num_samples, X.shape[0])):
        # Get the sample data
        sample_data = X[sample_idx]  # Shape: (timesteps, dimensions)
        
        # Extract real and imaginary parts for complex data (needed for both main plots and difference plots)
        if is_complex:
            real_data = np.real(sample_data)
            imag_data = np.imag(sample_data)
        
        if is_complex:
            # Create 2x2 grid for complex data
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Sample {sample_idx + 1} - Complex Data', fontsize=16)
            
            # Find global min/max for shared scales
            real_min, real_max = real_data.min(), real_data.max()
            imag_min, imag_max = imag_data.min(), imag_data.max()
            
            # Left column: Real parts
            for dim_idx in range(min(2, num_dimensions)):
                axes[dim_idx, 0].plot(time_axis, real_data[:, dim_idx], linewidth=1.5, color='blue')
                title = f'Real Part - Dimension {dim_idx + 1}'
                if y is not None:
                    title += f' (Target: {np.real(y[sample_idx, dim_idx]):.4f})'
                axes[dim_idx, 0].set_title(title)
                axes[dim_idx, 0].set_xlabel('Time Step')
                axes[dim_idx, 0].set_ylabel('Real Value')
                axes[dim_idx, 0].grid(True, alpha=0.3)
                # Share scale horizontally
                axes[dim_idx, 0].set_ylim(real_min, real_max)
            
            # Right column: Imaginary parts
            for dim_idx in range(min(2, num_dimensions)):
                axes[dim_idx, 1].plot(time_axis, imag_data[:, dim_idx], linewidth=1.5, color='red')
                title = f'Imaginary Part - Dimension {dim_idx + 1}'
                if y is not None:
                    title += f' (Target: {np.imag(y[sample_idx, dim_idx]):.4f})'
                axes[dim_idx, 1].set_title(title)
                axes[dim_idx, 1].set_xlabel('Time Step')
                axes[dim_idx, 1].set_ylabel('Imaginary Value')
                axes[dim_idx, 1].grid(True, alpha=0.3)
                # Share scale horizontally
                axes[dim_idx, 1].set_ylim(imag_min, imag_max)
            
            # Hide unused subplots if dimensions < 2
            if num_dimensions < 2:
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
            
        else:
            # Original plotting for real data
            fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 3*num_dimensions))
            if num_dimensions == 1:
                axes = [axes]
            
            # Plot each dimension
            for dim_idx in range(num_dimensions):
                dimension_data = sample_data[:, dim_idx]
                axes[dim_idx].plot(time_axis, dimension_data, linewidth=1.5)
                title = f'Sample {sample_idx + 1}, Dimension {dim_idx + 1}'
                if y is not None:
                    title += f' (Target: {y[sample_idx, dim_idx]:.4f})'
                axes[dim_idx].set_title(title)
                axes[dim_idx].set_xlabel('Time Step')
                axes[dim_idx].set_ylabel('Value')
                axes[dim_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'regression_sample_{sample_idx + 1}{seed_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot for sample {sample_idx + 1}")
        
        # Create difference plot for this sample
        if is_complex:
            # Create 2x2 grid for complex differences
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Sample {sample_idx + 1} - Complex Data Differences', fontsize=16)
            
            # Calculate differences for real and imaginary parts
            real_differences = np.diff(real_data, axis=0)
            imag_differences = np.diff(imag_data, axis=0)
            diff_time_axis = time_axis[1:]
            
            # Find global min/max for shared scales
            real_diff_min, real_diff_max = real_differences.min(), real_differences.max()
            imag_diff_min, imag_diff_max = imag_differences.min(), imag_differences.max()
            
            # Left column: Real part differences
            for dim_idx in range(min(2, num_dimensions)):
                axes[dim_idx, 0].plot(diff_time_axis, real_differences[:, dim_idx], linewidth=1.5, color='blue')
                title = f'Real Part Differences - Dimension {dim_idx + 1}'
                if y is not None:
                    title += f' (Target: {np.real(y[sample_idx, dim_idx]):.4f})'
                axes[dim_idx, 0].set_title(title)
                axes[dim_idx, 0].set_xlabel('Time Step')
                axes[dim_idx, 0].set_ylabel('Real Difference (t+1 - t)')
                axes[dim_idx, 0].grid(True, alpha=0.3)
                axes[dim_idx, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                # Share scale horizontally
                axes[dim_idx, 0].set_ylim(real_diff_min, real_diff_max)
            
            # Right column: Imaginary part differences
            for dim_idx in range(min(2, num_dimensions)):
                axes[dim_idx, 1].plot(diff_time_axis, imag_differences[:, dim_idx], linewidth=1.5, color='red')
                title = f'Imaginary Part Differences - Dimension {dim_idx + 1}'
                if y is not None:
                    title += f' (Target: {np.imag(y[sample_idx, dim_idx]):.4f})'
                axes[dim_idx, 1].set_title(title)
                axes[dim_idx, 1].set_xlabel('Time Step')
                axes[dim_idx, 1].set_ylabel('Imaginary Difference (t+1 - t)')
                axes[dim_idx, 1].grid(True, alpha=0.3)
                axes[dim_idx, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                # Share scale horizontally
                axes[dim_idx, 1].set_ylim(imag_diff_min, imag_diff_max)
            
            # Hide unused subplots if dimensions < 2
            if num_dimensions < 2:
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
                
        else:
            # Original difference plotting for real data
            fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 3*num_dimensions))
            if num_dimensions == 1:
                axes = [axes]
            
            # Plot differences for each dimension
            for dim_idx in range(num_dimensions):
                dimension_data = sample_data[:, dim_idx]
                # Calculate differences: i+1 - i for each timestep
                differences = np.diff(dimension_data)
                # Time axis for differences (one less point)
                diff_time_axis = time_axis[1:]
                
                axes[dim_idx].plot(diff_time_axis, differences, linewidth=1.5, color='red')
                title = f'Sample {sample_idx + 1}, Dimension {dim_idx + 1} - Differences'
                if y is not None:
                    title += f' (Target: {y[sample_idx, dim_idx]:.4f})'
                axes[dim_idx].set_title(title)
                axes[dim_idx].set_xlabel('Time Step')
                axes[dim_idx].set_ylabel('Difference (t+1 - t)')
                axes[dim_idx].grid(True, alpha=0.3)
                axes[dim_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'recurrence_diff_{sample_idx + 1}{seed_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved difference plot for sample {sample_idx + 1}")


def normalize_training_data(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize each sample individually by computing its own mean and std dev.
    For each sample: subtract its mean and divide by its std dev.
    Also normalize target variables y_train and y_test using the same per-sample approach.
    
    Args:
        X_train: Training dataset with shape (samples, timesteps, dimensions)
        X_test: Test dataset with shape (samples, timesteps, dimensions)
        y_train: Training target variables
        y_test: Test target variables
        
    Returns:
        Tuple of (normalized_X_train, normalized_X_test, normalized_y_train, normalized_y_test)
    """
    logger.info("Normalizing each sample individually...")
    
    num_train_samples, num_timesteps, num_dimensions = X_train.shape
    num_test_samples = X_test.shape[0]
    
    # Normalize X_train: for each sample and dimension, compute mean and std across timesteps only
    # Compute mean and std for each sample and dimension: shape (num_samples, num_dimensions)
    X_train_means = torch.mean(X_train, dim=1)  # Shape: (num_samples, num_dimensions)
    X_train_stds = torch.std(X_train, dim=1)    # Shape: (num_samples, num_dimensions)
    
    # Avoid division by zero
    X_train_stds = torch.where(X_train_stds == 0, torch.tensor(1.0, device=X_train.device), X_train_stds)
    zero_std_samples = torch.where(X_train_stds == 1.0)[0]
    for sample_idx in zero_std_samples:
        logger.warning(f"Training sample {sample_idx} has zero standard deviation, setting to 1.0")
    
    # Reshape for broadcasting: (num_samples, 1, num_dimensions) for broadcasting with (num_samples, num_timesteps, num_dimensions)
    X_train_means_broadcast = X_train_means.reshape(-1, 1, num_dimensions)  # Shape: (num_samples, 1, num_dimensions)
    X_train_stds_broadcast = X_train_stds.reshape(-1, 1, num_dimensions)    # Shape: (num_samples, 1, num_dimensions)
    
    normalized_X_train = (X_train - X_train_means_broadcast) / X_train_stds_broadcast
    
    # Normalize X_test: for each sample and dimension, compute mean and std across timesteps only
    X_test_means = torch.mean(X_test, dim=1)  # Shape: (num_test_samples, num_dimensions)
    X_test_stds = torch.std(X_test, dim=1)    # Shape: (num_test_samples, num_dimensions)
    
    # Avoid division by zero
    X_test_stds = torch.where(X_test_stds == 0, torch.tensor(1.0, device=X_test.device), X_test_stds)
    zero_std_samples_test = torch.where(X_test_stds == 1.0)[0]
    for sample_idx in zero_std_samples_test:
        logger.warning(f"Test sample {sample_idx} has zero standard deviation, setting to 1.0")
    
    # Reshape for broadcasting: (num_test_samples, 1, num_dimensions) for broadcasting with (num_test_samples, num_timesteps, num_dimensions)
    X_test_means_broadcast = X_test_means.reshape(-1, 1, num_dimensions)  # Shape: (num_test_samples, 1, num_dimensions)
    X_test_stds_broadcast = X_test_stds.reshape(-1, 1, num_dimensions)    # Shape: (num_test_samples, 1, num_dimensions)
    
    normalized_X_test = (X_test - X_test_means_broadcast) / X_test_stds_broadcast
    
    # Normalize y_train and y_test using the corresponding sample statistics
    # y should have shape (num_samples, dimensions) which matches X_train_means and X_train_stds
    # No need to reshape or take means across dimensions
    
    # For y_train: use the corresponding X_train sample means and stds
    normalized_y_train = (y_train - X_train_means) / X_train_stds
    
    # For y_test: use the corresponding X_test sample means and stds
    normalized_y_test = (y_test - X_test_means) / X_test_stds
    
    logger.info("y_train and y_test normalized using corresponding sample statistics")
    
    logger.info("Per-sample normalization completed!")
    logger.info(f"Normalized training set shape: {normalized_X_train.shape}")
    logger.info(f"Normalized test set shape: {normalized_X_test.shape}")
    logger.info(f"Normalized y_train shape: {normalized_y_train.shape}")
    logger.info(f"Normalized y_test shape: {normalized_y_test.shape}")
    
    return normalized_X_train, normalized_X_test, normalized_y_train, normalized_y_test, X_train_means, X_train_stds, X_test_means, X_test_stds


def time_augment(X):
    """
    Augment the input array by adding a time feature as the last dimension.
    
    Args:
        X: Input array of shape (n_samples, n_timesteps) or (n_samples, n_timesteps, n_features)
        
    Returns:
        Augmented array with time feature added as the last dimension
    """
    if len(X.shape) == 2:
        # If X is 2D (samples, timesteps), add time feature
        n_samples, length = X.shape
        time_feature = np.linspace(0, 1, length)
        time_feature = np.broadcast_to(time_feature, (n_samples, length))
        time_feature = time_feature[..., None]  # shape (n_samples, length, 1)
        X_expanded = X[..., None]  # shape (n_samples, length, 1)
        return np.concatenate([time_feature, X_expanded], axis=-1)
    elif len(X.shape) == 3:
        # If X is already 3D (samples, timesteps, features), add time feature
        n_samples, length, n_features = X.shape
        time_feature = np.linspace(0, 1, length)
        time_feature = np.broadcast_to(time_feature, (n_samples, length))
        time_feature = time_feature[..., None]  # shape (n_samples, length, 1)
        return np.concatenate([time_feature, X], axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def subsample_timesteps(X: torch.Tensor, max_timesteps: int) -> torch.Tensor:
    """
    Subsample timesteps from time series data by taking equally spaced points.
    
    Args:
        X: Time series data with shape (samples, timesteps, dimensions)
        max_timesteps: Maximum number of timesteps to keep
        
    Returns:
        Subsampled time series data with shape (samples, max_timesteps, dimensions)
    """
    if X.shape[1] <= max_timesteps:
        # No subsampling needed
        return X
    
    # Take max_timesteps equally spaced points from the full sequence
    original_length = X.shape[1]
    indices = torch.linspace(0, original_length - 1, max_timesteps, dtype=torch.long, device=X.device)
    X_subsampled = X[:, indices, :]
    
    logger.info(f"Subsampled from {original_length} to {max_timesteps} timesteps")
    logger.info(f"Original shape: {X.shape}, New shape: {X_subsampled.shape}")
    
    return X_subsampled


def sliding_window_subsample(X: torch.Tensor, y: torch.Tensor, window_length: int, stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding window subsequences from time series data and predict next timestep.
    
    Args:
        X: Time series data with shape (samples, timesteps, dimensions)
        y: Original targets with shape (samples, dimensions)
        window_length: Length of each sliding window
        stride: Step size between windows (default=1 for overlapping windows)
        
    Returns:
        Tuple of (X_windows, y_windows) where:
        - X_windows: Windowed time series data (original + new windows)
        - y_windows: Targets (original y for last window, next timestep for others)
    """
    if X.shape[1] <= window_length:
        # No windowing needed, return original data
        return X, y
    
    num_samples, num_timesteps, num_dimensions = X.shape
    num_windows_per_sample = (num_timesteps - window_length) // stride
    
    # Calculate total samples: original samples + new window samples
    total_samples = num_samples + (num_samples * num_windows_per_sample)
    
    # Initialize tensors for windowed data (including original samples)
    X_windows = torch.zeros((total_samples, window_length, num_dimensions), dtype=X.dtype, device=X.device)
    y_windows = torch.zeros((total_samples, num_dimensions), dtype=y.dtype, device=y.device)
    
    # First, add the original samples (using the last window of each original sample)
    for sample_idx in range(num_samples):
        # Use the last window for original samples
        last_window_start = num_timesteps - window_length
        X_windows[sample_idx] = X[sample_idx, last_window_start:last_window_start + window_length, :]
        y_windows[sample_idx] = y[sample_idx]  # Use original target
    
    # Then, add all the sliding windows as new samples
    window_idx = num_samples  # Start after original samples
    for sample_idx in range(num_samples):
        for window_start in range(0, num_timesteps - window_length, stride):
            # Extract window
            window_data = X[sample_idx, window_start:window_start + window_length, :]
            X_windows[window_idx] = window_data
            
            # Target is the next timestep after the window
            next_timestep = window_start + window_length
            y_windows[window_idx] = X[sample_idx, next_timestep, :]
            
            window_idx += 1
    
    logger.info(f"Created {num_windows_per_sample} windows per sample with length {window_length}")
    logger.info(f"Original shape: {X.shape}, Windowed shape: {X_windows.shape}")
    logger.info(f"Original y shape: {y.shape}, Windowed y shape: {y_windows.shape}")
    logger.info(f"Total samples: {num_samples} original + {num_samples * num_windows_per_sample} new = {total_samples}")
    
    return X_windows, y_windows


def print_dataset_statistics(X_train: np.ndarray, X_test: np.ndarray):
    """
    Print statistics for each dimension across training and test sets.
    
    Args:
        X_train: Training dataset with shape (samples, timesteps, dimensions) - can be real or complex
        X_test: Test dataset with shape (samples, timesteps, dimensions) - can be real or complex
    """
    logger.info("Dataset statistics:")
    logger.info("=" * 50)
    
    num_train_samples, num_timesteps, num_dimensions = X_train.shape
    num_test_samples = X_test.shape[0]
    
    # Check if data is complex
    train_is_complex = hasattr(X_train, 'is_complex') and X_train.is_complex() if hasattr(X_train, 'is_complex') else np.iscomplexobj(X_train)
    test_is_complex = hasattr(X_test, 'is_complex') and X_test.is_complex() if hasattr(X_test, 'is_complex') else np.iscomplexobj(X_test)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Training set dtype: {X_train.dtype}")
    logger.info(f"Test set dtype: {X_test.dtype}")
    logger.info(f"Training set is complex: {train_is_complex}")
    logger.info(f"Test set is complex: {test_is_complex}")
    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of test samples: {num_test_samples}")
    logger.info(f"Number of timesteps: {num_timesteps}")
    logger.info(f"Number of dimensions: {num_dimensions}")
    logger.info("=" * 50)
    
    # Calculate statistics for each dimension
    for dim_idx in range(num_dimensions):
        # Extract all values for this dimension across all samples and timesteps
        train_dimension_data = X_train[:, :, dim_idx].flatten()
        test_dimension_data = X_test[:, :, dim_idx].flatten()
        
        logger.info(f"Dimension {dim_idx + 1}:")
        
        if train_is_complex:
            # Handle complex training data
            train_real = train_dimension_data.real if hasattr(train_dimension_data, 'real') else np.real(train_dimension_data)
            train_imag = train_dimension_data.imag if hasattr(train_dimension_data, 'imag') else np.imag(train_dimension_data)
            train_magnitude = np.abs(train_dimension_data)
            
            logger.info(f"  Training Set:")
            logger.info(f"    Real - Min: {train_real.min():.6f}, Max: {train_real.max():.6f}, Mean: {train_real.mean():.6f}, Std: {train_real.std():.6f}")
            logger.info(f"    Imag - Min: {train_imag.min():.6f}, Max: {train_imag.max():.6f}, Mean: {train_imag.mean():.6f}, Std: {train_imag.std():.6f}")
            logger.info(f"    Magnitude - Min: {train_magnitude.min():.6f}, Max: {train_magnitude.max():.6f}, Mean: {train_magnitude.mean():.6f}, Std: {train_magnitude.std():.6f}")
        else:
            # Handle real training data
            logger.info(f"  Training Set:")
            logger.info(f"    Min: {train_dimension_data.min():.6f}")
            logger.info(f"    Max: {train_dimension_data.max():.6f}")
            logger.info(f"    Mean: {train_dimension_data.mean():.6f}")
            logger.info(f"    Std Dev: {train_dimension_data.std():.6f}")
        
        if test_is_complex:
            # Handle complex test data
            test_real = test_dimension_data.real if hasattr(test_dimension_data, 'real') else np.real(test_dimension_data)
            test_imag = test_dimension_data.imag if hasattr(test_dimension_data, 'imag') else np.imag(test_dimension_data)
            test_magnitude = np.abs(test_dimension_data)
            
            logger.info(f"  Test Set:")
            logger.info(f"    Real - Min: {test_real.min():.6f}, Max: {test_real.max():.6f}, Mean: {test_real.mean():.6f}, Std: {test_real.std():.6f}")
            logger.info(f"    Imag - Min: {test_imag.min():.6f}, Max: {test_imag.max():.6f}, Mean: {test_imag.mean():.6f}, Std: {test_imag.std():.6f}")
            logger.info(f"    Magnitude - Min: {test_magnitude.min():.6f}, Max: {test_magnitude.max():.6f}, Mean: {test_magnitude.mean():.6f}, Std: {test_magnitude.std():.6f}")
        else:
            # Handle real test data
            logger.info(f"  Test Set:")
            logger.info(f"    Min: {test_dimension_data.min():.6f}")
            logger.info(f"    Max: {test_dimension_data.max():.6f}")
            logger.info(f"    Mean: {test_dimension_data.mean():.6f}")
            logger.info(f"    Std Dev: {test_dimension_data.std():.6f}")
        
        logger.info("-" * 30)


def compute_gram_matrix_ksig_pde(X_train: np.ndarray, X_test: np.ndarray, kernel_type: KernelType = KERNEL_LINEAR, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute gram matrices using KSigPDE kernel with local caching.
    
    Args:
        X_train: Training data
        X_test: Test data
        kernel_type: Type of kernel to use (linear or rbf)
        
    Returns:
        Tuple of (train_gram, test_gram, computation_time) where computation_time is -1.0 if loaded from cache
    """
    # Generate cache filename
    cache_filename = generate_cache_filename("KSigPDE", kernel_type, X_train.shape, X_test.shape, seed=seed)
    
    # Check if cache exists and try to load
    logger.info(f"Checking for KSigPDE cache file: {cache_filename}")
    if os.path.exists(cache_filename):
        logger.info(f"Cache HIT: Loading cached KSigPDE gram matrices from {cache_filename}...")
        try:
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
                train_gram = cached_data['train_gram']
                test_gram = cached_data['test_gram']
            
            # Validate dimensions
            if validate_gram_matrices(train_gram, test_gram, X_train, X_test):
                logger.info("Successfully loaded cached KSigPDE gram matrices")
                logger.info(f"Train gram shape: {train_gram.shape}")
                logger.info(f"Test gram shape: {test_gram.shape}")
                
                # Print a small portion of the training gram matrix (from cache)
                print(f"KSigPDE Training gram shape (cached): {train_gram.shape}")
                print("Small portion of KSigPDE training gram matrix (5x5) - from cache:")
                print(train_gram[:5, :5])
                print(f"KSigPDE Training gram min: {train_gram.min():.6f}, max: {train_gram.max():.6f}, mean: {train_gram.mean():.6f}")
                
                # Calculate and log condition number for cached gram matrix
                try:
                    condition_number = np.linalg.cond(train_gram)
                    logger.info(f"KSigPDE cached train gram condition number: {condition_number:.2e}")
                    if condition_number > 1e12:
                        logger.warning(f"High condition number ({condition_number:.2e}) may cause training issues!")
                    elif condition_number > 1e8:
                        logger.warning(f"Moderately high condition number ({condition_number:.2e})")
                except Exception as e:
                    logger.warning(f"Could not compute condition number for cached gram matrix: {e}")
                
                return train_gram, test_gram, -1.0  # -1.0 indicates loaded from cache
            else:
                logger.warning("Cached gram matrices have incorrect dimensions, recomputing...")
        except Exception as e:
            logger.warning(f"Failed to load cached gram matrices: {e}")
    else:
        logger.info(f"Cache MISS: No cached KSigPDE gram matrices found at {cache_filename}")
    
    logger.info(f"Computing gram matrices using KSigPDE with {kernel_type} kernel...")
    start_time = time.time()
    
    try:
        # Initialize KSigPDE kernel based on kernel type
        if kernel_type == KERNEL_LINEAR:
            static_kernel = LinearKernel()
        elif kernel_type == KERNEL_RBF:
            static_kernel = RBFKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        ksig_pde_kernel = SignaturePDEKernel(normalize=False, static_kernel=static_kernel)
        
        # Convert to CuPy arrays for GPU acceleration
        X_train_cp = cp.array(X_train, dtype=cp.float64)
        X_test_cp = cp.array(X_test, dtype=cp.float64)
        
        # Compute gram matrices
        train_gram = ksig_pde_kernel(X_train_cp, X_train_cp)
        test_gram = ksig_pde_kernel(X_test_cp, X_train_cp)
        
        # Convert back to numpy
        train_gram = cp.asnumpy(train_gram)
        test_gram = cp.asnumpy(test_gram)
        
        # Add small epsilon * I to improve numerical stability
        epsilon = 1e-6  # Increased from 1e-8 for better stability
        n_train = train_gram.shape[0]
        train_gram += epsilon * np.eye(n_train)
        
        # Cache the results (ensure numpy arrays) - AFTER post-processing
        try:
            with open(cache_filename, 'wb') as f:
                pickle.dump({
                    'train_gram': ensure_numpy_array(train_gram),
                    'test_gram': ensure_numpy_array(test_gram)
                }, f)
            logger.info(f"Cached KSigPDE gram matrices to {cache_filename}")
            # Verify the file was actually created
            if os.path.exists(cache_filename):
                file_size = os.path.getsize(cache_filename)
                logger.info(f"Cache file created successfully: {cache_filename} (size: {file_size} bytes)")
            else:
                logger.warning(f"Cache file was not created: {cache_filename}")
        except Exception as e:
            logger.warning(f"Failed to cache gram matrices: {e}")
        
        # Print a small portion of the training gram matrix
        print(f"KSigPDE Training gram shape: {train_gram.shape}")
        print("Small portion of KSigPDE training gram matrix (5x5):")
        print(train_gram[:5, :5])
        print(f"KSigPDE Training gram min: {train_gram.min():.6f}, max: {train_gram.max():.6f}, mean: {train_gram.mean():.6f}")
        
        # Calculate and log condition number early
        try:
            # Convert to numpy array if it's a CuPy array
            train_gram_np = ensure_numpy_array(train_gram)
            condition_number = np.linalg.cond(train_gram_np)
            logger.info(f"KSigPDE train gram condition number: {condition_number:.2e}")
            if condition_number > 1e12:
                logger.warning(f"High condition number ({condition_number:.2e}) may cause training issues!")
            elif condition_number > 1e8:
                logger.warning(f"Moderately high condition number ({condition_number:.2e})")
        except Exception as e:
            logger.warning(f"Could not compute condition number: {e}")
        
        computation_time = time.time() - start_time
        logger.info(f"KSigPDE computation time: {computation_time:.3f}s")
        logger.info(f"Train gram shape: {train_gram.shape}")
        logger.info(f"Test gram shape: {test_gram.shape}")
        
        return train_gram, test_gram, computation_time
        
    except (MemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "out_of_memory" in str(e).lower() or "oom" in str(e).lower() or isinstance(e, MemoryError):
            logger.error(f"KSigPDE ran out of memory: {e}")
            logger.info("KSigPDE computation skipped due to OOM")
            # Return None to indicate OOM failure
            return None, None, -1.0
        else:
            logger.error(f"KSigPDE computation failed with error: {e}")
            raise
    except Exception as e:
        logger.error(f"KSigPDE computation failed with unexpected error: {e}")
        raise


def compute_gram_matrix_ksig_rfsf_trp(X_train: torch.Tensor, X_test: torch.Tensor, 
                                      n_levels: int = 21, n_features: int = 1000, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the train–train and test–train kernel matrices using
    Random Fourier Signature Features with Tensorized Random Projection (RFSF‑TRP) with local caching.

    Parameters
    ----------
    X_train : np.ndarray
        Array of shape (n_train, L_train, d), training time‐series.
    X_test : np.ndarray
        Array of shape (n_test, L_test, d), test time‐series.
    n_levels : int, default=21
        Truncation level for the signature feature map.
    n_features : int, default=1000
        Number of components for both the static RFF and the TRP.

    Returns
    -------
    K_train : np.ndarray
        Gram matrix on the training set, shape (n_train, n_train).
    K_test : np.ndarray
        Cross‐Gram matrix between test and train, shape (n_test, n_train).
    """
        # Generate cache filename
    cache_filename = generate_cache_filename("KSigRFSFTRP", X_train.shape, X_test.shape,
                                          seed=seed, n_levels=n_levels, n_features=n_features)
    
    # Check if cache exists and try to load
    logger.info(f"Checking for KSig RFSF-TRP cache file: {cache_filename}")
    if os.path.exists(cache_filename):
        logger.info(f"Cache HIT: Loading cached KSig RFSF-TRP gram matrices from {cache_filename}...")
        try:
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
                K_train = cached_data['train_gram']
                K_test = cached_data['test_gram']
            
            # Validate dimensions
            if validate_gram_matrices(K_train, K_test, X_train, X_test):
                logger.info("Successfully loaded cached KSig RFSF-TRP gram matrices")
                logger.info(f"Train gram shape: {K_train.shape}")
                logger.info(f"Test gram shape: {K_test.shape}")
                
                # Print a small portion of the training gram matrix (from cache)
                print(f"KSig RFSF-TRP Training gram shape (cached): {K_train.shape}")
                print("Small portion of KSig RFSF-TRP training gram matrix (5x5) - from cache:")
                print(K_train[:5, :5])
                print(f"KSig RFSF-TRP Training gram min: {K_train.min():.6f}, max: {K_train.max():.6f}, mean: {K_train.mean():.6f}")
                
                return K_train, K_test
            else:
                logger.warning("Cached gram matrices have incorrect dimensions, recomputing...")
        except Exception as e:
            logger.warning(f"Failed to load cached gram matrices: {e}")
    else:
        logger.info(f"Cache MISS: No cached KSig RFSF-TRP gram matrices found at {cache_filename}")
    
    logger.info("Computing gram matrices using KSig RFSF-TRP...")
    start_time = time.time()
    
    try:
        # 1) Static Random Fourier Features
        static_feat = ksig.static.features.RandomFourierFeatures(
            n_components=n_features
        )
        # 2) Tensorized Random Projection for coupling tensor products
        proj = ksig.projections.TensorizedRandomProjection(
            n_components=n_features
        )
        # 3) Wrap into the RFSF-TRP signature feature map
        rfsf_trp = ksig.kernels.SignatureFeatures(
            n_levels=n_levels,
            static_features=static_feat,
            projection=proj
        )
        # 4) Fit feature map on training data
        rfsf_trp.fit(X_train)
        # 5) Compute train–train Gram matrix
        K_train = rfsf_trp(X_train)
        # 6) Compute test–train Gram matrix
        K_test = rfsf_trp(X_test, X_train)
        
        # Cache the results (ensure numpy arrays)
        try:
            with open(cache_filename, 'wb') as f:
                pickle.dump({
                    'train_gram': ensure_numpy_array(K_train),
                    'test_gram': ensure_numpy_array(K_test)
                }, f)
            logger.info(f"Cached KSig RFSF-TRP gram matrices to {cache_filename}")
            # Verify the file was actually created
            if os.path.exists(cache_filename):
                file_size = os.path.getsize(cache_filename)
                logger.info(f"Cache file created successfully: {cache_filename} (size: {file_size} bytes)")
            else:
                logger.warning(f"Cache file was not created: {cache_filename}")
        except Exception as e:
            logger.warning(f"Failed to cache gram matrices: {e}")
        
        # Print a small portion of the training gram matrix
        print(f"KSig RFSF-TRP Training gram shape: {K_train.shape}")
        print("Small portion of KSig RFSF-TRP training gram matrix (5x5):")
        print(K_train[:5, :5])
        print(f"KSig RFSF-TRP Training gram min: {K_train.min():.6f}, max: {K_train.max():.6f}, mean: {K_train.mean():.6f}")
        
        # Calculate and log condition number early
        try:
            # Convert to numpy array if it's a CuPy array
            K_train_np = ensure_numpy_array(K_train)
            condition_number = np.linalg.cond(K_train_np)
            logger.info(f"KSig RFSF-TRP train gram condition number: {condition_number:.2e}")
            if condition_number > 1e12:
                logger.warning(f"High condition number ({condition_number:.2e}) may cause training issues!")
            elif condition_number > 1e8:
                logger.warning(f"Moderately high condition number ({condition_number:.2e})")
        except Exception as e:
            logger.warning(f"Could not compute condition number: {e}")
        
        logger.info(f"KSig RFSF-TRP computation time: {time.time() - start_time:.3f}s")
        logger.info(f"Train gram shape: {K_train.shape}")
        logger.info(f"Test gram shape: {K_test.shape}")
        
        return K_train, K_test
        
    except (MemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "out_of_memory" in str(e).lower() or "oom" in str(e).lower() or isinstance(e, MemoryError):
            logger.error(f"KSig RFSF-TRP ran out of memory: {e}")
            logger.info("KSig RFSF-TRP computation skipped due to OOM")
            # Return None to indicate OOM failure
            return None, None
        else:
            logger.error(f"KSig RFSF-TRP computation failed with error: {e}")
            raise
    except Exception as e:
        logger.error(f"KSig RFSF-TRP computation failed with unexpected error: {e}")
        raise


def compute_gram_matrix_powersig_jax(X_train: np.ndarray, X_test: np.ndarray, kernel_type: KernelType = KERNEL_LINEAR, order: int = 8, normalize: bool = False, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gram matrices using PowerSigJax with local caching.
    
    Args:
        X_train: Training data
        X_test: Test data
        kernel_type: Type of kernel to use (linear or rbf)
        order: Order for PowerSigJax
        
    Returns:
        Tuple of (train_gram, test_gram) matrices
    """
    # Generate cache filename
    cache_filename = generate_cache_filename("PowerSigJax", kernel_type, X_train.shape, X_test.shape, seed=seed, order=order)
    
    # Check if cache exists and try to load
    logger.info(f"Checking for PowerSigJax cache file: {cache_filename}")
    if os.path.exists(cache_filename):
        logger.info(f"Cache HIT: Loading cached PowerSigJax gram matrices from {cache_filename}...")
        try:
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
                train_gram = cached_data['train_gram']
                test_gram = cached_data['test_gram']
            
            # Validate dimensions
            if validate_gram_matrices(train_gram, test_gram, X_train, X_test):
                logger.info("Successfully loaded cached PowerSigJax gram matrices")
                logger.info(f"Train gram shape: {train_gram.shape}")
                logger.info(f"Test gram shape: {test_gram.shape}")
                
                # Compute and log condition number
                try:
                    condition_number = np.linalg.cond(train_gram)
                    logger.info(f"PowerSigJax Training gram condition number (cached): {condition_number:.2e}")
                except Exception as e:
                    logger.warning(f"Failed to compute condition number: {e}")
                    condition_number = np.inf
                
                # Print a small portion of the training gram matrix (from cache)
                print(f"PowerSigJax Training gram shape (cached): {train_gram.shape}")
                print("Small portion of PowerSigJax training gram matrix (5x5) - from cache:")
                print(train_gram[:5, :5])
                print(f"PowerSigJax Training gram min: {train_gram.min():.6f}, max: {train_gram.max():.6f}, mean: {train_gram.mean():.6f}")
                
                return train_gram, test_gram
            else:
                logger.warning("Cached gram matrices have incorrect dimensions, recomputing...")
        except Exception as e:
            logger.warning(f"Failed to load cached gram matrices: {e}")
    else:
        logger.info(f"Cache MISS: No cached PowerSigJax gram matrices found at {cache_filename}")
    
    logger.info(f"Computing gram matrices using PowerSigJax (order={order}) with {kernel_type} kernel...")
    start_time = time.time()
    
    # Select JAX device: prefer CUDA device 1, then CUDA device 0, then default
    devices = jax.devices()
    cuda_devices = [d for d in devices if d.platform == 'gpu']
    
    if len(cuda_devices) >= 2:
        device = cuda_devices[1]  # CUDA device 1
        logger.info(f"Using CUDA device 1: {device}")
    elif len(cuda_devices) >= 1:
        device = cuda_devices[0]  # CUDA device 0
        logger.info(f"Using CUDA device 0: {device}")
    else:
        device = devices[0]  # Default device
        logger.info(f"Using default device: {device}")
    
    # Initialize PowerSigJax with selected device and kernel type
    if kernel_type == KERNEL_LINEAR:
        static_kernel = static_kernels.linear_kernel
    elif kernel_type == KERNEL_RBF:
        static_kernel = static_kernels.rbf_kernel
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    powersig = PowerSigJax(order=order, static_kernel=static_kernel, device=device)
    
    # Convert to JAX arrays on the selected device
    X_train_jax = jnp.array(X_train, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test, dtype=jnp.float64, device=device)
    
    # Compute gram matrices
    train_gram = powersig.compute_gram_matrix(X_train_jax, X_train_jax)
    test_gram = powersig.compute_gram_matrix(X_test_jax, X_train_jax)

    # Convert back to numpy
    train_gram = np.array(train_gram, dtype=np.float64)
    test_gram = np.array(test_gram, dtype=np.float64)
    
    # Add small epsilon * I to improve numerical stability
    epsilon = 1e-6  # Increased from 1e-8 for better stability
    n_train = train_gram.shape[0]
    train_gram += epsilon * np.eye(n_train)
    # Check for NaN values after computation but before normalization
    if np.any(np.isnan(train_gram)):
        raise ValueError("train_gram contains NaN values after computation!")
    if np.any(np.isnan(test_gram)):
        raise ValueError("test_gram contains NaN values after computation!")
    
    if normalize:
        train_gram = normalize_kernel_matrix(train_gram)
        # For test gram, we need to normalize using the training gram diagonal
        # test_gram has shape (n_test, n_train), so we can't use normalize_kernel_matrix directly
        # Instead, we'll normalize it manually using the training gram diagonal
        train_diag = np.diag(train_gram)
        test_gram = test_gram / np.sqrt(np.outer(np.ones(test_gram.shape[0]), train_diag))

        if np.any(np.isnan(train_gram)):
            raise ValueError("train_gram contains NaN values after normalization!")
        if np.any(np.isnan(test_gram)):
            raise ValueError("test_gram contains NaN values after normalization!")
    # Cache the results (ensure numpy arrays)
    try:
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'train_gram': ensure_numpy_array(train_gram),
                'test_gram': ensure_numpy_array(test_gram)
            }, f)
        logger.info(f"Cached PowerSigJax gram matrices to {cache_filename}")
        # Verify the file was actually created
        if os.path.exists(cache_filename):
            file_size = os.path.getsize(cache_filename)
            logger.info(f"Cache file created successfully: {cache_filename} (size: {file_size} bytes)")
        else:
            logger.warning(f"Cache file was not created: {cache_filename}")
    except Exception as e:
        logger.warning(f"Failed to cache gram matrices: {e}")
    
    # Print a small portion of the training gram matrix
    print(f"PowerSigJax Training gram shape: {train_gram.shape}")
    print("Small portion of PowerSigJax training gram matrix (5x5):")
    print(train_gram[:5, :5])
    print(f"PowerSigJax Training gram min: {train_gram.min():.6f}, max: {train_gram.max():.6f}, mean: {train_gram.mean():.6f}")
    
    # Calculate and log condition number early
    try:
        # Convert to numpy array if it's a CuPy array
        train_gram_np = ensure_numpy_array(train_gram)
        condition_number = cond(train_gram_np)
        logger.info(f"PowerSigJax train gram condition number: {condition_number:.2e}")
        if condition_number > 1e12:
            logger.warning(f"High condition number ({condition_number:.2e}) may cause training issues!")
        elif condition_number > 1e8:
            logger.warning(f"Moderately high condition number ({condition_number:.2e})")
    except Exception as e:
        logger.warning(f"Could not compute condition number: {e}")
    
    logger.info(f"PowerSigJax computation time: {time.time() - start_time:.3f}s")
    logger.info(f"Train gram shape: {train_gram.shape}")
    logger.info(f"Test gram shape: {test_gram.shape}")
    
    return train_gram, test_gram


# MLP function removed - not available for regression


def run_grid_search_with_gram_matrices(train_gram: np.ndarray, test_gram: np.ndarray, 
                                     y_train: np.ndarray, y_test: np.ndarray, 
                                     kernel_name: str, X_test: np.ndarray, X_train_means: np.ndarray, X_train_stds: np.ndarray, X_test_means: np.ndarray, X_test_stds: np.ndarray, gram_computation_time: float = -1.0) -> Dict[str, Any]:
    """
    Shared function to run grid search with precomputed gram matrices for regression.
    Trains separate SVR models for each output dimension.
    
    Args:
        train_gram: Training gram matrix
        test_gram: Test gram matrix
        y_train: Training targets
        y_test: Test targets
        kernel_name: Name of the kernel
        gram_computation_time: Time taken to compute gram matrices (-1.0 if loaded from cache)
        
    Returns:
        Dictionary containing results
    """
    start_time = time.time()
    
    # Convert cupy arrays to numpy arrays if needed
    if hasattr(train_gram, 'get'):  # cupy array
        train_gram_np = train_gram.get()
    else:
        train_gram_np = train_gram
        
    if hasattr(test_gram, 'get'):  # cupy array
        test_gram_np = test_gram.get()
    else:
        test_gram_np = test_gram
    
    # Check for NaN values
    if np.any(np.isnan(train_gram_np)):
        print(f"ERROR: train_gram_np contains NaN values in grid search!")
        print(f"train_gram_np shape: {train_gram_np.shape}")
        print(f"train_gram_np min: {np.nanmin(train_gram_np)}, max: {np.nanmax(train_gram_np)}")
    if np.any(np.isnan(test_gram_np)):
        print(f"ERROR: test_gram_np contains NaN values in grid search!")
    if np.any(np.isnan(y_train)):
        print(f"ERROR: y_train contains NaN values in grid search!")
    if np.any(np.isnan(y_test)):
        print(f"ERROR: y_test contains NaN values in grid search!")
    
    # Handle multi-dimensional targets
    if y_train.ndim == 2:
        num_dimensions = y_train.shape[1]
        y_pred_svr = np.zeros_like(y_test)
        
        # Train separate SVR model for each dimension
        for dim in range(num_dimensions):
            # Run SVR grid search for this dimension
            param_grid = {'C': C_GRID}
            svr = SVR(kernel='precomputed')
            
            # Create custom fold that uses all indices
            full_idx = np.arange(len(y_train))  # All samples in same fold
            cv = [(full_idx, full_idx)]
            
            grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring='neg_max_error', n_jobs=len(C_GRID))
            grid_search.fit(train_gram_np, y_train[:, dim])
            
            # Get best model and predict for this dimension
            best_svr = grid_search.best_estimator_
            y_pred_svr[:, dim] = best_svr.predict(test_gram_np)
    else:
        # Single-dimensional targets
        param_grid = {'C': C_GRID}
        svr = SVR(kernel='precomputed')
        
        # Create custom fold that uses all indices
        full_idx = np.arange(len(y_train))  # All samples in same fold
        cv = [(full_idx, full_idx)]
        
        grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring='neg_max_error', n_jobs=len(C_GRID))
        grid_search.fit(train_gram_np, y_train)
        
        # Get best model and predict
        best_svr = grid_search.best_estimator_
        y_pred_svr = best_svr.predict(test_gram_np)
    
    svr_time = time.time() - start_time
    
    # Calculate SVR metrics
    print(f"y_test: {y_test[:5]}")
    print(f"y_pred_svr: {y_pred_svr[:5]}")
    print(f"X_test[:, -1, :]: {X_test[:, -1, :][:5]}")
    print(f"y_test-X_test[:, -1, :]: {(y_test-X_test[:, -1, :])[:5]}")
    print(f"y_pred_svr-X_test[:, -1, :]: {(y_pred_svr-X_test[:, -1, :])[:5]}")
    svr_mse = (np.abs(y_test -y_pred_svr) ** 2).mean()
    svr_mape = np.mean(np.abs((y_test - y_pred_svr) / (y_test + 1e-8))) * 100  # MAPE in percentage
    svr_r2 = r2_score(y_test, y_pred_svr)
    
    # total_nodes = 19
    # # test = (X_test_stds*y_test) + X_test_means - X_test[:, -1, :]
    # test = np.arccos(y_pred_svr)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test A: {test[:5]}") 
    # test = np.arccos(y_test)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test B: {test[:5]}") 
    # # Compute the inverse chebychev transform of y_test and y_pred_svr 
    # # Clip inputs to valid domain for arccos [-1, 1]
    # y_test_input = np.clip((y_test) + X_test_means - X_test[:, -1,:], -1.0, 1.0)
    # y_pred_svr_input = np.clip((y_pred_svr) + X_test_means - X_test[:, -1,:], -1.0, 1.0)
    
    # y_test_inv = (np.arccos(y_test_input)*(2*total_nodes/np.pi)-1)/2
    # y_pred_svr_inv = (np.arccos(y_pred_svr_input)*(2*total_nodes/np.pi)-1)/2
    # # Print out the inverse Chebyshev transform values
    # logger.info("Inverse Chebyshev transform values:")
    # logger.info(f"Raw preimage: {y_test - (X_test[:, -1, :] * X_test_stds + X_test_means)}")
    # logger.info(f"y_test_inv shape: {y_test_inv.shape}")
    # logger.info(f"y_test_inv sample: {y_test_inv[:5]}")
    # logger.info(f"y_pred_svr_inv shape: {y_pred_svr_inv.shape}")
    # logger.info(f"y_pred_svr_inv sample: {y_pred_svr_inv[:5]}")
    
    # Calculate metrics on the inverse transformed values
    
    # inv_mse = (np.abs(y_test_inv - y_pred_svr_inv) ** 2).mean()
    # inv_mape = np.mean(np.abs((y_test_inv - y_pred_svr_inv) / (y_test_inv + 1e-8))) * 100  # MAPE in percentage
    # inv_r2 = r2_score(y_test_inv, y_pred_svr_inv)
    # inv_r2 = 1 - (np.abs(y_test_inv - y_pred_svr_inv)).sum() / np.abs((y_test_inv - y_test_inv.mean()).sum())
    
    # Print metrics for inverse transformed values
    # logger.info("Metrics on inverse Chebyshev transform:")
    # logger.info(f"Inverse MSE: {inv_mse:.6f}")
    # logger.info(f"Inverse MAPE: {inv_mape:.6f}%")
    # logger.info(f"Inverse R²: {inv_r2:.6f}")
    # Calculate condition number for the training gram matrix
    try:
        condition_number = np.linalg.cond(train_gram_np)
        logger.info(f"{kernel_name} Training gram condition number: {condition_number:.2e}")
        if condition_number > 1e12:
            logger.warning(f"High condition number ({condition_number:.2e}) may cause training issues!")
        elif condition_number > 1e8:
            logger.warning(f"Moderately high condition number ({condition_number:.2e})")
    except Exception as e:
        logger.warning(f"Could not compute condition number: {e}")
        condition_number = np.inf
    
    # Return SVR results
    results = {
        'kernel_name': kernel_name,
        'mse': svr_mse,
        'mape': svr_mape,
        'r2_score': svr_r2,
        'gram_computation_time': gram_computation_time,
        'grid_search_time': svr_time,
        'total_time': gram_computation_time + svr_time if gram_computation_time >= 0 else svr_time,
        'condition_number': condition_number,
        'best_C': grid_search.best_params_['C'] if y_train.ndim == 1 else 'multiple',  # Multiple C values for multi-dim
        'cv_scores': grid_search.cv_results_ if y_train.ndim == 1 else 'multiple',  # Multiple CV results for multi-dim
        'y_pred': y_pred_svr,
        'y_true': y_test
    }
    
    return results


def run_ksig_pde_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor, 
                         y_train: np.ndarray, y_test: np.ndarray, kernel_type: KernelType, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run KSigPDE computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram, gram_computation_time = compute_gram_matrix_ksig_pde(X_train, X_test, kernel_type, seed=seed)
    
    # Check if OOM occurred
    if train_gram is None or test_gram is None:
        logger.warning("KSigPDE computation failed due to OOM, returning default results")
        return {
            'kernel_name': f"KSigPDE_{kernel_type}",
            'train_gram': None,
            'test_gram': None,
            'gram_computation_time': -1.0,
            'y_train': np.array(y_train, dtype=np.float64),
            'y_test': np.array(y_test, dtype=np.float64),
            'error': 'OOM - computation skipped'
        }
    
    # Return gram matrices and labels for main process grid search
    return {
        'kernel_name': f"KSigPDE_{kernel_type}",
        'train_gram': preserve_complex_or_float64(train_gram),
        'test_gram': preserve_complex_or_float64(test_gram),
        'gram_computation_time': gram_computation_time,
        'y_train': np.array(y_train, dtype=np.float64),
        'y_test': np.array(y_test, dtype=np.float64),
        'error': 'OK'
    }


def run_ksig_rfsf_trp_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                              y_train: np.ndarray, y_test: np.ndarray, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run KSig RFSF-TRP computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram = compute_gram_matrix_ksig_rfsf_trp(X_train, X_test, n_levels=21, n_features=1000, seed=seed)
    
    # Check if OOM occurred
    if train_gram is None or test_gram is None:
        logger.warning("KSig RFSF-TRP computation failed due to OOM, returning default results")
        return {
            'kernel_name': "KSig RFSF-TRP",
            'train_gram': None,
            'test_gram': None,
            'y_train': np.array(y_train, dtype=np.float64),
            'y_test': np.array(y_test, dtype=np.float64),
            'error': 'OOM - computation skipped'
        }
    
    # Return gram matrices and labels for main process grid search
    return {
        'kernel_name': "KSig RFSF-TRP",
        'train_gram': preserve_complex_or_float64(train_gram),
        'test_gram': preserve_complex_or_float64(test_gram),
        'y_train': np.array(y_train, dtype=np.float64),
        'y_test': np.array(y_test, dtype=np.float64),
        'error': 'OK'
    }


def run_powersig_jax_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                            y_train: torch.Tensor, y_test: torch.Tensor, kernel_type: KernelType, order: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run PowerSigJax computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram = compute_gram_matrix_powersig_jax(X_train, X_test, kernel_type, order, seed=seed)
    
    
    return {
        'kernel_name': f"PowerSigJax_{kernel_type}",
        'train_gram': preserve_complex_or_float64(train_gram),
        'test_gram': preserve_complex_or_float64(test_gram),
        'y_train': np.array(y_train, dtype=np.float64),
        'y_test': np.array(y_test, dtype=np.float64),
        'error': 'OK'
    }


def run_cuml_baseline_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                             y_train: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Any]:
    """
    Run cuML SVR baseline with RBF and linear kernels in separate process.
    """
    logger.info("Running cuML baseline process")
    # Convert torch tensors to cupy arrays and reshape for cuML SVR
    X_train_np = X_train_tensor.numpy()  # Shape: (samples, timesteps, features)
    X_test_np = X_test_tensor.numpy()    # Shape: (samples, timesteps, features)
    
    # Reshape to 2D: (samples, timesteps * features)
    X_train_2d = X_train_np.reshape(X_train_np.shape[0], -1)
    X_test_2d = X_test_np.reshape(X_test_np.shape[0], -1)
    
    # Ensure all data is properly converted to CuPy arrays
    X_train_cp = cp.array(X_train_2d, dtype=cp.float32)
    X_test_cp = cp.array(X_test_2d, dtype=cp.float32)
    y_train_cp = cp.array(y_train, dtype=cp.float32)
    y_test_cp = cp.array(y_test, dtype=cp.float32)
    
    # Verify that all arrays are CuPy arrays and not PyTorch tensors
    assert isinstance(X_train_cp, cp.ndarray), f"X_train_cp is {type(X_train_cp)}, expected cp.ndarray"
    assert isinstance(X_test_cp, cp.ndarray), f"X_test_cp is {type(X_test_cp)}, expected cp.ndarray"
    assert isinstance(y_train_cp, cp.ndarray), f"y_train_cp is {type(y_train_cp)}, expected cp.ndarray"
    assert isinstance(y_test_cp, cp.ndarray), f"y_test_cp is {type(y_test_cp)}, expected cp.ndarray"
        
    if not CUML_AVAILABLE:
        return {
            'kernel_name': "cuML_Baseline",
            'mse': np.inf,
            'mape': np.inf,
            'r2_score': -np.inf,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'best_kernel': 'none',
            'error': 'cuML not available'
        }
    
    # Test both RBF and linear kernels with grid search over C values
    kernels = ['rbf', 'linear']
    best_score = np.inf  # Lower MSE is better
    best_results = None
    best_kernel = None
    best_C = None
    
    for kernel in kernels:
        print(f"Running cuML grid search with {kernel} kernel")
        kernel_best_score = np.inf
        kernel_best_C = None
        kernel_best_results = None
        
        for C in C_GRID:
            print(f"  Testing C={C}")
            try:
                # Handle multi-dimensional targets
                if y_train_cp.ndim == 2:
                    num_dimensions = y_train_cp.shape[1]
                    y_pred_cp = cp.zeros_like(y_test_cp)
                    
                    # Train separate SVR model for each dimension
                    for dim in range(num_dimensions):
                        print(f"      Training SVR for dimension {dim}")
                        svr = cuMLSVR(kernel=kernel, C=C, verbose=0)
                        # Ensure inputs are CuPy arrays
                        X_train_dim = cp.asarray(X_train_cp, dtype=cp.float32)
                        y_train_dim = cp.asarray(y_train_cp[:, dim], dtype=cp.float32)
                        X_test_dim = cp.asarray(X_test_cp, dtype=cp.float32)
                        
                        svr.fit(X_train_dim, y_train_dim)
                        y_pred_cp[:, dim] = svr.predict(X_test_dim)
                    
                    y_pred = cp.asnumpy(y_pred_cp)
                else:
                    # Single-dimensional targets
                    print(f"      Training SVR for single dimension")
                    svr = cuMLSVR(kernel=kernel, C=C, verbose=0)
                    # Ensure inputs are CuPy arrays
                    X_train_single = cp.asarray(X_train_cp, dtype=cp.float32)
                    y_train_single = cp.asarray(y_train_cp, dtype=cp.float32)
                    X_test_single = cp.asarray(X_test_cp, dtype=cp.float32)
                    
                    svr.fit(X_train_single, y_train_single)
                    y_pred_cp = svr.predict(X_test_single)
                    y_pred = cp.asnumpy(y_pred_cp)
                
                # Calculate MSE - ensure both arrays are numpy arrays
                y_test_np = np.array(y_test, dtype=np.float64)
                y_pred_np = np.array(y_pred, dtype=np.float64)
                mse = mean_squared_error(y_test_np, y_pred_np)
                print(f"    C={C} MSE: {mse:.4f}")
                
                # Update best for this kernel if this is better
                if mse < kernel_best_score:
                    kernel_best_score = mse
                    kernel_best_C = C
                    
                    # Calculate all metrics for best result for this kernel
                    mape = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + 1e-8))) * 100  # MAPE in percentage
                    r2 = r2_score(y_test_np, y_pred_np)
                    
                    kernel_best_results = {
                        'kernel_name': f"cuML_Baseline_{kernel}",
                        'mse': mse,
                        'mape': mape,
                        'r2_score': r2,
                        'condition_number': np.inf,  # Not applicable for direct data
                        'gram_computation_time': -1.0,  # Not applicable for direct data
                        'grid_search_time': 0.0,  # Not applicable for direct data
                        'total_time': 0.0,  # Not applicable for direct data
                        'best_kernel': kernel,
                        'best_C': C,
                        'y_pred': y_pred,
                        'y_true': y_test.cpu().numpy()
                    }
                    
            except Exception as e:
                print(f"    cuML SVR failed with {kernel} kernel, C={C}: {e}")
                continue
        
        # Update overall best if this kernel's best is better
        if kernel_best_score < best_score:
            best_score = kernel_best_score
            best_kernel = kernel
            best_C = kernel_best_C
            best_results = kernel_best_results
        
        print(f"  Best for {kernel} kernel: C={kernel_best_C}, MSE={kernel_best_score:.4f}")
    
    if best_results is None:
        return {
            'kernel_name': "cuML_Baseline",
            'mse': np.inf,
            'mape': np.inf,
            'r2_score': -np.inf,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'error': 'All kernels failed'
        }
    
    # Log which kernel was selected
    logger.info(f"cuML_Baseline selected {best_kernel} kernel with C={best_C} and MSE: {best_results['mse']:.4f}")
    
    return best_results


def run_knn_dtw_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                        y_train: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Any]:
    """
    Run cuML KNeighborsRegressor in separate process.
    """
    try:
        # Check if cuML is available
        if not CUML_AVAILABLE:
            return {
                'kernel_name': "KNN_DTW",
                'mse': np.inf,
                'mae': np.inf,
                'r2_score': -np.inf,
                'condition_number': np.inf,
                'gram_computation_time': -1.0,
                'grid_search_time': 0.0,
                'total_time': 0.0,
                'best_kernel': 'none',
                'error': 'cuML not available'
            }
        
        # Import cuML KNeighborsRegressor
        from cuml.neighbors import KNeighborsRegressor
        
        # Convert torch tensors to numpy arrays
        X_train_np = X_train_tensor.numpy()  # Shape: (samples, timesteps, features)
        X_test_np = X_test_tensor.numpy()    # Shape: (samples, timesteps, features)
        
        # Reshape to 2D: (samples, timesteps * features) for cuML compatibility
        X_train_2d = X_train_np.reshape(X_train_np.shape[0], -1)
        X_test_2d = X_test_np.reshape(X_test_np.shape[0], -1)
        
        # Convert to cuPy arrays
        X_train_cp = cp.array(X_train_2d, dtype=cp.float64)
        X_test_cp = cp.array(X_test_2d, dtype=cp.float64)
        y_train_cp = cp.array(y_train, dtype=cp.float64)
        
        # Create and fit KNN regressor
        start_time = time.time()
        knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
        knn.fit(X_train_cp, y_train_cp)
        
        # Predict
        y_pred_cp = knn.predict(X_test_cp)
        y_pred = cp.asnumpy(y_pred_cp)
        y_test = y_test.cpu().numpy()
        end_time = time.time()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100  # MAPE in percentage
        r2 = r2_score(y_test, y_pred)
        
        total_time = end_time - start_time
        
        logger.info(f"KNN_DTW (cuML) completed with MSE: {mse:.4f}, R²: {r2:.4f} in {total_time:.3f}s")
        
        return {
            'kernel_name': "KNN_DTW",
            'mse': mse,
            'mape': mape,
            'r2_score': r2,
            'condition_number': np.inf,  # Not applicable for KNN
            'gram_computation_time': -1.0,  # Not applicable for KNN
            'grid_search_time': 0.0,  # Not applicable for KNN
            'total_time': total_time,
            'best_kernel': 'euclidean',  # cuML uses Euclidean distance by default
            'y_pred': y_pred,
            'y_true': y_test,
            'error': 'OK'
        }
        
    except Exception as e:
        logger.error(f"KNN_DTW (cuML) failed: {e}")
        return {
            'kernel_name': "KNN_DTW",
            'mse': np.inf,
            'mape': np.inf,
            'r2_score': -np.inf,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'best_kernel': 'none',
            'error': str(e)
        }

def generate_base_dataset(history_length: int, dimensions: int, dataset_type: DatasetType, 
                         num_samples: int, max_timesteps: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a base dataset at maximum timesteps for a given seed.
    This dataset will be subsampled for different timestep experiments.
    
    Args:
        history_length: Length of each time series
        dimensions: Number of dimensions
        dataset_type: Type of dataset to generate (DatasetType enum)
        num_samples: Number of samples to generate
        max_timesteps: Maximum number of timesteps (full length)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) at maximum timesteps
    """
    torch.manual_seed(seed)
    logger.info(f"Generating base dataset for seed {seed} with {max_timesteps} timesteps...")
    
    # Generate train and test datasets at maximum length
    X_train, y_train = build_regression_dataset(dataset_type=dataset_type, history_length=history_length, 
                                               num_samples=num_samples, num_timesteps=max_timesteps, dimensions=dimensions)
    X_test, y_test = build_regression_dataset(dataset_type=dataset_type, history_length=history_length, 
                                             num_samples=num_samples, num_timesteps=max_timesteps, dimensions=dimensions)
    
    logger.info(f"Base dataset generated: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Plot samples once after dataset generation
    logger.info("Plotting regression samples for base dataset...")
    plot_regression_samples(X_train, y_train, num_samples=5, seed=seed)
    
    return X_train, y_train, X_test, y_test


def run_long_time_dependency_experiment_with_data(X_train: torch.Tensor, y_train: torch.Tensor, 
                                                X_test: torch.Tensor, y_test: torch.Tensor,
                                                num_timesteps: int, max_timesteps: int = None, 
                                                seed: int = 69420) -> Dict[str, Any]:
    """
    Run long time dependency regression experiments with pre-generated data.
    
    Args:
        X_train: Pre-generated training data at maximum timesteps
        y_train: Pre-generated training targets
        X_test: Pre-generated test data at maximum timesteps  
        y_test: Pre-generated test targets
        num_timesteps: Number of timesteps to subsample to
        max_timesteps: Maximum timesteps in the data (for subsampling)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing experiment results
    """
    if max_timesteps is None:
        max_timesteps = X_train.shape[1]  # Use actual data length
        
    
    logger.info(f"Starting long time dependency regression experiment with {num_timesteps} timesteps...")
    logger.info(f"Kernels to run: {KERNELS_TO_RUN}")
    logger.info(f"Available kernels: {set(KERNEL_NAMES.keys())}")
    logger.info(f"Skipped kernels: {set(KERNEL_NAMES.keys()) - KERNELS_TO_RUN}")
    
    # Apply subsampling based on configured method
    logger.info(f"SUBSAMPLE_METHOD: {SUBSAMPLE_METHOD}")
    logger.info(f"max_timesteps: {max_timesteps}")
    logger.info(f"num_timesteps: {num_timesteps}")
    logger.info(f"X_train shape before subsampling: {X_train.shape}")
    
    if SUBSAMPLE_METHOD == SubsampleMethod.SLIDING_WINDOW:
        # Use sliding window approach
        logger.info("Using sliding window subsampling approach")
        X_train, y_train = sliding_window_subsample(X_train, y_train, num_timesteps, stride=1)
        X_test, y_test = sliding_window_subsample(X_test, y_test, num_timesteps, stride=1)
        
        logger.info(f"X_train shape after sliding window: {X_train.shape}")
        logger.info(f"y_train shape after sliding window: {y_train.shape}")
    elif SUBSAMPLE_METHOD == SubsampleMethod.EQUALLY_SPACED and num_timesteps < max_timesteps:
        # Use equally spaced subsampling
        logger.info("Using equally spaced subsampling approach")
        X_train = subsample_timesteps(X_train, num_timesteps)
        X_test = subsample_timesteps(X_test, num_timesteps)
        
        logger.info(f"X_train shape after equally spaced: {X_train.shape}")
    elif SUBSAMPLE_METHOD == SubsampleMethod.EQUALLY_SPACED and num_timesteps >= max_timesteps:
        logger.info("Using full dataset - no subsampling needed")
    elif SUBSAMPLE_METHOD == SubsampleMethod.NONE:
        logger.info("No subsampling - using full dataset")
    else:
        raise ValueError(f"Unknown SUBSAMPLE_METHOD: {SUBSAMPLE_METHOD}. Must be one of: {[sm.value for sm in SubsampleMethod]}")
    
    # total_nodes = 19
    # test = y_test - X_test[:, -1, :]
    # test = torch.arccos(test)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test: {test[:5]}")
    # Print statistics for both training and test sets (before normalization)
    logger.info("Dataset statistics BEFORE normalization:")
    print_dataset_statistics(X_train, X_test)
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    # Apply z-score normalization per dimension
    X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, X_train_means, X_train_stds, X_test_means, X_test_stds = normalize_training_data(X_train, X_test, y_train, y_test)
    
    # Handle complex data normalization
    if torch.is_complex(X_train):
        # For complex data, compute standard deviation of magnitude
        sigma = torch.abs(X_train).std()
        logger.info(f"Using magnitude-based normalization for complex data, sigma: {sigma:.6f}")
    else:
        sigma = X_train.std()
        logger.info(f"Using standard normalization for real data, sigma: {sigma:.6f}")
    
    # X_train_normalized = X_train / sigma
    # X_test_normalized = X_test / sigma
    # X_train_normalized = X_train
    # X_test_normalized = X_test
    y_train_normalized = y_train
    y_test_normalized = y_test 
    
    # Create appropriate dtype for means and stds based on data type
    # if torch.is_complex(X_train):
    #     X_train_means = torch.zeros((X_train.shape[0], X_train.shape[2]), dtype=torch.complex128, device=X_train.device)
    #     X_test_means = torch.zeros((X_test.shape[0], X_test.shape[2]), dtype=torch.complex128, device=X_test.device)
    #     X_train_stds = torch.ones((X_train.shape[0], X_train.shape[2]), dtype=torch.complex128, device=X_train.device) * sigma
    #     X_test_stds = torch.ones((X_test.shape[0], X_test.shape[2]), dtype=torch.complex128, device=X_test.device) * sigma
    # else:
    #     X_train_means = torch.zeros((X_train.shape[0], X_train.shape[2]), dtype=torch.float64, device=X_train.device)
    #     X_test_means = torch.zeros((X_test.shape[0], X_test.shape[2]), dtype=torch.float64, device=X_test.device)
    #     X_train_stds = torch.ones((X_train.shape[0], X_train.shape[2]), dtype=torch.float64, device=X_train.device) * sigma
    #     X_test_stds = torch.ones((X_test.shape[0], X_test.shape[2]), dtype=torch.float64, device=X_test.device) * sigma

    # test = (y_test_normalized* X_test_stds)+X_test_means - X_test[:, -1, :] 
    # test = torch.arccos(test)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test: {test[:5]}")
    # X_train_normalized = X_train
    # X_test_normalized = X_test
    # y_train_normalized = y_train
    # y_test_normalized = y_test
    
    # Print statistics for both training and test sets (after normalization)
    # logger.info("Dataset statistics AFTER normalization:")
    # print_dataset_statistics(X_train_normalized, X_test_normalized)
    # test = (y_test_normalized*X_test_stds)+X_test_means - X_test[:, -1, :] 
    # test = torch.arccos(test)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test: {test[:5]}")
    # Wrap normalized data in torch tensors for multiprocessing
    X_train_tensor = X_train_normalized.share_memory_()
    X_test_tensor = X_test_normalized.share_memory_()
    
    # Initialize results storage
    all_results = {}
    
    # Define kernel functions mapping
    kernel_functions = {
        # "KNN_DTW": (run_knn_dtw_process, (X_train_tensor, X_test_tensor, y_train_normalized, y_test_normalized)),
        # "cuML_Baseline": (run_cuml_baseline_process, (X_train_tensor, X_test_tensor, y_train_normalized, y_test_normalized)),
        # "KSigPDE": (run_ksig_pde_process, (X_train_tensor, X_test_tensor, y_train_normalized, y_test_normalized, KERNEL_LINEAR, seed)),
        "KSig RFSF-TRP": (run_ksig_rfsf_trp_process, (X_train_tensor, X_test_tensor, y_train_normalized, y_test_normalized, seed)),
        "PowerSigJax": (run_powersig_jax_process, (X_train_tensor, X_test_tensor, y_train_normalized, y_test_normalized, KERNEL_LINEAR, 9, seed)),
    }
    # test = (y_test_normalized* X_test_stds)+X_test_means - X_test[:, -1, :] 
    # test = torch.arccos(test)
    # test = (test * (2*total_nodes/torch.pi)-1)/2
    # logging.info(f"Test: {test[:5]}")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2, maxtasksperchild=1) as pool:
        for kernel_name in kernel_functions:
            if kernel_name in KERNELS_TO_RUN:
                # if kernel_name == "PowerSigJax":
                #        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
                results = pool.apply(kernel_functions[kernel_name][0], kernel_functions[kernel_name][1])
                all_results[kernel_name] = results
                logger.info(f"Completed {kernel_name}")
    
    # Run grid search for all successful kernels
    logger.info("\n" + "="*80)
    logger.info("RUNNING GRID SEARCH FOR ALL KERNELS")
    logger.info("="*80)
    
    final_results = {}
    for kernel_name, kernel_data in all_results.items():
        # Check if this kernel uses gram matrices or works directly with data
        uses_gram_matrices = 'train_gram' in kernel_data and kernel_data['train_gram'] is not None
        
        if kernel_name in ["cuML_Baseline", "KNN_DTW"] or not uses_gram_matrices:
            # These kernels already return final results, no grid search needed
            final_results[kernel_name] = kernel_data
        elif kernel_data.get('error', 'OK') == 'OK' and uses_gram_matrices:
            logger.info(f"Running grid search for {kernel_name}...")
            # test = (y_test_normalized - X_test_normalized[:, -1, :]) * X_test_stds + X_test_means
            # test = torch.arccos(test)
            # test = (test * (2*total_nodes/torch.pi)-1)/2
            # logging.info(f"Test: {test[:5]}")
            grid_results = run_grid_search_with_gram_matrices(
                kernel_data['train_gram'], 
                kernel_data['test_gram'], 
                kernel_data['y_train'], 
                kernel_data['y_test'], 
                kernel_name,
                X_test.cpu().numpy(),
                None, #X_train_means.numpy(),
                None, #X_train_stds.numpy(),
                None, #X_test_means.numpy(),
                None, #X_test_stds.numpy(),
                kernel_data.get('gram_computation_time', -1.0)
            )
            final_results[kernel_name] = grid_results
        else:
            logger.warning(f"Skipping grid search for {kernel_name} due to OOM or error")
            final_results[kernel_name] = {
                'kernel_name': kernel_name,
                'mse': np.inf,
                'mape': np.inf,
                'r2_score': -np.inf,
                'condition_number': np.inf,
                'best_kernel': 'none',
                'error': kernel_data.get('error', 'Unknown error')
            }
   
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    # Print header
    logger.info(f"{'Kernel':<20} {'MSE':<10} {'MAPE':<10} {'R²':<10} {'Grid(s)':<10} {'Cond#':<12}")
    logger.info("-" * 75)
    
    for kernel_name, results in final_results.items():
        condition_number = results.get('condition_number', np.inf)
        cond_str = f"{condition_number:.2e}" if condition_number != np.inf else "inf"
        status = results.get('error', 'OK')
        
        if status != 'OK':
            logger.info(f"{kernel_name:<20} {'OOM':<10} {'OOM':<10} {'OOM':<10} {'OOM':<10} {'OOM':<12}")
        else:
            grid_time = results.get('grid_search_time', 0.0)
            logger.info(f"{kernel_name:<20} {results['mse']:<10.4f} {results['mape']:<10.4f} "
                       f"{results['r2_score']:<10.4f} "
                       f"{grid_time:<10.3f} {cond_str:<12}")
    
    logger.info("-" * 75)
    
    # Print detailed timing information
    logger.info("\n" + "="*80)
    logger.info("DETAILED TIMING INFORMATION")
    logger.info("="*80)
    
    for kernel_name, results in final_results.items():
        if results.get('error', 'OK') == 'OK':
            gram_time = results.get('gram_computation_time', -1.0)
            grid_time = results.get('grid_search_time', 0.0)
            total_time = results.get('total_time', 0.0)
            
            if gram_time >= 0:
                logger.info(f"{kernel_name:<20}: Gram computation: {gram_time:.3f}s, Grid search: {grid_time:.3f}s, Total: {total_time:.3f}s")
            else:
                logger.info(f"{kernel_name:<20}: Gram computation: CACHED, Grid search: {grid_time:.3f}s, Total: {total_time:.3f}s")
    
    logger.info("="*80)
    
    # SVR-only results summary
    logger.info("\n" + "="*80)
    logger.info("SVR RESULTS SUMMARY")
    logger.info("="*80)
    
    for kernel_name, results in final_results.items():
        if results.get('error', 'OK') == 'OK':
            logger.info(f"{kernel_name:<20}: MSE: {results['mse']:.4f}, MAPE: {results['mape']:.4f}, R²: {results['r2_score']:.4f}")
    
    logger.info("="*80)
    
    # Find best performing kernels for each metric
    if final_results:
        # Filter out kernels that failed due to OOM
        successful_results = {k: v for k, v in final_results.items() if v.get('error', 'OK') == 'OK'}
        
        if successful_results:
            metrics = ['mse', 'mape', 'r2_score']
            logger.info("\nBEST PERFORMING KERNELS BY METRIC:")
            logger.info("="*50)
            
            for metric in metrics:
                if metric == 'r2_score':
                    # Higher R² is better
                    best_kernel = max(successful_results.keys(), key=lambda k: successful_results[k][metric])
                    best_value = successful_results[best_kernel][metric]
                else:
                    # Lower MSE/MAE is better
                    best_kernel = min(successful_results.keys(), key=lambda k: successful_results[k][metric])
                    best_value = successful_results[best_kernel][metric]
                logger.info(f"{metric.upper():<12}: {best_kernel:<20} ({best_value:.4f})")
            
            # Find kernel with best condition number (lowest)
            best_cond_kernel = min(successful_results.keys(), key=lambda k: successful_results[k]['condition_number'])
            best_cond = successful_results[best_cond_kernel]['condition_number']
            cond_str = f"{best_cond:.2e}" if best_cond != np.inf else "inf"
            logger.info(f"{'CONDITION':<12}: {best_cond_kernel:<20} ({cond_str})")
        else:
            logger.warning("No kernels completed successfully due to OOM errors")
        
        # Report OOM failures
        oom_kernels = [k for k, v in all_results.items() if v.get('error', 'OK') != 'OK']
        if oom_kernels:
            logger.info(f"\nKERNELS THAT FAILED DUE TO OOM:")
            logger.info("="*40)
            for kernel in oom_kernels:
                logger.info(f"- {kernel}")
    
    logger.info("\nExperiment completed!")
    
    # Return final results for CSV generation
    return final_results


def run_timestep_experiments():
    """
    Run experiments with different timestep values and generate CSV results.
    """
    import csv
    import os
    
    # Hardcoded parameters for experiments
    history_length = 17 #1289
    dataset_type = DatasetType.CHEBYCHEV_FROM_IR
    
    # Define timestep ranges
    # timestep_ranges = [3000]
     
    timestep_ranges = list(range(13,19)) + list(range(32,36))
    
    # # Range 1: 989 to 1289 in steps of 100
    # for timesteps in range(989, 1290, 100):
    #     timestep_ranges.append(timesteps)
    
    # # Range 2: 1290 to 2600 in steps of 200
    # for timesteps in range(1290, 2691, 200):
    #     timestep_ranges.append(timesteps)
    
    # # Range 3: 4000 to 5000 in steps of 200
    # for timesteps in range(5000, 5401, 200):
    #     timestep_ranges.append(timesteps)
    
    # CSV file setup with history_length and dataset_type in filename
    csv_filename = f"recurrence_reg_hl{history_length}_{dataset_type.value}.csv"
    csv_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['kernel_name', 'num_timesteps', 'history_length', 'dataset_type', 'seed', 'mse', 'mape', 'condition_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist
        if not csv_exists:
            writer.writeheader()
        
        # Define seeds to test
        seeds = [42, 123, 456, 789, 1776, 1863, 1903, 1947, 1969, 2015]
        # seeds = [1903, 1969]
        
        # Find maximum timesteps to generate base dataset
        max_timesteps = max(timestep_ranges)+1
        logger.info(f"Maximum timesteps for base dataset: {max_timesteps}")
        
        # Run experiments for each seed (generate data once per seed)
        for seed in seeds:
            logger.info(f"\n{'='*80}")
            logger.info(f"GENERATING BASE DATASET FOR SEED {seed}")
            logger.info(f"{'='*80}")
            
            # Generate base dataset once per seed at maximum timesteps
            X_train_base, y_train_base, X_test_base, y_test_base = generate_base_dataset(
                history_length=history_length, 
                dimensions=2, 
                dataset_type=dataset_type, 
                num_samples=25, 
                max_timesteps=max_timesteps, 
                seed=seed
            )
            
            # Run experiments for each timestep value using the same base dataset
            for num_timesteps in timestep_ranges:
                logger.info(f"\n{'='*80}")
                logger.info(f"RUNNING EXPERIMENT WITH {num_timesteps} TIMESTEPS, SEED {seed}")
                logger.info(f"{'='*80}")
                
                try:
                    # Run main experiment with pre-generated data
                    results = run_long_time_dependency_experiment_with_data(
                        X_train=X_train_base[:,:num_timesteps, :], 
                        y_train=X_train_base[:,num_timesteps, :], 
                        X_test=X_test_base[:,:num_timesteps, :], 
                        y_test=X_test_base[:,num_timesteps, :],
                        num_timesteps=num_timesteps, 
                        max_timesteps=max_timesteps, 
                        seed=seed
                    )
                    
                    # Write results to CSV
                    for kernel_name, kernel_results in results.items():
                        if kernel_results.get('error', 'OK') == 'OK':
                            row = {
                                'kernel_name': kernel_name,
                                'num_timesteps': num_timesteps,
                                'history_length': history_length,
                                'dataset_type': dataset_type.value,
                                'seed': seed,
                                'mse': kernel_results.get('mse', np.inf),
                                'mape': kernel_results.get('mape', np.inf),
                                'condition_number': kernel_results.get('condition_number', np.inf)
                            }
                            writer.writerow(row)
                            logger.info(f"Added {kernel_name} results to CSV: MSE={row['mse']:.6f}, MAPE={row['mape']:.6f}, Cond={row['condition_number']:.2e}")
                        else:
                            # Write failed experiments with inf values
                            row = {
                                'kernel_name': kernel_name,
                                'num_timesteps': num_timesteps,
                                'history_length': history_length,
                                'dataset_type': dataset_type.value,
                                'seed': seed,
                                'mse': np.inf,
                                'mape': np.inf,
                                'condition_number': np.inf
                            }
                            writer.writerow(row)
                            logger.warning(f"Added failed {kernel_name} results to CSV")
                    
                    # Flush the file to ensure data is written
                    csvfile.flush()
                    
                except Exception as e:
                    logger.error(f"Experiment failed for {num_timesteps} timesteps, seed {seed}: {e}")
                    # Write error entries for each kernel that was supposed to run
                    for kernel_name in KERNELS_TO_RUN:
                        row = {
                            'kernel_name': kernel_name,
                            'num_timesteps': num_timesteps,
                            'history_length': history_length,
                            'dataset_type': dataset_type.value,
                            'seed': seed,
                            'mse': np.inf,
                            'mape': np.inf,
                            'condition_number': np.inf
                        }
                        writer.writerow(row)
    
    logger.info(f"\nAll experiments completed! Results saved to {csv_filename}")





if __name__ == "__main__":
    
    run_timestep_experiments()
