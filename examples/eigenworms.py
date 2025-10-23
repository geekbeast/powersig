"""
Eigenworms Classification using SVC with Custom Kernels

This script implements Eigenworms classification using Support Vector Classification (SVC)
with custom signature kernels: KSigPDE, KSig RFSF-TRP, and PowerSigJax.
"""
import os

from examples.large_window import build_dataset
import powersig
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import numpy as np
from numpy.linalg import cond
import jax.numpy as jnp
import cupy as cp
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

import time
import logging
from typing import Tuple, Dict, Any, Optional, Literal
import os
import pickle
from enum import Enum
import csv

# Import PowerSig modules
from powersig.jax.algorithm import PowerSigJax
from powersig.jax import static_kernels

# Import KSig modules
import ksig
from ksig.kernels import SignaturePDEKernel, SignatureKernel
from ksig.static.kernels import LinearKernel, RBFKernel

from examples.neural import create_k_layer_mlp, create_k_layer_mlp_classification, train_mlp_model_classification

# Import AEON for dataset
from aeon.datasets import load_classification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import cuML SVC, fallback to sklearn if not available
try:
    from cuml.svm import SVC as cuMLSVC
    CUML_AVAILABLE = True
    logger.info("cuML SVC available - will use for baseline kernel")
except ImportError:
    CUML_AVAILABLE = False

# Constants for quick experiments
WINDOW_SIZE = 200  # Window size for subsampling
SUBSAMPLING_STRATEGY = "equally_spaced"  # Options: "sliding" or "equally_spaced"
# NUM_WINDOWS = 10
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
C_GRID = [10 ** (i-3) for i in range(9)]  # [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


def generate_cache_filename(kernel_name: str, X_train_shape: Tuple[int, ...], X_test_shape: Tuple[int, ...], 
                          kernel_type: Optional[KernelType] = None, **kwargs) -> str:
    """
    Generate a cache filename based on kernel name, data shapes, and parameters.
    
    Args:
        kernel_name: Name of the kernel
        X_train_shape: Shape of training data
        X_test_shape: Shape of test data
        kernel_type: Type of kernel (linear or rbf), optional
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


def download_aeon_dataset(dataset_name: str = "EigenWorms", split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Download and load the AEON dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        split: Which split to load ('train' or 'test')
        
    Returns:
        Tuple of (X, y) where X is the time series data and y are the labels
    """
    logger.info(f"Downloading {dataset_name} dataset from AEON ({split} split)...")
    
    X, y = load_classification(dataset_name, split=split)
    logger.info("Dataset loaded successfully!")

    # AEON returns shape (samples, channels, timesteps); we need (samples, timesteps, channels)
    if X.ndim == 3:
        X = X.transpose(0, 2, 1)
        logger.info("Transposed dataset to (samples, timesteps, channels) format.")

    # Ensure labels are integers
    if y.dtype.kind not in {'i', 'u'}:
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info("Encoded string labels to integers.")
    
    # Print label encoder classes
    logger.info(f"Label encoder classes: {le.classes_}")
    
    # Print label encoder mapping
    for i, label in enumerate(le.classes_):
        logger.info(f"Label mapping: {label} -> {i}")

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"Number of classes: {len(np.unique(y))}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def plot_eigenworms_samples(X: np.ndarray, y: np.ndarray = None, num_samples_per_class: int = 3, file_suffix: str = ""):
    """
    Plot Eigenworms samples grouped by class.
    
    Args:
        X: Dataset with shape (samples, timesteps, dimensions)
        y: Labels with shape (samples,) - optional
        num_samples_per_class: Number of samples to plot per class
        file_suffix: Suffix to add to file names to distinguish different plots
    
    Note:
        This function automatically detects complex data and creates side-by-side plots
        with real parts on the left and imaginary parts on the right for complex data.
        For real data, it uses the original single-column plotting format.
    """
    logger.info(f"Plotting Eigenworms samples grouped by class...")
    
    # Validate input data
    if X is None or X.size == 0:
        logger.error("Input data X is None or empty!")
        return
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Input data contains NaN or Inf values!")
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create time axis
    timesteps = X.shape[1]
    time_axis = np.linspace(0, timesteps - 1, timesteps)
    num_dimensions = X.shape[2]
    
    # Check if data is complex
    is_complex = np.iscomplexobj(X)
    logger.info(f"Data type: {'Complex' if is_complex else 'Real'}")
    
    if y is None:
        # Fallback to original behavior if no labels provided
        logger.info("No labels provided, plotting first few samples...")
        for sample_idx in range(min(3, X.shape[0])):
            # Get the sample data
            sample_data = X[sample_idx]  # Shape: (timesteps, dimensions)    
            
            if is_complex:
                # For complex data, create side-by-side plots (real on left, imaginary on right)
                fig, axes = plt.subplots(num_dimensions, 2, figsize=(20, 3*num_dimensions))
                if num_dimensions == 1:
                    axes = axes.reshape(1, -1)
                
                # Plot each dimension
                for dim_idx in range(num_dimensions):
                    dimension_data = sample_data[:, dim_idx]
                    real_data = np.real(dimension_data)
                    imag_data = np.imag(dimension_data)
                    
                    logger.info(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1}: shape={dimension_data.shape}")
                    logger.info(f"  Real part range: [{np.min(real_data):.6f}, {np.max(real_data):.6f}]")
                    logger.info(f"  Imaginary part range: [{np.min(imag_data):.6f}, {np.max(imag_data):.6f}]")
                    
                    # Check if data is valid
                    if np.any(np.isfinite(real_data)) and np.any(np.isfinite(imag_data)):
                        # Plot real part (left)
                        axes[dim_idx, 0].plot(time_axis, real_data, linewidth=1.5, color='blue')
                        axes[dim_idx, 0].set_title(f'Sample {sample_idx + 1}, Dimension {dim_idx + 1} - Real Part')
                        axes[dim_idx, 0].set_xlabel('Time Step')
                        axes[dim_idx, 0].set_ylabel('Real Value')
                        axes[dim_idx, 0].grid(True, alpha=0.3)
                        
                        # Plot imaginary part (right)
                        axes[dim_idx, 1].plot(time_axis, imag_data, linewidth=1.5, color='red')
                        axes[dim_idx, 1].set_title(f'Sample {sample_idx + 1}, Dimension {dim_idx + 1} - Imaginary Part')
                        axes[dim_idx, 1].set_xlabel('Time Step')
                        axes[dim_idx, 1].set_ylabel('Imaginary Value')
                        axes[dim_idx, 1].grid(True, alpha=0.3)
                    else:
                        logger.warning(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1} contains invalid data (NaN/Inf)")
            else:
                # For real data, use original plotting logic
                fig, axes = plt.subplots(num_dimensions, 1, figsize=(12, 3*num_dimensions))
                if num_dimensions == 1:
                    axes = [axes]
                
                # Plot each dimension
                for dim_idx in range(num_dimensions):
                    dimension_data = sample_data[:, dim_idx]
                    logger.info(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1}: shape={dimension_data.shape}, range=[{np.min(dimension_data):.6f}, {np.max(dimension_data):.6f}]")
                    
                    # Check if data is valid
                    if np.any(np.isfinite(dimension_data)):
                        axes[dim_idx].plot(time_axis, dimension_data, linewidth=1.5)
                        axes[dim_idx].set_title(f'Sample {sample_idx + 1}, Dimension {dim_idx + 1}')
                        axes[dim_idx].set_xlabel('Time Step')
                        axes[dim_idx].set_ylabel('Value')
                        axes[dim_idx].grid(True, alpha=0.3)
                    else:
                        logger.warning(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1} contains invalid data (NaN/Inf)")
            
            plt.tight_layout()
            suffix_part = f"_{file_suffix}" if file_suffix else ""
            plt.savefig(os.path.join(plots_dir, f'eigenworms_sample_{sample_idx + 1}{suffix_part}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot for sample {sample_idx + 1}")
        return
    
    # Get unique classes
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    logger.info(f"Found {num_classes} classes: {unique_classes}")
    
    # Create 5 plots, each corresponding to a specific class
    for plot_idx in range(min(5, num_classes)):
        class_label = unique_classes[plot_idx]
        
        if is_complex:
            # For complex data, create side-by-side plots (real on left, imaginary on right)
            fig, axes = plt.subplots(num_dimensions, 2, figsize=(25, 3*num_dimensions))
            if num_dimensions == 1:
                axes = axes.reshape(1, -1)
        else:
            # For real data, use original plotting logic
            fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3*num_dimensions))
            if num_dimensions == 1:
                axes = [axes]
        
        # Get samples for this specific class
        class_indices = np.where(y == class_label)[0]
        # Sample a few examples from this class
        num_samples_to_plot = min(num_samples_per_class, len(class_indices))
        selected_indices = np.random.choice(class_indices, num_samples_to_plot, replace=False)
        
        # Use different colors for different samples within the class
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples_to_plot))
        
        for dim_idx in range(num_dimensions):
            for sample_idx, color in zip(selected_indices, colors):
                dimension_data = X[sample_idx, :, dim_idx]
                
                if is_complex:
                    real_data = np.real(dimension_data)
                    imag_data = np.imag(dimension_data)
                    logger.info(f"Class {class_label}, Sample {sample_idx + 1}, Dimension {dim_idx + 1}: shape={dimension_data.shape}")
                    logger.info(f"  Real part range: [{np.min(real_data):.6f}, {np.max(real_data):.6f}]")
                    logger.info(f"  Imaginary part range: [{np.min(imag_data):.6f}, {np.max(imag_data):.6f}]")
                    
                    # Check if data is valid
                    if np.any(np.isfinite(real_data)) and np.any(np.isfinite(imag_data)):
                        # Plot real part (left)
                        axes[dim_idx, 0].plot(time_axis, real_data, 
                                             color=color, alpha=0.8, linewidth=1.5,
                                             label=f'Sample {sample_idx + 1}')
                        # Plot imaginary part (right)
                        axes[dim_idx, 1].plot(time_axis, imag_data, 
                                             color=color, alpha=0.8, linewidth=1.5,
                                             label=f'Sample {sample_idx + 1}')
                    else:
                        logger.warning(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1} contains invalid data (NaN/Inf)")
                else:
                    logger.info(f"Class {class_label}, Sample {sample_idx + 1}, Dimension {dim_idx + 1}: shape={dimension_data.shape}, range=[{np.min(dimension_data):.6f}, {np.max(dimension_data):.6f}]")
                    
                    # Check if data is valid
                    if np.any(np.isfinite(dimension_data)):
                        axes[dim_idx].plot(time_axis, dimension_data, 
                                         color=color, alpha=0.8, linewidth=1.5,
                                         label=f'Sample {sample_idx + 1}')
                    else:
                        logger.warning(f"Sample {sample_idx + 1}, Dimension {dim_idx + 1} contains invalid data (NaN/Inf)")
            
            if is_complex:
                # Set titles and labels for complex data (side-by-side)
                axes[dim_idx, 0].set_title(f'Class {class_label} - Dimension {dim_idx + 1} - Real Part')
                axes[dim_idx, 0].set_xlabel('Time Step')
                axes[dim_idx, 0].set_ylabel('Real Value')
                axes[dim_idx, 0].grid(True, alpha=0.3)
                axes[dim_idx, 0].legend()
                
                axes[dim_idx, 1].set_title(f'Class {class_label} - Dimension {dim_idx + 1} - Imaginary Part')
                axes[dim_idx, 1].set_xlabel('Time Step')
                axes[dim_idx, 1].set_ylabel('Imaginary Value')
                axes[dim_idx, 1].grid(True, alpha=0.3)
                axes[dim_idx, 1].legend()
            else:
                # Set titles and labels for real data (original format)
                axes[dim_idx].set_title(f'Class {class_label} - Dimension {dim_idx + 1}')
                axes[dim_idx].set_xlabel('Time Step')
                axes[dim_idx].set_ylabel('Value')
                axes[dim_idx].grid(True, alpha=0.3)
                axes[dim_idx].legend()
        
        plt.tight_layout()
        suffix_part = f"_{file_suffix}" if file_suffix else ""
        plt.savefig(os.path.join(plots_dir, f'eigenworms_plot_class_{class_label}{suffix_part}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot for class {class_label}")


def normalize_training_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize training data by subtracting mean and dividing by std dev for each dimension.
    Normalize test data using its own means and std dev for each dimension.
    
    Args:
        X_train: Training dataset with shape (samples, timesteps, dimensions)
        X_test: Test dataset with shape (samples, timesteps, dimensions)
        
    Returns:
        Tuple of (normalized_X_train, normalized_X_test)
    """
    logger.info("Normalizing training and test data...")
    
    num_train_samples, num_timesteps, num_dimensions = X_train.shape
    num_test_samples = X_test.shape[0]
    
    # Calculate mean and std for each dimension across all training samples and timesteps using broadcasting
    # Reshape to (num_dimensions,) for broadcasting
    train_means = np.mean(X_train, axis=(0, 1))  # Shape: (num_dimensions,)
    train_stds = 100 * np.std(X_train, axis=(0, 1))  # Shape: (num_dimensions,)
    
    # Avoid division by zero
    train_stds = np.where(train_stds == 0, 1.0, train_stds)
    zero_std_dims = np.where(train_stds == 1.0)[0]
    for dim_idx in zero_std_dims:
        logger.warning(f"Training dimension {dim_idx + 1} has zero standard deviation, setting to 1.0")
    
    logger.info("Training set normalization statistics (used for both train and test):")
    for dim_idx in range(num_dimensions):
        logger.info(f"  Dimension {dim_idx + 1}: mean={train_means[dim_idx]:.6f}, std={train_stds[dim_idx]:.6f}")
    
    # Normalize training data using training statistics with broadcasting
    # Reshape means and stds to (1, 1, num_dimensions) for broadcasting with (num_samples, num_timesteps, num_dimensions)
    train_means_broadcast = train_means.reshape(1, 1, -1)  # Shape: (1, 1, num_dimensions)
    train_stds_broadcast = train_stds.reshape(1, 1, -1)    # Shape: (1, 1, num_dimensions)
    normalized_X_train = (X_train - train_means_broadcast) / train_stds_broadcast
    
    # Normalize test data using training statistics with broadcasting (standard practice to avoid data leakage)
    normalized_X_test = (X_test - train_means_broadcast) / train_stds_broadcast
    
    logger.info("Normalization completed!")
    logger.info(f"Normalized training set shape: {normalized_X_train.shape}")
    logger.info(f"Normalized test set shape: {normalized_X_test.shape}")
    
    return normalized_X_train, normalized_X_test


def sliding_window_subsample(X: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply sliding window subsampling to create multiple samples from each original sample.
    
    Args:
        X: Input data of shape (n_samples, n_timesteps, n_features)
        y: Labels of shape (n_samples,)
        window_size: Size of the sliding window
        
    Returns:
        Tuple of (X_subsampled, y_subsampled) where X_subsampled has more samples
        but each sample has window_size timesteps
    """
    logger.info(f"Applying sliding window subsampling with window_size={window_size}")
    
    n_samples, n_timesteps, n_features = X.shape
    
    if n_timesteps <= window_size:
        logger.warning(f"Timesteps ({n_timesteps}) <= window_size ({window_size}), returning original data")
        return X, y
    
    # Calculate number of windows per sample
    n_windows_per_sample = n_timesteps - window_size + 1
    total_new_samples = n_samples * n_windows_per_sample
    
    logger.info(f"Creating {n_windows_per_sample} windows per sample, total new samples: {total_new_samples}")
    
    # Initialize output arrays
    X_subsampled = np.zeros((total_new_samples, window_size, n_features), dtype=X.dtype)
    y_subsampled = np.zeros(total_new_samples, dtype=y.dtype)
    
    # Apply sliding window to each sample
    new_sample_idx = 0
    for sample_idx in range(n_samples):
        for window_start in range(n_windows_per_sample):
            window_end = window_start + window_size
            
            # Extract window
            X_subsampled[new_sample_idx] = X[sample_idx, window_start:window_end, :]
            y_subsampled[new_sample_idx] = y[sample_idx]
            
            new_sample_idx += 1
    
    logger.info(f"Sliding window subsampling completed: {X.shape} -> {X_subsampled.shape}")
    logger.info(f"Labels shape: {y.shape} -> {y_subsampled.shape}")
    
    return X_subsampled, y_subsampled


def equally_spaced_subsample(X: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply equally spaced subsampling to reduce timesteps to window_size.
    
    Args:
        X: Input data of shape (n_samples, n_timesteps, n_features)
        y: Labels of shape (n_samples,)
        window_size: Target number of timesteps
        
    Returns:
        Tuple of (X_subsampled, y_subsampled) where X_subsampled has window_size timesteps
    """
    logger.info(f"Applying equally spaced subsampling with window_size={window_size}")
    
    n_samples, n_timesteps, n_features = X.shape
    
    if n_timesteps <= window_size:
        logger.warning(f"Timesteps ({n_timesteps}) <= window_size ({window_size}), returning original data")
        return X, y
    
    # Create equally spaced indices
    indices = np.linspace(0, n_timesteps - 1, window_size, dtype=int)
    
    logger.info(f"Selecting timesteps: {indices}")
    
    # Subsample using the indices
    X_subsampled = X[:, indices, :]
    
    logger.info(f"Equally spaced subsampling completed: {X.shape} -> {X_subsampled.shape}")
    logger.info(f"Labels shape unchanged: {y.shape}")
    
    return X_subsampled, y


def time_augment(X, scale: float = 1.0):
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
        time_feature = np.linspace(0, 1, length)*scale
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


def print_dataset_statistics(X_train: np.ndarray, X_test: np.ndarray):
    """
    Print statistics for each dimension across training and test sets.
    
    Args:
        X_train: Training dataset with shape (samples, timesteps, dimensions)
        X_test: Test dataset with shape (samples, timesteps, dimensions)
    """
    logger.info("Dataset statistics:")
    logger.info("=" * 50)
    
    num_train_samples, num_timesteps, num_dimensions = X_train.shape
    num_test_samples = X_test.shape[0]
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
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
        
        test_dimension_data = test_dimension_data.numpy() if isinstance(test_dimension_data, torch.Tensor) else test_dimension_data
        train_dimension_data = train_dimension_data.numpy() if isinstance(train_dimension_data, torch.Tensor) else train_dimension_data

        # Training set statistics
        train_min_val = np.min(train_dimension_data)
        train_max_val = np.max(train_dimension_data)
        train_mean_val = np.mean(train_dimension_data)
        train_std_val = np.std(train_dimension_data)
        
        # Test set statistics
        test_min_val = np.min(test_dimension_data)
        test_max_val = np.max(test_dimension_data)
        test_mean_val = np.mean(test_dimension_data)
        test_std_val = np.std(test_dimension_data)
        
        logger.info(f"Dimension {dim_idx + 1}:")
        logger.info(f"  Training Set:")
        logger.info(f"    Min: {train_min_val:.6f}")
        logger.info(f"    Max: {train_max_val:.6f}")
        logger.info(f"    Mean: {train_mean_val:.6f}")
        logger.info(f"    Std Dev: {train_std_val:.6f}")
        logger.info(f"  Test Set:")
        logger.info(f"    Min: {test_min_val:.6f}")
        logger.info(f"    Max: {test_max_val:.6f}")
        logger.info(f"    Mean: {test_mean_val:.6f}")
        logger.info(f"    Std Dev: {test_std_val:.6f}")
        logger.info("-" * 30)


def compute_gram_matrix_ksig_pde(X_train: np.ndarray, X_test: np.ndarray, kernel_type: KernelType = KERNEL_LINEAR) -> Tuple[np.ndarray, np.ndarray, float]:
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
    cache_filename = generate_cache_filename("KSigPDE", kernel_type, X_train.shape, X_test.shape)
    
    # Check if cache exists and try to load
    logger.info(f"Checking for KSigPDE cache file: {cache_filename}")
    if os.path.exists(cache_filename):
        logger.info(f"Cache HIT: Loading cached KSigPDE gram matrices from {cache_filename}...")
    else:
        logger.info(f"Cache MISS: No cached KSigPDE gram matrices found at {cache_filename}")
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
            condition_number = np.linalg.cond(train_gram)
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


def compute_gram_matrix_ksig_rfsf_trp(X_train: np.ndarray, X_test: np.ndarray, 
                                      n_levels: int = 21, n_features: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the train–train and test–train kernel matrices using
    Random Fourier Signature Features with Tensorized Random Projection (RFSF‑TRP) with local caching.

    Parameters
    ----------Pl
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
                                          n_levels=n_levels, n_features=n_features)
    
    # Check if cache exists and try to load
    logger.info(f"Checking for KSig RFSF-TRP cache file: {cache_filename}")
    if os.path.exists(cache_filename):
        logger.info(f"Cache HIT: Loading cached KSig RFSF-TRP gram matrices from {cache_filename}...")
    else:
        logger.info(f"Cache MISS: No cached KSig RFSF-TRP gram matrices found at {cache_filename}")
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
            condition_number = np.linalg.cond(ensure_numpy_array(K_train))
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


def compute_gram_matrix_powersig_jax(X_train: np.ndarray, X_test: np.ndarray, kernel_type: KernelType = KERNEL_LINEAR, order: int = 8) -> Tuple[np.ndarray, np.ndarray]:
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
    cache_filename = generate_cache_filename("PowerSigJax", kernel_type, X_train.shape, X_test.shape, order=order)
    
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
    train_gram = np.array(train_gram)
    test_gram = np.array(test_gram)
    
    # Add small epsilon * I to improve numerical stability
    epsilon = 1e-6  # Increased from 1e-8 for better stability
    n_train = train_gram.shape[0]
    train_gram += epsilon * np.eye(n_train)
    
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
        condition_number = cond(train_gram)
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




def run_mlp_with_gram_matrices(train_gram: np.ndarray, test_gram: np.ndarray, 
                              y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Run MLP training and prediction using gram matrices for multi-class classification.
    
    Args:
        train_gram: Training gram matrix
        test_gram: Test gram matrix  
        y_train: Training labels (integer)
        y_test: Test labels (integer)
        
    Returns:
        Dictionary containing results
    """
    start_time = time.time()
    
    # Get number of unique classes
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_classes)
    
    # Create label encoder to ensure sequential integer labels (0, 1, 2, ...)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to PyTorch tensors
    train_gram_tensor = torch.tensor(ensure_numpy_array(train_gram), dtype=torch.float32)
    test_gram_tensor = torch.tensor(ensure_numpy_array(test_gram), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    
    # Create MLP model for classification
    input_size = train_gram.shape[0]
    model = create_k_layer_mlp_classification(n=input_size, k=3, num_classes=num_classes)
    
    # Train the model using the classification training function
    trained_model, losses = train_mlp_model_classification(
        model=model,
        kernel=train_gram_tensor,
        y_train=y_train_tensor,
        epochs=100,
        lr=0.5,
        optimizer_type='lbfgs'
    )
    
    # Compute predictions on test set
    trained_model.eval()
    with torch.no_grad():
        test_outputs = trained_model(test_gram_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(test_outputs, dim=1)
        # Get predicted class indices
        y_pred_encoded = torch.argmax(probabilities, dim=1).numpy()
    
    # Convert back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    mlp_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Print MLP accuracy
    logger.info(f"MLP accuracy: {accuracy:.4f}")
    
    # Calculate condition number
    try:
        condition_number = np.linalg.cond(train_gram)
    except Exception as e:
        condition_number = np.inf
    
    results = {
        'kernel_name': 'MLP',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gram_computation_time': -1.0,  # Not applicable for MLP
        'grid_search_time': mlp_time,
        'total_time': mlp_time,
        'condition_number': condition_number,
        'y_pred': y_pred,
        'y_true': y_test,
        'final_loss': losses[-1] if losses else 0.0,
        'training_losses': losses
    }
    
    return results


def run_grid_search_with_gram_matrices(train_gram: np.ndarray, test_gram: np.ndarray, 
                                     y_train: np.ndarray, y_test: np.ndarray, 
                                     kernel_name: str, gram_computation_time: float = -1.0,
                                     run_mlp: bool = False) -> Dict[str, Any]:
    """
    Shared function to run grid search with precomputed gram matrices.
    Runs MLP first (if enabled), then SVC, returns the better performing one.
    
    Args:
        train_gram: Training gram matrix
        test_gram: Test gram matrix
        y_train: Training labels
        y_test: Test labels
        kernel_name: Name of the kernel
        gram_computation_time: Time taken to compute gram matrices (-1.0 if loaded from cache)
        run_mlp: Whether to run MLP training (default: False)
        
    Returns:
        Dictionary containing results
    """
    start_time = time.time()
    
    # Run MLP first (if enabled)
    mlp_results = None
    mlp_time = 0.0
    if run_mlp:
        mlp_start_time = time.time()
        mlp_results = run_mlp_with_gram_matrices(train_gram, test_gram, y_train, y_test)
        mlp_time = time.time() - mlp_start_time
    
    # Run SVC grid search
    param_grid = {'C': C_GRID, 'gamma': [10**(i-3) for i in range(0, 5)]}
    svc = SVC(kernel='precomputed', random_state=42)
    
    # Convert cupy arrays to numpy arrays if needed
    if hasattr(train_gram, 'get'):  # cupy array
        train_gram_np = train_gram.get()
    else:
        train_gram_np = train_gram
        
    if hasattr(test_gram, 'get'):  # cupy array
        test_gram_np = test_gram.get()
    else:
        test_gram_np = test_gram
    
    # Create custom fold that uses all indices
    full_idx = np.arange(len(y_train))  # All samples in same fold
    cv = [(full_idx, full_idx)]
    
    grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=len(C_GRID))
    grid_search.fit(train_gram_np, y_train)
    
    # Get best model and predict
    best_svc = grid_search.best_estimator_
    y_pred_svc = best_svc.predict(test_gram_np)
    
    svc_time = time.time() - start_time
    
    # Calculate SVC metrics
    svc_accuracy = accuracy_score(y_test, y_pred_svc)
    svc_precision = precision_score(y_test, y_pred_svc, average='weighted', zero_division=0)
    svc_recall = recall_score(y_test, y_pred_svc, average='weighted', zero_division=0)
    svc_f1 = f1_score(y_test, y_pred_svc, average='weighted', zero_division=0)
    
    # Calculate condition number for the training gram matrix
    try:
        condition_number = np.linalg.cond(train_gram_np)
    except Exception as e:
        condition_number = np.inf
    
    # Compare performances and return the better one (if MLP was run)
    if run_mlp and mlp_results['accuracy'] > svc_accuracy:
        # MLP performed better, return MLP results with updated timing
        mlp_results['grid_search_time'] = mlp_time + svc_time
        mlp_results['total_time'] = gram_computation_time + mlp_time + svc_time if gram_computation_time >= 0 else mlp_time + svc_time
        mlp_results['svc_accuracy'] = svc_accuracy  # Store SVC results for comparison
        mlp_results['mlp_vs_svc'] = 'MLP_WIN'
        mlp_results['condition_number'] = condition_number  # Always include condition number
        return mlp_results
    else:
        # SVC performed better (or MLP was disabled), return SVC results
        results = {
            'kernel_name': kernel_name,
            'accuracy': svc_accuracy,
            'precision': svc_precision,
            'recall': svc_recall,
            'f1_score': svc_f1,
            'gram_computation_time': gram_computation_time,
            'grid_search_time': mlp_time + svc_time,
            'total_time': gram_computation_time + mlp_time + svc_time if gram_computation_time >= 0 else mlp_time + svc_time,
            'condition_number': condition_number,
            'best_C': grid_search.best_params_['C'],
            'best_gamma': grid_search.best_params_['gamma'],
            'cv_scores': grid_search.cv_results_,
            'y_pred': y_pred_svc,
            'y_true': y_test,
            'mlp_vs_svc': 'SVC_WIN' if run_mlp else 'SVC_ONLY'
        }
        
        # Store MLP results for comparison if MLP was run
        if run_mlp:
            results['mlp_accuracy'] = mlp_results['accuracy']
        
        return results


def run_ksig_pde_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor, 
                         y_train: np.ndarray, y_test: np.ndarray, kernel_type: KernelType) -> Dict[str, Any]:
    """
    Run KSigPDE computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram, gram_computation_time = compute_gram_matrix_ksig_pde(X_train, X_test, kernel_type)
    
    # Check if OOM occurred
    if train_gram is None or test_gram is None:
        logger.warning("KSigPDE computation failed due to OOM, returning default results")
        return {
            'kernel_name': f"KSigPDE_{kernel_type}",
            'train_gram': None,
            'test_gram': None,
            'gram_computation_time': -1.0,
            'y_train': y_train,
            'y_test': y_test,
            'error': 'OOM - computation skipped'
        }
    
    # Return gram matrices and labels for main process grid search
    return {
        'kernel_name': f"KSigPDE_{kernel_type}",
        'train_gram': train_gram,
        'test_gram': test_gram,
        'gram_computation_time': gram_computation_time,
        'y_train': y_train,
        'y_test': y_test,
        'error': 'OK'
    }


def run_ksig_rfsf_trp_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                              y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Run KSig RFSF-TRP computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram = compute_gram_matrix_ksig_rfsf_trp(X_train, X_test, n_levels=21, n_features=1000)
    
    # Check if OOM occurred
    if train_gram is None or test_gram is None:
        logger.warning("KSig RFSF-TRP computation failed due to OOM, returning default results")
        return {
            'kernel_name': "KSig RFSF-TRP",
            'train_gram': None,
            'test_gram': None,
            'y_train': y_train,
            'y_test': y_test,
            'error': 'OOM - computation skipped'
        }
    
    # Return gram matrices and labels for main process grid search
    return {
        'kernel_name': "KSig RFSF-TRP",
        'train_gram': train_gram,
        'test_gram': test_gram,
        'y_train': y_train,
        'y_test': y_test,
        'error': 'OK'
    }


def run_powersig_jax_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                            y_train: np.ndarray, y_test: np.ndarray, kernel_type: KernelType, order: int) -> Dict[str, Any]:
    """
    Run PowerSigJax computation in separate process.
    """
    # Convert torch tensors to numpy arrays
    X_train = X_train_tensor.numpy()
    X_test = X_test_tensor.numpy()
    
    # Compute gram matrices
    train_gram, test_gram = compute_gram_matrix_powersig_jax(X_train, X_test, kernel_type, order)
    
    # Return gram matrices and labels for main process grid search
    return {
        'kernel_name': f"PowerSigJax_{kernel_type}",
        'train_gram': train_gram,
        'test_gram': test_gram,
        'y_train': y_train,
        'y_test': y_test,
        'error': 'OK'
    }


def run_cuml_baseline_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                             y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Run cuML SVC baseline with RBF and linear kernels in separate process.
    """
    # Convert torch tensors to cupy arrays and reshape for cuML SVC
    X_train_np = X_train_tensor.numpy()  # Shape: (samples, timesteps, features)
    X_test_np = X_test_tensor.numpy()    # Shape: (samples, timesteps, features)
    
    # Reshape to 2D: (samples, timesteps * features)
    X_train_2d = X_train_np.reshape(X_train_np.shape[0], -1)
    X_test_2d = X_test_np.reshape(X_test_np.shape[0], -1)
    
    data_dtype = cp.float64
    if X_train_tensor.dtype == torch.complex128:
       data_dtype = cp.complex128

    X_train_cp = cp.array(X_train_2d, dtype=data_dtype)
    X_test_cp = cp.array(X_test_2d, dtype=data_dtype)
    y_train_cp = cp.array(y_train, dtype=cp.int32)
    y_test_cp = cp.array(y_test, dtype=cp.int32)
        
    if not CUML_AVAILABLE:
        return {
            'kernel_name': "cuML_Baseline",
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'best_kernel': 'none',
            'error': 'cuML not available'
        }
    
    # Test both RBF and linear kernels with grid search over C and gamma values
    kernels = ['rbf', 'linear']
    best_score = -1
    best_results = None
    best_kernel = None
    best_C = None
    best_gamma = None
    
    for kernel in kernels:
        print(f"Running cuML grid search with {kernel} kernel")
        kernel_best_score = -1
        kernel_best_C = None
        kernel_best_gamma = None
        kernel_best_results = None
        
        # Define gamma grid (only for RBF kernel)
        gamma_grid = [10**(i-3) for i in range(0, 5)] if kernel == 'rbf' else ['scale']
        
        for C in C_GRID:
            for gamma in gamma_grid:
                print(f"  Testing C={C}, gamma={gamma}")
                try:
                    # Create cuML SVC with current kernel, C, and gamma values
                    svc = cuMLSVC(kernel=kernel, C=C, gamma=gamma, random_state=42, verbose=0)
                    svc.fit(X_train_cp, y_train_cp)
                    
                    # Predict
                    y_pred_cp = svc.predict(X_test_cp)
                    y_pred = cp.asnumpy(y_pred_cp)
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"    C={C}, gamma={gamma} accuracy: {accuracy:.4f}")
                    
                    # Update best for this kernel if this is better
                    if accuracy > kernel_best_score:
                        kernel_best_score = accuracy
                        kernel_best_C = C
                        kernel_best_gamma = gamma
                        
                        # Calculate all metrics for best result for this kernel
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        kernel_best_results = {
                            'kernel_name': f"cuML_Baseline_{kernel}",
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'condition_number': np.inf,  # Not applicable for direct data
                            'gram_computation_time': -1.0,  # Not applicable for direct data
                            'grid_search_time': 0.0,  # Not applicable for direct data
                            'total_time': 0.0,  # Not applicable for direct data
                            'best_kernel': kernel,
                            'best_C': C,
                            'best_gamma': gamma,
                            'y_pred': y_pred,
                            'y_true': y_test
                        }
                        
                except Exception as e:
                    print(f"    cuML SVC failed with {kernel} kernel, C={C}, gamma={gamma}: {e}")
                    continue
        
        # Update overall best if this kernel's best is better
        if kernel_best_score > best_score:
            best_score = kernel_best_score
            best_kernel = kernel
            best_C = kernel_best_C
            best_gamma = kernel_best_gamma
            best_results = kernel_best_results
        
        print(f"  Best for {kernel} kernel: C={kernel_best_C}, gamma={kernel_best_gamma}, accuracy={kernel_best_score:.4f}")
    
    if best_results is None:
        return {
            'kernel_name': "cuML_Baseline",
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'error': 'All kernels failed'
        }
    
    # Log which kernel was selected
    logger.info(f"cuML_Baseline selected {best_kernel} kernel with C={best_C}, gamma={best_gamma} and accuracy: {best_results['accuracy']:.4f}")
    
    return best_results


def run_knn_dtw_process(X_train_tensor: torch.Tensor, X_test_tensor: torch.Tensor,
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Run cuML KNeighborsClassifier in separate process.
    """
    try:
        # Check if cuML is available
        if not CUML_AVAILABLE:
            return {
                'kernel_name': "KNN_DTW",
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'condition_number': np.inf,
                'gram_computation_time': -1.0,
                'grid_search_time': 0.0,
                'total_time': 0.0,
                'best_kernel': 'none',
                'error': 'cuML not available'
            }
        
        # Import cuML KNeighborsClassifier
        from cuml.neighbors import KNeighborsClassifier
        
        # Convert torch tensors to numpy arrays
        X_train_np = X_train_tensor.numpy()  # Shape: (samples, timesteps, features)
        X_test_np = X_test_tensor.numpy()    # Shape: (samples, timesteps, features)
        
        # Reshape to 2D: (samples, timesteps * features) for cuML compatibility
        X_train_2d = X_train_np.reshape(X_train_np.shape[0], -1)
        X_test_2d = X_test_np.reshape(X_test_np.shape[0], -1)
        
        data_dtype = cp.float64
        
        if X_train_tensor.dtype == torch.complex128:
            data_dtype = cp.complex128

        # Convert to cuPy arrays
        X_train_cp = cp.array(X_train_2d, dtype=data_dtype)
        X_test_cp = cp.array(X_test_2d, dtype=data_dtype)
        y_train_cp = cp.array(y_train, dtype=cp.int32)
        
        # Create and fit KNN classifier
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        knn.fit(X_train_cp, y_train_cp)
        
        # Predict
        y_pred_cp = knn.predict(X_test_cp)
        y_pred = cp.asnumpy(y_pred_cp)
        end_time = time.time()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        total_time = end_time - start_time
        
        logger.info(f"KNN_DTW (cuML) completed with accuracy: {accuracy:.4f} in {total_time:.3f}s")
        
        return {
            'kernel_name': "KNN_DTW",
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
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
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'condition_number': np.inf,
            'gram_computation_time': -1.0,
            'grid_search_time': 0.0,
            'total_time': 0.0,
            'best_kernel': 'none',
            'error': str(e)
        }


def prepare_eigenworms_data():
    """
    Download, normalize, and prepare Eigenworms dataset for experiments.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) - raw data ready for processing
    """
    logger.info("Preparing Eigenworms dataset...")
    
    # 1. Download and load training dataset
    try:
        X_train, y_train = download_aeon_dataset("EigenWorms", split="train")
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        raise
    
    # 2. Download and load test dataset
    try:
        X_test, y_test = download_aeon_dataset("EigenWorms", split="test")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        raise
    
    # 3. Plot original samples and print statistics
    plot_eigenworms_samples(X_train, y_train, num_samples_per_class=3, file_suffix="original")
    
    # 4. Print statistics for both training and test sets (before normalization)
    logger.info("Dataset statistics BEFORE normalization:")
    print_dataset_statistics(X_train, X_test)
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    logger.info("Eigenworms dataset preparation completed!")
    return X_train, X_test, y_train, y_test


def run_experiment(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                   subsampling_strategy: str, window_size: int) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters.
    
    Args:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
        subsampling_strategy: "sliding" or "equally_spaced"
        window_size: Window size for subsampling
        
    Returns:
        Dictionary containing results for all kernels
    """
    logger.info(f"Running experiment: strategy={subsampling_strategy}, window_size={window_size}")
    
    # 1. Normalize training data
    X_train_normalized, X_test_normalized = normalize_training_data(X_train, X_test)
    
    # 2. Apply subsampling based on strategy
    logger.info(f"Applying subsampling strategy: {subsampling_strategy}")
    if subsampling_strategy == "sliding":
        # Sliding window applied before time augmentation
        logger.info("Applying sliding window subsampling before time augmentation...")
        X_train_normalized, y_train = sliding_window_subsample(X_train_normalized, y_train, window_size)
        X_test_normalized, y_test = sliding_window_subsample(X_test_normalized, y_test, window_size)
        logger.info(f"After sliding window - Train: {X_train_normalized.shape}, Test: {X_test_normalized.shape}")
        logger.info(f"Labels - Train: {y_train.shape}, Test: {y_test.shape}")
    elif subsampling_strategy == "equally_spaced":
        logger.info("Equally spaced subsampling will be applied after time augmentation")
    else:
        raise ValueError(f"Unknown subsampling strategy: {subsampling_strategy}")
    
    # 3. Apply time augmentation
    X_train_normalized, X_test_normalized = time_augment(X_train_normalized), time_augment(X_test_normalized)
    
    # 4. Apply equally spaced subsampling after time augmentation (if strategy is equally_spaced)
    if subsampling_strategy == "equally_spaced":
        logger.info("Applying equally spaced subsampling after time augmentation...")
        X_train_normalized, y_train = equally_spaced_subsample(X_train_normalized, y_train, window_size)
        X_test_normalized, y_test = equally_spaced_subsample(X_test_normalized, y_test, window_size)
        logger.info(f"After equally spaced subsampling - Train: {X_train_normalized.shape}, Test: {X_test_normalized.shape}")
        logger.info(f"Labels - Train: {y_train.shape}, Test: {y_test.shape}")
    
    # 5. Wrap normalized data in torch tensors for multiprocessing
    X_train_tensor = torch.from_numpy(X_train_normalized).share_memory_()
    X_test_tensor = torch.from_numpy(X_test_normalized).share_memory_()
    
    # 6. Initialize results storage
    all_results = {}
    
    # 7. Define kernel functions mapping
    kernel_functions = {
        "KNN_DTW": (run_knn_dtw_process, (X_train_tensor, X_test_tensor, y_train, y_test)),
        "cuML_Baseline": (run_cuml_baseline_process, (X_train_tensor, X_test_tensor, y_train, y_test)),
        "KSigPDE": (run_ksig_pde_process, (X_train_tensor, X_test_tensor, y_train, y_test, KERNEL_RBF)),
        "KSig RFSF-TRP": (run_ksig_rfsf_trp_process, (X_train_tensor, X_test_tensor, y_train, y_test)),
        "PowerSigJax": (run_powersig_jax_process, (X_train_tensor, X_test_tensor, y_train, y_test, KERNEL_RBF, 9)),
    }
    
    # 8. Run kernels using multiprocessing
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2, maxtasksperchild=1) as pool:
        for kernel_name in kernel_functions:
            if kernel_name in KERNELS_TO_RUN:
                if kernel_name == "PowerSigJax":
                    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
                results = pool.apply(kernel_functions[kernel_name][0], kernel_functions[kernel_name][1])
                all_results[kernel_name] = results
                logger.info(f"Completed {kernel_name}")
    
    # 9. Run grid search for all successful kernels
    logger.info(f"\nRunning grid search for window_size={window_size}, strategy={subsampling_strategy}")
    
    final_results = {}
    for kernel_name, kernel_data in all_results.items():
        # Check if this kernel uses gram matrices or works directly with data
        uses_gram_matrices = 'train_gram' in kernel_data and kernel_data['train_gram'] is not None
        
        if kernel_name in ["cuML_Baseline", "KNN_DTW"] or not uses_gram_matrices:
            # These kernels already return final results, no grid search needed
            final_results[kernel_name] = kernel_data
        elif kernel_data.get('error', 'OK') == 'OK' and uses_gram_matrices:
            logger.info(f"Running grid search for {kernel_name}...")
            grid_results = run_grid_search_with_gram_matrices(
                kernel_data['train_gram'], 
                kernel_data['test_gram'], 
                kernel_data['y_train'], 
                kernel_data['y_test'], 
                kernel_name,
                kernel_data.get('gram_computation_time', -1.0)
            )
            final_results[kernel_name] = grid_results
        else:
            logger.warning(f"Skipping grid search for {kernel_name} due to OOM or error")
            final_results[kernel_name] = {
                'kernel_name': kernel_name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'condition_number': np.inf,
                'best_kernel': 'none',
                'error': kernel_data.get('error', 'Unknown error')
            }
    
    return final_results


def write_results_to_csv(results: Dict[str, Any], window_size: int, subsampling_strategy: str, 
                         csv_filename: str = "eigenworms_results.csv"):
    """
    Write experiment results to CSV file.
    
    Args:
        results: Dictionary containing results for all kernels
        window_size: Window size used in the experiment
        subsampling_strategy: Subsampling strategy used
        csv_filename: Name of the CSV file to write to
    """
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            'kernel_name', 'static_kernel', 'window_size', 'subsampling_strategy',
            'accuracy', 'precision', 'recall', 'f1_score', 'condition_number',
            'gram_computation_time', 'grid_search_time', 'total_time', 'best_C', 'best_gamma', 'error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write results for each kernel
        for kernel_name, kernel_results in results.items():
            # Determine static kernel type
            static_kernel = 'none'
            if 'KSigPDE' in kernel_name:
                static_kernel = 'rbf'  # Default for KSigPDE
            elif 'PowerSigJax' in kernel_name:
                static_kernel = 'rbf'  # Default for PowerSigJax
            elif kernel_name == 'cuML_Baseline':
                static_kernel = kernel_results.get('best_kernel', 'none')
            
            row = {
                'kernel_name': kernel_name,
                'static_kernel': static_kernel,
                'window_size': window_size,
                'subsampling_strategy': subsampling_strategy,
                'accuracy': kernel_results.get('accuracy', 0.0),
                'precision': kernel_results.get('precision', 0.0),
                'recall': kernel_results.get('recall', 0.0),
                'f1_score': kernel_results.get('f1_score', 0.0),
                'condition_number': kernel_results.get('condition_number', np.inf),
                'gram_computation_time': kernel_results.get('gram_computation_time', -1.0),
                'grid_search_time': kernel_results.get('grid_search_time', 0.0),
                'total_time': kernel_results.get('total_time', 0.0),
                'best_C': kernel_results.get('best_C', 'N/A'),
                'best_gamma': kernel_results.get('best_gamma', 'N/A'),
                'error': kernel_results.get('error', 'OK')
            }
            writer.writerow(row)
    
    logger.info(f"Results written to {csv_filename}")


def main():
    """
    Main function to run Eigenworms classification experiments with window size sweep.
    """
    logger.info("Starting Eigenworms classification experiment with window size sweep...")
    logger.info(f"Kernels to run: {KERNELS_TO_RUN}")
    
    # 1. Prepare dataset once
    X_train, X_test, y_train, y_test = prepare_eigenworms_data()
    
    # 2. Define window sizes (powers of 2 from 16 to 4096)
    window_sizes = [2**i for i in range(13, 15)]  # 16, 32, 64, ..., 4096
    logger.info(f"Window sizes to test: {window_sizes}")
    
    # 3. Run experiments for each window size
    csv_filename = "eigenworms_results.csv"
    
    # Clear existing CSV file
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        logger.info(f"Removed existing {csv_filename}")
    
    for window_size in window_sizes:
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING EXPERIMENT: window_size={window_size}")
        logger.info(f"{'='*80}")
        
        try:
            # Run experiment with equally_spaced strategy
            results = run_experiment(X_train, X_test, y_train, y_test, 
                                   subsampling_strategy="equally_spaced", 
                                   window_size=window_size)
            
            # Write results to CSV
            write_results_to_csv(results, window_size, "equally_spaced", csv_filename)
            
            # Print summary for this window size
            logger.info(f"\nResults for window_size={window_size}:")
            for kernel_name, kernel_results in results.items():
                if kernel_results.get('error', 'OK') == 'OK':
                    logger.info(f"  {kernel_name}: accuracy={kernel_results['accuracy']:.4f}")
                else:
                    logger.info(f"  {kernel_name}: ERROR - {kernel_results.get('error', 'Unknown')}")
                    
        except Exception as e:
            logger.error(f"Experiment failed for window_size={window_size}: {e}")
            # Write error results to CSV
            error_results = {
                'ERROR': {
                    'kernel_name': 'ERROR',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'condition_number': np.inf,
                    'gram_computation_time': -1.0,
                    'grid_search_time': 0.0,
                    'total_time': 0.0,
                    'error': str(e)
                }
            }
            write_results_to_csv(error_results, window_size, "equally_spaced", csv_filename)
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPLETED!")
    logger.info(f"Results saved to: {csv_filename}")
    logger.info("You can now plot accuracy, precision, recall, F1 vs window_size")
    logger.info("with curves for different kernel, static kernel, and subsampling strategy combinations")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

# Example plotting code for the CSV results:
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('eigenworms_results.csv')

# Plot accuracy vs window_size for different kernels
plt.figure(figsize=(12, 8))
for kernel in df['kernel_name'].unique():
    kernel_data = df[df['kernel_name'] == kernel]
    plt.plot(kernel_data['window_size'], kernel_data['accuracy'], 
             marker='o', label=kernel, linewidth=2)

plt.xlabel('Window Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Window Size for Different Kernels')
plt.xscale('log', base=2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_vs_window_size.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot precision, recall, F1 vs window_size
metrics = ['precision', 'recall', 'f1_score']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    for kernel in df['kernel_name'].unique():
        kernel_data = df[df['kernel_name'] == kernel]
        axes[i].plot(kernel_data['window_size'], kernel_data[metric], 
                     marker='o', label=kernel, linewidth=2)
    
    axes[i].set_xlabel('Window Size')
    axes[i].set_ylabel(metric.title())
    axes[i].set_title(f'{metric.title()} vs Window Size')
    axes[i].set_xscale('log', base=2)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_vs_window_size.png', dpi=300, bbox_inches='tight')
plt.show()
"""
