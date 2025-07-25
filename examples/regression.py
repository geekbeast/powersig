import jax
import jax.numpy as jnp
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from powersig.jax.algorithm import PowerSigJax
from aeon.datasets import load_regression
from typing import Tuple, Optional, Dict
import time

def load_aeon_regression_dataset(
    dataset_name: str,
    split: str = "train",
    device: Optional[jax.Device] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.Device]:
    """
    Load an AEON regression dataset and convert it to JAX arrays.
    
    Args:
        dataset_name: Name of the AEON regression dataset to load
        split: Which split to load ('train' or 'test')
        device: Optional JAX device to place the arrays on. If None, uses default device.
        
    Returns:
        Tuple of (X, y, device) where:
            X: JAX array of shape (n_samples, n_timesteps, n_features)
            y: JAX array of shape (n_samples,) containing regression targets
    """
    # Get the CUDA device if available
    if device is None:
        try:
            device = jax.devices("cuda")[0]
            print(f"Using CUDA device: {device}")
        except (ValueError, RuntimeError):
            print("CUDA device not available, using default device")
            device = jax.devices()[0]
    
    # Load the dataset using AEON's regression loader
    X, y = load_regression(dataset_name, split=split)
    
    # Convert to JAX arrays
    X_jax = jnp.array(X, dtype=jnp.float64, device=device)
    y_jax = jnp.array(y, dtype=jnp.float64, device=device)
    
    # Move to specified device if provided
    if device is not None:
        X_jax = jax.device_put(X_jax, device)
        y_jax = jax.device_put(y_jax, device)
    
    return jnp.transpose(X_jax, (0, 2, 1)), y_jax, device

def time_augment(X: jnp.ndarray) -> jnp.ndarray:
    """
    Augment the input array by adding a time feature as the last dimension.
    The time feature is a linspace from 0 to 1 for each sample.
    Args:
        X: jnp.ndarray of shape (n_samples, length, dimension)
    Returns:
        jnp.ndarray of shape (n_samples, length, dimension + 1)
    """
    n_samples, length, _ = X.shape
    time_feature = jnp.linspace(0, 1, length)
    time_feature = jnp.broadcast_to(time_feature, (n_samples, length))
    time_feature = time_feature[..., None]  # shape (n_samples, length, 1)
    return jnp.concatenate([time_feature, X], axis=-1)

def main():
    # Set up device
    devices = jax.devices("cuda")
    device = devices[0] if devices else jax.devices()[0]
    print(f"Using device: {device}")
    
    # Load a large aeon regression dataset
    # Note: We'll use a classification dataset as regression target since aeon has limited regression datasets
    # In practice, you would use actual regression datasets
    dataset_name = "StandWalkJump"  # Using classification dataset as regression target (2500 timesteps)
    print(f"Loading {dataset_name} dataset for regression...")
    
    # Load training data
    X_train, y_train, device = load_aeon_regression_dataset(dataset_name, split="train", device=device)
    X_test, y_test, _ = load_aeon_regression_dataset(dataset_name, split="test", device=device)
    
    print(f"Original X_train.shape: {X_train.shape}")
    print(f"Original y_train.shape: {y_train.shape}")
    print(f"Original X_test.shape: {X_test.shape}")
    print(f"Original y_test.shape: {y_test.shape}")
    
    # Check if time series length is at least 1000
    if X_train.shape[1] < 1000:
        print(f"Warning: Time series length is only {X_train.shape[1]} timesteps, which is less than 1000")
        print("Proceeding anyway, but consider using a longer time series for better results")
    else:
        print(f"✓ Time series length is {X_train.shape[1]} timesteps (>= 1000)")
    
    # Normalize the data
    X_train_max = jnp.max(X_train)
    X_train /= X_train_max
    X_test /= X_train_max
    
    # Apply time augmentation
    X_train = time_augment(X_train)
    X_test = time_augment(X_test)
    
    print(f"After preprocessing:")
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train range: [{jnp.min(y_train):.3f}, {jnp.max(y_train):.3f}]")
    print(f"y_test range: [{jnp.min(y_test):.3f}, {jnp.max(y_test):.3f}]")
    
    # Convert targets to numpy for sklearn
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    
    # Initialize PowerSigJax with polynomial order 16
    print("\nInitializing PowerSigJax...")
    powersig = PowerSigJax(order=16, device=device)
    
    # Compute gram matrices
    print("Computing training gram matrix...")
    start_time = time.time()
    train_gram = powersig.compute_gram_matrix(X_train, X_train)
    train_time = time.time() - start_time
    print(f"Training gram matrix computed in {train_time:.2f} seconds")
    print(f"Training gram matrix shape: {train_gram.shape}")
    
    print("Computing test gram matrix...")
    start_time = time.time()
    test_gram = powersig.compute_gram_matrix(X_test, X_train)
    test_time = time.time() - start_time
    print(f"Test gram matrix computed in {test_time:.2f} seconds")
    print(f"Test gram matrix shape: {test_gram.shape}")
    
    # Convert to numpy for sklearn
    train_gram_np = np.array(train_gram)
    test_gram_np = np.array(test_gram)
    
    # Print gram matrix statistics
    print(f"\nGram matrix statistics:")
    print(f"Training gram matrix - min: {np.min(train_gram_np):.6f}, max: {np.max(train_gram_np):.6f}, mean: {np.mean(train_gram_np):.6f}")
    print(f"Test gram matrix - min: {np.min(test_gram_np):.6f}, max: {np.max(test_gram_np):.6f}, mean: {np.mean(test_gram_np):.6f}")
    
    # Train Kernel Ridge Regression
    print("\nTraining Kernel Ridge Regression...")
    
    # Try different alpha values for regularization
    alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alpha_values:
        print(f"Training with alpha = {alpha}...")
        krr = KernelRidge(alpha=alpha, kernel='precomputed')
        krr.fit(train_gram_np, y_train_np)
        
        # Predict on training set for validation
        y_pred_train = krr.predict(train_gram_np)
        r2 = r2_score(y_train_np, y_pred_train)
        
        print(f"  Training R² score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} (R² = {best_score:.4f})")
    
    # Train final model with best alpha
    print("Training final model...")
    final_krr = KernelRidge(alpha=best_alpha, kernel='precomputed')
    final_krr.fit(train_gram_np, y_train_np)
    
    # Predict on test set
    print("Making predictions on test set...")
    y_pred_test = final_krr.predict(test_gram_np)
    
    # Evaluate performance
    mse = mean_squared_error(y_test_np, y_pred_test)
    mae = mean_absolute_error(y_test_np, y_pred_test)
    r2_test = r2_score(y_test_np, y_pred_test)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2_test:.4f}")
    
    # Print some predictions vs actual values
    print(f"\nSample predictions (first 10):")
    print("Actual\tPredicted")
    for i in range(min(10, len(y_test_np))):
        print(f"{y_test_np[i]:.3f}\t{y_pred_test[i]:.3f}")
    
    # Print model coefficients summary
    print(f"\nModel Summary:")
    print(f"Number of dual coefficients: {len(final_krr.dual_coef_)}")
    print(f"Intercept: {final_krr.intercept_:.6f}")
    print(f"Alpha used: {best_alpha}")
    print(f"Polynomial order: 16")
    print(f"Total computation time: {train_time + test_time:.2f} seconds")

if __name__ == "__main__":
    main()
