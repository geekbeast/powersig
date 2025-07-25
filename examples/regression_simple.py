#!/usr/bin/env python3
"""
Simple regression example using PowerSig JAX with aeon datasets.
This example loads an aeon time series dataset with time series length >= 1000,
computes the gram matrix using PowerSig JAX, and performs kernel ridge regression.
"""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aeon.datasets import load_regression
import time

# Import PowerSigJax
try:
    from powersig.jax.algorithm import PowerSigJax
except ImportError:
    print("Error: Could not import PowerSigJax. Make sure powersig is installed.")
    exit(1)

def load_aeon_regression_dataset_simple(dataset_name="EigenWorms", split="train"):
    """
    Load an AEON regression dataset and convert to numpy arrays.
    
    Args:
        dataset_name: Name of the AEON regression dataset to load
        split: Which split to load ('train' or 'test')
        
    Returns:
        Tuple of (X, y) where X is shape (n_samples, n_timesteps, n_features)
    """
    print(f"Loading {dataset_name} regression dataset ({split} split)...")
    
    # Load the dataset using AEON's regression loader
    X, y = load_regression(dataset_name, split=split)
    
    # Convert to numpy arrays
    X_np = np.array(X, dtype=np.float64)
    y_np = np.array(y, dtype=np.float64)
    
    # Transpose to get (n_samples, n_timesteps, n_features)
    X_np = np.transpose(X_np, (0, 2, 1))
    
    print(f"Loaded {X_np.shape[0]} samples with {X_np.shape[1]} timesteps and {X_np.shape[2]} features")
    return X_np, y_np

def time_augment(X):
    """
    Augment the input array by adding a time feature as the last dimension.
    """
    n_samples, length, n_features = X.shape
    time_feature = np.linspace(0, 1, length)
    time_feature = np.broadcast_to(time_feature, (n_samples, length))
    time_feature = time_feature[..., None]  # shape (n_samples, length, 1)
    return np.concatenate([time_feature, X], axis=-1)

def main():
    print("=== PowerSig JAX Regression Example ===")
    
    # Set up JAX device
    try:
        devices = jax.devices("cuda")
        device = devices[0] if devices else jax.devices()[0]
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Warning: Could not set up CUDA device: {e}")
        device = jax.devices()[0]
        print(f"Using device: {device}")
    
    # Load dataset
    X_train, y_train = load_aeon_regression_dataset_simple("StandWalkJump", "train")
    X_test, y_test = load_aeon_regression_dataset_simple("StandWalkJump", "test")
    
    print(f"\nDataset info:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Time series length: {X_train.shape[1]}")
    print(f"Features: {X_train.shape[2]}")
    
    # Check if time series length is at least 1000
    if X_train.shape[1] < 1000:
        print(f"\nWarning: Time series length is only {X_train.shape[1]} timesteps, which is less than 1000")
        print("Proceeding anyway, but consider using a longer time series for better results")
    else:
        print(f"\n✓ Time series length is {X_train.shape[1]} timesteps (>= 1000)")
    
    # Normalize the data
    X_train_max = np.max(X_train)
    X_train = X_train / X_train_max
    X_test = X_test / X_train_max
    
    # Apply time augmentation
    X_train = time_augment(X_train)
    X_test = time_augment(X_test)
    
    print(f"\nAfter preprocessing:")
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train range: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")
    
    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test, dtype=jnp.float64, device=device)
    y_train_jax = jnp.array(y_train, dtype=jnp.float64, device=device)
    y_test_jax = jnp.array(y_test, dtype=jnp.float64, device=device)
    
    # Initialize PowerSigJax
    print("\nInitializing PowerSigJax...")
    powersig = PowerSigJax(order=8, device=device)  # Using order 8 for faster computation
    
    # Compute gram matrices
    print("Computing training gram matrix...")
    start_time = time.time()
    train_gram = powersig.compute_gram_matrix(X_train_jax, X_train_jax)
    train_time = time.time() - start_time
    print(f"✓ Training gram matrix computed in {train_time:.2f} seconds")
    print(f"Training gram matrix shape: {train_gram.shape}")
    
    print("Computing test gram matrix...")
    start_time = time.time()
    test_gram = powersig.compute_gram_matrix(X_test_jax, X_train_jax)
    test_time = time.time() - start_time
    print(f"✓ Test gram matrix computed in {test_time:.2f} seconds")
    print(f"Test gram matrix shape: {test_gram.shape}")
    
    # Convert to numpy for sklearn
    train_gram_np = np.array(train_gram)
    test_gram_np = np.array(test_gram)
    y_train_np = np.array(y_train_jax)
    y_test_np = np.array(y_test_jax)
    
    # Print gram matrix statistics
    print(f"\nGram matrix statistics:")
    print(f"Training gram matrix - min: {np.min(train_gram_np):.6f}, max: {np.max(train_gram_np):.6f}, mean: {np.mean(train_gram_np):.6f}")
    print(f"Test gram matrix - min: {np.min(test_gram_np):.6f}, max: {np.max(test_gram_np):.6f}, mean: {np.mean(test_gram_np):.6f}")
    
    # Train Kernel Ridge Regression
    print("\nTraining Kernel Ridge Regression...")
    
    # Try different alpha values for regularization
    alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alpha_values:
        print(f"  Training with alpha = {alpha}...")
        krr = KernelRidge(alpha=alpha, kernel='precomputed')
        krr.fit(train_gram_np, y_train_np)
        
        # Predict on training set for validation
        y_pred_train = krr.predict(train_gram_np)
        r2 = r2_score(y_train_np, y_pred_train)
        
        print(f"    Training R² score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_alpha = alpha
    
    print(f"\n✓ Best alpha: {best_alpha} (R² = {best_score:.4f})")
    
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
    
    print(f"\n=== Test Results ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2_test:.4f}")
    
    # Print some predictions vs actual values
    print(f"\nSample predictions (first 10):")
    print("Actual\tPredicted")
    for i in range(min(10, len(y_test_np))):
        print(f"{y_test_np[i]:.3f}\t{y_pred_test[i]:.3f}")
    
    # Print model summary
    print(f"\n=== Model Summary ===")
    print(f"Number of dual coefficients: {len(final_krr.dual_coef_)}")
    print(f"Intercept: {final_krr.intercept_:.6f}")
    print(f"Alpha used: {best_alpha}")
    print(f"Polynomial order: 8")
    print(f"Total computation time: {train_time + test_time:.2f} seconds")
    
    print(f"\n✓ Regression example completed successfully!")

if __name__ == "__main__":
    main() 