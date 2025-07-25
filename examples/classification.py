#!/usr/bin/env python3
"""
Classification example using PowerSig JAX with aeon datasets.
This example loads an aeon time series classification dataset,
computes the gram matrix using PowerSig JAX, and performs SVM classification.
"""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from aeon.datasets import load_classification
import time

# Import PowerSigJax
try:
    from powersig.jax.algorithm import PowerSigJax
except ImportError:
    print("Error: Could not import PowerSigJax. Make sure powersig is installed.")
    exit(1)

def load_aeon_classification_dataset(dataset_name="EigenWorms", split="train"):
    """
    Load an AEON classification dataset and convert to numpy arrays.
    
    Args:
        dataset_name: Name of the AEON classification dataset to load
        split: Which split to load ('train' or 'test')
        
    Returns:
        Tuple of (X, y, label_map) where:
            X: numpy array of shape (n_samples, n_timesteps, n_features)
            y: numpy array of shape (n_samples,) containing integer labels
            label_map: Dictionary mapping integer labels back to original string labels
    """
    print(f"Loading {dataset_name} classification dataset ({split} split)...")
    
    # Load the dataset using AEON's classification loader
    X, y = load_classification(dataset_name, split=split)
    
    # Convert to numpy arrays
    X_np = np.array(X, dtype=np.float64)
    
    # Handle string labels by converting to integers
    if isinstance(y[0], str):
        unique_labels = np.unique(y)
        label_map = {i: label for i, label in enumerate(unique_labels)}
        y_int = np.array([np.where(unique_labels == label)[0][0] for label in y])
        y_np = y_int
    else:
        y_np = np.array(y, dtype=np.int32)
        label_map = None
    
    # Transpose to get (n_samples, n_timesteps, n_features)
    X_np = np.transpose(X_np, (0, 2, 1))
    
    print(f"Loaded {X_np.shape[0]} samples with {X_np.shape[1]} timesteps and {X_np.shape[2]} features")
    print(f"Number of classes: {len(np.unique(y_np))}")
    if label_map is not None:
        print(f"Classes: {list(label_map.values())}")
    
    return X_np, y_np, label_map

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
    print("=== PowerSig JAX Classification Example ===")
    
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
    X_train, y_train, label_map = load_aeon_classification_dataset("EigenWorms", "train")
    X_test, y_test, _ = load_aeon_classification_dataset("EigenWorms", "test")
    
    print(f"\nDataset info:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Time series length: {X_train.shape[1]}")
    print(f"Features: {X_train.shape[2]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
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
    
    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test, dtype=jnp.float64, device=device)
    y_train_jax = jnp.array(y_train, dtype=jnp.int32, device=device)
    y_test_jax = jnp.array(y_test, dtype=jnp.int32, device=device)
    
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
    
    # Train SVM with precomputed kernel
    print("\nTraining SVM with precomputed kernel...")
    
    # Try different C values for regularization
    C_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_C = None
    best_score = -np.inf
    
    for C in C_values:
        print(f"  Training with C = {C}...")
        svm = SVC(kernel='precomputed', C=C, random_state=42)
        svm.fit(train_gram_np, y_train_np)
        
        # Predict on training set for validation
        y_pred_train = svm.predict(train_gram_np)
        accuracy = accuracy_score(y_train_np, y_pred_train)
        
        print(f"    Training accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_C = C
    
    print(f"\n✓ Best C: {best_C} (accuracy = {best_score:.4f})")
    
    # Train final model with best C
    print("Training final model...")
    final_svm = SVC(kernel='precomputed', C=best_C, random_state=42)
    final_svm.fit(train_gram_np, y_train_np)
    
    # Predict on test set
    print("Making predictions on test set...")
    y_pred_test = final_svm.predict(test_gram_np)
    
    # Evaluate performance
    test_accuracy = accuracy_score(y_test_np, y_pred_test)
    
    print(f"\n=== Test Results ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test_np, y_pred_test))
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_np, y_pred_test)
    print(cm)
    
    # Print some predictions vs actual values
    print(f"\nSample predictions (first 10):")
    print("Actual\tPredicted")
    for i in range(min(10, len(y_test_np))):
        actual_label = label_map[y_test_np[i]] if label_map else y_test_np[i]
        pred_label = label_map[y_pred_test[i]] if label_map else y_pred_test[i]
        print(f"{actual_label}\t{pred_label}")
    
    # Print model summary
    print(f"\n=== Model Summary ===")
    print(f"Number of support vectors: {len(final_svm.support_vectors_)}")
    print(f"C parameter used: {best_C}")
    print(f"Polynomial order: 8")
    print(f"Total computation time: {train_time + test_time:.2f} seconds")
    
    print(f"\n✓ Classification example completed successfully!")

if __name__ == "__main__":
    main() 