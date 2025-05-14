import jax
import jax.numpy as jnp
from aeon.datasets import load_classification
from typing import Tuple, Optional, Dict
from powersig.jax import PowerSigJax
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_aeon_dataset(
    dataset_name: str,
    split: str = "train",
    device: Optional[jax.Device] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[Dict[int, str]]]:
    """
    Load an AEON dataset and convert it to JAX arrays.
    
    Args:
        dataset_name: Name of the AEON dataset to load
        split: Which split to load ('train' or 'test')
        device: Optional JAX device to place the arrays on. If None, uses default device.
        
    Returns:
        Tuple of (X, y, label_map) where:
            X: JAX array of shape (n_samples, n_timesteps, n_features)
            y: JAX array of shape (n_samples,) containing integer labels
            label_map: Dictionary mapping integer labels back to original string labels (if applicable)
    """
    # Load the dataset using AEON's loader
    X, y = load_classification(dataset_name, split=split)
    
    # Convert to JAX arrays
    X_jax = jnp.array(X, dtype=jnp.float64)
    
    # Handle string labels by converting to integers
    if isinstance(y[0], str):
        unique_labels = np.unique(y)
        label_map = {i: label for i, label in enumerate(unique_labels)}
        y_int = np.array([np.where(unique_labels == label)[0][0] for label in y])
        y_jax = jnp.array(y_int, dtype=jnp.int32)
    else:
        y_jax = jnp.array(y, dtype=jnp.int32)
        label_map = None
    
    # Move to specified device if provided
    if device is not None:
        X_jax = jax.device_put(X_jax, device)
        y_jax = jax.device_put(y_jax, device)
    
    return X_jax, y_jax, label_map

def compute_gram_matrix(X: jnp.ndarray, order: int = 32) -> jnp.ndarray:
    """
    Compute the gram matrix for a dataset using PowerSigJax.
    This uses PowerSigJax's efficient batch computation method.
    
    Args:
        X: Input data of shape (n_samples, n_timesteps, n_features)
        order: Order of the polynomial approximation
        
    Returns:
        Gram matrix of shape (n_samples, n_samples)
    """
    # Initialize PowerSigJax with specified order
 
    
    # Use PowerSigJax's efficient batch computation
    return powersig.compute_gram_matrix(X, X)

if __name__ == "__main__":
    # Load DuckDuckGeese dataset
    print("Loading DuckDuckGeese dataset...")
    X_train, y_train, label_map = load_aeon_dataset("DuckDuckGeese", split="train")
    X_test, y_test, _ = load_aeon_dataset("DuckDuckGeese", split="test")
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Number of classes: {len(jnp.unique(y_train))}")
    if label_map is not None:
        print("\nLabel mapping:")
        for idx, label in label_map.items():
            print(f"{idx}: {label}")
    
    # Compute gram matrix for training
    print("\nComputing train gram matrix...")
    powersig = PowerSigJax(order=8)
    gram_matrix = compute_gram_matrix(X_train)
    powersig = PowerSigJax(order=8)
    test_gram_matrix = powersig.compute_gram_matrix(X_test, X_train)
    
    print(f"Gram matrix shape: {gram_matrix.shape}")
    print(f"Gram matrix min: {jnp.min(gram_matrix)}")
    print(f"Gram matrix max: {jnp.max(gram_matrix)}")
    print(f"Gram matrix mean: {jnp.mean(gram_matrix)}")


    # Convert to numpy for sklearn
    gram_matrix_np = np.array(gram_matrix)
    test_gram_matrix_np = np.array(test_gram_matrix)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    # Train SVC with precomputed kernel
    print("\nTraining SVC with precomputed kernel...")
    clf = SVC(kernel='precomputed')
    clf.fit(gram_matrix_np, y_train_np)

    # Predict on test set
    print("\nPredicting on test set...")
    y_pred = clf.predict(test_gram_matrix_np)
    acc = accuracy_score(y_test_np, y_pred)
    print(f"Test accuracy: {acc:.4f}")
