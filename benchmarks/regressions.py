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
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[Dict[int, str]], jax.Device]:
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
    # Get the CUDA device if available
    if device is None:
        try:
            device = jax.devices("cuda")[1]
            print(f"Using CUDA device: {device}")
        except (ValueError, RuntimeError):
            print("CUDA device not available, using default device")
            device = jax.devices()[1]
    # Load the dataset using AEON's loader
    X, y = load_classification(dataset_name, split=split)
    
    # Convert to JAX arrays
    X_jax = jnp.array(X, dtype=jnp.float64, device=device)
    
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
    
    return jnp.transpose(X_jax, (0, 2, 1)), y_jax, label_map, device

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
    print(f"time_feature.shape: {time_feature.shape}")
    time_feature = time_feature[..., None]  # shape (n_samples, length, 1)
    return jnp.concatenate([time_feature, X], axis=-1)

if __name__ == "__main__":
    # Load DuckDuckGeese dataset
    print("Loading DuckDuckGeese dataset...")
    X_train, y_train, label_map, device = load_aeon_dataset("DuckDuckGeese", split="train")
    X_test, y_test, _ , _ = load_aeon_dataset("DuckDuckGeese", split="test")

    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")

    print(f"X_train max: {jnp.max(X_train)}")
    print(f"X_train min: {jnp.min(X_train)}")
    print(f"X_train mean: {jnp.mean(X_train)}")

    X_train_max = jnp.max(X_train)
    X_train /= X_train_max
    X_train = time_augment(X_train)

    X_test /= X_train_max
    X_test = time_augment(X_test)
    
    print("Applied time augmentation")
    print(f"X_train[:,0,0]: {X_train[:,0,0]}")
    print(f"X_train[:,1,0]: {X_train[:,1,0]}")
    print(f"X_train[:,2,0]: {X_train[:,2,0]}")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"Dataset shape: {X_train.shape}")
    print(f"Number of classes: {len(jnp.unique(y_train))}")
    
    if label_map is not None:
        print("\nLabel mapping:")
        for idx, label in label_map.items():
            print(f"{idx}: {label}")
    
    # Compute gram matrix for training
    print("\nComputing train gram matrix...")
    powersig = PowerSigJax(order=16)
    gram_matrix = powersig.compute_gram_matrix(X_train, X_train)
    test_gram_matrix = powersig.compute_gram_matrix(X_test, X_train)
    
    print(f"Gram matrix: {gram_matrix}")
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

    for i in [ 10**j for j in range(-2,20)]:
        clf = SVC(kernel='precomputed',C=i)
        clf.fit(gram_matrix_np, y_train_np)

        # Predict on test set
        print(f"\nPredicting on test set with C={i}...")
        y_pred = clf.predict(test_gram_matrix_np)
        acc = accuracy_score(y_test_np, y_pred)
        print(f"Test accuracy: {acc:.4f}")
