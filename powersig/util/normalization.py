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
