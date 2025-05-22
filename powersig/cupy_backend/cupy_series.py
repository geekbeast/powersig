import cupy as cp
import numpy as np
from typing import Optional, Tuple


def double_length(array: cp.ndarray) -> cp.ndarray:
    new_array = cp.zeros([array.shape[0] * 2, array.shape[1] * 2], dtype=cp.float64)
    new_array[:array.shape[0], :array.shape[1]] = array
    return new_array


def resize(array: cp.ndarray, new_shape) -> cp.ndarray:
    new_array = cp.zeros(new_shape, dtype=cp.float64)
    new_array[:array.shape[0], :array.shape[1]] = array
    return new_array


def cupy_compute_derivative(X):
    diff = (X[1:,:] - X[:-1,:])
    if X.shape[0] == 1:
        return diff
    else:
        return diff * (X.shape[0] - 1)


def cupy_compute_dot_prod(X, Y):
    assert X.shape == Y.shape, "X and Y must have the same shape."
    return cp.sum(X * Y)


def cupy_compute_dot_prod_batch(X, Y):
    return cp.sum(X * Y, axis=len(X.shape) - 1)


def cupy_compute_derivative_batch(X, dt=None) -> cp.ndarray:
    """
    Compute the derivatives for a batch of time series.
    
    Args:
        X: Array of shape (batch_size, length, dim) containing the time series
        dt: Optional time step size
        
    Returns:
        Array of shape (batch_size, length-1, dim) containing the derivatives
    """
    diff = (X[:, 1:, :] - X[:, :-1, :])
    if dt is not None:
        diff = diff / dt
    elif X.shape[1] != 1:
        diff = diff * (X.shape[1] - 1)

    return diff 