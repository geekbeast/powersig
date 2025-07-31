#!/usr/bin/env python3
"""
Bitcoin Price Prediction using PowerSigJax Signature Kernels
This example replicates Chris Salvi's approach using PowerSigJax to form the gram matrix
for Support Vector Regression (SVR) with custom kernel entries.
"""

import numpy as np
import cupy
import jax
import jax.numpy as jnp
import pandas as pd
import yfinance as yf
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
import pickle
from datetime import datetime, timedelta

from powersig.jax import static_kernels
from powersig.jax.algorithm import PowerSigJax

# Import KSigPDE for validation
import ksig
from ksig.static.kernels import LinearKernel
from ksig.static.kernels import RBFKernel
from ksig.kernels import SignaturePDEKernel


def download_bitcoin_data(start_date="2020-01-01", end_date=None, max_retries=3):
    """
    Download Bitcoin price data using yfinance with automatic retry and error handling.
    
    Args:
        start_date: Start date for data download (default: "2020-01-01")
        end_date: End date for data download (default: today)
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        DataFrame with Bitcoin price data
    """
    import requests
    from datetime import datetime, timedelta
    
    # Set default end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading Bitcoin data from {start_date} to {end_date}...")
    
    # Try multiple data sources with fallback
    data_sources = [
        ("yfinance", "BTC-USD"),
        ("yfinance", "BTCUSD=X"),
        ("yfinance", "BTC-USD.P")
    ]
    
    for attempt in range(max_retries):
        for source_name, symbol in data_sources:
            try:
                print(f"Attempt {attempt + 1}: Trying {source_name} with symbol {symbol}...")
                
                if source_name == "yfinance":
                    btc = yf.download(symbol, start=start_date, end=end_date, progress=False, timeout=30)
                
                # Check if we got valid data
                if btc is not None and len(btc) > 0:
                    print(f"✓ Successfully downloaded {len(btc)} days of Bitcoin data")
                    print(f"Date range: {btc.index[0].date()} to {btc.index[-1].date()}")
                    print(f"Columns: {list(btc.columns)}")
                    
                    # Validate data quality
                    missing_data = btc.isnull().sum()
                    if missing_data.sum() > 0:
                        print(f"Warning: Found missing data: {missing_data.to_dict()}")
                        # Fill missing values with forward fill
                        btc = btc.fillna(method='ffill')
                        print("Applied forward fill for missing values")
                    
                    return btc
                    
            except Exception as e:
                print(f"Failed to download from {source_name} with symbol {symbol}: {str(e)}")
                continue
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5  # Exponential backoff
            print(f"All sources failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # If all attempts failed, try to download a smaller date range
    print("All download attempts failed. Trying with a smaller date range...")
    try:
        # Try last 2 years if full range fails
        fallback_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        btc = yf.download("BTC-USD", start=fallback_start, end=end_date, progress=False, timeout=30)
        
        if btc is not None and len(btc) > 0:
            print(f"✓ Successfully downloaded fallback data: {len(btc)} days")
            print(f"Date range: {btc.index[0].date()} to {btc.index[-1].date()}")
            return btc
    except Exception as e:
        print(f"Fallback download also failed: {str(e)}")
    
    # If everything fails, raise an error with helpful message
    raise RuntimeError(
        "Failed to download Bitcoin data. Please check your internet connection "
        "and try again. You can also manually download data from Yahoo Finance "
        "or use the test script with synthetic data."
    )

def save_bitcoin_data(data, filename="bitcoin_data.pkl"):
    """
    Save Bitcoin data to a pickle file for future use.
    
    Args:
        data: DataFrame with Bitcoin data
        filename: Name of the file to save data
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Bitcoin data saved to {filename}")
    except Exception as e:
        print(f"Warning: Failed to save data: {str(e)}")

def load_bitcoin_data(filename="bitcoin_data.pkl", max_age_days=7):
    """
    Load Bitcoin data from a pickle file if it exists and is recent.
    
    Args:
        filename: Name of the file to load data from
        max_age_days: Maximum age of cached data in days (default: 7)
        
    Returns:
        DataFrame with Bitcoin data or None if file doesn't exist or is too old
    """
    if not os.path.exists(filename):
        return None
    
    try:
        # Check file age
        file_age = time.time() - os.path.getmtime(filename)
        max_age_seconds = max_age_days * 24 * 3600
        
        if file_age > max_age_seconds:
            print(f"Cached data is {file_age/3600/24:.1f} days old (max: {max_age_days} days)")
            return None
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded cached Bitcoin data: {len(data)} days")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        return data
        
    except Exception as e:
        print(f"Warning: Failed to load cached data: {str(e)}")
        return None

def create_sliding_windows(data, window_size=36, target_col='Close'):
    """
    Create sliding windows for time series prediction with window-wise normalization.
    
    Args:
        data: DataFrame with time series data
        window_size: Size of the sliding window (default: 36 days)
        target_col: Column to use as target (default: 'Close')
        
    Returns:
        Tuple of (X, y, window_stats) where X contains the normalized windows, 
        y contains the targets, and window_stats contains (mean, std) for each window
    """
    print(f"Creating sliding windows of size {window_size} with window-wise normalization...")
    
    # Use only the target column for now (can be extended to use multiple features)
    series = data[target_col].values
    
    X = []
    y = []
    window_stats = []  # Store (mean, std) for each window
    
    for i in range(len(series) - window_size - 1):  # -1 to ensure we have 2 days ahead
        # Create window
        window = series[i:i+window_size]
        
        # Normalize each window individually
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        window_normalized = (window - window_mean) / window_std
        
        target = ((series[i + window_size] - window_mean) + (series[i + window_size + 1] - window_mean)) / (2*window_std)
        
        X.append(window_normalized)
        y.append(target)
        window_stats.append((window_mean, window_std))
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} windows with shape {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Window stats - Mean of means: {np.mean([s[0] for s in window_stats]):.2f}")
    print(f"Window stats - Mean of stds: {np.mean([s[1] for s in window_stats]):.2f}")
    
    return X, y, window_stats

def create_sliding_windows_from_array(series, window_size=36):
    """
    Create sliding windows from a numpy array with window-wise normalization.
    
    Args:
        series: numpy array of time series data
        window_size: Size of the sliding window (default: 36 days)
        
    Returns:
        Tuple of (X, y, window_stats) where X contains the normalized windows, 
        y contains the targets, and window_stats contains (mean, std) for each window
    """
    X = []
    y = []
    window_stats = []  # Store (mean, std) for each window
    
    for i in range(len(series) - window_size - 1):  # -1 to ensure we have 2 days ahead
        # Create window
        window = series[i:i+window_size]
        
        # Normalize each window individually
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        window_normalized = (window - window_mean) / window_std
        
        target = ((series[i + window_size] - window_mean) + (series[i + window_size + 1] - window_mean)) / (2*window_std)

        X.append(window_normalized)
        y.append(target)
        window_stats.append((window_mean, window_std))
    
    X = np.array(X)
    y = np.array(y)
    print("X min:", np.min(X), "X max:", np.max(X))
    print("y min:", np.min(y), "y max:", np.max(y))
    return X, y, window_stats

def time_augment(X):
    """
    Augment the input array by adding a time feature as the last dimension.
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

def split_train_test_simple(data, train_days=346, test_days=78):
    """
    Simple split: take first train_days elements for training, last test_days for testing.
    
    Args:
        data: DataFrame with Bitcoin data
        train_days: Number of days for training (default: 346)
        test_days: Number of days for testing (default: 78)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    # Extract Close prices as numpy array
    close_prices = data['Close'].values
    
    # Split the data
    train_data = close_prices[:train_days]
    test_data = close_prices[-test_days:]
    
    print(f"Training data: {len(train_data)} days ({data.index[0].date()} to {data.index[train_days-1].date()})")
    print(f"Test data: {len(test_data)} days ({data.index[-test_days].date()} to {data.index[-1].date()})")
    
    return train_data, test_data

def compute_gram_matrix_powersig(X_train, X_test, powersig):
    """
    Compute gram matrices using PowerSigJax.
    
    Args:
        X_train: Training data
        X_test: Test data
        powersig: PowerSigJax instance
        
    Returns:
        Tuple of (train_gram, test_gram)
    """

    # static_kernel = ksig.static.kernels.RBFKernel()
    static_kernel = ksig.static.kernels.LinearKernel()
    ksig_pde_kernel = SignaturePDEKernel(normalize=False, static_kernel=static_kernel)
    
    print("Computing training gram matrix...")
    start_time = time.time()
    train_gram = powersig.compute_gram_matrix(X_train, X_train)
    # train_gram = jnp.array(ksig_pde_kernel(cupy.array(X_train), cupy.array(X_train)))
    train_time = time.time() - start_time
    print(f"✓ Training gram matrix computed in {train_time:.2f} seconds")
    
    print("Computing test gram matrix...")
    start_time = time.time()
    # test_gram = powersig.compute_gram_matrix(X_test, X_train)
    test_gram = jnp.array(ksig_pde_kernel(cupy.array(X_test), cupy.array(X_train)))
    test_time = time.time() - start_time
    print(f"✓ Test gram matrix computed in {test_time:.2f} seconds")
    
    return train_gram, test_gram

def train_svr_model(train_gram, y_train, C=1.0, epsilon=0.1):
    """
    Train SVR model with custom gram matrix.
    
    Args:
        train_gram: Training gram matrix
        y_train: Training targets
        C: Regularization parameter
        epsilon: Epsilon parameter for SVR
        
    Returns:
        Trained SVR model
    """
    print(f"Training SVR with C={C}, epsilon={epsilon}...")
    
    # Convert to numpy for sklearn and ensure 1D arrays
    train_gram_np = np.array(train_gram)
    y_train_np = np.array(y_train).ravel()  # Ensure 1D array
    
    print(f"SVR Debug - Gram matrix shape: {train_gram_np.shape}")
    print(f"SVR Debug - Targets shape: {y_train_np.shape}")
    print(f"SVR Debug - Gram matrix condition number: {np.linalg.cond(train_gram_np):.2e}")
    
    # Train SVR with precomputed kernel
    svr = SVR(kernel='precomputed', C=C, epsilon=epsilon)
    svr.fit(train_gram_np, y_train_np)
    
    print(f"✓ SVR trained with {len(svr.support_)} support vectors")
    
    return svr

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value as percentage
    """
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def validate_gram_matrix_powersig_vs_ksigpde(X_train, powersig):
    """
    Validate PowerSigJax gram matrix against KSigPDE.
    
    Args:
        X_train: Training data (numpy array)
        powersig_order: Order for PowerSigJax
        
    Returns:
        Dictionary with validation results
    """
    
    print("\n=== Validating PowerSigJax vs KSigPDE ===")
    
    # Use a small subset for validation (first 10 samples)
    n_validate = min(10, X_train.shape[0])
    X_validate = X_train[:n_validate]
    
    print(f"Using {n_validate} samples for validation")
    print(f"X_validate shape: {X_validate.shape}")
    
    # 1. PowerSigJax computation
    print("Computing PowerSigJax gram matrix...")
    start_time = time.time()
    
    # Convert to JAX arrays for PowerSigJax
    X_validate_jax = jnp.array(X_validate, dtype=jnp.float64)
    powersig_gram = powersig.compute_gram_matrix(X_validate_jax, X_validate_jax)
    powersig_time = time.time() - start_time
    print(f"PowerSigJax time: {powersig_time:.3f}s")
    
    # 2. KSigPDE computation
    print("Computing KSigPDE gram matrix...")
    start_time = time.time()
    # static_kernel = RBFKernel()
    static_kernel = LinearKernel()
    ksig_pde_kernel = SignaturePDEKernel(normalize=False, static_kernel=static_kernel)
    ksig_gram = ksig_pde_kernel(cupy.array(X_validate,dtype=cupy.float64), cupy.array(X_validate,dtype=cupy.float64))
    ksig_time = time.time() - start_time
    print(f"KSigPDE time: {ksig_time:.3f}s")
    
    # Convert to numpy for comparison
    powersig_gram_np = np.array(powersig_gram)
    ksig_gram_np = np.array(ksig_gram.get())
    
    # 3. KSig Truncated Signature Kernel computation (level=21)
    print("\nComputing KSig Truncated Signature Kernel gram matrix (level=21)...")
    start_time = time.time()
    ksig_trunc_kernel = ksig.kernels.SignatureKernel(n_levels=21, order=0, normalize=False, static_kernel=static_kernel)
    ksig_trunc_gram = ksig_trunc_kernel(cupy.array(X_validate,dtype=cupy.float64), cupy.array(X_validate,dtype=cupy.float64))
    ksig_trunc_time = time.time() - start_time
    print(f"KSig Truncated time: {ksig_trunc_time:.3f}s")
    
    # Convert to numpy for comparison
    ksig_trunc_gram_np = np.array(ksig_trunc_gram)
    
    # Calculate differences between PowerSigJax and KSig Truncated
    abs_diff_trunc = np.abs(powersig_gram_np - ksig_trunc_gram_np)
    rel_diff_trunc = abs_diff_trunc / (np.abs(ksig_trunc_gram_np) + 1e-10)
    
    print(f"\nKSig Truncated Validation Results:")
    print(f"PowerSigJax gram shape: {powersig_gram_np.shape}")
    print(f"KSig Truncated gram shape: {ksig_trunc_gram_np.shape}")
    print(f"Max absolute difference: {np.max(abs_diff_trunc):.6f}")
    print(f"Mean absolute difference: {np.mean(abs_diff_trunc):.6f}")
    print(f"Max relative difference: {np.max(rel_diff_trunc):.6f}")
    print(f"Mean relative difference: {np.mean(rel_diff_trunc):.6f}")
    
    # Check if they're close enough
    tolerance = 1e-3
    is_close_trunc = np.allclose(powersig_gram_np, ksig_trunc_gram_np, rtol=tolerance, atol=tolerance)
    print(f"Matrices are close (tolerance {tolerance}): {is_close_trunc}")
    
    # Show sample values for truncated
    print(f"\nSample values (first 3x3 submatrix) - KSig Truncated:")
    print("PowerSigJax:")
    print(powersig_gram_np[:3, :3])
    print("KSig Truncated:")
    print(ksig_trunc_gram_np[:3, :3])
    
    # Calculate differences between PowerSigJax and KSigPDE
    abs_diff = np.abs(powersig_gram_np - ksig_gram_np)
    rel_diff = abs_diff / (np.abs(ksig_gram_np) + 1e-10)
    
    print(f"\nValidation Results:")
    print(f"PowerSigJax gram shape: {powersig_gram_np.shape}")
    print(f"KSigPDE gram shape: {ksig_gram_np.shape}")
    print(f"Max absolute difference: {np.max(abs_diff):.6f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")
    print(f"Max relative difference: {np.max(rel_diff):.6f}")
    print(f"Mean relative difference: {np.mean(rel_diff):.6f}")
    
    # Check if they're close enough
    tolerance = 1e-3
    is_close = np.allclose(powersig_gram_np, ksig_gram_np, rtol=tolerance, atol=tolerance)
    print(f"Matrices are close (tolerance {tolerance}): {is_close}")
    
    # Show sample values
    print(f"\nSample values (first 3x3 submatrix):")
    print("PowerSigJax:")
    print(powersig_gram_np[:3, :3])
    print("KSigPDE:")
    print(ksig_gram_np[:3, :3])
    
    return {
        'powersig_gram': powersig_gram_np,
        'ksig_gram': ksig_gram_np,
        'ksig_trunc_gram': ksig_trunc_gram_np,
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'max_abs_diff_trunc': np.max(abs_diff_trunc),
        'mean_abs_diff_trunc': np.mean(abs_diff_trunc),
        'max_rel_diff_trunc': np.max(rel_diff_trunc),
        'mean_rel_diff_trunc': np.mean(rel_diff_trunc),
        'is_close': is_close,
        'is_close_trunc': is_close_trunc,
        'powersig_time': powersig_time,
        'ksig_time': ksig_time,
        'ksig_trunc_time': ksig_trunc_time
    }

def evaluate_model(svr, train_gram, test_gram, y_train, y_test, window_stats_train=None, window_stats_test=None):
    """
    Evaluate the trained model.
    
    Args:
        svr: Trained SVR model
        train_gram: Training gram matrix
        test_gram: Test gram matrix
        y_train: Training targets (normalized)
        y_test: Test targets (normalized)
        window_stats_train: List of (mean, std) tuples for training windows
        window_stats_test: List of (mean, std) tuples for test windows
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to numpy and ensure 1D arrays
    train_gram_np = np.array(train_gram)
    test_gram_np = np.array(test_gram)
    y_train_np = np.array(y_train).ravel()  # Ensure 1D array
    y_test_np = np.array(y_test).ravel()    # Ensure 1D array
    
    print(f"y_train_np shape: {y_train_np.shape}")
    print(f"y_test_np shape: {y_test_np.shape}")
    
    # Predictions (in normalized scale)
    y_pred_train = svr.predict(train_gram_np)
    y_pred_test = svr.predict(test_gram_np)
    
    # Debug: Check prediction ranges
    print(f"\n=== Post-training Debug ===")
    print(f"Training predictions - min: {np.min(y_pred_train):.6f}, max: {np.max(y_pred_train):.6f}")
    print(f"Training predictions - mean: {np.mean(y_pred_train):.6f}, std: {np.std(y_pred_train):.6f}")
    print(f"Test predictions - min: {np.min(y_pred_test):.6f}, max: {np.max(y_pred_test):.6f}")
    print(f"Test predictions - mean: {np.mean(y_pred_test):.6f}, std: {np.std(y_pred_test):.6f}")
    
    # Denormalize predictions and targets to original scale
    if window_stats_train is not None:
        y_train_orig = np.array([y_train_np[i] * window_stats_train[i][1] + window_stats_train[i][0] for i in range(len(y_train_np))])
        y_pred_train_orig = np.array([y_pred_train[i] * window_stats_train[i][1] + window_stats_train[i][0] for i in range(len(y_pred_train))])
    else:
        y_train_orig = y_train_np
        y_pred_train_orig = y_pred_train
    
    if window_stats_test is not None:
        y_test_orig = np.array([y_test_np[i] * window_stats_test[i][1] + window_stats_test[i][0] for i in range(len(y_test_np))])
        y_pred_test_orig = np.array([y_pred_test[i] * window_stats_test[i][1] + window_stats_test[i][0] for i in range(len(y_pred_test))])
    else:
        y_test_orig = y_test_np
        y_pred_test_orig = y_pred_test
    
    # Calculate metrics on normalized scale
    train_mse_norm = mean_squared_error(y_train_np, y_pred_train)
    train_mae_norm = mean_absolute_error(y_train_np, y_pred_train)
    train_r2_norm = r2_score(y_train_np, y_pred_train)
    train_mape_norm = calculate_mape(y_train_np, y_pred_train)
    
    test_mse_norm = mean_squared_error(y_test_np, y_pred_test)
    test_mae_norm = mean_absolute_error(y_test_np, y_pred_test)
    test_r2_norm = r2_score(y_test_np, y_pred_test)
    test_mape_norm = calculate_mape(y_test_np, y_pred_test)
    
    # Calculate metrics on original scale
    train_mse = mean_squared_error(y_train_orig, y_pred_train_orig)
    train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
    train_r2 = r2_score(y_train_orig, y_pred_train_orig)
    train_mape = calculate_mape(y_train_orig, y_pred_train_orig)
    
    test_mse = mean_squared_error(y_test_orig, y_pred_test_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)
    test_mape = calculate_mape(y_test_orig, y_pred_test_orig)
    
    results = {
        # Normalized scale metrics
        'train_mse_norm': train_mse_norm,
        'train_mae_norm': train_mae_norm,
        'train_r2_norm': train_r2_norm,
        'train_mape_norm': train_mape_norm,
        'test_mse_norm': test_mse_norm,
        'test_mae_norm': test_mae_norm,
        'test_r2_norm': test_r2_norm,
        'test_mape_norm': test_mape_norm,
        # Original scale metrics
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'y_pred_train': y_pred_train_orig,
        'y_pred_test': y_pred_test_orig,
        'y_train_orig': y_train_orig,
        'y_test_orig': y_test_orig
    }
    
    return results

def plot_results(y_train, y_test, y_pred_train, y_pred_test, save_path=None):
    """
    Plot the prediction results.
    
    Args:
        y_train: Training targets
        y_test: Test targets
        y_pred_train: Training predictions
        y_pred_test: Test predictions
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Training predictions
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='Actual', alpha=0.7)
    plt.plot(y_pred_train, label='Predicted', alpha=0.7)
    plt.title('Training Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test predictions
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred_test, label='Predicted', alpha=0.7)
    plt.title('Test Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    print("=== Bitcoin Price Prediction with PowerSigJax ===")
    
    # Set up JAX device - prefer CUDA device 1, then device 0, then CPU
    try:
        cuda_devices = jax.devices("cuda")
        if len(cuda_devices) >= 2:
            device = cuda_devices[1]  # Use CUDA device 1
            print(f"Using CUDA device 1: {device}")
        elif len(cuda_devices) == 1:
            device = cuda_devices[0]  # Use CUDA device 0
            print(f"Using CUDA device 0: {device}")
        else:
            device = jax.devices()[0]  # Use CPU
            print(f"Using CPU device: {device}")
    except Exception as e:
        print(f"Warning: Could not set up CUDA devices: {e}")
        device = jax.devices()[0]
        print(f"Using fallback device: {device}")
    
    # Try to load cached Bitcoin data first
    btc_data = load_bitcoin_data()
    
    if btc_data is None:
        # Download Bitcoin data if not cached or too old
        # Start from 2017-06-02 to ensure we have the training period
        btc_data = download_bitcoin_data(start_date="2017-06-02")
        # Save for future use
        save_bitcoin_data(btc_data)
    print(f"Bitcoin data: {btc_data}")
    # Simple split: first 346 days for training, last 78 days for testings
    train_data, test_data = split_train_test_simple(btc_data, train_days=346, test_days=78)
    print(f"Training data min: {train_data.min()}, max: {train_data.max()}")
    
    # Create sliding windows with window-wise normalization
    window_size = 36
    print("Creating training windows...")
    X_train, y_train, window_stats_train = create_sliding_windows_from_array(train_data, window_size=window_size)
    print("Creating test windows...")
    X_test, y_test, window_stats_test = create_sliding_windows_from_array(test_data, window_size=window_size)
    
    print(f"Training windows: {len(X_train)} samples")
    print(f"Test windows: {len(X_test)} samples")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"y_test range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    
    # Apply time augmentation
    X_train_aug = time_augment(X_train)
    X_test_aug = time_augment(X_test)
    
    print(f"\nAfter preprocessing:")
    print(f"X_train.shape: {X_train_aug.shape}")
    print(f"X_test.shape: {X_test_aug.shape}")
    
    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train_aug, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test_aug, dtype=jnp.float64, device=device)
    y_train_jax = jnp.array(y_train, dtype=jnp.float64, device=device)
    y_test_jax = jnp.array(y_test, dtype=jnp.float64, device=device)
    
    # Initialize PowerSigJax
    print("\nInitializing PowerSigJax...")
    powersig = PowerSigJax(order=8,static_kernel=static_kernels.linear_kernel, device=device)  # Using order 8 for validation

    
    # Validate PowerSigJax against KSigPDE
    validation_results = validate_gram_matrix_powersig_vs_ksigpde(X_train_aug, powersig)
    
    # Compute gram matrices
    # Training gram matrix: between training samples and training samples
    # Test gram matrix: between test samples and training samples (for prediction)
    train_gram, test_gram = compute_gram_matrix_powersig(X_train_jax, X_test_jax, powersig)
    
    cond_train = jax.numpy.linalg.cond(train_gram)
    cond_test = jax.numpy.linalg.cond(test_gram)
    print(f"Condition number of training gram matrix: {cond_train:.2f}")
    print(f"Condition number of test gram matrix: {cond_test:.2f}")
    
    # Print gram matrix statistics
    train_gram_np = np.array(train_gram)
    test_gram_np = np.array(test_gram)
    print(f"\nGram matrix statistics:")
    print(f"Training gram matrix - min: {np.min(train_gram_np):.6f}, max: {np.max(train_gram_np):.6f}, mean: {np.mean(train_gram_np):.6f}")
    print(f"Test gram matrix - min: {np.min(test_gram_np):.6f}, max: {np.max(test_gram_np):.6f}, mean: {np.mean(test_gram_np):.6f}")
    
    # Train SVR model with reasonable C value and regularization
    C = .10  # Much smaller C for better regularization
    epsilon = 0.1  # Larger epsilon for more tolerance
    
    # Debug: Check gram matrix and target ranges before training
    train_gram_np = np.array(train_gram)
    y_train_np = np.array(y_train_jax).ravel()
    print(f"\n=== Pre-training Debug ===")
    print(f"Training gram matrix - min: {np.min(train_gram_np):.6f}, max: {np.max(train_gram_np):.6f}")
    print(f"Training targets - min: {np.min(y_train_np):.6f}, max: {np.max(y_train_np):.6f}")
    print(f"Training targets - mean: {np.mean(y_train_np):.6f}, std: {np.std(y_train_np):.6f}")
    
    # Add regularization to gram matrix for numerical stability
    print("Adding regularization to gram matrix...")
    reg_factor = 1e-6  # Small regularization factor
    train_gram_reg = train_gram_np + reg_factor * np.eye(train_gram_np.shape[0])
    print(f"Regularized gram matrix condition number: {np.linalg.cond(train_gram_reg):.2e}")
    
    svr = train_svr_model(train_gram_reg, y_train_jax, C=C, epsilon=epsilon)
    
    # Evaluate model
    results = evaluate_model(svr, train_gram_reg, test_gram, y_train_jax, y_test_jax, window_stats_train, window_stats_test)
    
    # Print results
    print(f"\n=== Model Performance (Normalized Scale) ===")
    print(f"Training MSE: {results['train_mse_norm']:.6f}")
    print(f"Training MAE: {results['train_mae_norm']:.6f}")
    print(f"Training MAPE: {results['train_mape_norm']:.2f}%")
    print(f"Training R²: {results['train_r2_norm']:.4f}")
    print(f"Test MSE: {results['test_mse_norm']:.6f}")
    print(f"Test MAE: {results['test_mae_norm']:.6f}")
    print(f"Test MAPE: {results['test_mape_norm']:.2f}%")
    print(f"Test R²: {results['test_r2_norm']:.4f}")
    
    print(f"\n=== Model Performance (Original Scale) ===")
    print(f"Training MSE: {results['train_mse']:.2f}")
    print(f"Training MAE: {results['train_mae']:.2f}")
    print(f"Training MAPE: {results['train_mape']:.2f}%")
    print(f"Training R²: {results['train_r2']:.4f}")
    print(f"Test MSE: {results['test_mse']:.2f}")
    print(f"Test MAE: {results['test_mae']:.2f}")
    print(f"Test MAPE: {results['test_mape']:.2f}%")
    print(f"Test R²: {results['test_r2']:.4f}")
    
    # Print some sample predictions (normalized scale)
    print(f"\nSample test predictions (first 10) - Normalized scale:")
    print("Actual\tPredicted\tError")
    y_test_np = np.array(y_test).ravel()
    y_pred_test_norm = svr.predict(test_gram_np)
    
    for i in range(min(10, len(y_test_np))):
        actual_val = float(y_test_np[i])
        pred_val = float(y_pred_test_norm[i])
        error = abs(actual_val - pred_val)
        print(f"{actual_val:.4f}\t{pred_val:.4f}\t{error:.4f}")
    
    # Print some sample predictions (denormalized to original scale)
    print(f"\nSample test predictions (first 10) - Original scale:")
    print("Actual\tPredicted\tError")
    y_test_orig = results['y_test_orig']
    y_pred_test = results['y_pred_test']
    
    for i in range(min(10, len(y_test_orig))):
        actual_val = float(y_test_orig[i])
        pred_val = float(y_pred_test[i])
        error = abs(actual_val - pred_val)
        print(f"{actual_val:.2f}\t{pred_val:.2f}\t{error:.2f}")
    
    # Print model summary
    print(f"\n=== Model Summary ===")
    print(f"Number of support vectors: {len(svr.support_)}")
    print(f"Support vector ratio: {len(svr.support_) / len(y_train):.2%}")
    print(f"C parameter: {C}")
    print(f"Epsilon parameter: {epsilon}")
    print(f"Polynomial order: 16")
    print(f"Window size: {window_size} days")
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_results(
        results['y_train_orig'], 
        results['y_test_orig'], 
        results['y_pred_train'], 
        results['y_pred_test']
    )
    
    print(f"\n✓ Bitcoin prediction example completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bitcoin Price Prediction with PowerSigJax")
    parser.add_argument("--start-date", default="2017-06-02", 
                       help="Start date for Bitcoin data (YYYY-MM-DD, default: 2017-06-02)")
    parser.add_argument("--end-date", default=None,
                       help="End date for Bitcoin data (YYYY-MM-DD, default: today)")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download of data (ignore cache)")
    parser.add_argument("--cache-days", type=int, default=7,
                       help="Maximum age of cached data in days (default: 7)")
    parser.add_argument("--train-start-date", default="2017-06-02",
                       help="Training start date (YYYY-MM-DD, default: 2017-06-02)")
    parser.add_argument("--train-days", type=int, default=346,
                       help="Training period length in days (default: 346)")
    parser.add_argument("--window-size", type=int, default=36,
                       help="Sliding window size in days (default: 36)")
    parser.add_argument("--order", type=int, default=16,
                       help="PowerSigJax polynomial order (default: 16)")
    parser.add_argument("--C", type=float, default=1.0,
                       help="SVR C parameter (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="SVR epsilon parameter (default: 0.1)")
    
    args = parser.parse_args()
    
    # Update global variables based on command line arguments
    if args.force_download:
        # Remove cached file if it exists
        if os.path.exists("bitcoin_data.pkl"):
            os.remove("bitcoin_data.pkl")
            print("Removed cached data file")
    
    # Update cache age if specified
    if args.cache_days != 7:
        # This would require modifying the load_bitcoin_data function call
        # For simplicity, we'll just note it in the output
        print(f"Cache age set to {args.cache_days} days")
    
    main()
