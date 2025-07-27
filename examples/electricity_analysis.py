import jax
import jax.numpy as jnp
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from powersig.jax.algorithm import PowerSigJax
import pandas as pd
import os
import requests
import zipfile
import io
import time

def download_and_extract(url, cache_dir):
    zip_path = os.path.join(cache_dir, "electricity_data.zip")
    data_path = os.path.join(cache_dir, "LD2011_2014.txt")
    
    if not os.path.exists(data_path):
        print(f"Downloading data to {zip_path}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save zip file
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
        
        # Extract the zip file
        print("Extracting data...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Extraction complete.")
    else:
        print(f"Using cached data from {data_path}")
    
    return data_path

def main():
    # Get JAX device with proper error handling
    try:
        devices = jax.devices("cuda")
        # Fix: Use device 2 only if we have more than 2 devices
        device = devices[1] if len(devices) > 1 else devices[0]
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Warning: Could not set up CUDA device: {e}")
        device = jax.devices()[0]
        print(f"Using device: {device}")
    
    # Load the electricity dataset directly from CSV
    print("Loading electricity dataset...")
    # Define URL and local cache path
    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download and extract data
    data_path = download_and_extract(url, cache_dir)
    
    # Load from local file
    df = pd.read_csv(data_path, sep=';', decimal=',', parse_dates=['Unnamed: 0'])
    df = df.rename(columns={df.columns[0]: 'timestamp'})
    
    # Split by date
    train_df = df[(df['timestamp'] >= '2011-01-01') & (df['timestamp'] <= '2013-12-31')]
    test_df = df[df['timestamp'] >= '2014-01-01']
    
    # Convert to numpy array, excluding the date column
    train_data = train_df.iloc[:, 1:].values.astype(np.float64).T  # (n_meters, n_train_timestamps)
    test_data = test_df.iloc[:, 1:].values.astype(np.float64).T    # (n_meters, n_test_timestamps)
    print(f"Number of meters: {train_data.shape[0]}, Train timestamps: {train_data.shape[1]}, Test timestamps: {test_data.shape[1]}")
    
    # Use 16000 timestamps for both training and testing
    train_len = min(16000, train_data.shape[1])
    test_len = min(16000, test_data.shape[1])
    X_train = train_data[:, :train_len]
    X_test = test_data[:, :test_len]
    
    print(f"Using {train_len} timestamps for training and {test_len} timestamps for testing")
    print(f"Note: This represents {train_len/4/24:.1f} days of data (at 15-minute intervals)")
    
    # Memory usage estimate for gram matrices
    estimated_memory_mb = (train_len * train_len * 8) / (1024 * 1024)  # float64 = 8 bytes
    print(f"Estimated gram matrix memory usage: {estimated_memory_mb:.1f} MB")
    if estimated_memory_mb > 1000:
        print("Warning: Large gram matrix may require significant memory!")
    
    # Normalize the data
    X_train_max = np.max(X_train)
    X_train = X_train / X_train_max
    X_test = X_test / X_train_max
    
    # Add singleton feature dimension: (n_meters, n_timestamps, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Convert to JAX arrays and place on device 1, using float64
    X_train_jax = jnp.array(X_train, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test, dtype=jnp.float64, device=device)
    
    print(f"Training data shape: {X_train_jax.shape}")
    print(f"Test data shape: {X_test_jax.shape}")
    
    # Create target variables: predict consumption for the next 24 hours (96 timestamps at 15-minute intervals)
    # Each target is a vector of 96 values representing the next 24 hours
    future_window = 96  # 24 hours * 4 intervals per hour = 96 timestamps
    
    # For training: use the next 96 timestamps as target
    if train_data.shape[1] > train_len + future_window:
        y_train = train_data[:, train_len:train_len+future_window]  # Shape: (n_meters, 96)
    else:
        y_train = train_data[:, -future_window:]  # Use last 96 timestamps
    
    # For testing: use the next 96 timestamps as target
    if test_data.shape[1] > test_len + future_window:
        y_test = test_data[:, test_len:test_len+future_window]  # Shape: (n_meters, 96)
    else:
        y_test = test_data[:, -future_window:]  # Use last 96 timestamps
    
    print(f"Target shape - y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"Target variable range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    
    # Initialize PowerSigJax with appropriate order for larger dataset
    # Lower order for computational efficiency with large datasets
    powersig = PowerSigJax(order=6)
    
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
    
    # Print gram matrix statistics
    print(f"\nGram matrix statistics:")
    print(f"Training gram matrix - min: {np.min(train_gram_np):.6f}, max: {np.max(train_gram_np):.6f}, mean: {np.mean(train_gram_np):.6f}")
    print(f"Test gram matrix - min: {np.min(test_gram_np):.6f}, max: {np.max(test_gram_np):.6f}, mean: {np.mean(test_gram_np):.6f}")
    
    # Train Kernel Ridge Regression with hyperparameter tuning
    print("Training Kernel Ridge Regression...")
    
    # Try different alpha values for regularization
    alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alpha_values:
        print(f"  Training with alpha = {alpha}...")
        krr = KernelRidge(alpha=alpha, kernel='precomputed')
        krr.fit(train_gram_np, y_train)
        
        # Predict on training set for validation
        y_pred_train = krr.predict(train_gram_np)
        r2 = r2_score(y_train, y_pred_train)
        
        print(f"    Training R² score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_alpha = alpha
    
    print(f"\n✓ Best alpha: {best_alpha} (R² = {best_score:.4f})")
    
    # Train final model with best alpha
    print("Training final model...")
    final_krr = KernelRidge(alpha=best_alpha, kernel='precomputed')
    final_krr.fit(train_gram_np, y_train)
    
    # Predict and evaluate
    print("Making predictions on test set...")
    y_pred = final_krr.predict(test_gram_np)
    
    # Calculate multiple metrics for multi-output regression
    # Reshape for sklearn metrics (flatten the time dimension)
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    mape = mean_absolute_percentage_error(y_test_flat, y_pred_flat) * 100
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    r2_test = r2_score(y_test_flat, y_pred_flat)
    
    print(f"\n=== Test Results ===")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2_test:.4f}")
    
    # Print some predictions vs actual values for first meter
    print(f"\nSample predictions for first meter (first 10 timestamps):")
    print("Timestamp\tActual\tPredicted")
    for i in range(min(10, y_test.shape[1])):
        print(f"{i*15:3d} min\t{y_test[0, i]:.3f}\t{y_pred[0, i]:.3f}")
    
    # Calculate per-meter metrics
    meter_mape = []
    meter_r2 = []
    for i in range(y_test.shape[0]):
        meter_mape.append(mean_absolute_percentage_error(y_test[i], y_pred[i]) * 100)
        meter_r2.append(r2_score(y_test[i], y_pred[i]))
    
    print(f"\nPer-meter statistics:")
    print(f"Average MAPE: {np.mean(meter_mape):.2f}% ± {np.std(meter_mape):.2f}%")
    print(f"Average R²: {np.mean(meter_r2):.4f} ± {np.std(meter_r2):.4f}")
    print(f"Best meter R²: {np.max(meter_r2):.4f}")
    print(f"Worst meter R²: {np.min(meter_r2):.4f}")
    
    # Print model summary
    print(f"\n=== Model Summary ===")
    print(f"Number of dual coefficients: {len(final_krr.dual_coef_)}")
    print(f"Alpha used: {best_alpha}")
    print(f"Polynomial order: 6")
    print(f"Total computation time: {train_time + test_time:.2f} seconds")
    
    print(f"\n✓ Electricity analysis completed successfully!")

if __name__ == "__main__":
    main() 