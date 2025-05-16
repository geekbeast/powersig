import jax
import jax.numpy as jnp
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_percentage_error
from powersig.jax import PowerSigJax
import pandas as pd
import os
import requests
import zipfile
import io

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
    # Get JAX device 1
    devices = jax.devices("cuda")
    device = devices[2] if len(devices) > 1 else devices[0]
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
    train_data = train_df.iloc[:, 1:].values.astype(np.float32).T  # (n_meters, n_train_timestamps)
    test_data = test_df.iloc[:, 1:].values.astype(np.float32).T    # (n_meters, n_test_timestamps)
    print(f"Number of meters: {train_data.shape[0]}, Train timestamps: {train_data.shape[1]}, Test timestamps: {test_data.shape[1]}")
    
    # Use only 256 timestamps for both training and testing
    train_len = min(256, train_data.shape[1])
    test_len = min(256, test_data.shape[1])
    X_train = train_data[:, :train_len]
    X_test = test_data[:, :test_len]
    
    print(f"Using {train_len} timestamps for training and {test_len} timestamps for testing")
    
    # Add singleton feature dimension: (n_meters, n_timestamps, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Convert to JAX arrays and place on device 1, using float64
    X_train_jax = jnp.array(X_train, dtype=jnp.float64, device=device)
    X_test_jax = jnp.array(X_test, dtype=jnp.float64, device=device)
    
    print(f"Training data shape: {X_train_jax.shape}")
    print(f"Test data shape: {X_test_jax.shape}")
    
    # For demonstration, let's use the mean of the next 96 timestamps in the test set as the target
    # (You can adjust this to your actual prediction target)
    y_train = train_data[:, train_len:train_len+96].mean(axis=1) if train_data.shape[1] > train_len+96 else train_data[:, -96:].mean(axis=1)
    y_test = test_data[:, test_len:test_len+96].mean(axis=1) if test_data.shape[1] > test_len+96 else test_data[:, -96:].mean(axis=1)
    
    # Initialize PowerSigJax
    powersig = PowerSigJax(order=8)
    
    # Compute gram matrices
    print("Computing gram matrices...")
    train_gram = powersig.compute_gram_matrix(X_train_jax, X_train_jax)
    test_gram = powersig.compute_gram_matrix(X_test_jax, X_train_jax)
    
    # Convert to numpy for sklearn
    train_gram_np = np.array(train_gram)
    test_gram_np = np.array(test_gram)
    
    # Train Kernel Ridge Regression
    print("Training Kernel Ridge Regression...")
    krr = KernelRidge(alpha=1e-3, kernel='precomputed')
    krr.fit(train_gram_np, y_train)
    
    # Predict and evaluate
    print("Making predictions...")
    y_pred = krr.predict(test_gram_np)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    print(f"Test MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main() 