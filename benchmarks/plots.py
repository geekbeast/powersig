from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR, 
    POWERSIG_RESULTS,
    SIGKERNEL_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    SIGNATURE_KERNEL,
    LENGTH,
    GPU_MEMORY,
    CUPY_MEMORY,
    DURATION
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_plots():
    data = load_csvs()
    results = get_accuracy(data)
    plot_mape(results['lengths'], results['mape_data'])
    plot_memory_usage(results['lengths'], data)
    plot_duration(results['lengths'], data)
    plot_memory_and_duration(results['lengths'], data)

def compute_mape(predictions, actuals):
    return np.mean(np.abs((predictions - actuals) / actuals))

def load_csvs():
    benchmark_files = [
        POWERSIG_RESULTS,
        SIGKERNEL_RESULTS,
        KSIG_RESULTS,
        KSIG_PDE_RESULTS
    ]
    data = {}
    for file in benchmark_files:
        data[file] = pd.read_csv(os.path.join(BENCHMARKS_RESULTS_DIR, file))
        
    return data

def get_accuracy(data):
    # Get both length and signature kernel values
    ksig_df = data[KSIG_RESULTS]
    ksig_pde_df = data[KSIG_PDE_RESULTS]
    powersig_df = data[POWERSIG_RESULTS]
    sigkernel_df = data[SIGKERNEL_RESULTS]
    
    # Find common lengths where we have all data for MAPE calculation
    count = min(len(ksig_df), len(ksig_pde_df))
    count = min(count, len(powersig_df))
    # count = min(count, len(sigkernel_df))
    
    # Only truncate the values used for MAPE comparison
    return {
        'lengths': ksig_df[LENGTH].to_numpy(),  # Keep full length array
        'mape_data': {
            'ksig': ksig_df[SIGNATURE_KERNEL].to_numpy()[:count],
            'ksig_pde': ksig_pde_df[SIGNATURE_KERNEL].to_numpy()[:count],
            'powersig': powersig_df[SIGNATURE_KERNEL].to_numpy()[:count],
            'sigkernel': sigkernel_df[SIGNATURE_KERNEL].to_numpy()[:count]
        }
    }

def plot_mape(lengths, values):
    # Use truncated lengths for MAPE plot
    truncated_lengths = lengths[:len(values['ksig'])]
    unique_lengths = np.unique(truncated_lengths)
    
    ksig_pde_mapes = []
    powersig_mapes = []
    
    # Calculate MAPE for each length
    for length in unique_lengths:
        length_mask = truncated_lengths == length
        
        # Get all values for this length
        ksig_vals = values['ksig'][length_mask]
        ksig_pde_vals = values['ksig_pde'][length_mask]
        powersig_vals = values['powersig'][length_mask]
        
        # Calculate individual MAPEs for this length
        ksig_pde_length_mapes = np.abs((ksig_pde_vals - ksig_vals) / ksig_vals)
        powersig_length_mapes = np.abs((powersig_vals - ksig_vals) / ksig_vals)
        
        # Store mean of MAPEs for this length
        ksig_pde_mapes.append(np.mean(ksig_pde_length_mapes))
        powersig_mapes.append(np.mean(powersig_length_mapes))
    
    # Print overall MAPE
    print(f"Overall MAPE relative to ksig:")
    print(f"KSig PDE: {np.mean(ksig_pde_mapes):.2%}")
    print(f"PowerSig: {np.mean(powersig_mapes):.2%}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy arrays for easier manipulation
    unique_lengths = np.array(unique_lengths)
    ksig_pde_mapes = np.array(ksig_pde_mapes)
    powersig_mapes = np.array(powersig_mapes)
    
    # Plot lines without error bars
    plt.plot(unique_lengths, ksig_pde_mapes, 'b-o', label='KSig PDE')
    plt.plot(unique_lengths, powersig_mapes, 'r-o', label='PowerSig')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('MAPE (relative to KSig)')
    plt.title('Mean Absolute Percentage Error')
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'mape_comparison.png'))
    plt.close()

def plot_memory_usage(lengths, data):
    plt.figure(figsize=(10, 6))
    
    # Get full memory usage data and compute statistics by length
    ksig_df = data[KSIG_RESULTS].groupby(LENGTH).agg({
        CUPY_MEMORY: ['mean', 'std']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS].groupby(LENGTH).agg({
        CUPY_MEMORY: ['mean', 'std']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean', 'std']
    }).reset_index()
    
    # Plot means with error bars
    plt.errorbar(
        ksig_df[LENGTH], 
        ksig_df[CUPY_MEMORY]['mean'],
        yerr=ksig_df[CUPY_MEMORY]['std'],
        fmt='g-o', label='KSig (CuPy)', capsize=5
    )
    
    plt.errorbar(
        ksig_pde_df[LENGTH], 
        ksig_pde_df[CUPY_MEMORY]['mean'],
        yerr=ksig_pde_df[CUPY_MEMORY]['std'],
        fmt='b-o', label='KSig PDE (CuPy)', capsize=5
    )
    
    plt.errorbar(
        powersig_df[LENGTH], 
        powersig_df[GPU_MEMORY]['mean'],
        yerr=powersig_df[GPU_MEMORY]['std'],
        fmt='r-o', label='PowerSig (PyTorch)', capsize=5
    )
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_comparison.png'))
    plt.close()

def plot_duration(lengths, data):
    plt.figure(figsize=(10, 6))
    
    # Compute statistics by length for each method
    ksig_df = data[KSIG_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    # Plot means with error bars
    plt.errorbar(
        ksig_df[LENGTH], 
        ksig_df[DURATION]['mean'],
        yerr=ksig_df[DURATION]['std'],
        fmt='g-o', label='KSig', capsize=5
    )
    
    plt.errorbar(
        ksig_pde_df[LENGTH], 
        ksig_pde_df[DURATION]['mean'],
        yerr=ksig_pde_df[DURATION]['std'],
        fmt='b-o', label='KSig PDE', capsize=5
    )
    
    plt.errorbar(
        powersig_df[LENGTH], 
        powersig_df[DURATION]['mean'],
        yerr=powersig_df[DURATION]['std'],
        fmt='r-o', label='PowerSig', capsize=5
    )
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('Duration (seconds)')
    plt.title('Computation Time vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'duration_comparison.png'))
    plt.close()

def plot_memory_and_duration(lengths, data):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get data and compute statistics for memory plot
    ksig_df = data[KSIG_RESULTS].groupby(LENGTH).agg({
        CUPY_MEMORY: ['mean']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS].groupby(LENGTH).agg({
        CUPY_MEMORY: ['mean']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean']
    }).reset_index()
    
    # Memory plot
    ax1.plot(ksig_df[LENGTH], ksig_df[CUPY_MEMORY]['mean'], 'g-o', label='KSig (CuPy)')
    ax1.plot(ksig_pde_df[LENGTH], ksig_pde_df[CUPY_MEMORY]['mean'], 'b-o', label='KSig PDE (CuPy)')
    ax1.plot(powersig_df[LENGTH], powersig_df[GPU_MEMORY]['mean'], 'r-o', label='PowerSig (PyTorch)')
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Length of Time Series')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage')
    ax1.legend()
    
    # Get data and compute statistics for duration plot
    ksig_df = data[KSIG_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    # Duration plot
    ax2.plot(ksig_df[LENGTH], ksig_df[DURATION]['mean'], 'g-o', label='KSig')
    ax2.plot(ksig_pde_df[LENGTH], ksig_pde_df[DURATION]['mean'], 'b-o', label='KSig PDE')
    ax2.plot(powersig_df[LENGTH], powersig_df[DURATION]['mean'], 'r-o', label='PowerSig')
    
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_title('Runtime')
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_and_duration_comparison.png'))
    plt.close()

if __name__ == "__main__": 
    generate_plots()
