from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR, 
    POWERSIG_RESULTS,
    SIGKERNEL_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    SIGNATURE_KERNEL,
    LENGTH,
    PYTORCH_MEMORY,
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
    count = min(count, len(sigkernel_df))
    
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
    # Use only the truncated data for MAPE calculations
    ksig_pde_mape = compute_mape(values['ksig_pde'], values['ksig'])
    powersig_mape = compute_mape(values['powersig'], values['ksig'])
    sigkernel_mape = compute_mape(values['sigkernel'], values['ksig'])
    
    print(f"Overall MAPE relative to ksig:")
    print(f"KSig PDE: {ksig_pde_mape:.2%}")
    print(f"PowerSig: {powersig_mape:.2%}")
    print(f"SigKernel: {sigkernel_mape:.2%}")
    
    # Create point-wise MAPE for plotting
    ksig_pde_mapes = np.abs((values['ksig_pde'] - values['ksig']) / values['ksig'])
    powersig_mapes = np.abs((values['powersig'] - values['ksig']) / values['ksig'])
    sigkernel_mapes = np.abs((values['sigkernel'] - values['ksig']) / values['ksig'])
    
    # Use truncated lengths for MAPE plot
    truncated_lengths = lengths[:len(ksig_pde_mapes)]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(truncated_lengths, ksig_pde_mapes, 'b-o', label='KSig PDE')
    plt.plot(truncated_lengths, powersig_mapes, 'r-o', label='PowerSig')
    #plt.plot(truncated_lengths, sigkernel_mapes, 'g-o', label='SigKernel')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('MAPE (relative to KSig)')
    plt.title('Mean Absolute Percentage Error vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'mape_comparison.png'))
    plt.close()
    
    return ksig_pde_mape, powersig_mape, sigkernel_mape

def plot_memory_usage(lengths, data):
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Get full memory usage data
    ksig_df = data[KSIG_RESULTS]
    ksig_pde_df = data[KSIG_PDE_RESULTS]
    powersig_df = data[POWERSIG_RESULTS]
    
    # Plot using all available data points for each method
    plt.plot(ksig_df[LENGTH], ksig_df[CUPY_MEMORY], 'g-o', label='KSig (CuPy)')
    plt.plot(ksig_pde_df[LENGTH], ksig_pde_df[CUPY_MEMORY], 'b-o', label='KSig PDE (CuPy)')
    plt.plot(powersig_df[LENGTH], powersig_df[PYTORCH_MEMORY], 'r-o', label='PowerSig (PyTorch)')
    
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
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Get full duration data
    ksig_df = data[KSIG_RESULTS]
    ksig_pde_df = data[KSIG_PDE_RESULTS]
    powersig_df = data[POWERSIG_RESULTS]
    
    # Plot using all available data points for each method
    plt.plot(ksig_df[LENGTH], ksig_df[DURATION], 'g-o', label='KSig')
    plt.plot(ksig_pde_df[LENGTH], ksig_pde_df[DURATION], 'b-o', label='KSig PDE')
    plt.plot(powersig_df[LENGTH], powersig_df[DURATION], 'r-o', label='PowerSig')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('Duration (seconds)')
    plt.title('Computation Time vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'duration_comparison.png'))
    plt.close()

if __name__ == "__main__": 
    generate_plots()
