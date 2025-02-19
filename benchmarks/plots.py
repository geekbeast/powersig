from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR, 
    POWERSIG_RESULTS,
    SIGKERNEL_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    SIGNATURE_KERNEL,
    LENGTH
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_plots():
    data = load_csvs()
    results = get_accuracy(data)
    plot_mape(results['lengths'], results['values'])

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
    
    # Return both lengths and values for each method
    return {
        'lengths': ksig_df[LENGTH].to_numpy(),
        'values': {
            'ksig': ksig_df[SIGNATURE_KERNEL].to_numpy(),
            'ksig_pde': ksig_pde_df[SIGNATURE_KERNEL].to_numpy(),
            'powersig': powersig_df[SIGNATURE_KERNEL].to_numpy(),
            'sigkernel': sigkernel_df[SIGNATURE_KERNEL].to_numpy()
        }
    }

def plot_mape(lengths, values):
    # Compute MAPE for each method using ksig as baseline
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
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, ksig_pde_mapes, 'b-o', label='KSig PDE')
    plt.plot(lengths, powersig_mapes, 'r-o', label='PowerSig')
    plt.plot(lengths, sigkernel_mapes, 'g-o', label='SigKernel')
    
    plt.xscale('log', base=2)  # Since lengths increase by powers of 2
    plt.yscale('log')  # Log scale for MAPE values
    
    plt.xlabel('Length of Time Series')
    plt.ylabel('MAPE (relative to KSig)')
    plt.title('Mean Absolute Percentage Error vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'mape_comparison.png'))
    plt.close()
    
    return ksig_pde_mape, powersig_mape, sigkernel_mape

if __name__ == "__main__": 
    generate_plots()
