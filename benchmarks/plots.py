from benchmarks.configuration import (
    BENCHMARKS_RESULTS_DIR, 
    POWERSIG_RESULTS,
    SIGKERNEL_RESULTS,
    KSIG_RESULTS,
    KSIG_PDE_RESULTS,
    SIGNATURE_KERNEL,
    LENGTH,
    GPU_MEMORY,
    DURATION,
    HURST,
    RUN_ID
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors

def generate_plots():
    # data = load_csvs()
    # results = get_accuracy(data)
    # plot_memory_usage(results['lengths'], data)
    # plot_duration(results['lengths'], data)
    # plot_memory_and_duration(results['lengths'], data)

    accuracy_data = load_accuracy_csvs()
    accuracy_results = get_accuracy(accuracy_data)
    
    # Use accuracy data for MAPE plot
    plot_mape(accuracy_results['lengths'], accuracy_results['mape_data'])
    
    rough_data = load_rough_csvs()
    # Add rough time series plots
    plot_rough_mape_vs_hurst(rough_data)
    # plot_rough_mape_heatmap(rough_data)
    
    # Create side-by-side plots
    plot_rough_and_accuracy_side_by_side(rough_data, accuracy_results)

def compute_mape(predictions, actuals):
    return np.mean(np.abs((predictions - actuals) / actuals))

def load_csvs():
    benchmark_files = [
        POWERSIG_RESULTS,
        SIGKERNEL_RESULTS,
        KSIG_RESULTS,
        KSIG_PDE_RESULTS,
        # POLYSIG_RESULTS  # Removed PolySig
    ]
    data = {}
    for file in benchmark_files:
        data[file] = pd.read_csv(os.path.join(BENCHMARKS_RESULTS_DIR, file))
        
    return data

def load_accuracy_csvs():
    benchmark_files = [
        POWERSIG_RESULTS,
        SIGKERNEL_RESULTS,
        KSIG_RESULTS,
        KSIG_PDE_RESULTS,
    ]
    data = {}
    for file in benchmark_files:
        data[file] = pd.read_csv(os.path.join(BENCHMARKS_RESULTS_DIR, 'accuracy', file))
        
    return data

def load_rough_csvs():
    benchmark_files = [
        POWERSIG_RESULTS,
        SIGKERNEL_RESULTS,
        KSIG_RESULTS,
        KSIG_PDE_RESULTS,
    ]
    data = {}
    for file in benchmark_files:
        data[file] = pd.read_csv(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough', file))
        print(data[file])
        
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
    # polysig_mapes = []  # Removed PolySig
    
    # Calculate MAPE for each length
    for length in unique_lengths:
        length_mask = truncated_lengths == length
        
        # Get all values for this length
        ksig_vals = values['ksig'][length_mask]
        ksig_pde_vals = values['ksig_pde'][length_mask]
        powersig_vals = values['powersig'][length_mask]
        # polysig_vals = values['polysig'][length_mask]  # Removed PolySig
        
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
    # polysig_mapes = np.array(polysig_mapes)  # Removed PolySig
    
    # Plot lines without error bars
    plt.plot(unique_lengths, ksig_pde_mapes, 'b-o', label='KSig PDE')
    plt.plot(unique_lengths, powersig_mapes, 'r-o', label='PowerSig')
    # plt.plot(unique_lengths, polysig_mapes, 'g-o', label='PolySig')  # Removed PolySig
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.xlabel('Time Series Length')
    plt.ylabel('MAPE (relative to KSig)')
    plt.title('Mean Absolute Percentage Error')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'mape_comparison.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'mape_comparison.svg'))
    plt.close()

def plot_memory_usage(lengths, data):
    plt.figure(figsize=(10, 6))
    
    # Get full memory usage data and compute statistics by length, excluding first run
    ksig_df = data[KSIG_RESULTS][data[KSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean', 'std']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS][data[KSIG_PDE_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean', 'std']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS][data[POWERSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean', 'std']
    }).reset_index()
    
    
    # Plot means with error bars
    plt.errorbar(
        ksig_df[LENGTH], 
        ksig_df[GPU_MEMORY]['mean'],
        yerr=ksig_df[GPU_MEMORY]['std'],
        fmt='g-o', label='KSig (CuPy)', capsize=5
    )
    
    plt.errorbar(
        ksig_pde_df[LENGTH], 
        ksig_pde_df[GPU_MEMORY]['mean'],
        yerr=ksig_pde_df[GPU_MEMORY]['std'],
        fmt='b-o', label='KSig PDE (CuPy)', capsize=5
    )
    
    plt.errorbar(
        powersig_df[LENGTH], 
        powersig_df[GPU_MEMORY]['mean'],
        yerr=powersig_df[GPU_MEMORY]['std'],
        fmt='r-o', label='PowerSig', capsize=5
    )

    
    plt.xscale('log', base=2)
    # plt.yscale('log')  # Removed logarithmic y-axis
    
    plt.xlabel('Time Series Length')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_comparison.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_comparison.svg'))
    plt.close()

def plot_duration(lengths, data):
    plt.figure(figsize=(10, 6))
    
    # Compute statistics by length for each method, excluding first run (run_id = 0)
    ksig_df = data[KSIG_RESULTS][data[KSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS][data[KSIG_PDE_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS][data[POWERSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean', 'std']
    }).reset_index()
    
    # polysig_df = data[POLYSIG_RESULTS][data[POLYSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({  # Removed PolySig
    #     DURATION: ['mean', 'std']
    # }).reset_index()
    
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
    
    # plt.errorbar(  # Removed PolySig
    #     polysig_df[LENGTH], 
    #     polysig_df[DURATION]['mean'],
    #     yerr=polysig_df[DURATION]['std'],
    #     fmt='m-o', label='PolySig', capsize=5
    # )
    
    plt.xscale('log', base=2)
    # plt.yscale('log')  # Removed logarithmic y-axis
    
    plt.xlabel('Time Series Length')
    plt.ylabel('Duration (seconds)')
    plt.title('Computation Time vs Time Series Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'duration_comparison.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'duration_comparison.svg'))
    plt.close()

def plot_memory_and_duration(lengths, data):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get data and compute statistics for memory plot, excluding first run
    ksig_df = data[KSIG_RESULTS][data[KSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS][data[KSIG_PDE_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS][data[POWERSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        GPU_MEMORY: ['mean']
    }).reset_index()
    
    # Memory plot
    ax1.plot(ksig_df[LENGTH], ksig_df[GPU_MEMORY]['mean'], 'g-o', label='KSig')
    ax1.plot(ksig_pde_df[LENGTH], ksig_pde_df[GPU_MEMORY]['mean'], 'b-o', label='KSig PDE')
    ax1.plot(powersig_df[LENGTH], powersig_df[GPU_MEMORY]['mean'], 'r-o', label='PowerSig')
    
    ax1.set_xscale('log', base=2)
    # Removed log scale for y-axis
    #ax1.set_yscale('log')
    ax1.set_xlabel('Time Series Length')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage')
    ax1.legend()
    
    # Get data and compute statistics for duration plot, excluding first run
    ksig_df = data[KSIG_RESULTS][data[KSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    ksig_pde_df = data[KSIG_PDE_RESULTS][data[KSIG_PDE_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    powersig_df = data[POWERSIG_RESULTS][data[POWERSIG_RESULTS][RUN_ID] > 0].groupby(LENGTH).agg({
        DURATION: ['mean']
    }).reset_index()
    
    # Duration plot
    ax2.plot(ksig_df[LENGTH], ksig_df[DURATION]['mean'], 'g-o', label='KSig')
    ax2.plot(ksig_pde_df[LENGTH], ksig_pde_df[DURATION]['mean'], 'b-o', label='KSig PDE')
    ax2.plot(powersig_df[LENGTH], powersig_df[DURATION]['mean'], 'r-o', label='PowerSig')
    
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time Series Length')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_title('Runtime')
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_and_duration_comparison.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'memory_and_duration_comparison.svg'))
    plt.close()

def plot_rough_mape_vs_hurst(data):
    # Choose a fixed length (e.g., 513)
    fixed_length = 51
    
    # Get data for each implementation
    ksig_df = data[KSIG_RESULTS]
    ksig_pde_df = data[KSIG_PDE_RESULTS]
    powersig_df = data[POWERSIG_RESULTS]
    # polysig_df = data[POLYSIG_RESULTS]  # Removed PolySig
    
    # Filter for fixed length
    ksig_df = ksig_df[ksig_df[LENGTH] == fixed_length]
    ksig_pde_df = ksig_pde_df[ksig_pde_df[LENGTH] == fixed_length]
    powersig_df = powersig_df[powersig_df[LENGTH] == fixed_length]
    # polysig_df = polysig_df[polysig_df[LENGTH] == fixed_length]  # Removed PolySig
    
    # Calculate MAPE for each Hurst index
    hurst_values = sorted(ksig_df[HURST].unique())
    ksig_pde_mapes = []
    powersig_mapes = []
    # polysig_mapes = []  # Removed PolySig

    for h in hurst_values:
        ksig_vals = ksig_df[ksig_df[HURST] == h][SIGNATURE_KERNEL].values
        ksig_pde_vals = ksig_pde_df[ksig_pde_df[HURST] == h][SIGNATURE_KERNEL].values
        powersig_vals = powersig_df[powersig_df[HURST] == h][SIGNATURE_KERNEL].values
        # polysig_vals = polysig_df[polysig_df[HURST] == h][SIGNATURE_KERNEL].values  # Removed PolySig
        
        ksig_pde_mapes.append(np.mean(np.abs((ksig_pde_vals - ksig_vals) / ksig_vals)))
        powersig_mapes.append(np.mean(np.abs((powersig_vals - ksig_vals) / ksig_vals)))
        # polysig_mapes.append(np.mean(np.abs((polysig_vals - ksig_vals) / ksig_vals)))  # Removed PolySig
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(hurst_values, ksig_pde_mapes, 'b-o', label='KSig PDE')
    plt.plot(hurst_values, powersig_mapes, 'r-o', label='PowerSig')
    # plt.plot(hurst_values, polysig_mapes, 'g-o', label='PolySig')  # Removed PolySig
    
    plt.xlabel('Hurst Index')
    plt.ylabel('MAPE (relative to KSig)')
    plt.title(f'Mean Absolute Percentage Error vs Hurst Index')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_mape_vs_hurst.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_mape_vs_hurst.svg'))
    plt.close()

def plot_rough_mape_heatmap(data):
    # Get data for each implementation
    ksig_df = data[KSIG_RESULTS]
    ksig_pde_df = data[KSIG_PDE_RESULTS]
    powersig_df = data[POWERSIG_RESULTS]
    # polysig_df = data[POLYSIG_RESULTS]  # Removed PolySig
    
    # Get unique lengths and Hurst indices
    lengths = sorted(ksig_df[LENGTH].unique())
    hurst_values = sorted(ksig_df[HURST].unique())
    
    # Find the minimum number of data points across all implementations
    min_length = min(
        len(ksig_df),
        len(ksig_pde_df),
        len(powersig_df),
        # len(polysig_df)  # Removed PolySig
    )
    
    # Truncate all dataframes to the minimum length
    ksig_df = ksig_df.iloc[:min_length]
    ksig_pde_df = ksig_pde_df.iloc[:min_length]
    powersig_df = powersig_df.iloc[:min_length]
    # polysig_df = polysig_df.iloc[:min_length]  # Removed PolySig
    
    # Create heatmap data for each implementation
    ksig_pde_mape = np.zeros((len(hurst_values), len(lengths)))
    powersig_mape = np.zeros((len(hurst_values), len(lengths)))
    # polysig_mape = np.zeros((len(hurst_values), len(lengths)))  # Removed PolySig
    
    # Calculate MAPE for each combination
    for i, h in enumerate(hurst_values):
        for j, l in enumerate(lengths):
            # Get values for this combination
            ksig_vals = ksig_df[(ksig_df[HURST] == h) & (ksig_df[LENGTH] == l)][SIGNATURE_KERNEL].values
            ksig_pde_vals = ksig_pde_df[(ksig_pde_df[HURST] == h) & (ksig_pde_df[LENGTH] == l)][SIGNATURE_KERNEL].values
            powersig_vals = powersig_df[(powersig_df[HURST] == h) & (powersig_df[LENGTH] == l)][SIGNATURE_KERNEL].values
            # polysig_vals = polysig_df[(polysig_df[HURST] == h) & (polysig_df[LENGTH] == l)][SIGNATURE_KERNEL].values  # Removed PolySig
            
            # Calculate MAPE for each implementation
            if len(ksig_vals) > 0:
                if len(ksig_pde_vals) > 0:
                    ksig_pde_mape[i, j] = np.mean(np.abs((ksig_pde_vals - ksig_vals) / ksig_vals))
                if len(powersig_vals) > 0:
                    powersig_mape[i, j] = np.mean(np.abs((powersig_vals - ksig_vals) / ksig_vals))
                # if len(polysig_vals) > 0:  # Removed PolySig
                #     polysig_mape[i, j] = np.mean(np.abs((polysig_vals - ksig_vals) / ksig_vals))
    
    # Create subplots for each implementation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Changed to 1x2 grid
    fig.suptitle('MAPE Heatmaps for Different Implementations', fontsize=16)
    
    # Select a subset of Hurst values for ticks
    n_ticks = 10  # Number of ticks to show
    tick_indices = np.linspace(0, len(hurst_values)-1, n_ticks, dtype=int)
    tick_labels = [f'{hurst_values[i]:.2f}' for i in tick_indices]
    
    # Find the global min and max for consistent color scaling
    valid_data = [ksig_pde_mape, powersig_mape]  # Removed PolySig
    # Filter out zeros and find min/max of non-zero values
    non_zero_data = [d[d > 0] for d in valid_data]
    if any(len(d) > 0 for d in non_zero_data):
        vmin = min(np.min(d) for d in non_zero_data if len(d) > 0)
        vmax = max(np.max(d) for d in non_zero_data if len(d) > 0)
    else:
        # If all values are zero, use small non-zero values for visualization
        vmin = 1e-10
        vmax = 1e-5
    
    # Plot heatmaps with log scale
    sns.heatmap(ksig_pde_mape, ax=axes[0], xticklabels=lengths, yticklabels=tick_labels,
                cmap='viridis', cbar_kws={'label': 'MAPE'},
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    axes[0].set_title('KSig PDE')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Hurst Index')
    axes[0].set_yticks(tick_indices)
    
    sns.heatmap(powersig_mape, ax=axes[1], xticklabels=lengths, yticklabels=tick_labels,
                cmap='viridis', cbar_kws={'label': 'MAPE'},
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    axes[1].set_title('PowerSig')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Hurst Index')
    axes[1].set_yticks(tick_indices)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_mape_heatmaps.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_mape_heatmaps.svg'))
    plt.close()

def plot_rough_and_accuracy_side_by_side(rough_data, accuracy_results):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy MAPE
    truncated_lengths = accuracy_results['lengths'][:len(accuracy_results['mape_data']['ksig'])]
    unique_lengths = np.unique(truncated_lengths)
    
    ksig_pde_mapes = []
    powersig_mapes = []
    
    # Calculate MAPE for each length
    for length in unique_lengths:
        length_mask = truncated_lengths == length
        
        # Get all values for this length
        ksig_vals = accuracy_results['mape_data']['ksig'][length_mask]
        ksig_pde_vals = accuracy_results['mape_data']['ksig_pde'][length_mask]
        powersig_vals = accuracy_results['mape_data']['powersig'][length_mask]
        
        # Calculate individual MAPEs for this length
        ksig_pde_length_mapes = np.abs((ksig_pde_vals - ksig_vals) / ksig_vals)
        powersig_length_mapes = np.abs((powersig_vals - ksig_vals) / ksig_vals)
        
        # Store mean of MAPEs for this length
        ksig_pde_mapes.append(np.mean(ksig_pde_length_mapes))
        powersig_mapes.append(np.mean(powersig_length_mapes))
    
    ax1.plot(unique_lengths, ksig_pde_mapes, 'b-o', label='KSig PDE')
    ax1.plot(unique_lengths, powersig_mapes, 'r-o', label='PowerSig')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Time Series Length')
    ax1.set_ylabel('MAPE (relative to KSig)')
    ax1.set_title('MAPE vs Time Series Length')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Plot 2: Rough Hurst MAPE
    fixed_length = 51
    ksig_df = rough_data[KSIG_RESULTS]
    ksig_pde_df = rough_data[KSIG_PDE_RESULTS]
    powersig_df = rough_data[POWERSIG_RESULTS]
    
    # Filter for fixed length and Hurst < 0.4
    ksig_df = ksig_df[(ksig_df[LENGTH] == fixed_length) & (ksig_df[HURST] < 0.4)]
    ksig_pde_df = ksig_pde_df[(ksig_pde_df[LENGTH] == fixed_length) & (ksig_pde_df[HURST] < 0.4)]
    powersig_df = powersig_df[(powersig_df[LENGTH] == fixed_length) & (powersig_df[HURST] < 0.4)]
    
    # Calculate MAPE for each Hurst index
    hurst_values = sorted(ksig_df[HURST].unique())
    ksig_pde_mapes = []
    powersig_mapes = []

    for h in hurst_values:
        ksig_vals = ksig_df[ksig_df[HURST] == h][SIGNATURE_KERNEL].values
        ksig_pde_vals = ksig_pde_df[ksig_pde_df[HURST] == h][SIGNATURE_KERNEL].values
        powersig_vals = powersig_df[powersig_df[HURST] == h][SIGNATURE_KERNEL].values
        
        ksig_pde_mapes.append(np.mean(np.abs((ksig_pde_vals - ksig_vals) / ksig_vals)))
        powersig_mapes.append(np.mean(np.abs((powersig_vals - ksig_vals) / ksig_vals)))
    
    ax2.plot(hurst_values, ksig_pde_mapes, 'b-o', label='KSig PDE')
    ax2.plot(hurst_values, powersig_mapes, 'r-o', label='PowerSig')
    ax2.set_xlabel('Hurst Index')
    ax2.set_ylabel('MAPE (relative to KSig)')
    ax2.set_title('MAPE vs Hurst Index')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_and_accuracy_mape_comparison.png'))
    plt.savefig(os.path.join(BENCHMARKS_RESULTS_DIR, 'rough_and_accuracy_mape_comparison.svg'))
    plt.close()

if __name__ == "__main__": 
    generate_plots()
