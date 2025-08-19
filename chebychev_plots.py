import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_powersig_condition_numbers():
    """Plot PowerSig condition numbers with normalized vs unnormalized comparison"""
    
    # Read the CSV data
    try:
        df = pd.read_csv('chebychev.csv')
        print("Loaded data from chebychev.csv")
    except FileNotFoundError:
        print("chebychev.csv not found. Please run chebychev.py first.")
        return
    
    # Filter for PowerSig data only
    powersig_df = df[df['Algorithm'] == 'PowerSig']
    
    # Debug: print the filtered data
    print("PowerSig data:")
    print(powersig_df)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots for each kernel
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('PowerSig Condition Numbers: Normalized vs Unnormalized', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    datasets = ['Original', 'Scaled', 'Chebyshev', 'Unity']
    
    # Plot 1: PowerSig Linear Kernel
    ax1 = axes[0]
    powersig_linear = powersig_df[powersig_df['Kernel'] == 'Linear']
    
    # Separate normalized and unnormalized data
    normalized_data = []
    unnormalized_data = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            orig_data = powersig_linear[powersig_linear['Data'] == 'Original']['Condition_Number'].iloc[0]
            normalized_data.append(np.nan)  # No normalized version for original
            unnormalized_data.append(orig_data)
        else:
            # For other datasets, get both normalized and unnormalized
            norm_data = powersig_linear[(powersig_linear['Data'] == dataset) & (powersig_linear['Normalized'] == True)]['Condition_Number']
            unnorm_data = powersig_linear[(powersig_linear['Data'] == dataset) & (powersig_linear['Normalized'] == False)]['Condition_Number']
            
            print(f"Dataset: {dataset}")
            print(f"  Normalized data: {norm_data.values}")
            print(f"  Unnormalized data: {unnorm_data.values}")
            
            if len(norm_data) > 0:
                normalized_data.append(norm_data.iloc[0])
            else:
                normalized_data.append(np.nan)
                
            if len(unnorm_data) > 0:
                unnormalized_data.append(unnorm_data.iloc[0])
            else:
                unnormalized_data.append(np.nan)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot bars
    bars1 = ax1.bar(x - width/2, normalized_data, width, label='Normalized', 
                     color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, unnormalized_data, width, label='Unnormalized', 
                     color='lightcoral', alpha=0.8)
    
    ax1.set_title('PowerSig Linear Kernel', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Condition Number (log scale)')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: PowerSig RBF Kernel
    ax2 = axes[1]
    powersig_rbf = powersig_df[powersig_df['Kernel'] == 'RBF']
    
    # Separate normalized and unnormalized data
    normalized_data = []
    unnormalized_data = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            orig_data = powersig_rbf[powersig_rbf['Data'] == 'Original']['Condition_Number'].iloc[0]
            normalized_data.append(np.nan)  # No normalized version for original
            unnormalized_data.append(orig_data)
        else:
            # For other datasets, get both normalized and unnormalized
            norm_data = powersig_rbf[(powersig_rbf['Data'] == dataset) & (powersig_rbf['Normalized'] == True)]['Condition_Number']
            unnorm_data = powersig_rbf[(powersig_rbf['Data'] == dataset) & (powersig_rbf['Normalized'] == False)]['Condition_Number']
            
            if len(norm_data) > 0:
                normalized_data.append(norm_data.iloc[0])
            else:
                normalized_data.append(np.nan)
                
            if len(unnorm_data) > 0:
                unnormalized_data.append(unnorm_data.iloc[0])
            else:
                unnormalized_data.append(np.nan)
    
    bars3 = ax2.bar(x - width/2, normalized_data, width, label='Normalized', 
                     color='lightgreen', alpha=0.8)
    bars4 = ax2.bar(x + width/2, unnormalized_data, width, label='Unnormalized', 
                     color='gold', alpha=0.8)
    
    ax2.set_title('PowerSig RBF Kernel', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Condition Number (log scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('powersig_condition_numbers.png', dpi=300, bbox_inches='tight')
    print("PowerSig plot saved as powersig_condition_numbers.png")
    
    # Show the plot
    plt.show()

def plot_ksig_condition_numbers():
    """Plot KSig condition numbers with normalized vs unnormalized comparison"""
    
    # Read the CSV data
    try:
        df = pd.read_csv('chebychev.csv')
        print("Loaded data from chebychev.csv")
    except FileNotFoundError:
        print("chebychev.csv not found. Please run chebychev.py first.")
        return
    
    # Filter for KSig data only
    ksig_df = df[df['Algorithm'] == 'KSigPDE']
    
    # Debug: print the filtered data
    print("KSig data:")
    print(ksig_df)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots for each kernel
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('KSig Condition Numbers: Normalized vs Unnormalized', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    datasets = ['Original', 'Scaled', 'Chebyshev', 'Unity']
    
    # Plot 1: KSig Linear Kernel
    ax1 = axes[0]
    ksig_linear = ksig_df[ksig_df['Kernel'] == 'Linear']
    
    # Separate normalized and unnormalized data
    normalized_data = []
    unnormalized_data = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            orig_data = ksig_linear[ksig_linear['Data'] == 'Original']['Condition_Number'].iloc[0]
            normalized_data.append(np.nan)  # No normalized version for original
            unnormalized_data.append(orig_data)
        else:
            # For other datasets, get both normalized and unnormalized
            norm_data = ksig_linear[(ksig_linear['Data'] == dataset) & (ksig_linear['Normalized'] == True)]['Condition_Number']
            unnorm_data = ksig_linear[(ksig_linear['Data'] == dataset) & (ksig_linear['Normalized'] == False)]['Condition_Number']
            
            if len(norm_data) > 0:
                normalized_data.append(norm_data.iloc[0])
            else:
                normalized_data.append(np.nan)
                
            if len(unnorm_data) > 0:
                unnormalized_data.append(unnorm_data.iloc[0])
            else:
                unnormalized_data.append(np.nan)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot bars
    bars1 = ax1.bar(x - width/2, normalized_data, width, label='Normalized', 
                     color='mediumorchid', alpha=0.8)
    bars2 = ax1.bar(x + width/2, unnormalized_data, width, label='Unnormalized', 
                     color='orange', alpha=0.8)
    
    ax1.set_title('KSig Linear Kernel', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Condition Number (log scale)')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: KSig RBF Kernel
    ax2 = axes[1]
    ksig_rbf = ksig_df[ksig_df['Kernel'] == 'RBF']
    
    # Separate normalized and unnormalized data
    normalized_data = []
    unnormalized_data = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            orig_data = ksig_rbf[ksig_rbf['Data'] == 'Original']['Condition_Number'].iloc[0]
            normalized_data.append(np.nan)  # No normalized version for original
            unnormalized_data.append(orig_data)
        else:
            # For other datasets, get both normalized and unnormalized
            norm_data = ksig_rbf[(ksig_rbf['Data'] == dataset) & (ksig_rbf['Normalized'] == True)]['Condition_Number']
            unnorm_data = ksig_rbf[(ksig_rbf['Data'] == dataset) & (ksig_rbf['Normalized'] == False)]['Condition_Number']
            
            if len(norm_data) > 0:
                normalized_data.append(norm_data.iloc[0])
            else:
                normalized_data.append(np.nan)
                
            if len(unnorm_data) > 0:
                unnormalized_data.append(unnorm_data.iloc[0])
            else:
                unnormalized_data.append(np.nan)
    
    bars3 = ax2.bar(x - width/2, normalized_data, width, label='Normalized', 
                     color='teal', alpha=0.8)
    bars4 = ax2.bar(x + width/2, unnormalized_data, width, label='Unnormalized', 
                     color='crimson', alpha=0.8)
    
    ax2.set_title('KSig RBF Kernel', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Condition Number (log scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('ksig_condition_numbers.png', dpi=300, bbox_inches='tight')
    print("KSig plot saved as ksig_condition_numbers.png")
    
    # Show the plot
    plt.show()

def create_summary_grid_plot():
    """Create a 2x2 grid summary plot with shared y-axes for easy comparison"""
    
    # Read the CSV data
    try:
        df = pd.read_csv('chebychev.csv')
        print("Loaded data from chebychev.csv")
    except FileNotFoundError:
        print("chebychev.csv not found. Please run chebychev.py first.")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), 
                             gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
    fig.suptitle('Condition Numbers Summary: PowerSig vs KSig (Normalized vs Unnormalized)', 
                 fontsize=18, fontweight='bold')
    
    # Prepare data for plotting
    datasets = ['Original', 'Scaled', 'Chebyshev', 'Unity']
    
    # Top row: Linear kernels (shared y-axis)
    # Top left: Linear kernels - Unnormalized
    ax_top_left = axes[0, 0]
    ax_top_right = axes[0, 1]
    
    # Get PowerSig and KSig linear unnormalized data
    ps_linear_unnorm = []
    ks_linear_unnorm = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            ps_linear_unnorm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'Linear') & (df['Data'] == 'Original')]['Condition_Number'].iloc[0])
            ks_linear_unnorm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'Linear') & (df['Data'] == 'Original')]['Condition_Number'].iloc[0])
        else:
            # For other datasets, get unnormalized
            ps_linear_unnorm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'Linear') & (df['Data'] == dataset) & (df['Normalized'] == False)]['Condition_Number'].iloc[0])
            ks_linear_unnorm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'Linear') & (df['Data'] == dataset) & (df['Normalized'] == False)]['Condition_Number'].iloc[0])
    
    # Get PowerSig and KSig linear normalized data
    ps_linear_norm = []
    ks_linear_norm = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data has no normalized version
            ps_linear_norm.append(np.nan)
            ks_linear_norm.append(np.nan)
        else:
            # For other datasets, get normalized
            ps_linear_norm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'Linear') & (df['Data'] == dataset) & (df['Normalized'] == True)]['Condition_Number'].iloc[0])
            ks_linear_norm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'Linear') & (df['Data'] == dataset) & (df['Normalized'] == True)]['Condition_Number'].iloc[0])
    
    # Bottom row: RBF kernels (shared y-axis)
    # Bottom left: RBF kernels - Unnormalized
    ax_bottom_left = axes[1, 0]
    ax_bottom_right = axes[1, 1]
    
    # Get PowerSig and KSig RBF unnormalized data
    ps_rbf_unnorm = []
    ks_rbf_unnorm = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data is always unnormalized
            ps_rbf_unnorm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'RBF') & (df['Data'] == 'Original')]['Condition_Number'].iloc[0])
            ks_rbf_unnorm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'RBF') & (df['Data'] == 'Original')]['Condition_Number'].iloc[0])
        else:
            # For other datasets, get unnormalized
            ps_rbf_unnorm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'RBF') & (df['Data'] == dataset) & (df['Normalized'] == False)]['Condition_Number'].iloc[0])
            ks_rbf_unnorm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'RBF') & (df['Data'] == dataset) & (df['Normalized'] == False)]['Condition_Number'].iloc[0])
    
    # Get PowerSig and KSig RBF normalized data
    ps_rbf_norm = []
    ks_rbf_norm = []
    
    for dataset in datasets:
        if dataset == 'Original':
            # Original data has no normalized version
            ps_rbf_norm.append(np.nan)
            ks_rbf_norm.append(np.nan)
        else:
            # For other datasets, get normalized
            ps_rbf_norm.append(df[(df['Algorithm'] == 'PowerSig') & (df['Kernel'] == 'RBF') & (df['Data'] == dataset) & (df['Normalized'] == True)]['Condition_Number'].iloc[0])
            ks_rbf_norm.append(df[(df['Algorithm'] == 'KSigPDE') & (df['Kernel'] == 'RBF') & (df['Data'] == dataset) & (df['Normalized'] == True)]['Condition_Number'].iloc[0])
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot 1: Top Left - Linear kernels, Unnormalized
    bars1 = ax_top_left.bar(x - width/2, ps_linear_unnorm, width, label='PowerSig', 
                             color='skyblue', alpha=0.8)
    bars2 = ax_top_left.bar(x + width/2, ks_linear_unnorm, width, label='KSig', 
                             color='mediumorchid', alpha=0.8)
    
    ax_top_left.set_title('Linear Kernels - Unnormalized', fontweight='bold', fontsize=14)
    ax_top_left.set_ylabel('Condition Number (log scale)')
    ax_top_left.set_yscale('log')
    ax_top_left.set_xticks(x)
    ax_top_left.set_xticklabels(datasets, rotation=45)
    ax_top_left.legend()
    ax_top_left.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax_top_left.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Top Right - Linear kernels, Normalized
    bars3 = ax_top_right.bar(x - width/2, ps_linear_norm, width, label='PowerSig', 
                              color='lightcoral', alpha=0.8)
    bars4 = ax_top_right.bar(x + width/2, ks_linear_norm, width, label='KSig', 
                              color='orange', alpha=0.8)
    
    ax_top_right.set_title('Linear Kernels - Normalized', fontweight='bold', fontsize=14)
    ax_top_right.set_ylabel('Condition Number (log scale)')
    ax_top_right.set_yscale('log')
    ax_top_right.set_xticks(x)
    ax_top_right.set_xticklabels(datasets, rotation=45)
    ax_top_right.legend()
    ax_top_right.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax_top_right.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Bottom Left - RBF kernels, Unnormalized
    bars5 = ax_bottom_left.bar(x - width/2, ps_rbf_unnorm, width, label='PowerSig', 
                                color='lightgreen', alpha=0.8)
    bars6 = ax_bottom_left.bar(x + width/2, ks_rbf_unnorm, width, label='KSig', 
                                color='teal', alpha=0.8)
    
    ax_bottom_left.set_title('RBF Kernels - Unnormalized', fontweight='bold', fontsize=14)
    ax_bottom_left.set_ylabel('Condition Number (log scale)')
    ax_bottom_left.set_yscale('log')
    ax_bottom_left.set_xticks(x)
    ax_bottom_left.set_xticklabels(datasets, rotation=45)
    ax_bottom_left.legend()
    ax_bottom_left.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax_bottom_left.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Bottom Right - RBF kernels, Normalized
    bars7 = ax_bottom_right.bar(x - width/2, ps_rbf_norm, width, label='PowerSig', 
                                 color='gold', alpha=0.8)
    bars8 = ax_bottom_right.bar(x + width/2, ks_rbf_norm, width, label='KSig', 
                                 color='crimson', alpha=0.8)
    
    ax_bottom_right.set_title('RBF Kernels - Normalized', fontweight='bold', fontsize=14)
    ax_bottom_right.set_ylabel('Condition Number (log scale)')
    ax_bottom_right.set_yscale('log')
    ax_bottom_right.set_xticks(x)
    ax_bottom_right.set_xticklabels(datasets, rotation=45)
    ax_bottom_right.legend()
    ax_bottom_right.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax_bottom_right.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1e}', ha='center', va='bottom', fontsize=9)
    
    # Share y-axis for top row (linear kernels)
    ax_top_left.get_shared_y_axes().joined = True
    ax_top_right.set_yticklabels([])  # Remove y-tick labels for right plot
    
    # Share y-axis for bottom row (RBF kernels)
    ax_bottom_left.get_shared_y_axes().joined = True
    ax_bottom_right.set_yticklabels([])  # Remove y-tick labels for right plot
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('summary_grid_plot.png', dpi=300, bbox_inches='tight')
    print("Summary grid plot saved as summary_grid_plot.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    print("Generating all plots...")
    
    # Always generate all plots
    plot_powersig_condition_numbers()
    plot_ksig_condition_numbers()
    create_summary_grid_plot()
    
    print("All plots generated successfully!")
