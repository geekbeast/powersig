import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_methods(sizes, batch_sizes, repeats=10):
    """
    Benchmark different methods for batched dot products.
    
    Args:
        sizes: List of vector sizes to test
        batch_sizes: List of batch sizes to test
        repeats: Number of repeats to average over
    
    Returns:
        Dictionary with timing results
    """
    results = {
        'manual_sum': np.zeros((len(batch_sizes), len(sizes))),
        'einsum': np.zeros((len(batch_sizes), len(sizes))),
        'bmm': np.zeros((len(batch_sizes), len(sizes))),
        'torch_dot': np.zeros((len(batch_sizes), len(sizes)))
    }
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    for b_idx, batch_size in enumerate(batch_sizes):
        for s_idx, size in enumerate(sizes):
            # Initialize tensors
            X = torch.randn(batch_size, size, device=device)
            Y = torch.randn(batch_size, size, device=device)
            
            # Warm-up
            _ = (X * Y).sum(dim=-1)
            _ = torch.einsum('bi,bi->b', X, Y)
            _ = torch.bmm(X.unsqueeze(1), Y.unsqueeze(2)).squeeze()
            
            # Manual sum method
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start = time.time()
                _ = (X * Y).sum(dim=-1)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times.append(time.time() - start)
            results['manual_sum'][b_idx, s_idx] = np.median(times) * 1000  # convert to ms
            
            # Einstein summation method
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start = time.time()
                _ = torch.einsum('bi,bi->b', X, Y)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times.append(time.time() - start)
            results['einsum'][b_idx, s_idx] = np.median(times) * 1000  # convert to ms
            
            # Batch matrix multiplication method
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start = time.time()
                _ = torch.bmm(X.unsqueeze(1), Y.unsqueeze(2)).squeeze()
                torch.cuda.synchronize() if device.type == 'cuda' else None
                times.append(time.time() - start)
            results['bmm'][b_idx, s_idx] = np.median(times) * 1000  # convert to ms
            
            # Torch built-in dot product (for completion)
            if batch_size <= 1:  # torch.dot doesn't handle batches
                results['torch_dot'][b_idx, s_idx] = np.nan
            else:
                times = []
                for _ in range(repeats):
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start = time.time()
                    _ = torch.stack([torch.dot(x, y) for x, y in zip(X, Y)])
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    times.append(time.time() - start)
                results['torch_dot'][b_idx, s_idx] = np.median(times) * 1000  # convert to ms
                
            # Print progress
            print(f"Completed batch_size={batch_size}, size={size}")
            
    return results

def plot_results(results, sizes, batch_sizes):
    """Plot benchmark results"""
    fig, axes = plt.subplots(len(batch_sizes), 1, figsize=(10, 4*len(batch_sizes)))
    if len(batch_sizes) == 1:
        axes = [axes]
        
    for b_idx, batch_size in enumerate(batch_sizes):
        ax = axes[b_idx]
        for method in results:
            if method == 'torch_dot' and batch_size <= 1:
                continue
            ax.plot(sizes, results[method][b_idx], label=method, marker='o')
        
        ax.set_title(f'Batch Size: {batch_size}')
        ax.set_xlabel('Vector Size')
        ax.set_ylabel('Time (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

def validate_methods():
    """Validate that all methods produce the same results"""
    batch_size = 16
    size = 32
    X = torch.randn(batch_size, size)
    Y = torch.randn(batch_size, size)
    
    result1 = (X * Y).sum(dim=-1)
    result2 = torch.einsum('bi,bi->b', X, Y)
    result3 = torch.bmm(X.unsqueeze(1), Y.unsqueeze(2)).squeeze()
    result4 = torch.stack([torch.dot(x, y) for x, y in zip(X, Y)])
    
    print("Validation differences:")
    print(f"manual_sum vs einsum: {torch.max(torch.abs(result1 - result2))}")
    print(f"manual_sum vs bmm: {torch.max(torch.abs(result1 - result3))}")
    print(f"manual_sum vs torch_dot: {torch.max(torch.abs(result1 - result4))}")
    
    return torch.allclose(result1, result2) and torch.allclose(result1, result3) and torch.allclose(result1, result4)

if __name__ == "__main__":
    print("Validating methods...")
    if validate_methods():
        print("All methods produce equivalent results.")
    else:
        print("WARNING: Methods produce different results!")
    
    # Test parameters
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    batch_sizes = [1, 16, 128, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    
    print("\nRunning benchmarks...")
    results = benchmark_methods(sizes, batch_sizes)
    
    print("\nPlotting results...")
    plot_results(results, sizes, batch_sizes)
    
    # Print summary of results
    print("\nSummary of fastest methods:")
    for b_idx, batch_size in enumerate(batch_sizes):
        for s_idx, size in enumerate(sizes):
            times = {method: results[method][b_idx, s_idx] for method in results if method != 'torch_dot' or batch_size > 1}
            fastest = min(times, key=times.get)
            print(f"Batch size: {batch_size}, Vector size: {size} - Fastest: {fastest} ({times[fastest]:.3f} ms)")
    
    
    # Print summary of results with detailed comparisons
    print("\nDetailed comparison of methods:")
    for b_idx, batch_size in enumerate(batch_sizes):
        print(f"\nBatch size: {batch_size}")
        for s_idx, size in enumerate(sizes):
            print(f"  Vector size: {size}")
            valid_methods = [method for method in results if method != 'torch_dot' or batch_size > 1]
            times = {method: results[method][b_idx, s_idx] for method in valid_methods}
            fastest = min(times, key=times.get)
            fastest_time = times[fastest]
            
            # Print all methods with times and speedup ratios
            for method in valid_methods:
                time_ms = times[method]
                speedup = time_ms / fastest_time if method != fastest else 1.0
                if method == fastest:
                    print(f"    {method:10s}: {time_ms:.3f} ms (fastest)")
                else:
                    print(f"    {method:10s}: {time_ms:.3f} ms ({speedup:.2f}x slower)")
            
            # Compare manual_sum vs bmm specifically
            manual_time = times['manual_sum']
            bmm_time = times['bmm']
            if manual_time < bmm_time:
                print(f"    manual_sum is {bmm_time/manual_time:.2f}x faster than bmm")
            else:
                print(f"    bmm is {manual_time/bmm_time:.2f}x faster than manual_sum")