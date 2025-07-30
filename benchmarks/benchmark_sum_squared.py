from powersig.jax.jax_config import configure_jax
configure_jax()

import jax
import jax.numpy as jnp
import time
import numpy as np

# Configure JAX
jax.config.update('jax_disable_jit', False)

# Device selection logic
def get_device():
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Try to find CUDA device 1 first
    cuda_devices = [d for d in devices if d.platform == 'gpu']
    if len(cuda_devices) > 1:
        print(f"Using CUDA device 1: {cuda_devices[1]}")
        return cuda_devices[1]
    elif len(cuda_devices) == 1:
        print(f"Using CUDA device 0: {cuda_devices[0]}")
        return cuda_devices[0]
    else:
        cpu_devices = [d for d in devices if d.platform == 'cpu']
        if cpu_devices:
            print(f"Using CPU device: {cpu_devices[0]}")
            return cpu_devices[0]
        else:
            print("No devices found, using default")
            return devices[0]

# Get the selected device
selected_device = get_device()

# Define the two methods
@jax.jit
def sum_squared_diff_dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    diff = x - y
    # For multi-dimensional arrays, we need to flatten or use einsum
    if diff.ndim == 1:
        return jnp.dot(diff, diff, precision = jax.lax.Precision.HIGHEST)
    else:
        # Flatten the arrays and then compute dot product
        diff_flat = diff.reshape(-1)
        return jnp.dot(diff_flat, diff_flat)

@jax.jit
def sum_squared_diff_sum_square(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(x - y))

@jax.jit
def sum_squared_diff_einsum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    diff = x - y
    return jnp.einsum('...i,...i->', diff, diff, precision = jax.lax.Precision.HIGHEST)

# Benchmark function
def benchmark_methods(array_sizes=[100, 1000, 10000, 100000, 1000000], num_runs=10000, dim=2):
    print(f"\nBenchmarking on device: {selected_device}")
    print(f"Vector dimensionality: {dim}")
    print("=" * 60)
    print(f"{'Size':<10} {'Dot (ms)':<12} {'Sum Square (ms)':<15} {'Einsum (ms)':<12} {'Best':<10}")
    print("-" * 60)
    
    for size in array_sizes:
        # Generate random data on the selected device with specified dimensionality
        x = jax.device_put(jnp.array(np.random.randn(size, dim), dtype=jnp.float64), selected_device)
        y = jax.device_put(jnp.array(np.random.randn(size, dim), dtype=jnp.float64), selected_device)
        
        # Warm up JIT
        _ = sum_squared_diff_dot(x, y)
        _ = sum_squared_diff_sum_square(x, y)
        _ = sum_squared_diff_einsum(x, y)
        
        # Benchmark dot method
        start_time = time.time()
        for _ in range(num_runs):
            result_dot = sum_squared_diff_dot(x, y)
        dot_time = (time.time() - start_time) * 1000 / num_runs
        
        # Benchmark sum square method
        start_time = time.time()
        for _ in range(num_runs):
            result_sum_square = sum_squared_diff_sum_square(x, y)
        sum_square_time = (time.time() - start_time) * 1000 / num_runs
        
        # Benchmark einsum method
        start_time = time.time()
        for _ in range(num_runs):
            result_einsum = sum_squared_diff_einsum(x, y)
        einsum_time = (time.time() - start_time) * 1000 / num_runs
        
        # Verify results are the same
        assert jnp.allclose(result_dot, result_sum_square), "Dot and Sum Square results don't match!"
        assert jnp.allclose(result_dot, result_einsum), "Dot and Einsum results don't match!"
        
        # Find the fastest method
        times = [dot_time, sum_square_time, einsum_time]
        methods = ['Dot', 'Sum Square', 'Einsum']
        best_method = methods[times.index(min(times))]
        
        print(f"{size:<10} {dot_time:<12.3f} {sum_square_time:<15.3f} {einsum_time:<12.3f} {best_method:<10}")
    
    print("-" * 60)
    print("Lower time is better. Best shows the fastest method for each size.")

# Run benchmark
if __name__ == "__main__":
    # You can change the dimensionality here
    benchmark_methods(dim=6) 