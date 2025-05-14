"""
JAX Configuration settings for PowerSig.
Import this module before any other JAX imports.
Optimized for high-performance systems with multiple cores, high memory, and GPUs.
"""
import os
import jax

# Assume high-end hardware
CPU_COUNT = 32  # High core count
TOTAL_MEMORY_GB = 64  # High memory (64GB)

def configure_jax():
    # Enable 64-bit precision
    jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_default_dtype_bits', '64')

    # Create XLA flags for GPU optimization
    xla_flags = [
        '--xla_gpu_autotune_level=4',
        '--xla_gpu_collective_permute_decomposer_threshold=128'
    ]

    # Set the XLA flags
    os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

    # Enable optimizations for speed
    jax.config.update('jax_disable_most_optimizations', False)
    jax.config.update('jax_exec_time_optimization_effort', 1.0)

    jax.config.update('jax_default_matmul_precision', 'highest')
    # Enable and configure compilation cache
    jax.config.update('jax_enable_compilation_cache', True)
    jax.config.update('jax_compilation_cache_max_size', 2048 * 1024 * 1024)  # 2GB cache

    # Set memory fitting effort for high-memory systems
    jax.config.update('jax_memory_fitting_effort', 0.3)

    # Set persistent cache directory
    if not os.path.exists('/tmp/jax_cache'):
        os.makedirs('/tmp/jax_cache', exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')

    # Print configuration summary
    print("JAX configured with high-performance settings:")
    print(f"- 64-bit enabled: {jax.config.jax_enable_x64}")
    print(f"- XLA Flags: {os.environ.get('XLA_FLAGS', '')}")
    print(f"- Default matmul precision: {jax.config.jax_default_matmul_precision}")

    try:
        devices = jax.devices()
        gpu_available = any(d.platform == 'gpu' for d in devices)
        print(f"- Available devices: {devices}")
        print(f"- GPU available: {gpu_available}")
    except:
        print("- Could not detect JAX devices") 