# Power-series based computation of Signature Kernels
Using ADM-derived Neumann series to compute signature kernels.

## Installation 
```bash
pip install git+https://github.com/geekbeast/powersig.git
```

Requires Python 3.12+ 

Requires PyTorch 2.5+, JAX 0.6.0+, or cupy 13.4.1+ depending on which implementation you prefer. 

## Getting Started
```python
import jax.numpy as jnp
from powersig.jax.utils import fractional_brownian_motion
from powersig.jax.algorithm import PowerSigJax

def main():
    # Generate fBM paths
    n_steps = 1000
    n_paths = 2
    hurst = 0.7
    
    # Generate fBM using the jax wrapper
    fbm_paths, dt = fractional_brownian_motion(
        n_steps=n_steps,
        n_paths=n_paths,
        hurst=hurst,
        dim=1
    )
    
    # Initialize PowerSigJax with polynomial order 8
    powersig = PowerSigJax(order=8)
    
    # Compute the signature kernel
    kernel_matrix = powersig(fbm_paths)
    
    print("Shape of fBM paths:", fbm_paths.shape)
    print("Shape of kernel matrix:", kernel_matrix.shape)
    print("\nKernel matrix:")
    print(kernel_matrix)

if __name__ == "__main__":
    main()
```
This example is also available in the repo under [examples](examples/simple.py).