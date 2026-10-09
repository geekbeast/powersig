import torch
from powersig.torch.utils import fractional_brownian_motion
from powersig.torch.algorithm import PowerSigTorch

def main():
    # Generate fBM paths
    n_steps = 1000
    n_paths = 2
    hurst = 0.7

    # Generate fBM using the torch wrapper
    fbm_paths, dt = fractional_brownian_motion(
        n_steps=n_steps,
        n_paths=n_paths,
        hurst=hurst,
        dim=1
    )

    # Initialize PowerSigTorch with polynomial order 8
    powersig = PowerSigTorch(order=8)

    # Compute the signature kernel
    kernel_matrix = powersig(fbm_paths)

    print("Shape of fBM paths:", tuple(fbm_paths.shape))
    print("Shape of kernel matrix:", tuple(kernel_matrix.shape))
    print("\nKernel matrix:")
    print(kernel_matrix)

if __name__ == "__main__":
    main()
