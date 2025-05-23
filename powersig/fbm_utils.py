import torch
from fbm import FBM


from typing import Tuple


def fractional_brownian_motion(n_steps, n_paths=1, cuda: bool = True, dim: int = 1, hurst: float = 0.5, T: float = 1.0) -> Tuple[torch.tensor, float]:
    """
    Generate multi-dimensional fractional Brownian motion paths using the fbm package.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        cuda (bool): Whether to use CUDA (GPU) or CPU
        dim (int): Dimension of the fractional Brownian motion (default: 1)
        hurst (float): Hurst parameter, must be in (0,1). H=0.5 gives standard Brownian motion.
        T (float): Total time period (default: 1.0)

    Returns:
        torch.Tensor: Fractional Brownian motion paths of shape (n_paths, n_steps + 1, dim)
        float: Time step size (dt)
    """
    dt = T/n_steps
    device = 'cuda' if cuda else 'cpu'

    # Initialize output tensor
    fbm_paths = torch.zeros(n_paths, n_steps + 1, dim, device=device, dtype=torch.float64)

    # Generate paths for each dimension
    for i in range(n_paths):
        for d in range(dim):
            # Create FBM instance
            f = FBM(n=n_steps, hurst=hurst, length=T, method='daviesharte')
            # Generate path
            path = f.fbm()
            # Convert to tensor and store
            fbm_paths[i, :, d] = torch.tensor(path, device=device, dtype=torch.float64)

    return fbm_paths, dt

