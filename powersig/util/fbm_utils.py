import numpy as np

from fbm import FBM


from typing import Tuple


def fractional_brownian_motion(n_steps, n_paths=1, cuda: bool = True, dim: int = 1, hurst: float = 0.5, t: float = 1.0) -> Tuple[np.ndarray, float]:

    """
    Generate multi-dimensional fractional Brownian motion paths using the fbm package.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        cuda (bool): Whether to use CUDA (GPU) or CPU
        dim (int): Dimension of the fractional Brownian motion (default: 1)
        hurst (float): Hurst parameter, must be in (0,1). H=0.5 gives standard Brownian motion.
        t (float): Total time period (default: 1.0)

    Returns:
        torch.Tensor: Fractional Brownian motion paths of shape (n_paths, n_steps + 1, dim)
        float: Time step size (dt)
    """
    dt = t / n_steps
    device = 'cuda' if cuda else 'cpu'

    # Initialize output tensor
    if cuda:
        fbm_paths = cupy.zeros((n_paths, n_steps + 1, dim), dtype=np.float64)
    else:
        fbm_paths = np.zeros((n_paths, n_steps + 1, dim), dtype=np.float64)

    # Generate paths for each dimension
    for i in range(n_paths):
        for d in range(dim):
            # Create FBM instance
            f = FBM(n=n_steps, hurst=hurst, length=t, method='daviesharte')
            # Generate path
            path = f.fbm()
            # Convert to appropriate array type and store
            if cuda:
                path = cupy.array(path)
            fbm_paths[i, :, d] = path

    return fbm_paths, dt