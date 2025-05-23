from typing import Tuple

import jax
import torch
import jax.numpy as jnp
import powersig.util.fbm_utils


def fractional_brownian_motion(n_steps, n_paths=1, device: jax.Device=None, dim: int = 1, hurst: float = 0.5, t: float = 1.0) -> \
Tuple[jnp.array, float]:
    """
    Generate multi-dimensional fractional Brownian motion paths using the fbm package.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        device (device): Device to use for the resulting tensor
        dim (int): Dimension of the fractional Brownian motion (default: 1)
        hurst (float): Hurst parameter, must be in (0,1). H=0.5 gives standard Brownian motion.
        t (float): Total time period (default: 1.0)

    Returns:
        torch.Tensor: Fractional Brownian motion paths of shape (n_paths, n_steps + 1, dim)
        float: Time step size (dt)
    """
    dt = t / n_steps

    # Get fBM paths and convert from CuPy to NumPy to JAX
    fbm_paths, _ = powersig.util.fbm_utils.fractional_brownian_motion(n_steps, n_paths, dim=dim, hurst=hurst, t=t)
    if hasattr(fbm_paths, 'get'):  # If it's a CuPy array
        fbm_paths = fbm_paths.get()
    
    return jnp.array(fbm_paths).to_device(device), dt
