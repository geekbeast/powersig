from typing import Tuple
import torch
import numpy as np

from powersig.torch.utils import fractional_brownian_motion


def set_seed(seed: int):
    """
    Set random seed for both NumPy and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_brownian_motion(n_steps, n_paths=1, cuda: bool = True, dim: int = 1, T: float = 1.0) -> Tuple[torch.tensor, float]:
    """
    Generate multi-dimensional Brownian motion paths.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        cuda (bool): Whether to use CUDA (GPU) or CPU
        dim (int): Dimension of the Brownian motion (default: 1)
        T (float): Total time period (default: 1.0)

    Returns:
        torch.Tensor: Brownian paths of shape (n_paths, n_steps + 1, dim)
        float: Time step size (dt)
    """
    dt = T/n_steps
    device = 'cuda' if cuda else 'cpu'
    
    # Generate random increments for each dimension
    dW = torch.normal(
        mean=0, 
        std=torch.sqrt(torch.tensor(dt, device=device, dtype=torch.float64)),
        size=(n_paths, n_steps, dim),
        device=device, 
        dtype=torch.float64
    )

    # Compute cumulative sum to get Brownian motion
    zeros = torch.zeros(n_paths, 1, dim, device=device, dtype=torch.float64)
    W = torch.cat([zeros, torch.cumsum(dW, dim=1)], dim=1)

    return W, dt

if __name__== '__main__':
    # Example usage:
    n_steps = 1000
    n_paths = 5
    T = 1.0  # Total time period

    # Set global seed for reproducibility
    set_seed(42)

    # Test standard Brownian motion
    paths, dt = generate_brownian_motion(n_steps, n_paths, T=T)

    # Test fractional Brownian motion
    fbm_paths, dt = fractional_brownian_motion(n_steps, n_paths, hurst=0.7, t=T)

    # If you want to visualize it:
    import matplotlib.pyplot as plt

    t = torch.arange(n_steps + 1) * dt
    print(f"t_max = {t.max()}")
    
    # Plot standard Brownian motion
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(n_paths):
        plt.plot(t, paths[i].cpu().numpy())
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Standard Brownian Motion Paths')
    
    # Plot fractional Brownian motion
    plt.subplot(1, 2, 2)
    for i in range(n_paths):
        plt.plot(t, fbm_paths[i].cpu().numpy())
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Fractional Brownian Motion Paths (H=0.25)')
    plt.tight_layout()
    plt.show()

    plt.savefig("bm_paths.svg")
