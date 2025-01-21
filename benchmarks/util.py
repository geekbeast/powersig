import csv
import os
from cupy.cuda.memory import MemoryPool
import torch


def save_stats(stats, filename):
    # Determine if file exists to decide if we need to write headers
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)


def generate_brownian_motion(n_steps, n_paths=1, cuda: bool = True):
    """
    Generate Brownian motion paths.

    Args:
        n_steps (int): Number of time steps
        n_paths (int): Number of paths to generate
        dt (float): Time step size

    Returns:
        torch.Tensor: Brownian paths of shape (n_paths, n_steps + 1)
    """


    dt = (4/n_steps)**2
    # Generate random increments
    dW = torch.normal(mean=0, std=torch.sqrt(torch.tensor(dt, device= 'cuda' if cuda else 'cpu', dtype=torch.float64)),
                      size=(n_paths, n_steps),device= 'cuda' if cuda else 'cpu', dtype=torch.float64)

    # Compute cumulative sum to get Brownian motion
    W = torch.cat([torch.zeros(n_paths, 1,device= 'cuda' if cuda else 'cpu', dtype=torch.float64), torch.cumsum(dW, dim=1)], dim=1)

    return W, dt


class TrackingMemoryPool(MemoryPool):
    def __init__(self, allocator=None):
        super().__init__(allocator)
        self.peak_usage = 0

    def reset_peak_usage(self):
        # Reset the peak usage to the current allocated memory (assumes memory is freed).
        self.peak_usage = self.used_bytes()

    def malloc(self, size):
        # Perform the allocation via parent class
        memptr = super().malloc(size)

        # Now check how much memory this pool is actually using in total
        self.peak_usage = max(self.peak_usage, self.used_bytes())

        return memptr

if __name__== '__main__':
    # Example usage:
    n_steps = 1000
    n_paths = 5
    dt = 1/(n_steps)

    paths, _ = generate_brownian_motion(n_steps, n_paths, dt)

    # If you want to visualize it:
    import matplotlib.pyplot as plt

    t = torch.arange(n_steps + 1) * dt
    print(f"t_max = {t.max()}")
    for i in range(n_paths):
        plt.plot(t, paths[i].cpu().numpy())
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Brownian Motion Paths')
    plt.show()