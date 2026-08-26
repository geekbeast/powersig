import torch


def linear_kernel(
    x2: torch.Tensor, x1: torch.Tensor, y2: torch.Tensor, y1: torch.Tensor
) -> torch.Tensor:
    return torch.sum((x2 - x1) * (y2 - y1), dim=-1)


def rbf_fn(diff: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    sq_dist = torch.sum(diff * diff, dim=-1)
    return torch.exp(-sq_dist / (2.0 * (bandwidth**2)))


def rbf_kernel(
    x2: torch.Tensor,
    x1: torch.Tensor,
    y2: torch.Tensor,
    y1: torch.Tensor,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    kx2y2 = rbf_fn(x2 - y2, bandwidth)
    kx1y1 = rbf_fn(x1 - y1, bandwidth)
    kx2y1 = rbf_fn(x2 - y1, bandwidth)
    kx1y2 = rbf_fn(x1 - y2, bandwidth)
    return (kx2y2 - kx1y2) - (kx2y1 - kx1y1)
