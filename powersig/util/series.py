from typing import Tuple

import torch

def double_length(tensor: torch.Tensor) -> torch.Tensor:
    new_tensor = torch.zeros([tensor.shape[0] * 2, tensor.shape[1] * 2],dtype=torch.float64,device=tensor.device)
    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
    return new_tensor

def resize(tensor: torch.Tensor, new_shape) -> torch.Tensor:
    new_tensor = torch.zeros(new_shape,dtype=torch.float64,device=tensor.device)
    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
    return new_tensor

@torch.compile(mode="max-autotune", fullgraph=True)
def torch_compute_derivative(X):
    return (X[1:,:] - X[:-1,:]) * (X.shape[0] - 1)

@torch.compile(mode="max-autotune", fullgraph=True)
def torch_compute_dot_prod(X,Y):
    assert X.shape == Y.shape, "X and Y must have the same shape."
    return (X*Y).sum()

# @torch.compile(mode="max-autotune", fullgraph=True,disable=False)
@torch.compile(dynamic=True)
def torch_compute_dot_prod_batch(X,Y):
    return (X*Y).sum(dim=len(X.shape)-1)
    # return torch.einsum("bd,bd->b", X, Y)

@torch.compile(mode="max-autotune", fullgraph=True,dynamic=True)
def torch_compute_derivative_batch(X) -> torch.Tensor:
    return (X[:, 1:, :] - X[:, :-1, :])*(X.shape[1]-1)
