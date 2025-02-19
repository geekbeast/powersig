import torch

def double_length(tensor: torch.Tensor) -> torch.Tensor:
    new_tensor = torch.zeros([tensor.shape[0] * 2, tensor.shape[1] * 2],dtype=torch.float64,device=tensor.device)
    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
    return new_tensor

def resize(tensor: torch.Tensor, new_shape) -> torch.Tensor:
    new_tensor = torch.zeros(new_shape,dtype=torch.float64,device=tensor.device)
    new_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
    return new_tensor

def torch_compute_derivative(X):
    diff = (X[1:,:] - X[:-1,:])
    if X.shape[0] == 1:
        return diff
    else:
        return diff * ( X.shape[0] - 1 )

def torch_compute_dot_prod(X,Y):
    assert X.shape == Y.shape, "X and Y must have the same shape."
    return (X*Y).sum()

def torch_compute_dot_prod_batch(X,Y):
    return (X*Y).sum(dim=len(X.shape)-1)
    # return torch.einsum("bd,bd->b", X, Y)

def torch_compute_derivative_batch(X,dt:float|None = None) -> torch.Tensor:
    # X = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # |X| = 9
    # X[0] @ t = 0
    # X[1] @ t = 1
    diff = (X[:, 1:, :] - X[:, :-1, :])
    if dt!= None:
        diff/=dt
    elif X.shape[1] != 1:
        diff *= ( X.shape[1] - 1 )

    return diff