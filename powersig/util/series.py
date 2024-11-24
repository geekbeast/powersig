import torch


def torch_compute_derivative(X):
    diff = (X[1:,:] - X[:-1,:])
    if X.shape[0] == 1:
        return diff
    else:
        return diff #/ ( 1 / ( X.shape[0] - 1 ) )

def torch_compute_dot_prod(X,Y):
    assert X.shape == Y.shape, "X and Y must have the same shape."
    return (X*Y).sum()

def torch_compute_derivative_batch(X) -> torch.Tensor:
    # X = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # |X| = 9
    # X[0] @ t = 0
    # X[1] @ t = 1
    diff = (X[:, 1:, :] - X[:, :-1, :])
    if X.shape[0] == 1:
        return diff
    else:
        return diff / ( 1 / ( X.shape[1] - 1 ) )