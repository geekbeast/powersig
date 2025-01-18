import os
import pickle

import torch
from sigkernel import sigkernel

_batch, _len_x, _len_y, _dim = 1, 100000, 100000, 2
_fresh = True
torch.random.manual_seed(1)
static_kernel = sigkernel.LinearKernel()
dyadic_order = 5
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

def get_test_config(fresh=_fresh):
    if os.path.exists("tests.run"):
        return pickle.load(open("tests.run", "rb"))
    else:
        config = TestRun()
        pickle.dump(config, open("tests.run", "wb"))
        return config

class TestRun:
    def __init__(self, batch: int = _batch,
                 len_x: int = _len_x, len_y: int = _len_y, dim: int = _dim,
                 cuda: bool = False):
        self.batch = batch
        self.len_x = len_x
        self.len_y = len_y
        self.dim = dim
        self.cuda = cuda
        self.X = torch.rand((batch, len_x, dim), dtype=torch.float64) / 1000 # shape (batch,len_x,dim)
        self.Y = torch.rand((batch, len_y, dim), dtype=torch.float64)  # shape (batch,len_y,dim)
        self.Z = torch.rand((batch, len_x, dim), dtype=torch.float64)  # shape

        if self.cuda:
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.Z = self.Z.cuda()
