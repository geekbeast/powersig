import unittest
import numpy as np
import time 
from numba import cuda

from powersig.powersig_cuda import compute_rho_diagonal
from powersig.util.cuda import get_number_threads


class TestPowerSigCuda(unittest.TestCase):
    def test_compute_rho_diagonal(self):
        # dX = np.asarray([[2,2],[4,4],[6,6],[8,8]]).astype(np.float64)
        # dY = np.asarray([[2,2],[4,4],[6,6],[8,8]]).astype(np.float64)

        dX = np.asarray([[2, 4, 6, 8]]).astype(np.float64)
        dY = np.asarray([[2, 4, 6, 8]]).astype(np.float64)
        # dX = np.random.random((1000,2047)).astype(np.float64)
        # dY = np.random.random((1000,2047)).astype(np.float64)

        actual_rho = (dX*dY).sum(1)
        print(f"Actual rho: {actual_rho}")

        dX = cuda.to_device(dX)
        dY = cuda.to_device(dY)

        rho_diagonal = cuda.to_device(np.zeros((dX.shape[0],), dtype=np.float64))
        threadsperblock = (32, get_number_threads(dX.shape[1]//32))
        blockspergrid = (dX.shape[0])
        print(f"Threads per block: {threadsperblock}")
        print(f"Blocks per grid: {blockspergrid}")

        #CUDA warmup
        compute_rho_diagonal[blockspergrid, threadsperblock](dX, dY, rho_diagonal)
        cuda.synchronize()

        start_time = time.time()
        for i in range(10000):
            compute_rho_diagonal[blockspergrid,threadsperblock](dX, dY, rho_diagonal)
        cuda.synchronize()
        print(f"Time taken: {(time.time() - start_time)/10000}s")
        print(f"Rho diagonal: {rho_diagonal.copy_to_host()}")

        assert np.allclose(actual_rho, rho_diagonal,rtol=0, atol=1e-10), "Rho diagonal is not close to actual rho"

    # def test_compute_sigkernel_diagonal(self):


if __name__ == "__main__":
    unittest.main()