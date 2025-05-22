import unittest
import cupy as cp
import numpy as np
import torch

from powersig.powersig_cupy import batch_ADM_for_diagonal
from powersig.powersig_cupy import batch_compute_boundaries
from powersig.powersig_cupy import compute_vandermonde_vectors
from powersig.powersig_cupy import build_stencil
from powersig.powersig_cupy import batch_compute_gram_entry
from powersig.cupy_backend.cupy_series import cupy_compute_derivative_batch
import ksig
import ksig.static.kernels
from sigkernel import sigkernel

class TestBatchADMForDiagonal(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.rho_2x2 = cp.array([0.5, 0.7], dtype=cp.float64)  # batch_size = 2
        self.S_2x2 = cp.array([
            [10.0, 30.0],
            [100.0,300.0]
        ], dtype=cp.float64)
        self.T_2x2 = cp.array([
            [10.0, 20.0],
            [100.0,200.0]
        ], dtype=cp.float64)
        self.stencil_2x2 = cp.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=cp.float64)  # n x n stencil
        # Pre-allocated buffer for 2x2 case
        self.U_buf_2x2 = cp.empty((2, 2, 2), dtype=cp.float64)

    def test_2x2_shape(self):
        """Test that the output shape is correct for 2x2 case"""
        result = batch_ADM_for_diagonal(self.rho_2x2, self.U_buf_2x2, self.S_2x2, self.T_2x2, self.stencil_2x2)
        self.assertEqual(result.shape, (2, 2, 2))  # batch_size x n x n

    def test_2x2_first_batch(self):
        """Test the first batch element of 2x2 case"""
        result = batch_ADM_for_diagonal(self.rho_2x2, self.U_buf_2x2, self.S_2x2, self.T_2x2, self.stencil_2x2)

        # For the first batch (rho = 0.5):
        # First level: first row/column use rho^0 = 1.0
        # Second level: remaining element uses rho^1 = 0.5
        expected_first = cp.array([
            [1.0 * 10.0 * 1.0, 2.0 * 30.0 * 1.0],  # First row: rho^0
            [3.0 * 20.0 * 1.0, 4.0 * 10.0 * 0.5]   # First column: rho^0, remaining: rho^1
        ], dtype=cp.float64)
        print(f"Expected: {expected_first}")
        print(f"Result: {result[0]}")
        self.assertTrue(cp.allclose(result[0], expected_first, rtol=1e-5))


class TestBuildStencil(unittest.TestCase):
    def setUp(self):
        self.order = 4  # Using a small order for easier testing
        self.dtype = cp.float64

    def test_stencil_shape(self):
        """Test that the stencil has the correct shape"""
        stencil = build_stencil(self.order, self.dtype)
        self.assertEqual(stencil.shape, (self.order, self.order))

    def test_stencil_values(self):
        """Test that the stencil values are computed correctly"""
        stencil = build_stencil(self.order, self.dtype)
        
        # First row and column should be ones (from initialization)
        self.assertTrue(cp.allclose(stencil[0, :], cp.ones(self.order, dtype=self.dtype)))
        self.assertTrue(cp.allclose(stencil[:, 0], cp.ones(self.order, dtype=self.dtype)))
        
        # Check some specific values in the stencil
        # For order=4, after cumulative product, the stencil should look like:
        # 1  1  1  1
        # 1 1/1 1/2 1/6
        # 1 1/2 1/4 1/12
        # 1 1/6 1/12 1/36
        expected_values = cp.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 1/3],
            [1.0, 0.5, 0.25, 1/12],
            [1.0, 1/3, 1/12, 1/36]
        ], dtype=self.dtype)
        
        self.assertTrue(cp.allclose(stencil, expected_values, rtol=1e-5))


class TestComputeBoundaries(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.U_2x2 = cp.array([
            [[1.0, 2.0],
             [3.0, 4.0]],
            [[5.0, 6.0],
             [7.0, 8.0]]
        ], dtype=cp.float64)  # batch_size=2, n=2
        self.ds_2x2 = 0.5
        self.dt_2x2 = 0.5
        self.v_s_2x2, self.v_t_2x2 = compute_vandermonde_vectors(self.ds_2x2, self.dt_2x2, 2, self.U_2x2.dtype)
        # Pre-allocated buffers for S and T for 2x2 case
        self.S_buf_2x2 = cp.empty((3, 2), dtype=cp.float64)  # Max size needed is batch_size+1
        self.T_buf_2x2 = cp.empty((3, 2), dtype=cp.float64)  # Max size needed is batch_size+1

    def test_2x2_growing(self):
        """Test 2x2 case with both skip_first and skip_last=False (growing)"""
        S, T = batch_compute_boundaries(
            self.U_2x2, 
            self.S_buf_2x2, 
            self.T_buf_2x2, 
            self.v_s_2x2, 
            self.v_t_2x2, 
            skip_first=False, 
            skip_last=False
        )
        
        # For the 2x2 case with skip_first and skip_last=False:
        # S: [1,0,0] for the first element (bottom boundary), v_t.U for the rest
        # T: U.v_s for all batches, [1,0,0] for the added element (right boundary)
        # v_s = [1, 0.5]
        # v_t = [1, 0.5]
        
        # S values (v_t.U):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        # [1,0.5] * [5,7] = 1*5 + 0.5*7 = 8.5
        # [1,0.5] * [6,8] = 1*6 + 0.5*8 = 10
        expected_S = cp.array([
            [1, 0],     # Initial bottom boundary
            [2.5, 4],   # First batch
            [8.5, 10]   # Second batch
        ], dtype=cp.float64)
        
        # T values (U.v_s):
        # v_s = [1, 0.5]
        # [1,2] * [1,0.5] = 1*1 + 2*0.5 = 2
        # [3,4] * [1,0.5] = 3*1 + 4*0.5 = 5
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        expected_T = cp.array([
            [2, 5],     # First batch
            [8, 11],   # Second batch
            [1, 0]     # Initial right boundary
        ], dtype=cp.float64)
        
        self.assertEqual(S.shape, (3, 2))  # (batch_size+1) x n
        self.assertEqual(T.shape, (3, 2))  # (batch_size+1) x n
        self.assertTrue(cp.allclose(S[:3,:2], expected_S, rtol=1e-5))
        self.assertTrue(cp.allclose(T[:3,:2], expected_T, rtol=1e-5))


class TestSignatureKernelConsistency(unittest.TestCase):
    def setUp(self):
        # Create two simple time series
        self.X = cp.array([[[0.0], [1.0], [2.0]],
                           [[1.0], [3.0], [3.0]],
                           [[2.0], [2.0], [4.0]], 
                           [[3.0], [4.0], [5.0]]], dtype=cp.float64)
        self.Y = cp.array([[[0.0], [0.5], [1.0]],
                           [[2.0], [1.0], [2.5]],
                           [[2.7], [3.0], [4.0]], 
                           [[3.0], [2.4], [3.20]], 
                           [[1.5], [2.0], [2.5]]], dtype=cp.float64)
        
        # Set up ksig static kernel
        self.static_kernel = ksig.static.kernels.LinearKernel()
        
        # Set up sigkernel
        self.dyadic_order = 0
        self.signature_kernel = sigkernel.SigKernel(sigkernel.LinearKernel(), self.dyadic_order)
        
    def test_signature_kernel_consistency(self):
        """Test that different signature kernel implementations give consistent results"""
        # Convert CuPy arrays to numpy/torch for sigkernel and ksig
        X_np = cp.asnumpy(self.X)
        Y_np = cp.asnumpy(self.Y)
        
        # 1. SigKernel implementation
        X_torch = torch.tensor(X_np)
        Y_torch = torch.tensor(Y_np)
        sig_kernel_result = self.signature_kernel.compute_Gram(X_torch, Y_torch)
        
        # 2. KSig PDE implementation
        ksig_pde_kernel = ksig.kernels.SignaturePDEKernel(normalize=False, static_kernel=self.static_kernel)
        ksig_pde_result = ksig_pde_kernel(X_np, Y_np)
        
        # 3. KSig truncated signature implementation
        ksig_trunc_kernel = ksig.kernels.SignatureKernel(n_levels=21, order=0, normalize=False, 
                                                        static_kernel=self.static_kernel)
        ksig_trunc_result = ksig_trunc_kernel(X_np, Y_np)

        # 3. PowerSig CuPy implementation
        # Convert to derivatives
        dX = cupy_compute_derivative_batch(self.X)
        dY = cupy_compute_derivative_batch(self.Y)
        
        # Compute gram matrix
        powersig_results = cp.zeros((self.X.shape[0], self.Y.shape[0]), dtype=cp.float64)
        order = 32
        
        # Using a loop since we don't have a batched version yet
        for i in range(dX.shape[0]):
            for j in range(dY.shape[0]):
                powersig_results[i, j] = batch_compute_gram_entry(dX[i], dY[j], None, order)
        
        # Convert CuPy array to numpy for comparison
        powersig_results_np = cp.asnumpy(powersig_results)
        
        # Print all results for comparison
        print(f"SigKernel result:\n{sig_kernel_result}")
        print(f"KSig PDE result:\n{ksig_pde_result}")
        print(f"KSig truncated signature result:\n{ksig_trunc_result}")
        print(f"PowerSig CuPy result:\n{powersig_results_np}")
        
        # Check that results are close to each other
        sig_kernel_np = sig_kernel_result.numpy()
        # self.assertTrue(np.allclose(sig_kernel_np, ksig_pde_result, rtol=1e-2), 
        #                 f"SigKernel and KSig PDE results differ significantly")
        # self.assertTrue(np.allclose(sig_kernel_np, powersig_results_np, rtol=1e-2), 
        #                 f"SigKernel and PowerSig CuPy results differ significantly")
        # self.assertTrue(np.allclose(sig_kernel_np, ksig_trunc_result, rtol=1e-2), 
        #                 f"SigKernel and KSig truncated signature results differ significantly") 
        self.assertTrue(np.allclose(powersig_results_np, ksig_trunc_result, rtol=1e-2), 
                        f"PowerSig CuPy and KSig truncated signature results differ significantly")


if __name__ == '__main__':
    unittest.main() 