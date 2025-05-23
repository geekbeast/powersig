import unittest
import os

import powersig

# Enable testing mode to use non-compiled versions of functions
os.environ["POWERSIG_TESTING"] = "1"

import torch

from powersig.torch import batch_ADM_for_diagonal, batch_compute_gram_entry_psi, build_increasing_matrix
from powersig.torch import batch_compute_boundaries
from powersig.torch import compute_vandermonde_vectors
from powersig.torch import build_stencil
from powersig.torch import batch_compute_gram_entry, compute_gram_entry
from powersig.torch.series import torch_compute_derivative_batch
from powersig.powersig_cupy import batch_compute_gram_entry as batch_compute_gram_entry_cupy
import cupy as cp
import ksig
import ksig.static.kernels
from sigkernel import sigkernel

class TestBatchADMForDiagonal(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.rho_2x2 = torch.tensor([0.5, 0.7], dtype=torch.float64)  # batch_size = 2
        self.S_2x2 = torch.tensor([
            [10.0, 30.0],
            [100.0,300.0]
        ], dtype=torch.float64)
        self.T_2x2 = torch.tensor([
            [10.0, 20.0],
            [100.0,200.0]
        ], dtype=torch.float64)
        self.stencil_2x2 = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=torch.float64)  # n x n stencil
        # Pre-allocated buffer for 2x2 case
        self.U_buf_2x2 = torch.empty((2, 2, 2), dtype=torch.float64)

        # Common test data for 3x3 case
        self.rho_3x3 = torch.tensor([0.3], dtype=torch.float64)  # batch_size = 1
        self.S_3x3 = torch.tensor([[1000.0, 4000.0, 5000.0]], dtype=torch.float64)
        self.T_3x3 = torch.tensor([[1000.0, 2000.0, 3000.0]], dtype=torch.float64)
        self.stencil_3x3 = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=torch.float64)  # n x n stencil
        # Pre-allocated buffer for 3x3 case
        self.U_buf_3x3 = torch.empty((1, 3, 3), dtype=torch.float64)

        # Common test data for 4x4 case
        self.rho_4x4 = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64)  # batch_size = 3
        self.S_4x4 = torch.tensor(
            [
                [10000.0, 50000.0, 60000.0, 70000.0],
                [100000.0, 500000.0, 600000.0, 700000.0],
                [1000000.0, 5000000.0, 6000000.0, 7000000.0],
            ],
            dtype=torch.float64,
        )
        self.T_4x4 = torch.tensor(
            [
                [10000.0, 20000.0, 30000.0, 40000.0],  # First batch
                [100000.0, 200000.0, 300000.0, 400000.0],  # Second batch
                [1000000.0, 2000000.0, 3000000.0, 4000000.0],  # Third batch
            ],
            dtype=torch.float64,
        )
        
        self.stencil_4x4 = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], dtype=torch.float64)  # n x n stencil
        # Pre-allocated buffer for 4x4 case
        self.U_buf_4x4 = torch.empty((3, 4, 4), dtype=torch.float64)

    def test_random(self):
        for batch_size in range(10):
            rho = torch.randn(batch_size, dtype=torch.float64)
            S = torch.randn(batch_size, 32, dtype=torch.float64)
            T = torch.randn(batch_size, 32, dtype=torch.float64)
            stencil = torch.randn(32, 32, dtype=torch.float64)
            U = torch.randn(batch_size, 32, 32, dtype=torch.float64)
            result = batch_ADM_for_diagonal(rho, U, S, T, stencil)

            for i in range(batch_size):
                actual = U[i]
                expected = torch.clone(stencil)
                for exponent in range(32):
                    expected[exponent, exponent+1:] *= S[i, 1:S.shape[1]-exponent] * (rho[i] ** exponent)
                    expected[exponent:, exponent] *= T[i, :T.shape[1]-exponent] * (rho[i] ** exponent)
                
                self.assertTrue(torch.allclose(actual, expected, rtol=1e-5))

            # Check that the result is a tensor of shape (batch_size, n, n)
            self.assertEqual(result.shape, (batch_size, 32, 32))

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
        expected_first = torch.tensor([
            [1.0 * 10.0 * 1.0, 2.0 * 30.0 * 1.0],  # First row: rho^0
            [3.0 * 20.0 * 1.0, 4.0 * 10.0 * 0.5]   # First column: rho^0, remaining: rho^1
        ], dtype=torch.float64)
        print(f"Expected: {expected_first}")
        print(f"Result: {result[0]}")
        self.assertTrue(torch.allclose(result[0], expected_first, rtol=1e-5))

    def test_2x2_second_batch(self):
        """Test the second batch element of 2x2 case"""
        result = batch_ADM_for_diagonal(self.rho_2x2, self.U_buf_2x2, self.S_2x2, self.T_2x2, self.stencil_2x2)

        # For the second batch (rho = 0.7):
        # First level: first row/column use rho^0 = 1.0
        # Second level: remaining element uses rho^1 = 0.7
        # Diagonals and their coefficients:
        # 0: [1,4] uses C[0] = 100.0
        # -1: [3] uses C[1] = 200.0
        # 1: [2] uses C[2] = 300.0
        expected_second = torch.tensor([
            [1.0 * 100.0 * 1.0, 2.0 * 300.0 * 1.0],  # First row: rho^0
            [3.0 * 200.0 * 1.0, 4.0 * 100.0 * 0.7]   # First column: rho^0, remaining: rho^1
        ], dtype=torch.float64)

        self.assertTrue(torch.allclose(result[1], expected_second, rtol=1e-5))

    def test_3x3_shape(self):
        """Test that the output shape is correct for 3x3 case"""
        result = batch_ADM_for_diagonal(self.rho_3x3, self.U_buf_3x3, self.S_3x3, self.T_3x3, self.stencil_3x3)
        self.assertEqual(result.shape, (1, 3, 3))  # batch_size x n x n

    def test_3x3_values(self):
        """Test the values for 3x3 case"""
        result = batch_ADM_for_diagonal(self.rho_3x3, self.U_buf_3x3, self.S_3x3, self.T_3x3, self.stencil_3x3)

        # For the 3x3 case (rho = 0.3):
        # First level: first row/column use rho^0 = 1.0
        # Second level: first row/column of remaining 2x2 use rho^1 = 0.3
        # Third level: remaining element uses rho^2 = 0.09
        # Diagonals and their coefficients:
        # 0: [1,5,9] uses C[0] = 1000.0
        # -1: [4,8] uses C[1] = 2000.0
        # -2: [7] uses C[2] = 3000.0
        # 1: [2,6] uses C[3] = 4000.0
        # 2: [3] uses C[4] = 5000.0
        expected = torch.tensor([
            [1.0 * 1000.0 * 1.0, 2.0 * 4000.0 * 1.0, 3.0 * 5000.0 * 1.0],  # First row: rho^0
            [4.0 * 2000.0 * 1.0, 5.0 * 1000.0 * 0.3, 6.0 * 4000.0 * 0.3],  # First column: rho^0, remaining row/column: rho^1
            [7.0 * 3000.0 * 1.0, 8.0 * 2000.0 * 0.3, 9.0 * 1000.0 * 0.09]  # First column: rho^0, remaining row/column: rho^1, remaining: rho^2
        ], dtype=torch.float64)

        self.assertTrue(torch.allclose(result[0], expected, rtol=1e-5))

    def test_4x4_values(self):
        """Test the values for 4x4 case"""
        result = batch_ADM_for_diagonal(self.rho_4x4, self.U_buf_4x4, self.S_4x4, self.T_4x4, self.stencil_4x4)

        # For the 4x4 case with batch size 3:
        # First batch (rho = 0.4):
        # First level: first row/column use rho^0 = 1.0
        # Second level: first row/column of remaining 3x3 use rho^1 = 0.4
        # Third level: first row/column of remaining 2x2 use rho^2 = 0.16
        # Fourth level: remaining element uses rho^3 = 0.064
        # Diagonals and their coefficients:
        # 0: [1,6,11,16] uses C[0] = 10000.0
        # -1: [5,10,15] uses C[1] = 20000.0
        # -2: [9,14] uses C[2] = 30000.0
        # -3: [13] uses C[3] = 40000.0
        # 1: [2,7,12] uses C[4] = 50000.0
        # 2: [3,8] uses C[5] = 60000.0
        # 3: [4] uses C[6] = 70000.0
        expected_first = torch.tensor([
            [1.0 * 10000.0 * 1.0, 2.0 * 50000.0 * 1.0, 3.0 * 60000.0 * 1.0, 4.0 * 70000.0 * 1.0],  # First row: rho^0
            [5.0 * 20000.0 * 1.0, 6.0 * 10000.0 * 0.4, 7.0 * 50000.0 * 0.4, 8.0 * 60000.0 * 0.4],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 30000.0 * 1.0, 10.0 * 20000.0 * 0.4, 11.0 * 10000.0 * 0.16, 12.0 * 50000.0 * 0.16],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 40000.0 * 1.0, 14.0 * 30000.0 * 0.4, 15.0 * 20000.0 * 0.16, 16.0 * 10000.0 * 0.064]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=torch.float64)

        # Second batch (rho = 0.6):
        # Same pattern but with rho = 0.6 and coefficients scaled by 10
        expected_second = torch.tensor([
            [1.0 * 100000.0 * 1.0, 2.0 * 500000.0 * 1.0, 3.0 * 600000.0 * 1.0, 4.0 * 700000.0 * 1.0],  # First row: rho^0
            [5.0 * 200000.0 * 1.0, 6.0 * 100000.0 * 0.6, 7.0 * 500000.0 * 0.6, 8.0 * 600000.0 * 0.6],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 300000.0 * 1.0, 10.0 * 200000.0 * 0.6, 11.0 * 100000.0 * 0.36, 12.0 * 500000.0 * 0.36],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 400000.0 * 1.0, 14.0 * 300000.0 * 0.6, 15.0 * 200000.0 * 0.36, 16.0 * 100000.0 * 0.216]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=torch.float64)

        # Third batch (rho = 0.8):
        # Same pattern but with rho = 0.8 and coefficients scaled by 100
        expected_third = torch.tensor([
            [1.0 * 1000000.0 * 1.0, 2.0 * 5000000.0 * 1.0, 3.0 * 6000000.0 * 1.0, 4.0 * 7000000.0 * 1.0],  # First row: rho^0
            [5.0 * 2000000.0 * 1.0, 6.0 * 1000000.0 * 0.8, 7.0 * 5000000.0 * 0.8, 8.0 * 6000000.0 * 0.8],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 3000000.0 * 1.0, 10.0 * 2000000.0 * 0.8, 11.0 * 1000000.0 * 0.64, 12.0 * 5000000.0 * 0.64],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 4000000.0 * 1.0, 14.0 * 3000000.0 * 0.8, 15.0 * 2000000.0 * 0.64, 16.0 * 1000000.0 * 0.512]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=torch.float64)

        print(f"Expected: {expected_first}")
        print(f"Result: {result[0]}")

        print(f"Expected: {expected_second}")
        print(f"Result: {result[1]}")

        print(f"Expected: {expected_third}")
        print(f"Result: {result[2]}")

        self.assertTrue(torch.allclose(result[0], expected_first, rtol=1e-5))
        self.assertTrue(torch.allclose(result[1], expected_second, rtol=1e-5))
        self.assertTrue(torch.allclose(result[2], expected_third, rtol=1e-5))

class TestBatchComputeBoundaries(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.U_2x2 = torch.tensor([
            [[1.0, 2.0],
             [3.0, 4.0]],
            [[5.0, 6.0],
             [7.0, 8.0]]
        ], dtype=torch.float64)  # batch_size=2, n=2
        self.ds_2x2 = torch.tensor([0.5])
        self.dt_2x2 = torch.tesnor([0.5])
        self.v_s_2x2, self.v_t_2x2 = compute_vandermonde_vectors(self.ds_2x2, self.dt_2x2, 2)
        # Pre-allocated buffers for S and T for 2x2 case
        self.S_buf_2x2 = torch.empty((3, 2), dtype=torch.float64)  # Max size needed is batch_size+1
        self.T_buf_2x2 = torch.empty((3, 2), dtype=torch.float64)  # Max size needed is batch_size+1

        # Common test data for 3x3 case
        self.U_3x3 = torch.tensor([
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        ], dtype=torch.float64)  # batch_size=1, n=3
        self.ds_3x3 = torch.tensor([0.3])
        self.dt_3x3 = torch.tensor([0.3])
        self.v_s_3x3, self.v_t_3x3 = compute_vandermonde_vectors(self.ds_3x3, self.dt_3x3, 3)
        # Pre-allocated buffers for S and T for 3x3 case
        self.S_buf_3x3 = torch.empty((2, 3), dtype=torch.float64)  # Max size needed is batch_size+1
        self.T_buf_3x3 = torch.empty((2, 3), dtype=torch.float64)  # Max size needed is batch_size+1

        # Common test data for 4x4 case with batch size 3
        self.U_4x4 = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]],
            [[17.0, 18.0, 19.0, 20.0],
             [21.0, 22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0, 28.0],
             [29.0, 30.0, 31.0, 32.0]],
            [[33.0, 34.0, 35.0, 36.0],
             [37.0, 38.0, 39.0, 40.0],
             [41.0, 42.0, 43.0, 44.0],
             [45.0, 46.0, 47.0, 48.0]]
        ], dtype=torch.float64)  # batch_size=3, n=4
        self.ds_4x4 = torch.tensor([0.25])
        self.dt_4x4 = torch.tensor([0.25])
        self.v_s_4x4, self.v_t_4x4 = compute_vandermonde_vectors(self.ds_4x4, self.dt_4x4, 4)
        # Pre-allocated buffers for S and T for 4x4 case
        self.S_buf_4x4 = torch.empty((4, 4), dtype=torch.float64)  # Max size needed is batch_size+1
        self.T_buf_4x4 = torch.empty((4, 4), dtype=torch.float64)  # Max size needed is batch_size+1

    def test_random(self):
        ZERO = torch.tensor([0], dtype=torch.float64)
        ONE = torch.tensor([1], dtype=torch.float64)
        order = 5
        v_s = torch.randn(order,dtype=torch.float64)
        v_t = torch.randn(order,dtype=torch.float64)
        for batch_size in range(2,10):
            U = torch.randn((batch_size, order, order), dtype=torch.float64)
            S = torch.randn((batch_size+1, order), dtype=torch.float64)
            T = torch.randn((batch_size+1, order), dtype=torch.float64)
            S[batch_size] = 1
            T[batch_size] = 1
            
            U_vs = U @ v_s
            vt_U = v_t @ U

            # Shrinking case
            skip_first = True
            skip_last = True
            S_result, T_result = batch_compute_boundaries(U,S,T,v_s,v_t,skip_first, skip_last)
            self.assertTrue( torch.allclose(U_vs[1:], T_result), f"Shrinking: U_vs != T\nU_vs = {U_vs}\nT_result={T_result}")
            self.assertTrue( torch.allclose(vt_U[:-1], S_result), f"Shrinking: vt_U !=S\nvt_U = {vt_U}\nS_result={S_result}")
            
            # Growing case
            skip_first = False
            skip_last = False
            S[0,:] = 0
            S[0,0] = 1
            T[batch_size,:] = 0
            T[batch_size,0] = 1
            S_result, T_result = batch_compute_boundaries(U,S,T,v_s,v_t,skip_first, skip_last)
            self.assertEqual(T_result.shape, (batch_size+1, order), f"T_result.shape = {T_result.shape}\nT_result = {T_result}")
            self.assertEqual(S_result.shape, (batch_size+1, order), f"S_result.shape = {S_result.shape}\nS_result = {S_result}")
            self.assertTrue( torch.allclose(U_vs, T_result[:-1]), f"Growing: U_vs != T\nU_vs = {U_vs}\nT_result={T_result}")
            self.assertTrue( torch.allclose(vt_U, S_result[1:]), f"Growing: vt_U !=S\nvt_U = {vt_U}\nS_result={S_result}")
            self.assertTrue( torch.allclose(S[0,0],ONE), f"S = {S}")
            self.assertTrue( torch.allclose(S[0,1:].sum(),ZERO), f"sum = {S[0,1:].sum()}\nS = {S}")
            self.assertTrue( torch.allclose(T[batch_size,0],ONE), f"T = {T}")
            self.assertTrue( torch.allclose(T[batch_size,1:].sum(),ZERO),f"sum = {T[0,1:].sum()}\nT = {T}")

            # Staying the same cases 
            skip_first = False
            skip_last = True
            S[batch_size,:] = 0
            S[batch_size,0] = 1
            S_result, T_result = batch_compute_boundaries(U,S,T,v_s,v_t,skip_first, skip_last)
            self.assertTrue( torch.allclose(U_vs, T_result), f"Same(skip_first=False, skip_last=True): U_vs != T\nU_vs = {U_vs}\nT_result={T_result}")
            self.assertTrue( torch.allclose(vt_U[:-1], S_result[1:]), f"Same(skip_first=False, skip_last=True): vt_U !=S\nvt_U = {vt_U}\nS_result={S_result}")
            self.assertTrue( torch.allclose(S[0,0],ONE), f"S = {S}")
            self.assertTrue( torch.allclose(S[0,1:].sum(),ZERO), f"sum = {S[0,1:].sum()}\nS = {S}")

            skip_first = True
            skip_last = False
            T[0,:] = 0
            T[0,0] = 1
            S_result, T_result = batch_compute_boundaries(U,S,T,v_s,v_t,skip_first, skip_last)
            self.assertTrue( torch.allclose(U_vs[1:], T_result[:-1]), f"Same(skip_first=True, skip_last=False): U_vs != T\nU_vs = {U_vs}\nT_result={T_result}")
            self.assertTrue( torch.allclose(vt_U, S_result), f"Same(skip_first=True, skip_last=False): vt_U !=S\nvt_U = {vt_U}\nS_result={S_result}")
            self.assertTrue( torch.allclose(T[batch_size,0],ONE),f"T = {T}")
            self.assertTrue( torch.allclose(T[batch_size,1:].sum(),ZERO),f"T = {T}")
            


class TestBuildStencil(unittest.TestCase):
    def setUp(self):
        self.order = 4  # Using a small order for easier testing
        self.device = torch.device('cpu')
        self.dtype = torch.float64

    def test_stencil_shape(self):
        """Test that the stencil has the correct shape"""
        stencil = build_stencil(self.order, self.device, self.dtype)
        self.assertEqual(stencil.shape, (self.order, self.order))

    def test_stencil_values(self):
        """Test that the stencil values are computed correctly"""
        stencil = build_stencil(self.order, self.device, self.dtype)
        
        # First row and column should be ones (from initialization)
        self.assertTrue(torch.allclose(stencil[0, :], torch.ones(self.order, dtype=self.dtype)))
        self.assertTrue(torch.allclose(stencil[:, 0], torch.ones(self.order, dtype=self.dtype)))
        
        # Check some specific values in the stencil
        # For order=4, after cumulative product, the stencil should look like:
        # 1  1  1  1
        # 1 1/1 1/2 1/6
        # 1 1/2 1/4 1/12
        # 1 1/6 1/12 1/36
        expected_values = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 1/3],
            [1.0, 0.5, 0.25, 1/12],
            [1.0, 1/3, 1/12, 1/36]
        ], dtype=self.dtype)
        
        self.assertTrue(torch.allclose(stencil, expected_values, rtol=1e-5))

    def test_stencil_diagonals(self):
        """Test that the diagonals are correctly computed"""
        stencil = build_stencil(self.order, self.device, self.dtype)
        
        # Check main diagonal
        main_diag = torch.diagonal(stencil)
        expected_main_diag = torch.tensor([1.0, 1.0, 0.25, 1/36], dtype=self.dtype)
        self.assertTrue(torch.allclose(main_diag, expected_main_diag, rtol=1e-5))
        
        # Check first upper diagonal
        upper_diag = torch.diagonal(stencil, offset=1)
        expected_upper_diag = torch.tensor([1.0, 0.5, 1/12], dtype=self.dtype)
        self.assertTrue(torch.allclose(upper_diag, expected_upper_diag, rtol=1e-5))
        
        # Check first lower diagonal (should be same as upper diagonal due to symmetry)
        lower_diag = torch.diagonal(stencil, offset=-1)
        expected_lower_diag = torch.tensor([1.0, 0.5, 1/12], dtype=self.dtype)
        self.assertTrue(torch.allclose(lower_diag, expected_lower_diag, rtol=1e-5))

    def test_stencil_symmetry(self):
        """Test that the stencil is symmetric around the main diagonal"""
        stencil = build_stencil(self.order, self.device, self.dtype)
        
        # Check that the matrix is symmetric
        for k in range(1, self.order):
            upper_diag = torch.diagonal(stencil, offset=k)
            lower_diag = torch.diagonal(stencil, offset=-k)
            self.assertTrue(torch.allclose(upper_diag, lower_diag, rtol=1e-5))

    def test_stencil_dtype_device(self):
        """Test that the stencil has the correct dtype and device"""
        stencil = build_stencil(self.order, self.device, self.dtype)
        self.assertEqual(stencil.dtype, self.dtype)
        self.assertEqual(stencil.device, self.device)

class TestSignatureKernelConsistency(unittest.TestCase):
    def setUp(self):
        # Determine device to use (CUDA if available, otherwise CPU)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"self.device: {self.device}")
        # Create two simple time series
        # self.X = torch.tensor([[[0.0], [1.0], [2.0]],
        #                    [[1.0], [3.0], [3.0]],
        #                    [[2.0], [2.0], [4.0]], 
        #                    [[3.0], [4.0], [5.0]]], dtype=torch.float64).to(self.device)
        # self.Y = torch.tensor([[[0.0], [0.5], [1.0]],
        #                    [[2.0], [1.0], [2.5]],
        #                    [[2.7], [3.0], [4.0]], 
        #                    [[3.0], [2.4], [3.20]], 
        #                    [[1.5], [2.0], [2.5]]], dtype=torch.float64).to(self.device)
        self.X = torch.randn(4, 3, 2, dtype=torch.float64).to(self.device)/2
        self.Y = torch.randn(4, 3, 2, dtype=torch.float64).to(self.device)/2
        
        # Set up ksig static kernel
        self.static_kernel = ksig.static.kernels.LinearKernel()
        self.order = 32
        self.dtype = torch.float64

        # Set up sigkernel
        self.dyadic_order = 0
        self.signature_kernel = sigkernel.SigKernel(sigkernel.LinearKernel(), self.dyadic_order)
        
    def test_signature_kernel_consistency(self):
        """Test that different signature kernel implementations give consistent results"""
        # Move data to CPU for sigkernel and ksig if necessary
        X_cpu = self.X.cpu() if self.X.is_cuda else self.X
        Y_cpu = self.Y.cpu() if self.Y.is_cuda else self.Y
        
        # 1. SigKernel implementation
        sig_kernel_result = self.signature_kernel.compute_Gram(X_cpu, Y_cpu)
        
        # 2. KSig PDE implementation
        ksig_pde_kernel = ksig.kernels.SignaturePDEKernel(normalize=False, static_kernel=self.static_kernel)
        ksig_pde_result = ksig_pde_kernel(X_cpu, Y_cpu)
        
        # 3. KSig truncated signature implementation
        ksig_trunc_kernel = ksig.kernels.SignatureKernel(n_levels=50, order=0, normalize=False, 
                                                        static_kernel=self.static_kernel)
        ksig_trunc_result = ksig_trunc_kernel(X_cpu, Y_cpu)
        order = self.order
        
        # 5. PowerSig implementation
        # Convert to derivatives
        dX = torch_compute_derivative_batch(self.X).clone()
        dY = torch_compute_derivative_batch(self.Y).clone()
        ds = 1/dX.shape[1]
        dt = 1/dY.shape[1]
        v_s, v_t = compute_vandermonde_vectors(ds, dt, self.order, dX.dtype, dX.device)
        v_s = v_s.clone()
        v_t = v_t.clone()
        psi_s = powersig.torch.torch.build_stencil_s(v_s, self.order, self.device, self.dtype).clone()
        psi_t = powersig.torch.torch.build_stencil_t(v_t, self.order, self.device, self.dtype).clone()
        ic = torch.zeros((self.order,), dtype=self.dtype, device=self.device)
        ic[0] = 1
        exponents = build_increasing_matrix(self.order, torch.int8, self.device).clone()
        longest_diagonal = min(dX.shape[1], dY.shape[1])
        diagonal_count = dX.shape[1] + dY.shape[1] - 1
        indices = torch.arange(longest_diagonal, dtype=torch.int32, device=self.device)
        

        # Compute gram matrix
        powersig_results = torch.zeros((self.X.shape[0], self.Y.shape[0]), dtype=torch.float64, device=self.device)
        powersig_cupy_results = torch.zeros((self.X.shape[0], self.Y.shape[0]), dtype=torch.float64, device=self.device)
        
        for i in range(dX.shape[0]):
            for j in range(dY.shape[0]):
                powersig_results[i, j] = compute_gram_entry(dX[i], dY[j],v_s, v_t, psi_s, psi_t, diagonal_count, 4, longest_diagonal, ic, indices,exponents,order)
                powersig_cupy_results[i, j] = torch.tensor(batch_compute_gram_entry_cupy(cp.array(dX[i].cpu().numpy()), cp.array(dY[j].cpu().numpy()), order),dtype=torch.float64, device=self.device)
        # # Move results to CPU for comparison
        # if powersig_results.is_cuda:
        #     powersig_results = powersig_results.cpu()
        
        # Print all results for comparison
        print(f"SigKernel result:\n{sig_kernel_result}")
        print(f"KSig PDE result:\n{ksig_pde_result}")
        print(f"KSig truncated signature result:\n{ksig_trunc_result}")
        print(f"PowerSig result:\n{powersig_results}")
        print(f"PowerSig_cupy result:\n{powersig_cupy_results}")

        # Convert numpy arrays to PyTorch tensors if needed
        if not isinstance(ksig_pde_result, torch.Tensor):
            ksig_pde_result = torch.tensor(ksig_pde_result, device=self.device, dtype=torch.float64)
        if not isinstance(ksig_trunc_result, torch.Tensor):
            ksig_trunc_result = torch.tensor(ksig_trunc_result, device=self.device, dtype=torch.float64)
        if not isinstance(sig_kernel_result, torch.Tensor):
            sig_kernel_result = torch.tensor(sig_kernel_result, device=self.device, dtype=torch.float64)    
        else: 
            sig_kernel_result = sig_kernel_result.to(self.device)

        print(f"sig_kernel_result.device: {sig_kernel_result.device}")
        print(f"ksig_pde_result.device: {ksig_pde_result.device}")
        print(f"ksig_trunc_result.device: {ksig_trunc_result.device}")
        print(f"powersig_results.device: {powersig_results.device}")
        print(f"powersig_cupy_results.device: {powersig_cupy_results.device}")
        # Check that results are close to each other
        self.assertTrue(torch.allclose(sig_kernel_result, ksig_pde_result, rtol=1e-2), 
                        f"SigKernel and KSig PDE results differ significantly\n{sig_kernel_result}\n{ksig_pde_result}")
        # self.assertTrue(torch.allclose(sig_kernel_result, ksig_trunc_result, rtol=1e-2), 
        #                 f"SigKernel and KSig truncated results differ significantly")
        # self.assertTrue(torch.allclose(sig_kernel_result, powersig_results, rtol=1e-2), 
        #                 f"SigKernel and PowerSig results differ significantly")
        print(f"ksig_trunc_result-powersig_results total error= {torch.sum(abs(ksig_trunc_result-powersig_results))}")
        print(f"ksig_trunc_result-powersig_cupy_results total error= {torch.sum(abs(ksig_trunc_result-powersig_cupy_results))}")
        print(f"powersig_results-powersig_cupy_results total error= {torch.sum(abs(powersig_results-powersig_cupy_results))}")

        self.assertTrue(torch.allclose(ksig_trunc_result, powersig_results, rtol=1e-3),   
                        f"KSig truncated and PowerSig results differ significantly\nksig_trunc_result = {ksig_trunc_result}\npowersig_results = {powersig_results}\nksig_trunc_result-powersig_results = {ksig_trunc_result-powersig_results}\nTotal error: {torch.sum(abs(ksig_trunc_result-powersig_results))}")
        self.assertTrue(torch.allclose(ksig_trunc_result, powersig_cupy_results, rtol=1e-7),   
                        f"KSig truncated and PowerSig Cupy results differ significantly\nksig_trunc_result = {ksig_trunc_result}\npowersig_cupy_results = {powersig_cupy_results}\nksig_trunc_result-powersig_cupy_results = {ksig_trunc_result-powersig_cupy_results}\nTotal error: {torch.sum(abs(ksig_trunc_result-powersig_cupy_results))}")
        self.assertTrue(torch.allclose(powersig_results, powersig_cupy_results, rtol=1e-7),   
                        f"PowerSig and PowerSig_cupy results differ significantly\npowersig_results = {powersig_results}\npowersig_cupy_results = {powersig_cupy_results}\npowersig_results-powersig_cupy_results = {powersig_results-powersig_cupy_results}\nTotal error: {torch.sum(abs(powersig_results-powersig_cupy_results))}")

class TestStencilSimple(unittest.TestCase):
    """Simple direct tests for build_stencil_s and build_stencil_t functions."""
    
    def test_build_stencil_s_direct(self):
        """Test build_stencil_s directly with random vector."""
        from powersig.torch import build_stencil, build_stencil_s
        
        order = 5
        device = torch.device("cpu")
        dtype = torch.float64
        
        # Create a random vector for v_s
        v_s = torch.rand(order, dtype=dtype, device=device)
        
        # Get the base stencil
        base_stencil = build_stencil(order, device, dtype)
        
        # Get the stencil with columns multiplied by v_s
        stencil_s = build_stencil_s(v_s, order, device, dtype)
        
        # Manually multiply each col and check
        for i in range(order):
            expected_col = base_stencil[:,i] * v_s[i]
            actual_col = stencil_s[:,i]
            self.assertTrue(torch.allclose(actual_col, expected_col),
                          f"Row {i} multiplication failed:\nExpected: {expected_col}\nActual: {actual_col}")
    
    def test_build_stencil_t_direct(self):
        """Test build_stencil_t directly with random vector."""
        from powersig.torch import build_stencil, build_stencil_t
        
        order = 5
        device = torch.device("cpu")
        dtype = torch.float64
        
        # Create a random vector for v_t
        v_t = torch.rand(order, dtype=dtype, device=device)
        
        # Get the base stencil
        base_stencil = build_stencil(order, device, dtype)
        
        # Get the stencil with rows multiplied by v_t
        stencil_t = build_stencil_t(v_t, order, device, dtype)
        
        # Manually multiply each rows and check
        for j in range(order):
            expected_row = base_stencil[j,:] * v_t[j]
            actual_row = stencil_t[j,:]
            self.assertTrue(torch.allclose(actual_row, expected_row),
                          f"Row {j} multiplication failed:\nExpected: {expected_row}\nActual: {actual_row}")
   

if __name__ == '__main__':
    # To run all tests
    # unittest.main()
    
    # To run only TestBuildStencil
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBuildStencil)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchComputeBoundaries)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchADMForDiagonal)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSignatureKernelConsistency)
    unittest.TextTestRunner().run(suite)
    
    # To run a specific test method
    # suite = unittest.TestLoader().loadTestsFromName('TestBatchComputeBoundaries.test_2x2_values')
    # unittest.TextTestRunner().run(suite)
