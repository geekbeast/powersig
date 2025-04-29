import unittest
import torch

from powersig.torch import batch_ADM_for_diagonal
from powersig.torch import batch_compute_boundaries
from powersig.torch import compute_vandermonde_vectors
from powersig.torch import build_stencil
from powersig.torch import batch_compute_gram_entry
from powersig.util.series import torch_compute_derivative_batch
from powersig.matrixsig import build_scaling_for_integration
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
        self.ds_2x2 = 0.5
        self.dt_2x2 = 0.5
        self.v_s_2x2, self.v_t_2x2 = compute_vandermonde_vectors(self.ds_2x2, self.dt_2x2, 2, self.U_2x2.dtype, self.U_2x2.device)

        # Common test data for 3x3 case
        self.U_3x3 = torch.tensor([
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        ], dtype=torch.float64)  # batch_size=1, n=3
        self.ds_3x3 = 0.3
        self.dt_3x3 = 0.3
        self.v_s_3x3, self.v_t_3x3 = compute_vandermonde_vectors(self.ds_3x3, self.dt_3x3, 3, self.U_3x3.dtype, self.U_3x3.device)

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
        self.ds_4x4 = 0.25
        self.dt_4x4 = 0.25
        self.v_s_4x4, self.v_t_4x4 = compute_vandermonde_vectors(self.ds_4x4, self.dt_4x4, 4, self.U_4x4.dtype, self.U_4x4.device)

    def test_2x2_shrinking(self):
        """Test 2x2 case with both skip_first and skip_last=True (shrinking)"""
        S, T = batch_compute_boundaries(self.U_2x2, self.v_s_2x2, self.v_t_2x2, skip_first=True, skip_last=True)
        
        # For first batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        
        # T values (U.v_s):
        # v_s = [1, 0.5]
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        
        expected_S = torch.tensor([
            [2.5, 4.0]  # First batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [8.0, 11.0]  # First batch
        ], dtype=torch.float64)
        
        # print("==== test_2x2_shrinking ====")
        # print(f"U = {self.U_2x2}")
        # print(f"v_s = {self.v_s_2x2}")
        # print(f"v_t = {self.v_t_2x2}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (1, 2))  # batch_size x (n-1)
        self.assertEqual(T.shape, (1, 2))  # batch_size x (n-1)
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_2x2_growing(self):
        """Test 2x2 case with both skip_first and skip_last=False (growing)"""
        S, T = batch_compute_boundaries(self.U_2x2, self.v_s_2x2, self.v_t_2x2, skip_first=False, skip_last=False)
        
        # For first batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        
        # T values (U.v_s):
        # v_s = [1, 0.5]
        # [1,2] * [1,0.5] = 1*1 + 2*0.5 = 2
        # [3,4] * [1,0.5] = 3*1 + 4*0.5 = 5
        
        # For second batch:
        # S values (v_t^T.U):
        # [1,0.5] * [5,7] = 1*5 + 0.5*7 = 8.5
        # [1,0.5] * [6,8] = 1*6 + 0.5*8 = 10
        
        # T values (U.v_s):
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        
        expected_S = torch.tensor([
            [1.0, 0.0],   # New batch with initial condition (bottom boundary)
            [2.5, 4.0],   # First batch
            [8.5, 10.0]   # Second batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [2.0, 5.0],  # First batch
            [8.0, 11.0],  # Second batch
            [1.0, 0.0]   # New batch with initial condition
        ], dtype=torch.float64)
        
        # print(f"U = {self.U_2x2}")
        # print(f"v_s = {self.v_s_2x2}")
        # print(f"v_t = {self.v_t_2x2}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (3, 2))  # (batch_size+1) x n
        self.assertEqual(T.shape, (3, 2))  # (batch_size+1) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_2x2_staying_same(self):
        """Test 2x2 case with skip_first=True and skip_last=False (staying same size)"""
        S, T = batch_compute_boundaries(self.U_2x2, self.v_s_2x2, self.v_t_2x2, skip_first=True, skip_last=False)
        
        # For first batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        
        # T values (U.v_s):
        # [1, 0] from initial condition
        
        # For second batch:
        # S values (v_t^T.U):
        # [1,0.5] * [5,7] = 1*5 + 0.5*7 = 8.5
        # [1,0.5] * [6,8] = 1*6 + 0.5*8 = 10
        
        # T values (U.v_s):
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        
        expected_S = torch.tensor([
            [2.5, 4.0],  # First batch
            [8.5, 10.0]  # Second batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [1.0, 0.0],  # First batch
            [8.0, 11.0]  # Second batch
        ], dtype=torch.float64)
        
        # print("==== test_2x2_staying_same ====")
        # print(f"U = {self.U_2x2}")
        # print(f"v_s = {self.v_s_2x2}")
        # print(f"v_t = {self.v_t_2x2}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (2, 2))  # batch_size x n
        self.assertEqual(T.shape, (2, 2))  # batch_size x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_3x3_shrinking(self):
        """Test 3x3 case with both skip_first and skip_last=True (shrinking)"""
        S, T = batch_compute_boundaries(self.U_3x3, self.v_s_3x3, self.v_t_3x3, skip_first=True, skip_last=True)
        
        # For the single batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.3, 0.09]
        # [1,0.3,0.09] * [1,4,7] = 1*1 + 0.3*4 + 0.09*7 = 2.83
        # [1,0.3,0.09] * [2,5,8] = 1*2 + 0.3*5 + 0.09*8 = 4.22
        # [1,0.3,0.09] * [3,6,9] = 1*3 s
        expected_S = torch.tensor([[2.83, 4.22, 5.61]], dtype=torch.float64)
        expected_T = torch.tensor([[1.87, 6.04, 10.21]], dtype=torch.float64)
        
        # print("==== test_3x3_shrinking ====")
        # print(f"U = {self.U_3x3}")
        # print(f"v_s = {self.v_s_3x3}")
        # print(f"v_t = {self.v_t_3x3}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (0, 3))  # (batch_size-1) x n
        self.assertEqual(T.shape, (0, 3))  # (batch_size-1) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_3x3_growing(self):
        """Test 3x3 case with both skip_first and skip_last=False (growing)"""
        S, T = batch_compute_boundaries(self.U_3x3, self.v_s_3x3, self.v_t_3x3, skip_first=False, skip_last=False)
        
        # For the single batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.3, 0.09]
        # [1,0.3,0.09] * [1,4,7] = 1*1 + 0.3*4 + 0.09*7 = 2.83
        # [1,0.3,0.09] * [2,5,8] = 1*2 + 0.3*5 + 0.09*8 = 4.22
        # [1,0.3,0.09] * [3,6,9] = 1*3 + 0.3*6 + 0.09*9 = 5.61
        
        # T values (U.v_s):
        # v_s = [1, 0.3, 0.09]
        # [1,2,3] * [1,0.3,0.09] = 1*1 + 2*0.3 + 3*0.09 = 1.87
        # [4,5,6] * [1,0.3,0.09] = 4*1 + 5*0.3 + 6*0.09 = 6.04
        # [7,8,9] * [1,0.3,0.09] = 7*1 + 8*0.3 + 9*0.09 = 10.21
        
        expected_S = torch.tensor([
            [1.0, 0.0, 0.0],      # New batch with initial condition (bottom boundary)
            [2.83, 4.22, 5.61]    # Original batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [1.87, 6.04, 10.21],  # Original batch
            [1.0, 0.0, 0.0]       # New batch with initial condition
        ], dtype=torch.float64)
        
        # print("==== test_3x3_growing ====")
        # print(f"U = {self.U_3x3}")
        # print(f"v_s = {self.v_s_3x3}")
        # print(f"v_t = {self.v_t_3x3}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (2, 3))  # (batch_size+1) x n
        self.assertEqual(T.shape, (2, 3))  # (batch_size+1) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_3x3_staying_same(self):
        """Test 3x3 case with skip_first=True and skip_last=False (staying same size)"""
        S, T = batch_compute_boundaries(self.U_3x3, self.v_s_3x3, self.v_t_3x3, skip_first=True, skip_last=False)
        
        # For the single batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.3, 0.09]
        # [1,0.3,0.09] * [1,4,7] = 1*1 + 0.3*4 + 0.09*7 = 2.83
        # [1,0.3,0.09] * [2,5,8] = 1*2 + 0.3*5 + 0.09*8 = 4.22
        # [1,0.3,0.09] * [3,6,9] = 1*3 + 0.3*6 + 0.09*9 = 5.61
        
        # T values (U.v_s):
        # [1,0,0] from initial conditions
        
        expected_S = torch.tensor([[2.83, 4.22, 5.61]], dtype=torch.float64)
        expected_T = torch.tensor([[1.00, 0.0, 0.0]], dtype=torch.float64)
        
        # print("==== test_3x3_staying_same ====")
        # print(f"U = {self.U_3x3}")
        # print(f"v_s = {self.v_s_3x3}")
        # print(f"v_t = {self.v_t_3x3}")
        # print(f"S = {S}")
        # print(f"T = {T}")

        self.assertEqual(S.shape, (1, 3))  # (batch_size) x n
        self.assertEqual(T.shape, (1, 3))  # (batch_size) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_4x4_shrinking(self):
        """Test 4x4 case with both skip_first and skip_last=True (shrinking)"""
        S, T = batch_compute_boundaries(self.U_4x4, self.v_s_4x4, self.v_t_4x4, skip_first=True, skip_last=True)
        
        # For the first batch:
        # S values (v_t^T.U):s1,15] = 1*3 + 0.25*7 + 0.0625*11 + 0.015625*15 = 5.203125
        # [1,0.25,0.0625,0.015625] * [4,8,12,16] = 1*4 + 0.25*8 + 0.0625*12 + 0.015625*16 = 6.203125
        
        # T values (U.v_s):
        # v_s = [1, 0.25, 0.0625, 0.015625]
        # [1,2,3,4] * [1,0.25,0.0625,0.015625] = 1*1 + 2*0.25 + 3*0.0625 + 4*0.015625 = 3.015625
        # [5,6,7,8] * [1,0.25,0.0625,0.015625] = 5*1 + 6*0.25 + 7*0.0625 + 8*0.015625 = 4.34375
        # [9,10,11,12] * [1,0.25,0.0625,0.015625] = 9*1 + 10*0.25 + 11*0.0625 + 12*0.015625 = 5.671875
        # [13,14,15,16] * [1,0.25,0.0625,0.015625] = 13*1 + 14*0.25 + 15*0.0625 + 16*0.015625 = 7.0
        
        # Similar computations for second and third batches with scaled values
        
        expected_S = torch.tensor([
            [3.015625, 4.34375, 5.671875, 7],  # First batch
            [24.2656, 25.5938, 26.9219, 28.2500]  # Second batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [23.0000, 28.3125, 33.6250, 38.9375],  # First batch
            [44.2500, 49.5625, 54.8750, 60.1875]  # Second batch
        ], dtype=torch.float64)
        
        # print("==== test_4x4_shrinking ====")
        # print(f"U = {self.U_4x4}")
        # print(f"v_s = {self.v_s_4x4}")
        # print(f"v_t = {self.v_t_4x4}")
        # print(f"S = {S}")
        # print(f"T = {T}")
        
        self.assertEqual(S.shape, (2, 4))  # (batch_size-1) x n
        self.assertEqual(T.shape, (2, 4))  # (batch_size-1) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

    def test_4x4_growing(self):
        """Test 4x4 case with both skip_first and skip_last=False (growing)"""
        S, T = batch_compute_boundaries(self.U_4x4, self.v_s_4x4, self.v_t_4x4, skip_first=False, skip_last=False)
        
        # For the first batch:
        # S values (v_t^T.U):
        # v_t = [1, 0.25, 0.0625, 0.015625]
        # [1,0.25,0.0625,0.015625] * [1,5,9,13] = 1*1 + 0.25*5 + 0.0625*9 + 0.015625*13 = 3.015625
        # [1,0.25,0.0625,0.015625] * [2,6,10,14] = 1*2 + 0.25*6 + 0.0625*10 + 0.015625*14 = 4.34375
        # [1,0.25,0.0625,0.015625] * [3,7,11,15] = 1*3 + 0.25*7 + 0.0625*11 + 0.015625*15 = 5.671875
        # [1,0.25,0.0625,0.015625] * [4,8,12,16] = 1*4 + 0.25*8 + 0.0625*12 + 0.015625*16 = 7
        
        # T values (U.v_s):
        # v_s = [1, 0.25, 0.0625, 0.015625]
        # [1,2,3,4] * [1,0.25,0.0625,0.015625] = 1*1 + 2*0.25 + 3*0.0625 + 4*0.015625 = 1.75
        # [5,6,7,8] * [1,0.25,0.0625,0.015625] = 5*1 + 6*0.25 + 7*0.0625 + 8*0.015625 = 7.0625
        # [9,10,11,12] * [1,0.25,0.0625,0.015625] = 9*1 + 10*0.25 + 11*0.0625 + 12*0.015625 = 12.3750
        # [13,14,15,16] * [1,0.25,0.0625,0.015625] = 13*1 + 14*0.25 + 15*0.0625 + 16*0.015625 = 17.6875
        
        # Similar computations for second and third batches with scaled values
        
        expected_S = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],   # New batch with initial condition (bottom boundary)
            [3.015625, 4.34375, 5.671875, 7],  # First batch
            [24.2656, 25.5938, 26.9219, 28.2500],  # Second batch
            [45.5156, 46.8438, 48.1719, 49.5000] # Third batch
        ], dtype=torch.float64)
        
        expected_T = torch.tensor([
            [1.75, 7.0625, 12.3750, 17.6875],  # First batch
            [23.0000, 28.3125, 33.6250, 38.9375],  # Second batch
            [44.2500, 49.5625, 54.8750, 60.1875],  # Third batch
            [1.0, 0.0, 0.0, 0.0]   # New batch with initial condition
        ], dtype=torch.float64)

        # print("==== test_4x4_growing ====")
        # print(f"U = {self.U_4x4}")
        # print(f"v_s = {self.v_s_4x4}")
        # print(f"v_t = {self.v_t_4x4}")
        # print(f"S = {S}")
        # print(f"T = {T}")
        
        self.assertEqual(S.shape, (4, 4))  # (batch_size+1) x n
        self.assertEqual(T.shape, (4, 4))  # (batch_size+1) x n
        self.assertTrue(torch.allclose(S, expected_S, rtol=1e-5))
        self.assertTrue(torch.allclose(T, expected_T, rtol=1e-5))

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
        
        # Create two simple time series
        self.X = torch.tensor([[[0.0], [1.0], [2.0]], 
                              [[3.0], [4.0], [5.0]]], dtype=torch.float64).to(self.device)
        self.Y = torch.tensor([[[0.0], [0.5], [1.0]],
                              [[1.5], [2.0], [2.5]]], dtype=torch.float64).to(self.device)
        
        # Set up ksig static kernel
        self.static_kernel = ksig.static.kernels.LinearKernel()
        
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
        ksig_trunc_kernel = ksig.kernels.SignatureKernel(n_levels=21, order=0, normalize=False, 
                                                        static_kernel=self.static_kernel)
        ksig_trunc_result = ksig_trunc_kernel(X_cpu, Y_cpu)
        
        # 4. PowerSig implementation
        # Convert to derivatives
        dX = torch_compute_derivative_batch(self.X)
        dY = torch_compute_derivative_batch(self.Y)
        
        # Compute gram matrix
        powersig_results = torch.zeros((self.X.shape[0], self.Y.shape[0]), dtype=torch.float64, device=self.device)
        order = 32
        
        for i in range(dX.shape[0]):
            for j in range(dY.shape[0]):
                powersig_results[i, j] = batch_compute_gram_entry(dX[i], dY[j], None, order)
        
        # Move results to CPU for comparison
        if powersig_results.is_cuda:
            powersig_results = powersig_results.cpu()
        
        # Print all results for comparison
        print(f"SigKernel result:\n{sig_kernel_result}")
        print(f"KSig PDE result:\n{ksig_pde_result}")
        print(f"KSig truncated signature result:\n{ksig_trunc_result}")
        print(f"PowerSig result:\n{powersig_results}")
        
        # Convert numpy arrays to PyTorch tensors if needed
        if not isinstance(ksig_pde_result, torch.Tensor):
            ksig_pde_result = torch.tensor(ksig_pde_result, dtype=torch.float64)
        
        if not isinstance(ksig_trunc_result, torch.Tensor):
            ksig_trunc_result = torch.tensor(ksig_trunc_result, dtype=torch.float64)
        
        # Check that results are close to each other
        self.assertTrue(torch.allclose(sig_kernel_result, ksig_pde_result, rtol=1e-2), 
                        f"SigKernel and KSig PDE results differ significantly\n{sig_kernel_result}\n{ksig_pde_result}")
        self.assertTrue(torch.allclose(sig_kernel_result, ksig_trunc_result, rtol=1e-2), 
                        f"SigKernel and KSig truncated results differ significantly")
        self.assertTrue(torch.allclose(sig_kernel_result, powersig_results, rtol=1e-2), 
                        f"SigKernel and PowerSig results differ significantly")


if __name__ == '__main__':
    # To run all tests
    unittest.main()
    
    # To run only TestBuildStencil
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBuildStencil)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchComputeBoundaries)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchADMForDiagonal)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestSignatureKernelConsistency)
    # unittest.TextTestRunner().run(suite)
    
    # To run a specific test method
    # suite = unittest.TestLoader().loadTestsFromName('TestBatchComputeBoundaries.test_2x2_values')
    # unittest.TextTestRunner().run(suite)
