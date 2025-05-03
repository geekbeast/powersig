import unittest
import jax.numpy as jnp
import numpy as np
import torch

from powersig.jax import batch_ADM_for_diagonal
from powersig.jax import batch_compute_boundaries
from powersig.jax import compute_vandermonde_vectors
from powersig.jax import build_stencil
from powersig.jax import batch_compute_gram_entry
from powersig.util.jax_series import jax_compute_derivative_batch
import ksig
import ksig.static.kernels
from sigkernel import sigkernel

class TestBatchADMForDiagonal(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.rho_2x2 = jnp.array([0.5, 0.7], dtype=jnp.float64)  # batch_size = 2
        self.S_2x2 = jnp.array([
            [10.0, 30.0],
            [100.0,300.0]
        ], dtype=jnp.float64)
        self.T_2x2 = jnp.array([
            [10.0, 20.0],
            [100.0,200.0]
        ], dtype=jnp.float64)
        self.stencil_2x2 = jnp.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=jnp.float64)  # n x n stencil
        # Pre-allocated buffer for 2x2 case
        self.U_buf_2x2 = jnp.empty((2, 2, 2), dtype=jnp.float64)

        # Common test data for 3x3 case
        self.rho_3x3 = jnp.array([0.3], dtype=jnp.float64)  # batch_size = 1
        self.S_3x3 = jnp.array([[1000.0, 4000.0, 5000.0]], dtype=jnp.float64)
        self.T_3x3 = jnp.array([[1000.0, 2000.0, 3000.0]], dtype=jnp.float64)
        self.stencil_3x3 = jnp.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=jnp.float64)  # n x n stencil
        # Pre-allocated buffer for 3x3 case
        self.U_buf_3x3 = jnp.empty((1, 3, 3), dtype=jnp.float64)

        # Common test data for 4x4 case
        self.rho_4x4 = jnp.array([0.4, 0.6, 0.8], dtype=jnp.float64)  # batch_size = 3
        self.S_4x4 = jnp.array(
            [
                [10000.0, 50000.0, 60000.0, 70000.0],
                [100000.0, 500000.0, 600000.0, 700000.0],
                [1000000.0, 5000000.0, 6000000.0, 7000000.0],
            ],
            dtype=jnp.float64,
        )
        self.T_4x4 = jnp.array(
            [
                [10000.0, 20000.0, 30000.0, 40000.0],  # First batch
                [100000.0, 200000.0, 300000.0, 400000.0],  # Second batch
                [1000000.0, 2000000.0, 3000000.0, 4000000.0],  # Third batch
            ],
            dtype=jnp.float64,
        )
        
        self.stencil_4x4 = jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], dtype=jnp.float64)  # n x n stencil
        # Pre-allocated buffer for 4x4 case
        self.U_buf_4x4 = jnp.empty((3, 4, 4), dtype=jnp.float64)

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
        expected_first = jnp.array([
            [1.0 * 10.0 * 1.0, 2.0 * 30.0 * 1.0],  # First row: rho^0
            [3.0 * 20.0 * 1.0, 4.0 * 10.0 * 0.5]   # First column: rho^0, remaining: rho^1
        ], dtype=jnp.float64)
        print(f"Expected: {expected_first}")
        print(f"Result: {result[0]}")
        self.assertTrue(np.allclose(np.array(result[0]), np.array(expected_first), rtol=1e-5))

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
        expected_second = jnp.array([
            [1.0 * 100.0 * 1.0, 2.0 * 300.0 * 1.0],  # First row: rho^0
            [3.0 * 200.0 * 1.0, 4.0 * 100.0 * 0.7]   # First column: rho^0, remaining: rho^1
        ], dtype=jnp.float64)

        self.assertTrue(np.allclose(np.array(result[1]), np.array(expected_second), rtol=1e-5))

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
        expected = jnp.array([
            [1.0 * 1000.0 * 1.0, 2.0 * 4000.0 * 1.0, 3.0 * 5000.0 * 1.0],  # First row: rho^0
            [4.0 * 2000.0 * 1.0, 5.0 * 1000.0 * 0.3, 6.0 * 4000.0 * 0.3],  # First column: rho^0, remaining row/column: rho^1
            [7.0 * 3000.0 * 1.0, 8.0 * 2000.0 * 0.3, 9.0 * 1000.0 * 0.09]  # First column: rho^0, remaining row/column: rho^1, remaining: rho^2
        ], dtype=jnp.float64)

        self.assertTrue(np.allclose(np.array(result[0]), np.array(expected), rtol=1e-5))

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
        expected_first = jnp.array([
            [1.0 * 10000.0 * 1.0, 2.0 * 50000.0 * 1.0, 3.0 * 60000.0 * 1.0, 4.0 * 70000.0 * 1.0],  # First row: rho^0
            [5.0 * 20000.0 * 1.0, 6.0 * 10000.0 * 0.4, 7.0 * 50000.0 * 0.4, 8.0 * 60000.0 * 0.4],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 30000.0 * 1.0, 10.0 * 20000.0 * 0.4, 11.0 * 10000.0 * 0.16, 12.0 * 50000.0 * 0.16],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 40000.0 * 1.0, 14.0 * 30000.0 * 0.4, 15.0 * 20000.0 * 0.16, 16.0 * 10000.0 * 0.064]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=jnp.float64)

        # Second batch (rho = 0.6):
        # Same pattern but with rho = 0.6 and coefficients scaled by 10
        expected_second = jnp.array([
            [1.0 * 100000.0 * 1.0, 2.0 * 500000.0 * 1.0, 3.0 * 600000.0 * 1.0, 4.0 * 700000.0 * 1.0],  # First row: rho^0
            [5.0 * 200000.0 * 1.0, 6.0 * 100000.0 * 0.6, 7.0 * 500000.0 * 0.6, 8.0 * 600000.0 * 0.6],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 300000.0 * 1.0, 10.0 * 200000.0 * 0.6, 11.0 * 100000.0 * 0.36, 12.0 * 500000.0 * 0.36],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 400000.0 * 1.0, 14.0 * 300000.0 * 0.6, 15.0 * 200000.0 * 0.36, 16.0 * 100000.0 * 0.216]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=jnp.float64)

        # Third batch (rho = 0.8):
        # Same pattern but with rho = 0.8 and coefficients scaled by 100
        expected_third = jnp.array([
            [1.0 * 1000000.0 * 1.0, 2.0 * 5000000.0 * 1.0, 3.0 * 6000000.0 * 1.0, 4.0 * 7000000.0 * 1.0],  # First row: rho^0
            [5.0 * 2000000.0 * 1.0, 6.0 * 1000000.0 * 0.8, 7.0 * 5000000.0 * 0.8, 8.0 * 6000000.0 * 0.8],  # First column: rho^0, remaining row/column: rho^1
            [9.0 * 3000000.0 * 1.0, 10.0 * 2000000.0 * 0.8, 11.0 * 1000000.0 * 0.64, 12.0 * 5000000.0 * 0.64],  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2
            [13.0 * 4000000.0 * 1.0, 14.0 * 3000000.0 * 0.8, 15.0 * 2000000.0 * 0.64, 16.0 * 1000000.0 * 0.512]  # First column: rho^0, remaining row/column: rho^1, remaining row/column: rho^2, remaining: rho^3
        ], dtype=jnp.float64)

        print(f"Expected: {expected_first}")
        print(f"Result: {result[0]}")

        print(f"Expected: {expected_second}")
        print(f"Result: {result[1]}")

        print(f"Expected: {expected_third}")
        print(f"Result: {result[2]}")

        self.assertTrue(np.allclose(np.array(result[0]), np.array(expected_first), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(result[1]), np.array(expected_second), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(result[2]), np.array(expected_third), rtol=1e-5))

class TestBatchComputeBoundaries(unittest.TestCase):
    def setUp(self):
        # Common test data for 2x2 case
        self.U_2x2 = jnp.array([
            [[1.0, 2.0],
             [3.0, 4.0]],
            [[5.0, 6.0],
             [7.0, 8.0]]
        ], dtype=jnp.float64)  # batch_size=2, n=2
        self.ds_2x2 = 0.5
        self.dt_2x2 = 0.5
        self.v_s_2x2, self.v_t_2x2 = compute_vandermonde_vectors(self.ds_2x2, self.dt_2x2, 2, self.U_2x2.dtype)
        # Pre-allocated buffers for S and T for 2x2 case
        self.S_buf_2x2 = jnp.empty((3, 2), dtype=jnp.float64)  # Max size needed is batch_size+1
        self.T_buf_2x2 = jnp.empty((3, 2), dtype=jnp.float64)  # Max size needed is batch_size+1

        # Common test data for 3x3 case
        self.U_3x3 = jnp.array([
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        ], dtype=jnp.float64)  # batch_size=1, n=3
        self.ds_3x3 = 0.3
        self.dt_3x3 = 0.3
        self.v_s_3x3, self.v_t_3x3 = compute_vandermonde_vectors(self.ds_3x3, self.dt_3x3, 3, self.U_3x3.dtype)
        # Pre-allocated buffers for S and T for 3x3 case
        self.S_buf_3x3 = jnp.empty((2, 3), dtype=jnp.float64)  # Max size needed is batch_size+1
        self.T_buf_3x3 = jnp.empty((2, 3), dtype=jnp.float64)  # Max size needed is batch_size+1

        # Common test data for 4x4 case with batch size 3
        self.U_4x4 = jnp.array([
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
        ], dtype=jnp.float64)  # batch_size=3, n=4
        self.ds_4x4 = 0.25
        self.dt_4x4 = 0.25
        self.v_s_4x4, self.v_t_4x4 = compute_vandermonde_vectors(self.ds_4x4, self.dt_4x4, 4, self.U_4x4.dtype)
        # Pre-allocated buffers for S and T for 4x4 case
        self.S_buf_4x4 = jnp.empty((4, 4), dtype=jnp.float64)  # Max size needed is batch_size+1
        self.T_buf_4x4 = jnp.empty((4, 4), dtype=jnp.float64)  # Max size needed is batch_size+1 

    def test_2x2_shrinking(self):
        """Test 2x2 case with both skip_first and skip_last=True (shrinking)"""
        S, T = batch_compute_boundaries(
            self.U_2x2, 
            self.S_buf_2x2, 
            self.T_buf_2x2, 
            self.v_s_2x2, 
            self.v_t_2x2, 
            skip_first=True, 
            skip_last=True
        )
        
        # For the 2x2 case with skip_first and skip_last=True:
        # S: v_t . U[0]
        # T: U[1] . v_s
        # v_s = [1, 0.5]
        # v_t = [1, 0.5]
        
        # S values (v_t.U[0]):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        expected_S = jnp.array([[2.5, 4]], dtype=jnp.float64)
        
        # T values (U[1].v_s):
        # v_s = [1, 0.5]
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        expected_T = jnp.array([[8, 11]], dtype=jnp.float64)
        
        # self.assertEqual(S.shape, (1, 2))  # (batch_size-1) x n
        # self.assertEqual(T.shape, (1, 2))  # (batch_size-1) x n
        self.assertTrue(np.allclose(np.array(S[:1,:2]), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T[:1,:2]), np.array(expected_T), rtol=1e-5))

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
        expected_S = jnp.array([
            [1, 0],     # Initial bottom boundary
            [2.5, 4],   # First batch
            [8.5, 10]   # Second batch
        ], dtype=jnp.float64)
        
        # T values (U.v_s):
        # v_s = [1, 0.5]
        # [1,2] * [1,0.5] = 1*1 + 2*0.5 = 2
        # [3,4] * [1,0.5] = 3*1 + 4*0.5 = 5
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        expected_T = jnp.array([
            [2, 5],     # First batch
            [8, 11],   # Second batch
            [1, 0]     # Initial right boundary
        ], dtype=jnp.float64)
        
        # self.assertEqual(S.shape, (3, 2))  # (batch_size+1) x n
        # self.assertEqual(T.shape, (3, 2))  # (batch_size+1) x n
        self.assertTrue(np.allclose(np.array(S[:3,:2]), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T[:3,:2]), np.array(expected_T), rtol=1e-5))

    def test_2x2_staying_same(self):
        """Test 2x2 case with only skip_first=True (staying same size)"""
        S, T = batch_compute_boundaries(
            self.U_2x2, 
            self.S_buf_2x2, 
            self.T_buf_2x2, 
            self.v_s_2x2, 
            self.v_t_2x2, 
            skip_first=True, 
            skip_last=False
        )
        
        # For the 2x2 case with skip_first=True, skip_last=False:
        # S: v_t.U for all batches
        # T: [1,0,0] for the first element (bottom boundary), U[1:].v_s for the rest
        # v_s = [1, 0.5]
        # v_t = [1, 0.5]
        
        # S values (v_t.U):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        # [1,0.5] * [5,7] = 1*5 + 0.5*7 = 8.5
        # [1,0.5] * [6,8] = 1*6 + 0.5*8 = 10
        expected_S = jnp.array([
            [2.5, 4],   # First batch
            [8.5, 10]   # Second batch
        ], dtype=jnp.float64)
        
        # T values (U[1:].v_s):
        # v_s = [1, 0.5]
        # Initial boundary first, then:
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        expected_T = jnp.array([
            [1, 0],     # Initial bottom boundary
            [8, 11]     # Second batch
        ], dtype=jnp.float64)
        
        # Test with skip_last=True, skip_first=False
        S2, T2 = batch_compute_boundaries(
            self.U_2x2, 
            self.S_buf_2x2, 
            self.T_buf_2x2, 
            self.v_s_2x2, 
            self.v_t_2x2, 
            skip_first=False, 
            skip_last=True
        )
        
        # For the 2x2 case with skip_first=False, skip_last=True:
        # S: v_t.U[:-1] for all batches except the last, which gets [1,0,0] (right boundary)
        # T: U.v_s for all batches
        # v_s = [1, 0.5]
        # v_t = [1, 0.5]
        
        # S values (v_t.U[:-1]):
        # v_t = [1, 0.5]
        # [1,0.5] * [1,3] = 1*1 + 0.5*3 = 2.5
        # [1,0.5] * [2,4] = 1*2 + 0.5*4 = 4
        # Last element is [1,0]
        expected_S2 = jnp.array([
            [2.5, 4],   # First batch
            [1, 0]      # Initial right boundary
        ], dtype=jnp.float64)
        
        # T values (U.v_s):
        # v_s = [1, 0.5]
        # [1,2] * [1,0.5] = 1*1 + 2*0.5 = 2
        # [3,4] * [1,0.5] = 3*1 + 4*0.5 = 5
        # [5,6] * [1,0.5] = 5*1 + 6*0.5 = 8
        # [7,8] * [1,0.5] = 7*1 + 8*0.5 = 11
        expected_T2 = jnp.array([
            [2, 5],     # First batch
            [8, 11]     # Second batch
        ], dtype=jnp.float64)
        
        # self.assertEqual(S.shape, (2, 2))  # batch_size x n
        # self.assertEqual(T.shape, (2, 2))  # batch_size x n
        self.assertTrue(np.allclose(np.array(S[:2,:2]), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T[:2,:2]), np.array(expected_T), rtol=1e-5))
        
        # self.assertEqual(S2.shape, (2, 2))  # batch_size x n
        # self.assertEqual(T2.shape, (2, 2))  # batch_size x n
        self.assertTrue(np.allclose(np.array(S2[:2,:2]), np.array(expected_S2), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T2[:2,:2]), np.array(expected_T2), rtol=1e-5))

    def test_3x3_shrinking(self):
        """Test 3x3 case with both skip_first and skip_last=True (shrinking)"""
        # For shrinking, we would need at least 2 batches, but our 3x3 test case has only 1 batch
        # So we don't expect meaningful results
        # But we can test that the function runs without errors and returns the expected shapes
        S, T = batch_compute_boundaries(
            self.U_3x3, 
            self.S_buf_3x3, 
            self.T_buf_3x3, 
            self.v_s_3x3, 
            self.v_t_3x3, 
            skip_first=True, 
            skip_last=True
        )
        
        # We expect empty outputs because we're trying to shrink a batch of size 1
        # self.assertEqual(S.shape, (0, 3))  # (batch_size-1) x n
        # self.assertEqual(T.shape, (0, 3))  # (batch_size-1) x n
        self.assertTrue(np.allclose(np.array(S), np.array(self.S_buf_3x3), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T), np.array(self.T_buf_3x3), rtol=1e-5))
        
    def test_3x3_growing(self):
        """Test 3x3 case with both skip_first and skip_last=False (growing)"""
        S, T = batch_compute_boundaries(
            self.U_3x3, 
            self.S_buf_3x3, 
            self.T_buf_3x3, 
            self.v_s_3x3, 
            self.v_t_3x3, 
            skip_first=False, 
            skip_last=False
        )
        
        # For the 3x3 case with skip_first and skip_last=False:
        # S: [1,0,0,0] for the first element (bottom boundary), v_t.U for the rest
        # T: U.v_s for all batches, [1,0,0,0] for the added element (right boundary)
        # v_s = [1, 0.3, 0.09]
        # v_t = [1, 0.3, 0.09]
        
        # S values (v_t.U):
        # v_t = [1, 0.3, 0.09]
        # [1,0.3,0.09] * [1,4,7] = 1*1 + 0.3*4 + 0.09*7 = 2.83
        # [1,0.3,0.09] * [2,5,8] = 1*2 + 0.3*5 + 0.09*8 = 4.22
        # [1,0.3,0.09] * [3,6,9] = 1*3 + 0.3*6 + 0.09*9 = 5.61
        expected_S = jnp.array([
            [1, 0, 0],     # Initial bottom boundary
            [2.83, 4.22, 5.61]   # First batch
        ], dtype=jnp.float64)
        
        # T values (U.v_s):
        # v_s = [1, 0.3, 0.09]
        # [1,2,3] * [1,0.3,0.09] = 1*1 + 2*0.3 + 3*0.09 = 1.87
        # [4,5,6] * [1,0.3,0.09] = 4*1 + 5*0.3 + 6*0.09 = 6.04
        # [7,8,9] * [1,0.3,0.09] = 7*1 + 8*0.3 + 9*0.09 = 10.21
        expected_T = jnp.array([
            [1.87, 6.04, 10.21],   # First batch
            [1, 0, 0]              # Initial right boundary
        ], dtype=jnp.float64)
        
        self.assertEqual(S.shape, (2, 3))  # (batch_size+1) x n
        self.assertEqual(T.shape, (2, 3))  # (batch_size+1) x n

        # print(f"===== test_3x3_growing =====")
        # print(f"S: {S}")
        # print(f"expected_S: {expected_S}")
        # print(f"T: {T}")
        # print(f"expected_T: {expected_T}")
        self.assertTrue(np.allclose(np.array(S), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T), np.array(expected_T), rtol=1e-5))

    def test_3x3_staying_same(self):
        """Test 3x3 case with only skip_first=True (staying same size)"""
        S, T = batch_compute_boundaries(
            self.U_3x3, 
            self.S_buf_3x3, 
            self.T_buf_3x3, 
            self.v_s_3x3, 
            self.v_t_3x3, 
            skip_first=True, 
            skip_last=False
        )
        
        # For the 3x3 case with skip_first=True, skip_last=False:
        # This is a special case because we only have 1 batch in the 3x3 case
        # Skipping the first means we have to provide the bottom boundary [1,0,0]
        # Not skipping the last means we compute v_t.U
        # But since this is the only element, those would go to the same output!
        
        # S values (v_t.U):
        # v_t = [1, 0.3, 0.09]
        # [1,0.3,0.09] * [1,4,7] = 1*1 + 0.3*4 + 0.09*7 = 2.83
        # [1,0.3,0.09] * [2,5,8] = 1*2 + 0.3*5 + 0.09*8 = 4.22
        # [1,0.3,0.09] * [3,6,9] = 1*3 + 0.3*6 + 0.09*9 = 5.61
        expected_S = jnp.array([[2.83, 4.22, 5.61]], dtype=jnp.float64)
        
        # T values:
        # Should be [1,0,0] for the bottom boundary
        expected_T = jnp.array([[1, 0, 0]], dtype=jnp.float64)
        
        # self.assertEqual(S.shape, (1, 3))  # batch_size x n
        # self.assertEqual(T.shape, (1, 3))  # batch_size x n
        self.assertTrue(np.allclose(np.array(S[:1, :3]), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T[:1, :3]), np.array(expected_T), rtol=1e-5))
        
        # Test with skip_last=True, skip_first=False
        S2, T2 = batch_compute_boundaries(
            self.U_3x3, 
            self.S_buf_3x3, 
            self.T_buf_3x3, 
            self.v_s_3x3, 
            self.v_t_3x3, 
            skip_first=False, 
            skip_last=True
        )
        
        # For the 3x3 case with skip_first=False, skip_last=True:
        # Not skipping the first means we compute U.v_s
        # Skipping the last means we have to provide the right boundary [1,0,0]
        
        # S values:
        # Should be [1,0,0] for the right boundary
        expected_S2 = jnp.array([[1, 0, 0]], dtype=jnp.float64)
        
        # T values (U.v_s):
        # v_s = [1, 0.3, 0.09]
        # [1,2,3] * [1,0.3,0.09] = 1*1 + 2*0.3 + 3*0.09 = 1.87
        # [4,5,6] * [1,0.3,0.09] = 4*1 + 5*0.3 + 6*0.09 = 6.04
        # [7,8,9] * [1,0.3,0.09] = 7*1 + 8*0.3 + 9*0.09 = 10.21
        expected_T2 = jnp.array([[1.87, 6.04, 10.21]], dtype=jnp.float64)
        
        # self.assertEqual(S2.shape, (1, 3))  # batch_size x n
        # self.assertEqual(T2.shape, (1, 3))  # batch_size x n
        self.assertTrue(np.allclose(np.array(S2[:1, :3]), np.array(expected_S2), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T2[:1, :3]), np.array(expected_T2), rtol=1e-5))

    def test_4x4_shrinking(self):
        """Test 4x4 case with both skip_first and skip_last=True (shrinking)"""
        S, T = batch_compute_boundaries(
            self.U_4x4, 
            self.S_buf_4x4, 
            self.T_buf_4x4, 
            self.v_s_4x4, 
            self.v_t_4x4, 
            skip_first=True, 
            skip_last=True
        )
        
        # For the 4x4 case with skip_first and skip_last=True:
        # S: v_t . U[:-1]
        # T: U[1:] . v_s
        # v_s = [1, 0.25, 0.0625, 0.015625]
        # v_t = [1, 0.25, 0.0625, 0.015625]
        
        # Compute some expected values:
        # S values (v_t.U[0]):
        # v_t = [1, 0.25, 0.0625, 0.015625]
        # [1,0.25,0.0625,0.015625] * [1,5,9,13] = 1*1 + 0.25*5 + 0.0625*9 + 0.015625*13 = 3.015625
        # [1,0.25,0.0625,0.015625] * [2,6,10,14] = 1*2 + 0.25*6 + 0.0625*10 + 0.015625*14 = 4.34375
        # [1,0.25,0.0625,0.015625] * [3,7,11,15] = 1*3 + 0.25*7 + 0.0625*11 + 0.015625*15 = 5.671875
        # [1,0.25,0.0625,0.015625] * [4,8,12,16] = 1*4 + 0.25*8 + 0.0625*12 + 0.015625*16 = 7
        
        # T values (U[1:].v_s):
        # v_s = [1, 0.25, 0.0625, 0.015625]
        # [17,18,19,20] * [1,0.25,0.0625,0.015625] = 17*1 + 18*0.25 + 19*0.0625 + 20*0.015625 = 23
        # [21,22,23,24] * [1,0.25,0.0625,0.015625] = 21*1 + 22*0.25 + 23*0.0625 + 24*0.015625 = 28.3125
        # [25,26,27,28] * [1,0.25,0.0625,0.015625] = 25*1 + 26*0.25 + 27*0.0625 + 28*0.015625 = 33.625 
        # [29,30,31,32] * [1,0.25,0.0625,0.015625] = 29*1 + 30*0.25 + 31*0.0625 + 32*0.015625 = 38.9375
        
        expected_S = jnp.array([
            [3.015625, 4.34375, 5.671875, 7],  # First batch
            [24.2656, 25.5938, 26.9219, 28.2500]  # Second batch
        ], dtype=jnp.float64)
        
        expected_T = jnp.array([
            [23, 28.3125, 33.625, 38.9375],  # Second batch
            [44.25, 49.5625, 54.875, 60.1875]  # Third batch
        ], dtype=jnp.float64)
        
        print(f"===== test_4x4_shrinking =====")
        print(f"S: {S}")
        print(f"expected_S: {expected_S}")
        print(f"T: {T}")
        print(f"expected_T: {expected_T}")

        # self.assertEqual(S.shape, (2, 4))  # (batch_size-1) x n
        # self.assertEqual(T.shape, (2, 4))  # (batch_size-1) x n
        self.assertTrue(np.allclose(np.array(S[:2, :4]), np.array(expected_S), rtol=1e-5))
        self.assertTrue(np.allclose(np.array(T[:2, :4]), np.array(expected_T), rtol=1e-5)) 

class TestBuildStencil(unittest.TestCase):
    def setUp(self):
        self.order = 4  # Using a small order for easier testing
        self.dtype = jnp.float64

    def test_stencil_shape(self):
        """Test that the stencil has the correct shape"""
        stencil = build_stencil(self.order, self.dtype)
        self.assertEqual(stencil.shape, (self.order, self.order))

    def test_stencil_values(self):
        """Test that the stencil values are computed correctly"""
        stencil = build_stencil(self.order, self.dtype)
        
        # First row and column should be ones (from initialization)
        self.assertTrue(np.allclose(np.array(stencil[0, :]), np.ones(self.order, dtype=np.float64)))
        self.assertTrue(np.allclose(np.array(stencil[:, 0]), np.ones(self.order, dtype=np.float64)))
        
        # Check some specific values in the stencil
        # For order=4, after cumulative product, the stencil should look like:
        # 1  1  1  1
        # 1 1/1 1/2 1/6
        # 1 1/2 1/4 1/12
        # 1 1/6 1/12 1/36
        expected_values = jnp.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 1/3],
            [1.0, 0.5, 0.25, 1/12],
            [1.0, 1/3, 1/12, 1/36]
        ], dtype=self.dtype)
        
        self.assertTrue(np.allclose(np.array(stencil), np.array(expected_values), rtol=1e-5))

    def test_stencil_diagonals(self):
        """Test that the diagonals are correctly computed"""
        stencil = build_stencil(self.order, self.dtype)
        
        # Check main diagonal
        main_diag = jnp.diag(stencil)
        expected_main_diag = jnp.array([1.0, 1.0, 0.25, 1/36], dtype=self.dtype)
        self.assertTrue(np.allclose(np.array(main_diag), np.array(expected_main_diag), rtol=1e-5))
        
        # Extract diagonals manually for upper and lower since JAX doesn't have diagonal with offset
        # Check first upper diagonal
        upper_diag = jnp.array([stencil[i, i+1] for i in range(self.order-1)])
        expected_upper_diag = jnp.array([1.0, 0.5, 1/12], dtype=self.dtype)
        self.assertTrue(np.allclose(np.array(upper_diag), np.array(expected_upper_diag), rtol=1e-5))
        
        # Check first lower diagonal (should be same as upper diagonal due to symmetry)
        lower_diag = jnp.array([stencil[i+1, i] for i in range(self.order-1)])
        expected_lower_diag = jnp.array([1.0, 0.5, 1/12], dtype=self.dtype)
        self.assertTrue(np.allclose(np.array(lower_diag), np.array(expected_lower_diag), rtol=1e-5))

    def test_stencil_symmetry(self):
        """Test that the stencil is symmetric around the main diagonal"""
        stencil = build_stencil(self.order, self.dtype)
        
        # Check that the matrix is symmetric
        for i in range(self.order):
            for j in range(i+1, self.order):
                self.assertAlmostEqual(float(stencil[i, j]), float(stencil[j, i]), places=10)

    def test_stencil_dtype(self):
        """Test that the stencil has the correct dtype"""
        stencil = build_stencil(self.order, self.dtype)
        self.assertEqual(stencil.dtype, self.dtype)

class TestSignatureKernelConsistency(unittest.TestCase):
    def setUp(self):
        # Create two simple time series
        self.X = jnp.array([[[0.0], [1.0], [2.0]],
                           [[1.0], [3.0], [3.0]],
                           [[2.0], [2.0], [4.0]], 
                           [[3.0], [4.0], [5.0]]], dtype=jnp.float64)
        self.Y = jnp.array([[[0.0], [0.5], [1.0]],
                           [[2.0], [1.0], [2.5]],
                           [[2.7], [3.0], [4.0]], 
                           [[3.0], [2.4], [3.20]], 
                           [[1.5], [2.0], [2.5]]], dtype=jnp.float64)
        
        # Set up ksig static kernel
        self.static_kernel = ksig.static.kernels.LinearKernel()
        
        # Set up sigkernel
        self.dyadic_order = 0
        self.signature_kernel = sigkernel.SigKernel(sigkernel.LinearKernel(), self.dyadic_order)
        
    def test_signature_kernel_consistency(self):
        """Test that different signature kernel implementations give consistent results"""
        # Convert JAX arrays to numpy/torch for sigkernel and ksig
        X_np = np.array(self.X)
        Y_np = np.array(self.Y)
        
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
        
        # 4. PowerSig JAX implementation
        # Convert to derivatives
        from powersig.util.jax_series import jax_compute_derivative_batch
        dX = jax_compute_derivative_batch(self.X)
        dY = jax_compute_derivative_batch(self.Y)
        
        # Compute gram matrix
        powersig_results = jnp.zeros((self.X.shape[0], self.Y.shape[0]), dtype=jnp.float64)
        order = 32
        
        # Using a loop since we don't have a batched version yet
        for i in range(dX.shape[0]):
            for j in range(dY.shape[0]):
                powersig_results = powersig_results.at[i, j].set(
                    batch_compute_gram_entry(dX[i], dY[j], None, order)
                )
        
        # Convert JAX array to numpy for comparison
        powersig_results_np = np.array(powersig_results)
        
        # Print all results for comparison
        print(f"SigKernel result:\n{sig_kernel_result}")
        print(f"KSig PDE result:\n{ksig_pde_result}")
        print(f"KSig truncated signature result:\n{ksig_trunc_result}")
        print(f"PowerSig JAX result:\n{powersig_results_np}")
        
        # Convert numpy arrays to JAX arrays if needed
        if not isinstance(ksig_pde_result, jnp.ndarray):
            ksig_pde_result = jnp.array(ksig_pde_result, dtype=jnp.float64)
        
        if not isinstance(ksig_trunc_result, jnp.ndarray):
            ksig_trunc_result = jnp.array(ksig_trunc_result, dtype=jnp.float64)
        
        if not isinstance(sig_kernel_result, jnp.ndarray):
            sig_kernel_result = jnp.array(sig_kernel_result, dtype=jnp.float64)
        
        # Check that results are close to each other
        self.assertTrue(np.allclose(sig_kernel_result, ksig_pde_result, rtol=1e-2), 
                        f"SigKernel and KSig PDE results differ significantly\n{sig_kernel_result}\n{ksig_pde_result}")
        self.assertTrue(np.allclose(sig_kernel_result, ksig_trunc_result, rtol=1e-2), 
                        f"SigKernel and KSig truncated results differ significantly")
        self.assertTrue(np.allclose(sig_kernel_result, powersig_results_np, rtol=1e-2), 
                        f"SigKernel and PowerSig JAX results differ significantly")


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