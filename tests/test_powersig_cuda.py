import unittest
import numpy as np
import cupy as cp

from powersig.cuda import cuda_compute_gram_entry_cooperative, cuda_compute_next_boundary_inplace, MAX_ORDER, cuda_compute_gram_entry
from powersig.powersig_cupy import batch_ADM_for_diagonal, batch_compute_boundaries, build_stencil, build_stencil_s, build_stencil_t, compute_vandermonde_vectors

class TestCudaKernel(unittest.TestCase):
    def setUp(self):
        # Create test data for all tests
        self.order = 4  # Small order for testing
        
        # Create a simple stencil
        self.stencil = build_stencil(self.order)
        self.S_debug = cp.zeros((3, self.order), dtype=cp.float64)
        self.T_debug = cp.zeros((3, self.order), dtype=cp.float64)
        # Create Vandermonde vectors
        ds = 10
        dt = 10
        self.v_s, self.v_t = compute_vandermonde_vectors(ds, dt, self.order, cp.float64)
        self.psi_s = build_stencil_s(self.v_s, self.order, cp.float64)
        self.psi_t = build_stencil_t(self.v_t, self.order, cp.float64)
        
        # Create dummy rho values
        self.rho = cp.array([0.5, 0.6, 0.7], dtype=cp.float64)

    def test_growing_diagonal(self):
        """Test case 1: Diagonal length > 1, skip_first=False, skip_last=False (Growing diagonal)"""
        # Setup
        skip_first = False
        skip_last = False
        dlen = 3  # Diagonal length
        
        # Initialize input/output arrays with random values
        S_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        T_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        S_out = cp.zeros((dlen+1, self.order), dtype=cp.float64)  # Growing diagonal
        T_out = cp.zeros((dlen+1, self.order), dtype=cp.float64)  # Growing diagonal
        

        # Run kernel
        threadsperblock = (32, 32)
        blockspergrid = (dlen,)
        
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            self.v_s, 
            self.v_t, 
            self.psi_s, 
            self.psi_t, 
            self.rho[:dlen], 
            S_in, 
            T_in, 
            S_out, 
            T_out, 
            skip_first, 
            skip_last
        )
        


    def test_staying_same_skip_first(self):
        """Test case 2: Diagonal length > 1, skip_first=True, skip_last=False (Staying same size)"""
        # Setup
        skip_first = True
        skip_last = False
        dlen = 3  # Diagonal length
        
        # Initialize input/output arrays with random values
        S_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        T_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        S_out = cp.zeros((dlen, self.order), dtype=cp.float64)  # Same size
        T_out = cp.zeros((dlen, self.order), dtype=cp.float64)  # Same size
        

        
        # Run kernel
        threadsperblock = (32, 32)
        blockspergrid = (dlen,)
        
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            self.v_s, 
            self.v_t, 
            self.psi_s, 
            self.psi_t, 
            self.rho[:dlen], 
            S_in, 
            T_in, 
            S_out, 
            T_out, 
            skip_first, 
            skip_last,
        )
        
     

    def test_staying_same_skip_last(self):
        """Test case 3: Diagonal length > 1, skip_first=False, skip_last=True (Staying same size)"""
        # Setup
        skip_first = False
        skip_last = True
        dlen = 3  # Diagonal length
        
        # Initialize input/output arrays with random values
        S_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        T_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        S_out = cp.zeros((dlen, self.order), dtype=cp.float64)  # Same size
        T_out = cp.zeros((dlen, self.order), dtype=cp.float64)  # Same size
        cp1_s = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        cp2_t = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U_s = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U_t = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U_s[:] = self.psi_s
        U_t[:] = self.psi_t
        # Run kernel
        threadsperblock = (32, 32)
        blockspergrid = (dlen,)
        
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            self.v_s, 
            self.v_t, 
            self.psi_s, 
            self.psi_t, 
            self.rho[:dlen], 
            S_in, 
            T_in, 
            S_out, 
            T_out, 
            skip_first, 
            skip_last,
            cp1_s,
            cp2_t,
        )

        U = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U = batch_ADM_for_diagonal(self.rho[:dlen], U, S_in, T_in, self.stencil)
        rho_powers = self.rho[:dlen].reshape(-1,1) ** cp.arange(self.order)
        # Loop over exponents
        for exponent in range(self.order):
                        
            # # Update rows using broadcasting
            # U[:, exponent, exponent+1:] *= S_in[:, exponent+1:] * rho_powers[:,exponent].reshape(-1,1)
            
            # # Update columns using broadcasting
            # U[:, exponent:, exponent] *= T_in[:, exponent:] * rho_powers[:,exponent].reshape(-1,1)
            rho_power =  rho_powers[:,exponent].reshape(-1,1)
            s = S_in[:, 1:S_in.shape[1]-exponent]  
            t = T_in[:, :T_in.shape[1]-exponent]
            U_s[:, exponent, exponent+1:] *= s
            U_s[:, exponent, exponent+1:] *= rho_power
            U_s[:, exponent:, exponent] *= t
            U_s[:, exponent:, exponent] *= rho_power
        
            U_t[:, exponent, exponent+1:] *= s
            U_t[:, exponent, exponent+1:] *= rho_power
            U_t[:, exponent:, exponent] *= t
            U_t[:, exponent:, exponent] *= rho_power

        expected_U_t = U_t.sum(axis=1)
        expected_U_s = U_s.sum(axis=2)
        expected_S = self.v_t @ U
        expected_T = U @ self.v_s
        
        # batch_compute_boundaries(U, S_in, T_in, self.v_s, self.v_t, skip_first, skip_last)
        
        # Assertions

        self.assertTrue(cp.allclose(expected_U_s, cp1_s.sum(axis=2)))
        self.assertTrue(cp.allclose(expected_U_t, cp2_t.sum(axis=1)))
        self.assertTrue(cp.allclose(expected_U_s, T_out))
        self.assertTrue(cp.allclose(expected_U_t, S_out))
       
        # Assertions for test_staying_same_skip_last
        # Check that S_in and S_out are equal
        self.assertTrue(cp.allclose(expected_S, S_out), f"expected_S and S_out should be equal.\nexpected_S:\n{expected_S}\nS_out:\n{S_out}")
        
        # Check that T_in and T_out are equal
        self.assertTrue(cp.allclose(expected_T, T_out), f"expected_T and T_out should be equal.\nexpected_T:\n{expected_T}\nT_out:\n{T_out}")
        

    def test_shrinking_diagonal(self):
        """Test case 4: Diagonal length > 1, skip_first=True, skip_last=True (Shrinking diagonal)"""
        # Setup
        skip_first = True
        skip_last = True
        dlen = 3  # Diagonal length
        
        # Initialize input/output arrays with random values
        S_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        T_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        S_out = cp.zeros((dlen-1, self.order), dtype=cp.float64)  # Shrinking diagonal
        T_out = cp.zeros((dlen-1, self.order), dtype=cp.float64)  # Shrinking diagonal
        
        # Run kernel
        threadsperblock = (32, 32)
        blockspergrid = (dlen,)
        
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            self.v_s, 
            self.v_t, 
            self.psi_s, 
            self.psi_t, 
            self.rho[:dlen], 
            S_in, 
            T_in, 
            S_out, 
            T_out, 
            skip_first, 
            skip_last,
        )
        
        U = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U = batch_ADM_for_diagonal(self.rho[:dlen], U, S_in, T_in, self.stencil)
        batch_compute_boundaries(U, S_in, T_in, self.v_s, self.v_t, skip_first, skip_last)
    

    def test_terminal_case(self):
        """Test case 5: Diagonal length = 1, skip_first=True, skip_last=True (Terminal case)"""
        # Setup
        skip_first = True
        skip_last = True
        dlen = 1  # Single element diagonal
        
        # Initialize input/output arrays with random values
        S_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        T_in = cp.random.random((dlen, self.order), dtype=cp.float64)
        S_out = cp.zeros((dlen, self.order), dtype=cp.float64)
        T_out = cp.zeros((dlen, self.order), dtype=cp.float64)
        
        
        # Run kernel
        threadsperblock = (32, 32)
        blockspergrid = (dlen,)
        
        cuda_compute_next_boundary_inplace[blockspergrid, threadsperblock](
            self.v_s, 
            self.v_t, 
            self.psi_s, 
            self.psi_t, 
            self.rho[:dlen], 
            S_in, 
            T_in, 
            S_out, 
            T_out, 
            skip_first, 
            skip_last,
            self.S_debug,
            self.T_debug
        )
        
        U = cp.zeros((dlen, self.order, self.order), dtype=cp.float64)
        U = batch_ADM_for_diagonal(self.rho[:dlen], U, S_in, T_in, self.stencil)
        batch_compute_boundaries(U, S_in, T_in, self.v_s, self.v_t, skip_first, skip_last)
        # Assertions
        # The terminal case should calculate the final value in T_out[0,:]
        # The value should be non-zero due to the computation
        self.assertTrue(cp.any(T_out[0, :] != 0.0))
        
        # Check that T_out has received the computed value from the special terminal case code path
        self.assertFalse(cp.allclose(T_out[0, :], T_in[0, :]))


class TestCudaGramEntry(unittest.TestCase):
    def test_cuda_compute_gram_entry(self):
        """Test the full cuda_compute_gram_entry function with small random paths"""
        # Create small random paths
        dX = cp.random.random((8, 2)).astype(cp.float64)
        dY = cp.random.random((8, 2)).astype(cp.float64)
        
        # Set a small order for testing
        order = 4
        
        # Run the function
        result = cuda_compute_gram_entry_cooperative(dX, dY, order)
        
        # Assertions
        # Result should be a scalar
        self.assertIn(type(result), (float, np.float64, cp.ndarray))
        
        # If it's a CuPy array, it should be a scalar (size 1)
        if isinstance(result, cp.ndarray):
            self.assertEqual(result.size, 1)
            
        # Result should be finite
        if isinstance(result, cp.ndarray):
            self.assertTrue(cp.isfinite(result).all())
        else:
            self.assertTrue(np.isfinite(result))
        
        # Test different path lengths
        dX2 = cp.random.random((12, 2)).astype(cp.float64)
        dY2 = cp.random.random((6, 2)).astype(cp.float64)
        
        result2 = cuda_compute_gram_entry_cooperative(dX2, dY2, order)
        self.assertIn(type(result2), (float, np.float64, cp.ndarray))
        
        # Result should be different for different inputs
        if isinstance(result, cp.ndarray) and isinstance(result2, cp.ndarray):
            self.assertFalse(cp.allclose(result, result2))
        else:
            self.assertNotEqual(result, result2)


if __name__ == '__main__':
    # unittest.main() 
    suite = unittest.TestSuite()
    # suite.addTest(TestCudaKernel('test_growing_diagonal'))
    # suite.addTest(TestCudaKernel('test_staying_same_skip_first'))
    # suite.addTest(TestCudaKernel('test_staying_same_skip_last'))
    # suite.addTest(TestCudaKernel('test_shrinking_diagonal'))
    # suite.addTest(TestCudaKernel('test_terminal_case'))
    suite.addTest(TestCudaGramEntry('test_cuda_compute_gram_entry'))
    unittest.TextTestRunner().run(suite)