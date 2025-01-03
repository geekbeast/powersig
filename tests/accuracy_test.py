import time
import unittest
from math import factorial
from multiprocessing import set_start_method

import numpy
import numpy as np
import torch
from numba.cpython.setobj import set_len
from sigkernel import sigkernel

from powersig.matrixsig import MatrixSig, build_tile_power_series_stencil, build_scaling_for_integration, \
    build_vandermonde_matrix_s, build_vandermonde_matrix_t, diagonal_to_string, get_diagonal_length
from powersig.power_series import SimplePowerSeries, MatrixPowerSeries, build_A1, build_A2, \
    build_integration_gather_matrix_s, build_integration_gather_matrix_t
from powersig.simpesig import SimpleSig
from tests.configuration import TestRun, signature_kernel
from tests.utils import setup_torch


class TestSimplePowerSeriesAccuracy(unittest.TestCase):
    configuration = TestRun()

    @classmethod
    def setUpClass(cls):
        setup_torch()

    def test_power_series_add_and_multiply(self):
        coeffs = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        coeffs = [1.0 / factorial(c) for c in coeffs]
        coeffs = torch.tensor(coeffs).float().cuda()
        threshold = .000001
        series_A = SimplePowerSeries(coeffs, torch.tensor(
            [[0, 0], [1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]).cuda())

        series_B = series_A.deep_clone()

        series_B *= 2

        series_C = series_A + series_B
        a = series_A(1.0, 1.0)
        b = series_B(1.0, 1.0)
        c = series_C(1.0, 1.0)
        assert (b - 2 * a) < threshold, "Values must match under multiplication"
        print(f"Expected: {2 * a}")
        print(f"Actual: {b}")
        expected = a + b
        residual = (c - expected)
        print(f"Expected: {expected}")
        print(f"Actual: {c}")

        assert residual < threshold, f"Values must match under addition. Residual = {residual}"

    def test_power_series(self):
        coeffs = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        coeffs = [1.0 / factorial(c) for c in coeffs]
        coeffs = torch.tensor(coeffs).float().cuda()
        threshold = .000001
        series = SimplePowerSeries(coeffs, torch.tensor(
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]).cuda())
        print(f"Series(1,1) = {series(1, 1)}")
        print(f"Error = {(series(1, 1) - np.e) ** 2}")
        assert ((series(1, 1) - np.e) ** 2 < threshold)
        ps = series.integrate(0, 0)
        r = ps(1, 1)
        print(f"Integrated power series: {ps}")
        print(f"Series at (1,1): {r}")

    def setUp(self):
        print(f"Data shape: {self.__class__.configuration.X.shape}")

    def test_build_gram_matrix(self):
        config = self.__class__.configuration
        max_batch = 10
        start = time.time()
        sk = signature_kernel.compute_Gram(config.X.cpu(), config.X.cpu(), max_batch)
        print(f"SigKernel computation took: {time.time() - start}s")
        print(f"SigKernel Gram Matrix: \n {sk.tolist()}")
        start = time.time()
        simple = SimpleSig(config.X, config.X).compute_gram_matrix()
        print(f"Simple Sig computation took: {time.time() - start}s")
        print(f"Simple Sig computation of gram Matrix: \n {simple.tolist()}")
        mse = torch.mean((sk.cpu() - simple.cpu()) ** 2)
        print(f"MSE SimpleSig versus SigKernel: {mse}")

        start = time.time()
        m = MatrixSig(config.X, config.X).compute_gram_matrix()
        print(f"Matrix Sig computation took: {time.time() - start}s")
        print(f"Matrix Sig computation of gram Matrix: \n {m.tolist()}")
        mse = torch.mean((sk.cpu() - m.cpu()) ** 2)
        print(f"MSE MatrixSig versus SigKernel: {mse}")

        start = time.time()
        sk = signature_kernel.compute_Gram(config.X.cpu(), config.X.cpu(), max_batch)
        print(f"SigKernel computation took: {time.time() - start}s")
        print(f"SigKernel Gram Matrix: \n {sk.tolist()}")


class TestMatrixPowerSeriesAccuracy(unittest.TestCase):
    configuration = TestRun()

    @classmethod
    def setUpClass(cls):
        setup_torch()

    def setUp(self):
        print(f"Data shape: {self.__class__.configuration.X.shape}")

    def test_diagonal_length(self):
        d, rows, cols = 3, 5, 5
        assert get_diagonal_length(d, rows, cols) == 3

    def test_integration_scaling(self):
        '''
        This test just makes sure that matrix that will be broadcast for computing the s,t integral is constructed properly.
        '''
        rho = 16
        s_len = 3
        t_len = 3

        u = torch.zeros([1, 5, 5], dtype=torch.float64)
        u[0, 0, 0] = 1

        s = torch.linspace(0, 1, s_len, dtype=u.dtype, device=u.device)
        t = torch.linspace(0, 1, t_len, dtype=u.dtype, device=u.device)

        print(f"s = {s}")
        print(f"t = {t}")

        scales = build_scaling_for_integration(5,u.device, u.dtype)
        print(f"scales = {scales}")

        print("anti-diagonal starting at 0,0")
        v_s = build_vandermonde_matrix_s(s[:1], 5, u.device, u.dtype,1)
        v_t = build_vandermonde_matrix_t(t[:1], 5, u.device, u.dtype,1)
        print(f"vandermonde matrix s: {v_s}")
        print(f"vandermonde matrix t: {v_t}")

        u_n = torch.clone(u)

        for i in range(5):
            u_step = rho*u_n*scales

            u_n[:, 1:, 1:] = u_step[:,:-1, :-1]
            u_n[:,:1,:] = torch.bmm(v_t, u_step)
            u_n[:,:,:1] = torch.bmm(u_step, v_s)
            u_n[:,:1,:1] = torch.bmm(v_t, u_n[:,:,:1])
            print(f"u_n = {u_n}")
            u+=u_n

        print(f"u = {diagonal_to_string(u)}")

        # Build the next diagonal
        s0 = build_vandermonde_matrix_s(s[:1], 5, u.device, u.dtype)
        t0 = build_vandermonde_matrix_t(t[:1], 5, u.device, u.dtype)
        # u_next =

        print("anti-diagonal starting at  1,0")
        v_s = build_vandermonde_matrix_s(s[range(1,-1,-1)],5,u.device, u.dtype,1)
        v_t = build_vandermonde_matrix_s(t[:2], 5, u.device, u.dtype,1)
        print(f"vandermonde matrix s: {v_s}")
        print(f"vandermonde matrix t: {v_t}")

        for i in range(5):
            u_step = rho * u_n * scales

            u_n[:, 1:, 1:] = u_step[:, :-1, :-1]
            u_n[:, :1, :] = torch.bmm(v_t, u_step)
            u_n[:, :, :1] = torch.bmm(u_step, v_s)
            u_n[:, :1, :1] = torch.bmm(v_t, u_n[:, :, :1])
            print(f"u_n = {u_n}")
            u += u_n

        print(f"u = {diagonal_to_string(u)}")




    def test_integration(self):
        rho = 16
        s_len = 3
        t_len = 3

        initial = torch.zeros([5, 5], dtype=torch.float64)
        initial[0, 0] = 1

        s = torch.linspace(0, 1, s_len, dtype=initial.dtype, device=initial.device)
        t = torch.linspace(0, 1, t_len, dtype=initial.dtype, device=initial.device)

        print(f"s = {s}")
        print(f"t = {t}")

        u = MatrixPowerSeries(initial)
        u_n = u.deep_clone()

        print(f"u_n device = {u_n.coefficients.device}")
        print(f"u_0 = {u}")

        # Derive stencil
        min_ij, denominator = build_tile_power_series_stencil(initial.shape,initial.device)


        g1 = u.build_gather_s(s_min)
        g2 = u.build_gather_t(t_min)
        C = u.coefficients

        min_ij_log_rho = min_ij * math.log(rho)
        new_entries = torch.exp(min_ij_log_rho - denom)

        new_entries.diagonal().__imul__(C[0,0])

        for i in range(1,new_entries.shape[0]):
            # print(f"C[0,{i}] = {C[0,i]}")
            new_entries.diagonal(i).__imul__(C[0,i])
        for j in range(1,new_entries.shape[1]):
            new_entries.diagonal(-j).__imul__(C[j,0])

        print(f"new_entries = {new_entries.tolist()}")

        C[1:,1:] = new_entries
        C[1:, :1] -= torch.mm(new_entries, g1[1:,:])
        C[:1, 1:] -= torch.mm(g2[:,1:],new_entries)

        print(f"Elapsed time: {time.time() - start}")
        print(f"u = {u}")

    def test_matrix_integration(self):
        rho = 16
        s_len = 3
        t_len = 3
        initial = torch.zeros([5, 5], dtype=torch.float64)
        initial[0, 0] = 1
        u_11 = MatrixPowerSeries(torch.diag_embed(torch.tensor([1, 16, 64, 1024 / 9, 1024 / 9], dtype=torch.float64)))

        A1 = build_A1(initial.shape[1], initial.device)
        A2 = build_A2(initial.shape[0], initial.device)

        print(f"A1 = {A1.to_dense().tolist()}")
        print(f"A2 = {A2.to_dense().tolist()}")

        s = torch.linspace(0, 1, s_len, dtype=initial.dtype, device=initial.device)
        t = torch.linspace(0, 1, t_len, dtype=initial.dtype, device=initial.device)
        print(f"s = {s}")
        print(f"t = {t}")
        # torch.diag_embed(torch.rand(4, dtype=torch.float64), offset=0)

        u = MatrixPowerSeries(initial)
        u_n = u.deep_clone()

        print(f"u_n device = {u_n.coefficients.device}")
        print(f"u_0 = {u}")

        IminusG1 = build_integration_gather_matrix_s(s[0], u_n.coefficients.shape[1], u_n.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t[0], u_n.coefficients.shape[0], u_n.coefficients.device)

        print(f"IminusG1 = {IminusG1.to_dense().tolist()}")
        print(f"IminusG2 = {IminusG2.to_dense().tolist()}")
        print(f"L = {torch.mm(IminusG2, A2).to_dense().tolist()}")
        print(f"R = {torch.mm(A1, IminusG1).to_dense().tolist()}")

        for i in range(5):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
            u_n *= rho
            u += u_n
            print(f"u_n = {u_n}")

        # Solution for bottom left tile.
        print(f"u = {u}")
        print(f"u_11 = {u_11}")
        u_11_actual = u.deep_clone()
        assert torch.allclose(u.coefficients - u_11.coefficients, torch.zeros(initial.shape, dtype=initial.dtype),
                              rtol=0, atol=1e-8), "Must be close to zero."

        IminusG1 = build_integration_gather_matrix_s(s[1], u.coefficients.shape[1], u.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t[0], u.coefficients.shape[0], u.coefficients.device)

        print(f"IminusG1 = {IminusG1.to_dense().tolist()}")
        print(f"IminusG2 = {IminusG2.to_dense().tolist()}")
        print(f"L = {torch.mm(IminusG2, A2).to_dense().tolist()}")
        print(f"R = {torch.mm(A1, IminusG1).to_dense().tolist()}")

        v = u.deep_clone()
        u = u.bind_s(s[1].item())
        u_n = u.deep_clone()
        u_21_lb = MatrixPowerSeries(torch.zeros(initial.shape, dtype=torch.float64))
        u_21_lb.coefficients[:, 0] = torch.tensor([1, 8, 16, 128 / 9, 64 / 9], dtype=torch.float64)
        print(f"u[1,0]_0 = {u}")
        print(f"u_21_bb = {u_21_lb}")
        assert torch.allclose(u.coefficients - u_21_lb.coefficients, torch.zeros(initial.shape, dtype=initial.dtype),
                              rtol=0, atol=1e-8), "Must be close to zero."
        for i in range(5):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
            u_n *= rho
            u += u_n
            print(f"u_n = {u_n}")
        print(f"u = {u}")

        u_21_actual = u.deep_clone()
        u_21 = MatrixPowerSeries(torch.tensor(
            [[1, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 64, 0, 0], [0, 0, 0, 1024 / 9, 0], [0, 0, 0, 0, 1024 / 9]],
            dtype=torch.float64))
        print(f"u_21 = {u_21}")

        assert torch.allclose(u.coefficients - u_21.coefficients, torch.zeros(initial.shape, dtype=initial.dtype),
                              rtol=0, atol=1e-8), "Must be close to zero."

        IminusG1 = build_integration_gather_matrix_s(s[0], u.coefficients.shape[1], u.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t[1], u.coefficients.shape[0], u.coefficients.device)

        u = v.deep_clone()
        u = u.bind_t(t[1].item())
        u_n = u.deep_clone()
        u_12_lb = MatrixPowerSeries(torch.zeros(initial.shape, dtype=torch.float64))
        u_12_lb.coefficients[0, :] = torch.tensor([1, 4, 4, 16 / 9, 4 / 9], dtype=torch.float64)

        print(f"u[0,1]_0 = {u}")
        print(f"u_12_lb = {u_12_lb}")

        for i in range(5):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
            u_n *= rho
            u += u_n
            print(f"u_n = {u_n}")
            print(f"u = {u}")

        print(f"u = {u}")

        u_12_actual = u.deep_clone()
        u_12 = MatrixPowerSeries(torch.tensor(
            [[1, 0, 0, 0, 0], [0, 16, 0, 0, 0], [0, 0, 64, 0, 0], [0, 0, 0, 1024 / 9, 0], [0, 0, 0, 0, 1024 / 9]],
            dtype=torch.float64))
        print(f"u_12 = {u_12}")

        assert torch.allclose(u.coefficients - u_12.coefficients, torch.zeros(initial.shape, dtype=initial.dtype),
                              rtol=0, atol=1e-8), "Must be close to zero."

        IminusG1 = build_integration_gather_matrix_s(s[1], u.coefficients.shape[1], u.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t[1], u.coefficients.shape[0], u.coefficients.device)

        print("\n===== Final Tile =====\n")
        print(f"u_21_actual = {u_21_actual}")
        print(f"u_12_actual = {u_12_actual}")
        print(f"u_12_rb = {u_12_actual.bind_s(s[1].item())}")
        print(f"u_21_tb = {u_21_actual.bind_t(t[1].item())}")
        print(f"u_11_actual = {u_11_actual}")
        print(f"u_11_actual@1,1 = {u_11_actual(s[1], t[1]).item()}")
        u = u_12_actual.bind_s(s[1].item())
        print(f"1) u[2,2]_0 = {u}")
        u += u_21_actual.bind_t(t[1].item())
        print(f"2) u[2,2]_0 = {u}")
        u -= u_11_actual(s[1], t[1]).item()
        print(f"3) u[2,2]_0 = {u}")
        u_n = u.deep_clone()

        for i in range(5):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
            u_n *= rho
            u += u_n
            print(f"u_n = {u_n}")
            print(f"u = {u}")

        print(f"u = {u}")
        u_22 = MatrixPowerSeries(torch.tensor(
            [[1 / 81, 0, 512 / 81, -(512 / 27), 256 / 9], [0, 16 / 81, 4096 / 81, -(8192 / 81),
                                                           10240 / 81], [512 / 81, 4096 / 81, -(11456 / 81), 4096 /
                                                                         9, -(48128 / 81)],
             [-(512 / 27), -(8192 / 81), 4096 / 9, -(25600 / 27),
              114688 / 81], [256 / 9, 10240 / 81, -(48128 / 81), 114688 / 81, -(146432 / 81)]]
            ,
            dtype=torch.float64))
        print(f"u_22 = {u_22}")

        assert torch.allclose(u.coefficients - u_22.coefficients, torch.zeros(initial.shape, dtype=initial.dtype),
                              rtol=0, atol=1e-8), "Must be close to zero."

    def test_build_gram_matrix(self):
        config = self.__class__.configuration
        max_batch = 10
        start = time.time()
        sk = signature_kernel.compute_Gram(config.X.cpu(), config.X.cpu(), max_batch)
        print(f"SigKernel computation took: {time.time() - start}s")
        print(f"SigKernel Gram Matrix: \n {sk.tolist()}")
        start = time.time()
        m = MatrixSig(config.X.cuda(), config.X.cuda()).compute_gram_matrix()
        print(f"Matrix Sig computation took: {time.time() - start}s")
        print(f"Matrix Sig computation of gram Matrix: \n {m.tolist()}")
        # mse = torch.mean((sk.cpu() - m.cpu()) ** 2)
        # print(f"MSE MatrixSig versus SigKernel: {mse}")


if __name__ == '__main__':
    setup_torch()
    unittest.main()
