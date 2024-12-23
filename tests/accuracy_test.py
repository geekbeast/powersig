import time
import unittest
from math import factorial
from multiprocessing import set_start_method

import numpy
import numpy as np
import torch
from sigkernel import sigkernel

from powersig.matrixsig import MatrixSig
from powersig.power_series import SimplePowerSeries
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


if __name__== '__main__':
    setup_torch()
    unittest.main()