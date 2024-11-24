import unittest

from tests import accuracy, benchmarks
from tests.utils import setup_torch

if __name__== '__main__':
    setup_torch()
    accuracy_tests = unittest.TestLoader().loadTestsFromModule(accuracy)
    benchmark_tests = unittest.TestLoader().loadTestsFromModule(benchmarks)
    unittest.TextTestRunner().run(accuracy_tests)
    unittest.TextTestRunner().run(benchmark_tests)