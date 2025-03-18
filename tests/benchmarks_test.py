import os
import unittest
from contextlib import contextmanager

import psutil

from tests.utils import setup_torch


@contextmanager
def track_peak_memory():
    process = psutil.Process(os.getpid())
    peak_mem = 0
    try:
        yield
    finally:
        peak_mem = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"Peak memory usage: {peak_mem:.1f} MB")


if __name__== '__main__':
    setup_torch()
    unittest.main()