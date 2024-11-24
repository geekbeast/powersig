from multiprocessing import set_start_method

import torch


def setup_torch():
    torch.set_num_threads(64)
    set_start_method("spawn")
    print(f"Number of threads: {torch.get_num_threads()}")
    print(f"Number of interop threads: {torch.get_num_interop_threads()}")


