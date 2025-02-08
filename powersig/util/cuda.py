from numba import cuda, float32


def get_number_threads(x: int) -> int:
    p = 1
    while p < x and p < 32:
        p <<= 1
    return p


@cuda.jit
def print_shared_matrix(shared):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Print the shared memory contents
    cuda.syncthreads()

    # Print matrix - one element at a time to ensure it works
    if tx == 0 and ty == 0:
        for i in range(16):
            for j in range(16):
                print(shared[i, j])
            print(" ")  # New line after each row

    cuda.syncthreads()