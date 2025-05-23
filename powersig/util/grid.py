from typing import Tuple

def get_diagonal_range(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    # d, s_start, t_start are 0 based indexes while rows/cols are shapes.

    if d < cols:
        # if d < cols, then we haven't hit the right edge of the grid
        t_start = 0
        s_start = d
    else:
        # if d >= cols then we have the right edge and wrapped around the corner
        t_start = d - cols + 1  # diag index - cols + 1
        s_start = cols - 1

    return s_start, t_start, min(rows - t_start, s_start + 1)
