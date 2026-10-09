from typing import Tuple

def get_diagonal_range(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    # d, s_start, t_start are 0 based indexes while rows/cols are shapes.
    if d < rows:
        # We have not yet hit the bottom edge of the grid.
        s_start = d
        t_start = 0
    else:
        # Once we reach the bottom edge, keep s pinned and advance t.
        s_start = rows - 1
        t_start = d - rows + 1

    return s_start, t_start, min(s_start + 1, cols - t_start)
