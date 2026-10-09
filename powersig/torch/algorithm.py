from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from powersig.util.grid import get_diagonal_range

from . import static_kernels


DIAGONAL_CHUNK_SIZE = 1024
JIT_BOUNDARY_THRESHOLD = 64
_MAX_BLOCK_SIZE_SMALL = 256
_MAX_BLOCK_SIZE_LARGE = 16384


def get_diagonal_range_bool(d: int, rows: int, cols: int) -> Tuple[int, int, int]:
    # Benchmark-only version matching the corrected boolean arithmetic used in JAX.
    t_start = (d < rows) * 0 + (d >= rows) * (d - rows + 1)
    s_start = (d < rows) * d + (d >= rows) * (rows - 1)
    dlen = min(s_start + 1, cols - t_start)
    return int(s_start), int(t_start), int(dlen)


def get_max_block_size(device: torch.device) -> int:
    if device.type != "cuda":
        return _MAX_BLOCK_SIZE_SMALL

    try:
        props = torch.cuda.get_device_properties(_device_index(device))
        if props.total_memory > 24 * 1024**3:
            return _MAX_BLOCK_SIZE_LARGE
    except Exception:
        pass

    return _MAX_BLOCK_SIZE_SMALL


def estimate_bytes_per_pair(longest_diagonal: int, order: int, dtype: torch.dtype) -> int:
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    buffers = 2 * longest_diagonal * order
    toeplitz = 3 * longest_diagonal * order * order
    aux = 5 * longest_diagonal * order
    return elem_bytes * (buffers + toeplitz + aux)


def get_available_gpu_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 4 * 1024**3

    try:
        free, _ = torch.cuda.mem_get_info(_device_index(device))
        return int(free)
    except Exception:
        pass

    try:
        props = torch.cuda.get_device_properties(_device_index(device))
        return int(props.total_memory)
    except Exception:
        return 8 * 1024**3


def _round_to_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_block_size(
    longest_diagonal: int,
    order: int,
    dtype: torch.dtype,
    device: torch.device,
    total_pairs: int,
    safety_factor: float = 0.7,
) -> int:
    max_bs = get_max_block_size(device)
    per_pair = estimate_bytes_per_pair(longest_diagonal, order, dtype)
    available = get_available_gpu_memory(device)
    budget = int(available * safety_factor)

    if per_pair > 0:
        raw = max(1, budget // per_pair)
    else:
        raw = max_bs

    raw = min(raw, total_pairs, max_bs)
    return _round_to_power_of_2(raw)


def compute_vandermonde_vectors(
    ds: float,
    dt: float,
    n: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    powers = torch.arange(n, dtype=dtype, device=device)
    v_s = torch.pow(torch.as_tensor(ds, dtype=dtype, device=device), powers)
    v_t = torch.pow(torch.as_tensor(dt, dtype=dtype, device=device), powers)
    return v_s, v_t


def build_psi_stencil(
    order: int,
    delta: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    psi = torch.zeros((order, order), dtype=dtype, device=device)
    psi[0, :] = 1.0

    i_indices = torch.arange(1, order, dtype=dtype, device=device).reshape(-1, 1)
    j_indices = torch.arange(order, dtype=dtype, device=device).reshape(1, -1)
    psi[1:, :] = 1.0 / (i_indices * (j_indices + i_indices))

    powers = torch.arange(order, dtype=dtype, device=device)
    psi[0, :] *= delta**powers
    for row_idx in range(1, order):
        psi[row_idx, :] = psi[row_idx, :] * psi[row_idx - 1, :] * delta

    return psi


def build_stencil(
    order: int = 32,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    stencil = torch.ones((order, order), dtype=dtype, device=device)

    i_indices = torch.arange(1, order, dtype=dtype, device=device).reshape(-1, 1)
    j_indices = torch.arange(1, order, dtype=dtype, device=device).reshape(1, -1)
    stencil[1:, 1:] = 1.0 / (i_indices * j_indices)

    for k in range(-(order - 1), order):
        diagonal = torch.diagonal(stencil, offset=k)
        diagonal.copy_(torch.cumprod(diagonal, dim=0))

    return stencil


def build_stencil_s(
    v_s: torch.Tensor,
    order: int = 32,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return build_stencil(order=order, dtype=dtype, device=device) * v_s


def build_stencil_t(
    v_t: torch.Tensor,
    order: int = 32,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return build_stencil(order=order, dtype=dtype, device=device) * v_t[:, None]


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


class PowerSigTorch:
    def __init__(
        self,
        order: int = 32,
        static_kernel: Callable = static_kernels.linear_kernel,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        compile_forward: bool = False,
    ):
        self.order = order
        self.dtype = dtype
        self.static_kernel = static_kernel
        self.compile_forward = compile_forward

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.exponents = torch.arange(self.order, dtype=torch.int64, device=self.device)
        self.psi_s = build_psi_stencil(self.order, dtype=self.dtype, device=self.device)
        for i in range(1, self.order):
            self.psi_s[i, -i:] = 0.0

        self.psi_t = build_stencil(self.order, dtype=self.dtype, device=self.device)
        self.psi_s_t = self.psi_s.transpose(0, 1).contiguous()
        self.psi_t_upper = torch.triu(self.psi_t, diagonal=1).contiguous()
        self.psi_t_lower = torch.tril(self.psi_t, diagonal=-1).contiguous()

        self.ic = torch.zeros(self.order, dtype=self.dtype, device=self.device)
        self.ic[0] = 1.0
        self.v_s_unit, self.v_t_unit = compute_vandermonde_vectors(
            1.0, 1.0, self.order, dtype=self.dtype, device=self.device
        )

        indices = torch.arange(self.order, dtype=torch.long, device=self.device)
        diff = indices[None, :] - indices[:, None]
        self._toeplitz_is_upper = (diff >= 0).contiguous()
        self._toeplitz_upper_index = diff.clamp_min(0).contiguous()
        self._toeplitz_lower_index = (-diff).clamp_min(0).contiguous()
        jac_row_idx = torch.arange(self.order, dtype=torch.long, device=self.device)[:, None]
        jac_col_idx = torch.arange(self.order, dtype=torch.long, device=self.device)[None, :]
        jac_diff = jac_row_idx - jac_col_idx
        self._jac_row_idx = jac_row_idx.expand(self.order, self.order).contiguous()
        self._jac_safe_diff = jac_diff.clamp_min(0).contiguous()
        self._jac_mask = ((jac_col_idx >= 1) & (jac_row_idx >= jac_col_idx)).contiguous()

        self._minimum_sweep = self._compute_gram_entry_batch_minimum
        self._boolean_sweep = self._compute_gram_entry_batch_bool
        if self.device.type == "cuda" and self.compile_forward:
            try:
                self._minimum_sweep = torch.compile(
                    self._compute_gram_entry_batch_minimum,
                    dynamic=False,
                    mode="reduce-overhead",
                )
                self._boolean_sweep = torch.compile(
                    self._compute_gram_entry_batch_bool,
                    dynamic=False,
                    mode="reduce-overhead",
                )
            except Exception:
                self._minimum_sweep = self._compute_gram_entry_batch_minimum
                self._boolean_sweep = self._compute_gram_entry_batch_bool

    def __call__(
        self,
        X,
        Y=None,
        symmetric: bool = False,
        block_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        return self.compute_gram_matrix(
            X, Y, symmetric=symmetric, block_size=block_size, show_progress=show_progress
        )

    def compute_signature_kernel(
        self, X: torch.Tensor, Y: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        X_i = self._as_single_path(X, device=device)
        Y_j = self._as_single_path(Y, device=device)
        return self._detach_from_graph_buffer(
            self._minimum_sweep(X_i[None, ...], Y_j[None, ...])[0]
        )

    def _detach_from_graph_buffer(self, out: torch.Tensor) -> torch.Tensor:
        """Copy a compiled sweep's result out of its CUDA-graph static buffer.

        compile_forward uses torch.compile(mode="reduce-overhead"), which replays
        a CUDA graph writing into a fixed output buffer. Without this copy the
        tensor a caller is holding is silently overwritten by their next call, so
        collecting results in a list yields the last value repeated. The clone is
        a scalar, so it costs nothing next to the sweep itself.
        """
        if self.compile_forward and out.is_cuda:
            return out.clone()
        return out

    def compute_signature_kernel_bool_geometry(
        self, X: torch.Tensor, Y: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        X_i = self._as_single_path(X, device=device)
        Y_j = self._as_single_path(Y, device=device)
        return self._boolean_sweep(X_i[None, ...], Y_j[None, ...])[0]

    def compute_signature_kernel_chunked(
        self, X: torch.Tensor, Y: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        # Unlike the JAX backend, the Torch sweep is already shape-stable and does
        # not need a separate long-diagonal implementation.
        return self.compute_signature_kernel(X, Y, device=device)

    def compute_gram_matrix(
        self,
        X,
        Y=None,
        symmetric: bool = False,
        block_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        X_batch = self._as_path_batch(X)
        Y_batch = X_batch if Y is None else self._as_path_batch(Y)

        gram_matrix = torch.zeros(
            (X_batch.shape[0], Y_batch.shape[0]), dtype=self.dtype, device=self.device
        )

        rows = X_batch.shape[1] - 1
        cols = Y_batch.shape[1] - 1
        if rows <= 0 or cols <= 0:
            raise ValueError("paths must have length at least 2")

        pairs_i = []
        pairs_j = []
        for i in range(X_batch.shape[0]):
            for j in range(i if symmetric else 0, Y_batch.shape[0]):
                pairs_i.append(i)
                pairs_j.append(j)

        total_pairs = len(pairs_i)
        if total_pairs == 0:
            return gram_matrix

        longest_diagonal = min(rows, cols)
        if block_size is None:
            block_size = compute_block_size(
                longest_diagonal, self.order, self.dtype, self.device, total_pairs
            )
        else:
            block_size = _round_to_power_of_2(min(block_size, total_pairs))

        i_all = torch.tensor(pairs_i, dtype=torch.long, device=self.device)
        j_all = torch.tensor(pairs_j, dtype=torch.long, device=self.device)

        pbar = tqdm(total=total_pairs, desc="Computing Gram Matrix", disable=not show_progress)
        offset = 0
        while offset < total_pairs:
            end = min(offset + block_size, total_pairs)
            batch_i = i_all[offset:end]
            batch_j = j_all[offset:end]
            actual_count = end - offset

            if actual_count < block_size:
                pad_count = block_size - actual_count
                batch_i = torch.cat([batch_i, batch_i[-1:].expand(pad_count)])
                batch_j = torch.cat([batch_j, batch_j[-1:].expand(pad_count)])

            results = self._minimum_sweep(X_batch[batch_i], Y_batch[batch_j])
            actual_i = i_all[offset:end]
            actual_j = j_all[offset:end]
            gram_matrix[actual_i, actual_j] = results[:actual_count]
            if symmetric:
                gram_matrix[actual_j, actual_i] = results[:actual_count]

            pbar.update(end - offset)
            offset = end

        pbar.close()
        return gram_matrix

    def compute_prefix_family(
        self,
        state_path,
        refs,
        min_prefix_len: int = 2,
        max_prefix_len: Optional[int] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> torch.Tensor:
        from .autodiff import compute_prefix_family

        return compute_prefix_family(
            self,
            state_path,
            refs,
            min_prefix_len=min_prefix_len,
            max_prefix_len=max_prefix_len,
            checkpoint_interval=checkpoint_interval,
        )

    def compute_prefix_family_fast_diff(
        self,
        state_path,
        refs,
        min_prefix_len: int = 2,
        max_prefix_len: Optional[int] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> torch.Tensor:
        return self.compute_prefix_family(
            state_path,
            refs,
            min_prefix_len=min_prefix_len,
            max_prefix_len=max_prefix_len,
            checkpoint_interval=checkpoint_interval,
        )

    def compute_signature_kernel_fast_diff(
        self,
        X,
        Y,
        checkpoint_interval: Optional[int] = None,
    ) -> torch.Tensor:
        from .autodiff import compute_sig_kernel_fast_diff

        return compute_sig_kernel_fast_diff(
            self,
            X,
            Y,
            checkpoint_interval=checkpoint_interval,
        )

    def compute_gram_fast_diff(
        self,
        X,
        Y,
        symmetric: bool = False,
        block_size: Optional[int] = None,
        checkpoint_interval: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        from .autodiff import compute_gram_fast_diff

        return compute_gram_fast_diff(
            self,
            X,
            Y,
            symmetric=symmetric,
            block_size=block_size,
            checkpoint_interval=checkpoint_interval,
            show_progress=show_progress,
        )

    def _as_tensor(self, X, device: Optional[torch.device] = None) -> torch.Tensor:
        target_device = self.device if device is None else torch.device(device)
        if torch.is_tensor(X):
            return X.to(device=target_device, dtype=self.dtype)
        return torch.as_tensor(X, dtype=self.dtype, device=target_device)

    def _as_single_path(self, X, device: Optional[torch.device] = None) -> torch.Tensor:
        path = self._as_tensor(X, device=device)
        if path.ndim == 3:
            if path.shape[0] != 1:
                raise ValueError("expected a single path with shape (length, dim)")
            path = path[0]
        if path.ndim != 2:
            raise ValueError("expected a path with shape (length, dim)")
        if path.shape[0] < 2:
            raise ValueError("paths must have length at least 2")
        return path

    def _as_path_batch(self, X) -> torch.Tensor:
        paths = self._as_tensor(X)
        if paths.ndim == 2:
            paths = paths[None, ...]
        if paths.ndim != 3:
            raise ValueError("expected paths with shape (batch, length, dim)")
        if paths.shape[1] < 2:
            raise ValueError("paths must have length at least 2")
        return paths

    def _compute_gram_entry_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        geometry_fn: Callable[[int, int, int], Tuple[int, int, int]] = get_diagonal_range,
    ) -> torch.Tensor:
        pair_batch = X_batch.shape[0]
        rows = X_batch.shape[1] - 1
        cols = Y_batch.shape[1] - 1
        longest_diagonal = min(rows, cols)
        diagonal_count = rows + cols - 1

        S_buf = torch.zeros(
            (pair_batch, longest_diagonal, self.order), dtype=self.dtype, device=self.device
        )
        T_buf = torch.zeros_like(S_buf)
        S_buf[:, :, 0] = 1.0
        T_buf[:, :, 0] = 1.0

        ic = self.ic.view(1, 1, self.order)
        max_index = max(longest_diagonal - 1, 0)

        for d in range(diagonal_count):
            s_start, t_start, dlen = geometry_fn(d, rows, cols)
            diagonal_indices = torch.arange(dlen, dtype=torch.long, device=self.device)
            row_indices = s_start - diagonal_indices
            col_indices = t_start + diagonal_indices
            is_before_wrap = d < rows

            s_index = torch.clamp(diagonal_indices - int(is_before_wrap), 0, max_index)
            t_index = torch.clamp(diagonal_indices + int(not is_before_wrap), 0, max_index)

            s_prev = S_buf[:, s_index, :]
            t_prev = T_buf[:, t_index, :]

            s = torch.where((t_start + diagonal_indices).view(1, dlen, 1) == 0, ic, s_prev)
            t = torch.where((s_start - diagonal_indices).view(1, dlen, 1) == 0, ic, t_prev)

            rho = self._evaluate_static_kernel(
                X_batch[:, row_indices + 1, :],
                X_batch[:, row_indices, :],
                Y_batch[:, col_indices + 1, :],
                Y_batch[:, col_indices, :],
            )

            S_next, T_next = self._map_diagonal_entry_batch(rho, s, t)
            pad_rows = longest_diagonal - dlen
            S_buf = F.pad(S_next, (0, 0, 0, pad_rows))
            T_buf = F.pad(T_next, (0, 0, 0, pad_rows))

        return torch.matmul(S_buf[:, 0, :], self.v_s_unit)

    def _compute_gram_entry_batch_minimum(
        self, X_batch: torch.Tensor, Y_batch: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_gram_entry_batch(X_batch, Y_batch, geometry_fn=get_diagonal_range)

    def _compute_gram_entry_batch_bool(
        self, X_batch: torch.Tensor, Y_batch: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_gram_entry_batch(X_batch, Y_batch, geometry_fn=get_diagonal_range_bool)

    def _evaluate_static_kernel(
        self,
        x2: torch.Tensor,
        x1: torch.Tensor,
        y2: torch.Tensor,
        y1: torch.Tensor,
    ) -> torch.Tensor:
        try:
            out = self.static_kernel(x2, x1, y2, y1)
            if out.shape == x2.shape[:-1]:
                return out
        except Exception:
            pass

        x2_flat = x2.reshape(-1, x2.shape[-1])
        x1_flat = x1.reshape(-1, x1.shape[-1])
        y2_flat = y2.reshape(-1, y2.shape[-1])
        y1_flat = y1.reshape(-1, y1.shape[-1])
        out = torch.vmap(self.static_kernel)(x2_flat, x1_flat, y2_flat, y1_flat)
        return out.reshape(x2.shape[:-1])

    def _map_diagonal_entry_batch(
        self, rho: torch.Tensor, s: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        leading_shape = rho.shape
        rho_flat = rho.reshape(-1)
        s_flat = s.reshape(-1, self.order)
        t_flat = t.reshape(-1, self.order)

        r = torch.pow(rho_flat[:, None], self.exponents)
        toeplitz = self._build_toeplitz_batch(t_flat, s_flat)

        s_dense = torch.matmul(t_flat, self.psi_s_t)
        t_dense = torch.matmul(s_flat, self.psi_s_t)

        s_next = torch.bmm(r.unsqueeze(1), toeplitz * self.psi_t_upper).squeeze(1)
        s_next = s_next + (s_dense * r)

        t_next = torch.bmm(toeplitz * self.psi_t_lower, r.unsqueeze(-1)).squeeze(-1)
        t_next = t_next + (t_dense * r)

        new_shape = (*leading_shape, self.order)
        return s_next.reshape(new_shape), t_next.reshape(new_shape)

    def _build_toeplitz_batch(self, first_col: torch.Tensor, first_row: torch.Tensor) -> torch.Tensor:
        upper = first_row[:, self._toeplitz_upper_index]
        lower = first_col[:, self._toeplitz_lower_index]
        return torch.where(self._toeplitz_is_upper.unsqueeze(0), upper, lower)
