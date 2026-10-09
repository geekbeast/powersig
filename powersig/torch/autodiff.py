from functools import partial
from math import ceil, sqrt
from typing import Optional, Tuple

import torch
from tqdm.auto import tqdm

from powersig.util.grid import get_diagonal_range

from . import static_kernels


def _compute_checkpoint_interval(diagonal_count: int) -> int:
    return max(1, min(int(sqrt(diagonal_count)), diagonal_count))


def _extract_rbf_bandwidth(static_kernel) -> Optional[float]:
    if static_kernel is static_kernels.rbf_kernel:
        return 1.0
    if isinstance(static_kernel, partial) and static_kernel.func is static_kernels.rbf_kernel:
        if static_kernel.keywords and "bandwidth" in static_kernel.keywords:
            return float(static_kernel.keywords["bandwidth"])
        if static_kernel.args:
            return float(static_kernel.args[0])
        return 1.0
    return None


def _prefix_step(
    ps,
    state_path: torch.Tensor,
    refs: torch.Tensor,
    S_buf: torch.Tensor,
    T_buf: torch.Tensor,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = state_path.shape[0] - 1
    cols = refs.shape[1] - 1
    s_start, t_start, dlen = get_diagonal_range(d, rows, cols)
    diag_idx = torch.arange(dlen, dtype=torch.long, device=state_path.device)
    row_idx = s_start - diag_idx
    col_idx = t_start + diag_idx

    ic = ps.ic.view(1, 1, ps.order)
    left_prev = S_buf[:, row_idx, :]
    left = torch.where((col_idx == 0).view(1, dlen, 1), ic, left_prev)

    bottom_src_idx = torch.clamp(row_idx - 1, min=0)
    bottom_prev = T_buf[:, bottom_src_idx, :]
    bottom = torch.where((row_idx == 0).view(1, dlen, 1), ic, bottom_prev)

    x2 = state_path[row_idx + 1].unsqueeze(0).expand(refs.shape[0], -1, -1)
    x1 = state_path[row_idx].unsqueeze(0).expand_as(x2)
    y2 = refs[:, col_idx + 1, :]
    y1 = refs[:, col_idx, :]

    rho = ps._evaluate_static_kernel(x2, x1, y2, y1)
    s_next, t_next = ps._map_diagonal_entry_batch(rho, left, bottom)

    new_S = S_buf.clone()
    new_T = T_buf.clone()
    new_S[:, row_idx, :] = s_next
    new_T[:, row_idx, :] = t_next
    return new_S, new_T


def _forward_prefix_family_with_checkpoints(
    ps,
    state_path: torch.Tensor,
    refs: torch.Tensor,
    min_prefix_len: int,
    max_prefix_len: int,
    checkpoint_interval: int,
):
    rows = state_path.shape[0] - 1
    cols = refs.shape[1] - 1
    diagonal_count = rows + cols - 1
    num_prefixes = max_prefix_len - min_prefix_len + 1
    num_refs = refs.shape[0]
    num_checkpoints = ceil(diagonal_count / checkpoint_interval)

    S_buf = ps.ic.view(1, 1, ps.order).expand(num_refs, rows, ps.order).clone()
    T_buf = ps.ic.view(1, 1, ps.order).expand(num_refs, rows, ps.order).clone()
    emissions = torch.zeros(
        (num_prefixes, num_refs), dtype=state_path.dtype, device=state_path.device
    )
    S_checkpoints = torch.zeros(
        (num_checkpoints, num_refs, rows, ps.order),
        dtype=state_path.dtype,
        device=state_path.device,
    )
    T_checkpoints = torch.zeros_like(S_checkpoints)

    for d in range(diagonal_count):
        if d % checkpoint_interval == 0:
            ckpt_idx = d // checkpoint_interval
            S_checkpoints[ckpt_idx] = S_buf
            T_checkpoints[ckpt_idx] = T_buf

        S_buf, T_buf = _prefix_step(ps, state_path, refs, S_buf, T_buf, d)

        emit_i = d - cols + 1
        emit_prefix_len = emit_i + 2
        emit_valid = (
            0 <= emit_i < rows
            and min_prefix_len <= emit_prefix_len <= max_prefix_len
        )
        if emit_valid:
            emit_idx = emit_prefix_len - min_prefix_len
            emissions[emit_idx] = torch.matmul(T_buf[:, emit_i, :], ps.v_t_unit)

    return emissions, S_checkpoints, T_checkpoints


def _map_diagonal_entry_bwd_batch(
    ps,
    rho: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    bar_s_next: torch.Tensor,
    bar_t_next: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leading_shape = rho.shape
    rho_flat = rho.reshape(-1)
    s_flat = s.reshape(-1, ps.order)
    t_flat = t.reshape(-1, ps.order)
    bar_s_flat = bar_s_next.reshape(-1, ps.order)
    bar_t_flat = bar_t_next.reshape(-1, ps.order)

    r = torch.pow(rho_flat[:, None], ps.exponents)
    dr = torch.zeros_like(r)
    if ps.order > 1:
        orders = torch.arange(1, ps.order, dtype=ps.dtype, device=ps.device)
        dr[:, 1:] = orders * r[:, :-1]

    toeplitz = ps._build_toeplitz_batch(t_flat, s_flat) * ps.psi_t
    upper = torch.triu(toeplitz, diagonal=1)
    lower = torch.tril(toeplitz, diagonal=-1)

    psi_s_t = torch.matmul(t_flat, ps.psi_s_t)
    psi_s_s = torch.matmul(s_flat, ps.psi_s_t)

    rho_term = torch.bmm(upper, bar_s_flat.unsqueeze(-1)).squeeze(-1)
    rho_term = rho_term + torch.bmm(lower.transpose(1, 2), bar_t_flat.unsqueeze(-1)).squeeze(-1)
    rho_term = rho_term + psi_s_t * bar_s_flat + psi_s_s * bar_t_flat
    bar_rho = torch.sum(dr * rho_term, dim=-1)

    J_ss = torch.where(
        ps._jac_mask.unsqueeze(0),
        r[:, ps._jac_safe_diff] * ps.psi_t[ps._jac_safe_diff, ps._jac_row_idx],
        torch.zeros((1, ps.order, ps.order), dtype=ps.dtype, device=ps.device),
    )
    J_tt = torch.where(
        ps._jac_mask.unsqueeze(0),
        r[:, ps._jac_safe_diff] * ps.psi_t[ps._jac_row_idx, ps._jac_safe_diff],
        torch.zeros((1, ps.order, ps.order), dtype=ps.dtype, device=ps.device),
    )

    r_bar_s_next = r * bar_s_flat
    r_bar_t_next = r * bar_t_flat
    bar_s = torch.bmm(J_ss.transpose(1, 2), bar_s_flat.unsqueeze(-1)).squeeze(-1)
    bar_s = bar_s + torch.matmul(r_bar_t_next, ps.psi_s)
    bar_t = torch.bmm(J_tt.transpose(1, 2), bar_t_flat.unsqueeze(-1)).squeeze(-1)
    bar_t = bar_t + torch.matmul(r_bar_s_next, ps.psi_s)

    new_shape = (*leading_shape, ps.order)
    return bar_s.reshape(new_shape), bar_t.reshape(new_shape), bar_rho.reshape(leading_shape)


def _static_kernel_vjp_batch(
    ps,
    x2: torch.Tensor,
    x1: torch.Tensor,
    y2: torch.Tensor,
    y1: torch.Tensor,
    bar_rho: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ps.static_kernel is static_kernels.linear_kernel:
        dy = y2 - y1
        dx = x2 - x1
        scale = bar_rho.unsqueeze(-1)
        return scale * dy, -scale * dy, scale * dx, -scale * dx

    bandwidth = _extract_rbf_bandwidth(ps.static_kernel)
    if bandwidth is not None:
        inv_bw_sq = 1.0 / (bandwidth**2)

        diff_x2y2 = x2 - y2
        diff_x1y2 = x1 - y2
        diff_x2y1 = x2 - y1
        diff_x1y1 = x1 - y1

        k_x2y2 = static_kernels.rbf_fn(diff_x2y2, bandwidth)
        k_x1y2 = static_kernels.rbf_fn(diff_x1y2, bandwidth)
        k_x2y1 = static_kernels.rbf_fn(diff_x2y1, bandwidth)
        k_x1y1 = static_kernels.rbf_fn(diff_x1y1, bandwidth)

        grad_a_x2y2 = -diff_x2y2 * (inv_bw_sq * k_x2y2.unsqueeze(-1))
        grad_b_x2y2 = diff_x2y2 * (inv_bw_sq * k_x2y2.unsqueeze(-1))
        grad_a_x1y2 = -diff_x1y2 * (inv_bw_sq * k_x1y2.unsqueeze(-1))
        grad_b_x1y2 = diff_x1y2 * (inv_bw_sq * k_x1y2.unsqueeze(-1))
        grad_a_x2y1 = -diff_x2y1 * (inv_bw_sq * k_x2y1.unsqueeze(-1))
        grad_b_x2y1 = diff_x2y1 * (inv_bw_sq * k_x2y1.unsqueeze(-1))
        grad_a_x1y1 = -diff_x1y1 * (inv_bw_sq * k_x1y1.unsqueeze(-1))
        grad_b_x1y1 = diff_x1y1 * (inv_bw_sq * k_x1y1.unsqueeze(-1))

        g_x2 = grad_a_x2y2 - grad_a_x2y1
        g_x1 = -grad_a_x1y2 + grad_a_x1y1
        g_y2 = grad_b_x2y2 - grad_b_x1y2
        g_y1 = -grad_b_x2y1 + grad_b_x1y1
        scale = bar_rho.unsqueeze(-1)
        return scale * g_x2, scale * g_x1, scale * g_y2, scale * g_y1

    with torch.enable_grad():
        x2_req = x2.detach().requires_grad_(True)
        x1_req = x1.detach().requires_grad_(True)
        y2_req = y2.detach().requires_grad_(True)
        y1_req = y1.detach().requires_grad_(True)
        rho = ps._evaluate_static_kernel(x2_req, x1_req, y2_req, y1_req)
        grads = torch.autograd.grad(
            rho,
            (x2_req, x1_req, y2_req, y1_req),
            grad_outputs=bar_rho,
            allow_unused=False,
        )
    return grads


def _reverse_prefix_family_with_replay(
    ps,
    state_path: torch.Tensor,
    refs: torch.Tensor,
    grad_output: torch.Tensor,
    S_checkpoints: torch.Tensor,
    T_checkpoints: torch.Tensor,
    min_prefix_len: int,
    max_prefix_len: int,
    checkpoint_interval: int,
):
    rows = state_path.shape[0] - 1
    cols = refs.shape[1] - 1
    diagonal_count = rows + cols - 1
    num_refs = refs.shape[0]
    num_checkpoints = ceil(diagonal_count / checkpoint_interval)

    grad_state = torch.zeros_like(state_path)
    grad_refs = torch.zeros_like(refs)
    bar_S = torch.zeros((num_refs, rows, ps.order), dtype=ps.dtype, device=ps.device)
    bar_T = torch.zeros_like(bar_S)

    for b in range(num_checkpoints - 1, -1, -1):
        block_start = b * checkpoint_interval
        block_end = min(block_start + checkpoint_interval, diagonal_count)
        block_len = block_end - block_start

        S_buf = S_checkpoints[b]
        T_buf = T_checkpoints[b]
        S_tape = []
        T_tape = []

        for d in range(block_start, block_end):
            S_tape.append(S_buf)
            T_tape.append(T_buf)
            S_buf, T_buf = _prefix_step(ps, state_path, refs, S_buf, T_buf, d)

        for local_idx in range(block_len - 1, -1, -1):
            d = block_start + local_idx
            emit_i = d - cols + 1
            emit_prefix_len = emit_i + 2
            emit_valid = (
                0 <= emit_i < rows
                and min_prefix_len <= emit_prefix_len <= max_prefix_len
            )
            if emit_valid:
                emit_idx = emit_prefix_len - min_prefix_len
                bar_T[:, emit_i, :] = bar_T[:, emit_i, :] + (
                    grad_output[emit_idx].unsqueeze(-1) * ps.v_t_unit
                )

            S_saved = S_tape[local_idx]
            T_saved = T_tape[local_idx]

            s_start, t_start, dlen = get_diagonal_range(d, rows, cols)
            diag_idx = torch.arange(dlen, dtype=torch.long, device=ps.device)
            row_idx = s_start - diag_idx
            col_idx = t_start + diag_idx

            ic = ps.ic.view(1, 1, ps.order)
            left_prev = S_saved[:, row_idx, :]
            left = torch.where((col_idx == 0).view(1, dlen, 1), ic, left_prev)

            bottom_src_idx = torch.clamp(row_idx - 1, min=0)
            bottom_prev = T_saved[:, bottom_src_idx, :]
            bottom = torch.where((row_idx == 0).view(1, dlen, 1), ic, bottom_prev)

            x2 = state_path[row_idx + 1].unsqueeze(0).expand(num_refs, -1, -1)
            x1 = state_path[row_idx].unsqueeze(0).expand_as(x2)
            y2 = refs[:, col_idx + 1, :]
            y1 = refs[:, col_idx, :]
            rho = ps._evaluate_static_kernel(x2, x1, y2, y1)

            bar_s_next = bar_S[:, row_idx, :]
            bar_t_next = bar_T[:, row_idx, :]
            bar_left, bar_bottom, bar_rho = _map_diagonal_entry_bwd_batch(
                ps, rho, left, bottom, bar_s_next, bar_t_next
            )

            bar_S_prev = bar_S.clone()
            bar_T_prev = bar_T.clone()
            bar_S_prev[:, row_idx, :] = 0.0
            bar_T_prev[:, row_idx, :] = 0.0

            left_mask = col_idx > 0
            if torch.any(left_mask):
                left_rows = row_idx[left_mask]
                bar_S_prev[:, left_rows, :] = bar_S_prev[:, left_rows, :] + bar_left[:, left_mask, :]

            bottom_mask = row_idx > 0
            if torch.any(bottom_mask):
                bottom_rows = row_idx[bottom_mask] - 1
                bar_T_prev[:, bottom_rows, :] = bar_T_prev[:, bottom_rows, :] + bar_bottom[:, bottom_mask, :]

            g_x2, g_x1, g_y2, g_y1 = _static_kernel_vjp_batch(ps, x2, x1, y2, y1, bar_rho)
            grad_state[row_idx + 1] = grad_state[row_idx + 1] + g_x2.sum(dim=0)
            grad_state[row_idx] = grad_state[row_idx] + g_x1.sum(dim=0)
            grad_refs[:, col_idx + 1, :] = grad_refs[:, col_idx + 1, :] + g_y2
            grad_refs[:, col_idx, :] = grad_refs[:, col_idx, :] + g_y1

            bar_S = bar_S_prev
            bar_T = bar_T_prev

    return grad_state, grad_refs


class _PrefixFamilyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state_path: torch.Tensor,
        refs: torch.Tensor,
        ps,
        min_prefix_len: int,
        max_prefix_len: int,
        checkpoint_interval: int,
    ) -> torch.Tensor:
        out, S_ckpt, T_ckpt = _forward_prefix_family_with_checkpoints(
            ps,
            state_path,
            refs,
            min_prefix_len,
            max_prefix_len,
            checkpoint_interval,
        )
        ctx.ps = ps
        ctx.min_prefix_len = min_prefix_len
        ctx.max_prefix_len = max_prefix_len
        ctx.checkpoint_interval = checkpoint_interval
        ctx.save_for_backward(state_path, refs, S_ckpt, T_ckpt)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state_path, refs, S_ckpt, T_ckpt = ctx.saved_tensors
        grad_state, grad_refs = _reverse_prefix_family_with_replay(
            ctx.ps,
            state_path,
            refs,
            grad_output,
            S_ckpt,
            T_ckpt,
            ctx.min_prefix_len,
            ctx.max_prefix_len,
            ctx.checkpoint_interval,
        )
        return grad_state, grad_refs, None, None, None, None


def compute_prefix_family(
    ps,
    state_path,
    refs,
    min_prefix_len: int = 2,
    max_prefix_len: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
) -> torch.Tensor:
    state_path_t = ps._as_single_path(state_path)
    refs_t = ps._as_path_batch(refs)

    if min_prefix_len < 2:
        raise ValueError(f"min_prefix_len must be >= 2 (got {min_prefix_len})")

    T = state_path_t.shape[0]
    if max_prefix_len is None:
        max_prefix_len = T
    if max_prefix_len > T:
        raise ValueError(f"max_prefix_len ({max_prefix_len}) > state_path length ({T})")
    if max_prefix_len < min_prefix_len:
        raise ValueError(
            f"max_prefix_len ({max_prefix_len}) < min_prefix_len ({min_prefix_len})"
        )
    if refs_t.shape[1] < 2:
        raise ValueError("reference paths must have length at least 2")

    rows = state_path_t.shape[0] - 1
    cols = refs_t.shape[1] - 1
    diagonal_count = rows + cols - 1
    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(diagonal_count)

    return _PrefixFamilyFunction.apply(
        state_path_t,
        refs_t,
        ps,
        min_prefix_len,
        max_prefix_len,
        checkpoint_interval,
    )


def compute_prefix_family_fast_diff(
    ps,
    state_path,
    refs,
    min_prefix_len: int = 2,
    max_prefix_len: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
) -> torch.Tensor:
    return compute_prefix_family(
        ps,
        state_path,
        refs,
        min_prefix_len=min_prefix_len,
        max_prefix_len=max_prefix_len,
        checkpoint_interval=checkpoint_interval,
    )


def compute_sig_kernel_fast_diff(
    ps,
    X_i,
    Y_j,
    checkpoint_interval: Optional[int] = None,
) -> torch.Tensor:
    X_i_t = ps._as_single_path(X_i)
    Y_j_t = ps._as_single_path(Y_j)
    out = compute_prefix_family(
        ps,
        X_i_t,
        Y_j_t[None, ...],
        min_prefix_len=X_i_t.shape[0],
        max_prefix_len=X_i_t.shape[0],
        checkpoint_interval=checkpoint_interval,
    )
    return out[0, 0]


def compute_gram_fast_diff(
    ps,
    X,
    Y,
    symmetric: bool = False,
    block_size: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    X_batch = ps._as_path_batch(X)
    Y_batch = ps._as_path_batch(Y)

    pairs_i = []
    pairs_j = []
    for i in range(X_batch.shape[0]):
        for j in range(i if symmetric else 0, Y_batch.shape[0]):
            pairs_i.append(i)
            pairs_j.append(j)

    total_pairs = len(pairs_i)
    if total_pairs == 0:
        return torch.zeros(
            (X_batch.shape[0], Y_batch.shape[0]), dtype=ps.dtype, device=ps.device
        )

    values = []
    pbar = tqdm(total=total_pairs, desc="Computing Gram (diff)", disable=not show_progress)
    for i, j in zip(pairs_i, pairs_j):
        values.append(
            compute_sig_kernel_fast_diff(
                ps,
                X_batch[i],
                Y_batch[j],
                checkpoint_interval=checkpoint_interval,
            )
        )
        pbar.update(1)
    pbar.close()

    vals = torch.stack(values)
    i_all = torch.tensor(pairs_i, dtype=torch.long, device=ps.device)
    j_all = torch.tensor(pairs_j, dtype=torch.long, device=ps.device)
    gram = torch.zeros((X_batch.shape[0], Y_batch.shape[0]), dtype=ps.dtype, device=ps.device)
    gram = gram.index_put((i_all, j_all), vals)
    if symmetric:
        gram = gram.index_put((j_all, i_all), vals)
    return gram
