import string
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import math
import torch
from math import isclose

from torch.onnx.symbolic_opset9 import unsqueeze

from powersig.power_series import build_integration_gather_matrix_t, build_integration_gather_matrix_s, \
    MatrixPowerSeries, build_A1, build_A2, build_integration_limit_matrix_s, build_integration_limit_matrix_t
from powersig.util.series import torch_compute_dot_prod, torch_compute_derivative_batch, double_length, \
    torch_compute_dot_prod_batch


class FrontierParameters:
    def __init__(self, left_bc_ps: Optional[MatrixPowerSeries], bottom_bc_ps: Optional[MatrixPowerSeries], ic: float,
                 lb_samples=None, bb_samples=None):
        self.left_bc_ps = left_bc_ps
        self.bottom_bc_ps = bottom_bc_ps
        self.ic = ic
        self.lb_samples = lb_samples
        self.bb_samples = bb_samples

    def is_ready(self) -> bool:
        return self.left_bc_ps is not None and self.bottom_bc_ps is not None and self.ic is not None


class SignatureKernelParameters:
    def __init__(self, i: int, j: int, dX_i: torch.Tensor, dY_j: torch.Tensor):
        self.i = i
        self.j = j
        self.dX_i = dX_i
        self.dY_j = dY_j


class MatrixSig:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.device == Y.device, "X and Y must be on same device."

        self.dX = torch_compute_derivative_batch(X)
        self.dY = torch_compute_derivative_batch(Y)
        # Hard coding for now.
        self.executor = ProcessPoolExecutor(2 * torch.cuda.device_count() if X.is_cuda else 8)
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    def compute_gram_matrix(self) -> torch.Tensor:
        gram_matrix = torch.zeros((self.dX.shape[0], self.dY.shape[0]), dtype=torch.float64)
        device_index = 0

        entries = []
        for i in range(self.dX.shape[0]):
            for j in range(self.dY.shape[0]):
                # print(
                #     f"rho = {torch.exp(torch.matmul(self.dX[i], torch.t(self.dY[j])).sum() / (self.dX[i].shape[0] * self.dY[j].shape[0])).item()}")
                # gram_matrix[i, j] = compute_gram_entry( self.dX[i], self.dY[j])
                if len(self.devices) > 0 and self.dX.is_cuda:
                    entries.append(SignatureKernelParameters(i, j,
                                                             self.dX[i].to(device=self.devices[device_index]),
                                                             self.dY[j].to(device=self.devices[device_index])))
                    device_index = (device_index + 1) % len(self.devices)
                else:
                    entries.append(SignatureKernelParameters(i, j, self.dX[i], self.dY[j]))

        for i, j, entry in self.executor.map(compute_gram_matrix_entry, entries):
            gram_matrix[i, j] = entry  # entry.item()

        return gram_matrix

    def compute_rho(self, i, j) -> torch.Tensor:
        return torch.matmul(self.dX[i], torch.t(self.dY[j]))


def compute_gram_matrix_entry(params: SignatureKernelParameters) -> (int, int, float):
    return (
        params.i,
        params.j,
        compute_gram_entry(params.dX_i, params.dY_j)
    )


def compute_gram_matrix_entry_iteration(params: SignatureKernelParameters) -> (int, int, float):
    ds = 1 / params.dX_i.shape[0]
    dt = 1 / params.dY_j.shape[0]

    # Initial boundary conditions
    ocs = torch.eye(64, device=params.dX_i.device, dtype=torch.float64)  # Hard coded truncation order
    # ocs[0, 0] = 1
    A1 = build_A1(ocs.shape[0], params.dX_i.device)
    A2 = build_A2(ocs.shape[1], params.dY_j.device)

    for i in range(10):
        acc = torch.zeros((64, 64), device=params.dX_i.device, dtype=torch.float64)
        for i in range(params.dX_i.shape[0]):
            for j in range(params.dY_j.shape[0]):
                L = None
                R = None
                if i == ocs.shape[0] - 1 and j == ocs.shape[1] - 1:
                    L = torch.matmul(
                        build_integration_gather_matrix_t(j / params.dY_j.shape[0], ocs.shape[1], ocs.device), A2)
                    R = torch.matmul(A1, build_integration_gather_matrix_s(i / params.dX_i.shape[0], ocs.shape[1],
                                                                           ocs.device))

                elif i == ocs.shape[0] - 1:
                    R = torch.matmul(A1, build_integration_gather_matrix_s(i / params.dX_i.shape[0], ocs.shape[1],
                                                                           ocs.device))
                    L = torch.matmul(
                        build_integration_limit_matrix_t(j / params.dY_j.shape[0], (j + 1) / params.dY_j.shape[0],
                                                         ocs.shape[1], ocs.device), A2)

                elif j == ocs.shape[1] - 1:
                    R = torch.matmul(A1,
                                     build_integration_limit_matrix_s(i / params.dX_i.shape[0],
                                                                      (i + 1) / params.dX_i.shape[0],
                                                                      ocs.shape[1],
                                                                      ocs.device))
                    L = torch.matmul(
                        build_integration_gather_matrix_t(j / params.dY_j.shape[0], ocs.shape[1], ocs.device), A2)

                else:
                    R = torch.matmul(A1,
                                     build_integration_limit_matrix_s(i / params.dX_i.shape[0],
                                                                      (i + 1) / params.dX_i.shape[0],
                                                                      ocs.shape[1],
                                                                      ocs.device))
                    L = torch.matmul(
                        build_integration_limit_matrix_t(j / params.dY_j.shape[0], (j + 1) / params.dY_j.shape[0],
                                                         ocs.shape[1],
                                                         ocs.device), A2)
                acc += torch_compute_dot_prod(params.dX_i[i], params.dY_j[j]) * torch.matmul(torch.matmul(L, ocs), R)
            ocs = acc
    one = MatrixPowerSeries(ocs)
    return params.i, params.j, one(1, 1)


def build_tile_power_series_stencil(shape: torch.Size, device: torch.device):
    i_vals = torch.arange(start=1, end=shape[0], dtype=torch.int64, device=device)
    j_vals = torch.arange(start=1, end=shape[1], dtype=torch.int64, device=device)
    fact_i = torch.lgamma(i_vals + 1)  # shape [N]
    fact_j = torch.lgamma(j_vals + 1)  # shape [M]3

    # Create 2D mesh: i_grid[i,j] = i, j_grid[i,j] = j
    i_grid, j_grid = torch.meshgrid(i_vals, j_vals, indexing='ij')
    fact_grid_i, fact_grid_j = torch.meshgrid(fact_i, fact_j, indexing='ij')
    # min(i,j) in a 2D tensor determines the power of row based off
    min_ij = torch.minimum(i_grid, j_grid).to(dtype=torch.float64)
    denominator = fact_grid_i + fact_grid_j

    return min_ij, denominator


def build_tile_power_series(left_bc_ps: MatrixPowerSeries, bottom_bc_ps: MatrixPowerSeries, rho: float,
                            s_min: float, t_min: float, ic: float,
                            ds: float, dt: float,
                            min_ij, denom,
                            a: dict[int, torch.Tensor]) -> MatrixPowerSeries:
    # Gather the constants into a new power series
    if torch.cuda.is_available():
        u = MatrixPowerSeries(left_bc_ps.coefficients.cuda()) + MatrixPowerSeries(
            bottom_bc_ps.coefficients.cuda()) - ic
    else:
        u = left_bc_ps + bottom_bc_ps - ic
    print(f"rho = {rho}")
    print(f"u_0 = {u}")
    start = time.time()
    g1 = u.build_gather_s(s_min)
    g2 = u.build_gather_t(t_min)
    # C = u.coefficients
    #
    # min_ij_log_rho = min_ij * math.log(rho)
    # new_entries = torch.exp(min_ij_log_rho - denom)
    #
    # new_entries.diagonal().__imul__(C[0,0])
    #
    # for i in range(1,new_entries.shape[0]):
    #     # print(f"C[0,{i}] = {C[0,i]}")
    #     new_entries.diagonal(i).__imul__(C[0,i])
    # for j in range(1,new_entries.shape[1]):
    #     new_entries.diagonal(-j).__imul__(C[j,0])
    #
    # print(f"new_entries = {new_entries.tolist()}")
    #
    # C[1:,1:] = new_entries
    # C[1:, :1] -= torch.mm(new_entries, g1[1:,:])
    # C[:1, 1:] -= torch.mm(g2[:,1:],new_entries)
    #
    # print(f"Elapsed time: {time.time() - start}")
    # print(f"u = {u}")
    #
    # return MatrixPowerSeries(u.coefficients.cpu())
    u_n = u.deep_clone()
    IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1], u_n.coefficients.device)
    IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0], u_n.coefficients.device)
    g1 = u.build_gather_s(s_min + ds)
    g2 = u.build_gather_t(t_min + dt)

    A1 = None
    A2 = None

    if u_n.coefficients.shape[0] not in a:
        A1 = build_A1(u_n.coefficients.shape[1], u_n.coefficients.device)
        A2 = build_A2(u_n.coefficients.shape[0], u_n.coefficients.device)
        a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])] = (A1, A2)

    A1, A2 = a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])]

    if torch.cuda.is_available():
        IminusG1 = IminusG1.cuda()
        IminusG2 = IminusG2.cuda()

    # estimate = None
    # if not isclose(0, s_min, rel_tol=0, abs_tol=1e-10) > 0 and not isclose(0, t_min, rel_tol=0, abs_tol=1e-10):
    #     L = torch.matmul(IminusG2, A2)
    #     R = torch.matmul(A1, IminusG1)
    # print(f"L={L.to_dense()}")
    # print(f"R={R.to_dense()}")
    # IminusLR = torch.eye(L.shape[0], dtype=torch.float64, device=IminusG1.device) - rho * torch.matmul(L, R)
    # print(f"IminusLR={IminusLR.to_dense()}")
    # estimate = torch.matmul(torch.linalg.inv(IminusLR), u.coefficients)

    # print(f"u_0={u}")
    # Repeatedly integrate to generate the new power series
    # Truncate if necessary using tbd utility functions to eliminate terms with really small coefficients.
    # prev = u.evaluate(g1, g2).item()
    # print(f"Initial power series solution for tile @ ({s_min},{t_min}): {prev}")
    truncation_order = 0
    prev = u_n.evaluate(g1, g2).item()
    converged = False
    while True:
        while not converged and truncation_order < (u.coefficients.shape[0] - 1):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
            u_n *= rho
            next = u_n.evaluate(g1, g2).item()
            # print(f"u_{step}={u_n}")
            u += u_n
            truncation_order += 1
            if math.isclose(0, next - prev, rel_tol=0.0, abs_tol=1e-10):
                converged = True
                prev = next
                break
            prev = next
            # print(f"u={u}")

        # is_converged, prev = u.is_converged(prev, g1, g2, 1e-2)

        if converged:
            break
        else:
            print(f"Rho on resize: {rho}")
            u_n = MatrixPowerSeries(double_length(u_n.coefficients))
            u = MatrixPowerSeries(double_length(u.coefficients))
            A1 = build_A1(u_n.coefficients.shape[1], u_n.coefficients.device)
            A2 = build_A2(u_n.coefficients.shape[0], u_n.coefficients.device)
            a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])] = (A1, A2)
            IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1], u_n.coefficients.device)
            IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0], u_n.coefficients.device)
            g1 = u.build_gather_s(s_min + ds)
            g2 = u.build_gather_t(t_min + dt)
            # L = torch.matmul(IminusG2, A2)
            # R = torch.matmul(A1, IminusG1)
            # print(f"L={L.to_dense()}")
            # print(f"R={R.to_dense()}")
            # IminusLR = torch.eye(L.shape[0], dtype=torch.float64, device=IminusG1.device) - rho * torch.matmul(L, R)
            # print(f"IminusLR={IminusLR.to_dense()}")
            # estimate = torch.matmul(torch.linalg.inv(IminusLR), u_n.coefficients)
            print(f"Resized coefficient matrix to {u_n.coefficients.shape} for convergence.")
            if torch.cuda.is_available():
                IminusG1 = IminusG1.cuda()
                IminusG2 = IminusG2.cuda()

    # print(f"Final size of coefficient matrix: {u.coefficients.shape}")
    # Return the resulting power series

    # if estimate is not None:
    #     mse = torch.mean((u.coefficients[:estimate.shape[0], :estimate.shape[1]].cpu() - estimate) ** 2)
    #     # print(f"Estimate: {estimate}")
    print(f"u = {u}")
    #     print(f"MSE for estimate = {mse}")

    return MatrixPowerSeries(u.coefficients.cpu())


def compute_gram_entry(dX_i, dY_j) -> float:
    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]
    # Initial boundary conditions
    ocs = torch.zeros([32, 32], device=dX_i.device, dtype=torch.float64)  # Hard coded initial truncation order
    ocs[0, 0] = 1
    one = MatrixPowerSeries(ocs)
    Amap = {(ocs.shape[0], ocs.shape[1]): (build_A1(ocs.shape[1], ocs.device), build_A2(ocs.shape[0], ocs.device))}
    min_ij, denom = build_tile_power_series_stencil(ocs.shape, dX_i.device)

    initial_left_bc_ps = one.deep_clone()
    initial_bottom_bc_ps = one.deep_clone()

    frontiers = {}
    frontiers.setdefault((0, 0), FrontierParameters(initial_left_bc_ps, initial_bottom_bc_ps, None,
                                                    lb_samples=(1 / (2 * dX_i.shape[0]), 1),
                                                    bb_samples=(1 / (2 * dY_j.shape[0]), 1)))
    next_frontiers = {}
    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    kg = torch.zeros([dX_i.shape[0], dY_j.shape[0]], device=dX_i.device, dtype=torch.float64)
    kg[:, 0] = 1
    kg[0, :] = 1

    for d_i in range(diagonal_count):
        for grid_point in frontiers:
            frontier = frontiers[grid_point]
            s_i, t_i = grid_point
            s_base = s_i * ds
            t_base = t_i * dt
            # print(f"Processing tile {s_i}, {t_i} with rho = {rho[s_i, t_i]} and kg[{s_i}][{t_i}] = {kg[s_i, t_i]}")
            # print(f"Left initial: {frontier.left_bc_ps(s_base, t_base)}, Right initial: {frontier.bottom_bc_ps(s_base, t_base)}")
            # print(f"Sum check: {frontier.left_bc_ps(s_base, t_base) + frontier.bottom_bc_ps(s_base, t_base)} == { (frontier.left_bc_ps+frontier.bottom_bc_ps)(s_base,t_base)}")
            streams = []
            start = time.time()

            ps = build_tile_power_series(frontier.left_bc_ps, frontier.bottom_bc_ps,
                                         torch_compute_dot_prod(dX_i[s_i], dY_j[t_i]).item(),
                                         s_base, t_base, kg[s_i, t_i].item(),
                                         ds, dt,
                                         min_ij, denom,
                                         {})
            # print(f"Solving one tile took {time.time() - start:.2f}s")
            # print(f"Series for {s_i},{t_i} = {ps.human_readable()}")
            if frontier.lb_samples is not None:
                lbsp, k_lb = frontier.lb_samples
                # print(f"Left Boundary({s_base},{lbsp}) = {k_lb}")
                # print(f"Left Solution({s_base},{lbsp}) = {ps(s_base, lbsp)}")

            if frontier.bb_samples is not None:
                bbsp, k_bb = frontier.bb_samples
                # print(f"Bottom Boundary({bbsp},{t_base}) = {k_bb}")
                # print(f"Bottom Solution({bbsp},{t_base}) = {ps(bbsp, t_base)}")

            if s_i == (dX_i.shape[0] - 1) and t_i == (dY_j.shape[0] - 1):
                return ps(1, 1)

            # Register boundary conditions with tiles to the right and above, if they need computing
            next_s_i = s_i + 1
            next_s_base = s_base + ds
            next_t_base = t_base + dt
            next_t_i = t_i + 1
            # Sample points for testing
            # lb_sample_point = t_base + dt * random.uniform(0, 1)
            # bb_sample_point = s_base + ds * random.uniform(0, 1)

            if next_s_i < dX_i.shape[0]:
                # k_lb = ps(next_s_base, lb_sample_point)
                nf = next_frontiers.setdefault((next_s_i, t_i), FrontierParameters(None, None, None))
                if t_i == 0:
                    nf.bottom_bc_ps = one.deep_clone()
                nf.left_bc_ps = ps.deep_clone().bind_s(next_s_base)
                # nf.lb_samples = (lb_sample_point, k_lb)

            if next_t_i < dY_j.shape[0]:
                # k_bb = ps(bb_sample_point, next_t_base)
                nf = next_frontiers.setdefault((s_i, next_t_i), FrontierParameters(None, None, None))
                if s_i == 0:
                    nf.left_bc_ps = one.deep_clone()
                nf.bottom_bc_ps = ps.bind_t(next_t_base)
                # nf.bb_samples = (bb_sample_point, k_bb)

            # Use most accurate value for ic
            if next_s_i < dX_i.shape[0] and next_t_i < dY_j.shape[0]:
                kg[next_s_i, next_t_i] = ps(next_s_base, next_t_base)

        frontiers = next_frontiers
        next_frontiers = {}

    raise RuntimeError("Unable to compute gram matrix")


def build_scaling_for_integration(order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scales = torch.arange(1, order + 1, device=device, dtype=dtype) ** -1
    return torch.mm(scales.view(-1, 1), scales.view(1, -1))


def build_vandermonde_matrix_s(s: torch.Tensor, order: int, device: torch.device, dtype: torch.dtype,
                               shift: int = 0) -> torch.Tensor:
    powers = torch.arange(shift, order + shift, device=device, dtype=dtype)
    return s.unsqueeze(1).pow(powers).unsqueeze(-1)


def build_vandermonde_matrix_t(t: torch.Tensor, order: int, device: torch.device, dtype: torch.dtype,
                               shift: int = 0) -> torch.Tensor:
    powers = torch.arange(shift, order + shift, device=device, dtype=dtype)
    return t.unsqueeze(1).pow(powers).unsqueeze(1)


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


def reverse_linspace_0_1(steps: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    inbetween_count = steps - 1
    s = torch.arange(inbetween_count, -1, -1, device=device, dtype=dtype)
    s /= inbetween_count
    return s


def diagonal_to_string(v: torch.Tensor):
    for d in range(v.shape[0]):
        print(f"Diagonal index: {d}")
        ps = ""
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                c = v[d, i, j].item()
                if abs(c) > 0:
                    if ps == "":
                        ps += f"{c}*s^{j}*t^{i} "
                    else:
                        ps += f"+ {c}*s^{j}*t^{i} "
        print(ps)


@torch.compile()
def tensor_compute_gram_entry(dX_i: torch.Tensor, dY_j: torch.Tensor, scales: torch.Tensor, order: int = 32) -> float:
    dX_i[:] = dX_i.flip(0)
    # Initial tile
    u = torch.zeros([1, order, order], dtype=dX_i.dtype, device=dX_i.device)
    prev_u = None

    s = reverse_linspace_0_1(dX_i.shape[0] + 1, dtype=u.dtype, device=u.device)
    t = torch.linspace(0, 1, dY_j.shape[0] + 1, dtype=u.dtype, device=u.device)

    v_s = None
    v_t = None

    diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1

    for d in range(diagonal_count):
        s_start, t_start, dlen = get_diagonal_range(d, dX_i.shape[0], dY_j.shape[0])
        u = torch.zeros([dlen, order, order], dtype=dX_i.dtype, device=dX_i.device)

        # This is for the left / bottom boundaries of the current set of diagonals
        s_L = len(s) - (s_start + 1)
        s_i = s[s_L:s_L + dlen]
        t_j = t[t_start:(t_start+dlen)]
        v_s = build_vandermonde_matrix_s(s_i, order, u.device, u.dtype, 1)
        v_t = build_vandermonde_matrix_t(t_j, order, u.device, u.dtype, 1)

        # print(f"vandermonde matrix s: {v_s}")
        # print(f"vandermonde matrix t: {v_t}")

        # Compute the initial power series that will be iterated for tile on the diagonal
        if d == 0:
            u[0, 0, 0] = 1
        else:
            s_b = build_vandermonde_matrix_s(s_i, order, u.device, u.dtype, 0)
            t_b = build_vandermonde_matrix_t(t_j, order, u.device, u.dtype, 0)
            # Build the next diagonal. We will only use the boundaries for computational efficiency reasons.
            # While we could directly use the top right corner of each tile to compute the initial condition,
            # This would require us storing values for diagonals we aren't directly working on at the moment.
            # We can instead use either of the boundaries for a tile to compute the initial value.

            # Use the left / bottom boundaries to set the initial conditions based on previous solution for u
            # This is for the left / bottom boundaries of the current set of diagonals
            #s_b = build_vandermonde_matrix_s(s[-(s_start+1):], order, u.device, u.dtype)
            #t_b = build_vandermonde_matrix_t(t[t_start:], order, u.device, u.dtype)

            if d < dX_i.shape[0]:
                # If you haven't reached the right edge, diagonal tiles in u will be the same length or longer than prev_u
                # We only care about prev_u.shape[0] propagations from L -> R at v_s points
                torch.bmm(prev_u, s_b[:prev_u.shape[0]], out=u[:prev_u.shape[0], :, :1])
            else:
                # Skip the first one of the previous as you've reached the right edge.
                # If you have reached the right edge diagonals only get shorter.
                torch.bmm(prev_u[1:,:,:], s_b, out=u[:, :, :1])

            start_offset, stop_offset = 0, 0

            if t_start == 0:
                start_offset = 1
            if s_start - dlen + 1 == 0:
                stop_offset = 1
            if start_offset + stop_offset < dlen:
                u[start_offset:(start_offset+dlen-stop_offset),:1, :1] -= torch.bmm(t_b[start_offset:(start_offset+dlen-stop_offset)],u[start_offset:(start_offset+dlen-stop_offset), :, :1])

            # We can't do these in place since we are adding
            if d < dY_j.shape[0]:
                # Always propagate all tiles up, skip v_t[1,:,:] and u[1,:,:] since there is no corresponding tile below
                # u will always be the same or longer than prev_u for this case
                u[1:, :1, :]+=torch.bmm(t_b[1:,:,:], prev_u)
            else:
                # We don't want to propagate the last tile in diagonal up.
                # Need to figure out whether we want to skip first tile.
                if t_start == 0:
                    # t_start = 0 and first tile doesn't need bottom boundary propagated
                    u[1:, :1, :]+=torch.bmm(t_b[1:, :, :] , prev_u[:-1, :, :])
                else:
                    # The first tile needs to have bottom boundary propagated
                    u[:, :1, :]+=torch.bmm(t_b, prev_u[:-1, :, :])

            # We need to subtract out all the boundary conditions.
            # prev_u

            # print(f"u_0 = {u}")
            # diagonal_to_string(u)

            # Some clean  up
            to_delete = prev_u
            prev_u = None
            del to_delete

        dX_L = dX_i.shape[0] - (s_start + 1)
        # print(f"dX_L = {dX_L}")
        # print(f"s_start = {s_start}")
        rho = torch_compute_dot_prod_batch(dX_i[dX_L:dX_L + dlen].unsqueeze(1), dY_j[t_start:(t_start+dlen)].unsqueeze(1))

        # print(f"rho = {rho}")

        u_n = torch.clone(u)

        for i in range(order - 1):
            u_step = rho.view(-1,1,1) * u_n
            u_step *= scales
            # print(f"u_step = {u_step}")
            u_n[:, 1:, 1:] = u_step[:, :-1, :-1]
            u_n[:, :1, 1:] = -torch.bmm(v_t, u_step)[:, :, :-1]
            u_step_s = torch.bmm(u_step, v_s)
            u_n[:, 1:, :1] = -u_step_s[:, :-1, :]
            # print(f"(v_t . u_n[:, :, :1]) = {torch.bmm(v_t, u_n[:, :, :1])}")
            u_n[:, :1, :1] = torch.bmm(v_t, u_step_s)
            # print(f"u_n = {u_n}")
            u += u_n
            # print(f"u = {u}")

        prev_u = u

    return u.sum().item()
