import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import math
import torch
from math import isclose


from powersig.power_series import build_integration_gather_matrix_t, build_integration_gather_matrix_s, \
    MatrixPowerSeries, build_A1, build_A2, build_integration_limit_matrix_s, build_integration_limit_matrix_t
from powersig.util.series import torch_compute_dot_prod, torch_compute_derivative_batch, double_length


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
                print(
                    f"rho = {torch.exp(torch.matmul(self.dX[i], torch.t(self.dY[j])).sum() / (self.dX[i].shape[0] * self.dY[j].shape[0])).item()}")
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
    i_vals = torch.arange(start=1, end=shape[0], dtype=torch.int32, device=device)
    j_vals = torch.arange(start=1, end=shape[1], dtype=torch.int32, device=device)
    fact_i = torch.lgamma(i_vals + 1)  # shape [N]
    fact_j = torch.lgamma(j_vals + 1)  # shape [M]


    # Create 2D mesh: i_grid[i,j] = i, j_grid[i,j] = j
    i_grid, j_grid = torch.meshgrid(i_vals, j_vals, indexing='ij')
    fact_grid_i, fact_grid_j = torch.meshgrid(fact_i, fact_j, indexing='ij')
    # min(i,j) in a 2D tensor determines the power of row based off
    min_ij = torch.minimum(i_grid, j_grid)
    denominator = fact_grid_i + fact_grid_j

    return min_ij, denominator

def build_tile_power_series(left_bc_ps: MatrixPowerSeries, bottom_bc_ps: MatrixPowerSeries, rho: float,
                            s_min: float, t_min: float, ic: float,
                            ds: float, dt: float,
                            a: dict[int, torch.Tensor]) -> MatrixPowerSeries:
    # Gather the constants into a new power series
    if torch.cuda.is_available():
        u = MatrixPowerSeries(left_bc_ps.coefficients.cuda()) + MatrixPowerSeries(
            bottom_bc_ps.coefficients.cuda()) - ic
    else:
        u = left_bc_ps + bottom_bc_ps - ic

    print(f"u_0 = {u}")
    start = time.time()
    g1 = u.build_gather_s(s_min)
    g2 = u.build_gather_t(t_min)
    C = u.coefficients

    i_vals = torch.arange(start=1, end=C.shape[0], dtype=C.dtype, device=C.device)
    j_vals = torch.arange(start=1, end=C.shape[1], dtype=C.dtype, device=C.device)

    # factorial(i) = gamma(i+1)
    fact_i = torch.lgamma(i_vals + 1)  # shape [N]
    fact_j = torch.lgamma(j_vals + 1)  # shape [M]
    fact_grid_i, fact_grid_j = torch.meshgrid(fact_i, fact_j, indexing = 'ij')
    # Create 2D mesh: i_grid[i,j] = i, j_grid[i,j] = j
    i_grid, j_grid = torch.meshgrid(i_vals, j_vals, indexing='ij')

    # min(i,j) in a 2D tensor
    min_ij = torch.minimum(i_grid, j_grid)

    # min(i,j) * log(rho)
    min_ij_log_rho = min_ij * math.log(rho)

    # factor(i)*factor(j) in broadcast form
    denom = fact_grid_i + fact_grid_j #fact_i[i_grid.long()] + fact_j[j_grid.long()]
    print(f"denom shape = {denom.shape}")
    print(f"min_ij_log_rho shape = {min_ij_log_rho.shape}")
    new_entries = torch.exp(min_ij_log_rho - denom)

    new_entries.diagonal().__imul__(C[0,0])

    for i in range(1,new_entries.shape[0]):
        # print(f"C[0,{i}] = {C[0,i]}")
        new_entries.diagonal(i).__imul__(C[0,i])
    for j in range(1,new_entries.shape[0]):
        new_entries.diagonal(-j).__imul__(C[j,0])

    print(f"new_entries = {new_entries.tolist()}")

    C[1:,1:] = new_entries
    C[1:, :1] -= torch.mm(new_entries, g1[1:,:])
    C[:1, 1:] -= torch.mm(g2[:,1:],new_entries)

    print(f"Elapsed time: {time.time() - start}")
    print(f"u = {u}")

    return MatrixPowerSeries(u.coefficients.cpu())
    # u_n = u.deep_clone()
    # IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1], u_n.coefficients.device)
    # IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0], u_n.coefficients.device)
    # g1 = u.build_gather_s(s_min + ds)
    # g2 = u.build_gather_t(t_min + dt)
    #
    # A1 = None
    # A2 = None
    #
    # if u_n.coefficients.shape[0] not in a:
    #     A1 = build_A1(u_n.coefficients.shape[1], u_n.coefficients.device)
    #     A2 = build_A2(u_n.coefficients.shape[0], u_n.coefficients.device)
    #     a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])] = (A1, A2)
    #
    # A1, A2 = a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])]
    #
    # if torch.cuda.is_available():
    #     IminusG1 = IminusG1.cuda()
    #     IminusG2 = IminusG2.cuda()
    #
    # # estimate = None
    # # if not isclose(0, s_min, rel_tol=0, abs_tol=1e-10) > 0 and not isclose(0, t_min, rel_tol=0, abs_tol=1e-10):
    # #     L = torch.matmul(IminusG2, A2)
    # #     R = torch.matmul(A1, IminusG1)
    # # print(f"L={L.to_dense()}")
    # # print(f"R={R.to_dense()}")
    # # IminusLR = torch.eye(L.shape[0], dtype=torch.float64, device=IminusG1.device) - rho * torch.matmul(L, R)
    # # print(f"IminusLR={IminusLR.to_dense()}")
    # # estimate = torch.matmul(torch.linalg.inv(IminusLR), u.coefficients)
    #
    # # print(f"u_0={u}")
    # # Repeatedly integrate to generate the new power series
    # # Truncate if necessary using tbd utility functions to eliminate terms with really small coefficients.
    # # prev = u.evaluate(g1, g2).item()
    # # print(f"Initial power series solution for tile @ ({s_min},{t_min}): {prev}")
    # truncation_order = 0
    # prev = u_n.evaluate(g1, g2).item()
    # converged = False
    # while True:
    #     while not converged and truncation_order < (u.coefficients.shape[0]-1):
    #         u_n.inplace_matrix_integrate(IminusG1, IminusG2, A1, A2)
    #         u_n *= rho
    #         next = u_n.evaluate(g1, g2).item()
    #         # print(f"u_{step}={u_n}")
    #         u += u_n
    #         truncation_order += 1
    #         if math.isclose(0, next-prev,rel_tol = 0.0, abs_tol = 1e-5):
    #             converged = True
    #             prev = next
    #             break
    #         prev = next
    #         # print(f"u={u}")
    #
    #
    #     # is_converged, prev = u.is_converged(prev, g1, g2, 1e-2)
    #
    #     if converged:
    #         break
    #     else:
    #         print(f"Rho on resize: {rho}")
    #         u_n = MatrixPowerSeries(double_length(u_n.coefficients))
    #         u = MatrixPowerSeries(double_length(u.coefficients))
    #         A1 = build_A1(u_n.coefficients.shape[1], u_n.coefficients.device)
    #         A2 = build_A2(u_n.coefficients.shape[0], u_n.coefficients.device)
    #         a[(u_n.coefficients.shape[0], u_n.coefficients.shape[1])] = (A1, A2)
    #         IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1], u_n.coefficients.device)
    #         IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0], u_n.coefficients.device)
    #         g1 = u.build_gather_s(s_min + ds)
    #         g2 = u.build_gather_t(t_min + dt)
    #         # L = torch.matmul(IminusG2, A2)
    #         # R = torch.matmul(A1, IminusG1)
    #         # print(f"L={L.to_dense()}")
    #         # print(f"R={R.to_dense()}")
    #         # IminusLR = torch.eye(L.shape[0], dtype=torch.float64, device=IminusG1.device) - rho * torch.matmul(L, R)
    #         # print(f"IminusLR={IminusLR.to_dense()}")
    #         # estimate = torch.matmul(torch.linalg.inv(IminusLR), u_n.coefficients)
    #         print(f"Resized coefficient matrix to {u_n.coefficients.shape} for convergence.")
    #         if torch.cuda.is_available():
    #             IminusG1 = IminusG1.cuda()
    #             IminusG2 = IminusG2.cuda()
    #
    # # print(f"Final size of coefficient matrix: {u.coefficients.shape}")
    # # Return the resulting power series
    #
    # # if estimate is not None:
    # #     mse = torch.mean((u.coefficients[:estimate.shape[0], :estimate.shape[1]].cpu() - estimate) ** 2)
    # #     # print(f"Estimate: {estimate}")
    # #     # print(f"u = {u}")
    # #     print(f"MSE for estimate = {mse}")
    #
    # return MatrixPowerSeries(u.coefficients.cpu())


def compute_gram_entry(dX_i, dY_j) -> float:
    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]
    # Initial boundarey conditions
    ocs = torch.zeros([32, 32], device=dX_i.device, dtype=torch.float64)  # Hard coded initial truncation order
    ocs[0, 0] = 1
    one = MatrixPowerSeries(ocs)
    Amap = {(ocs.shape[0], ocs.shape[1]): (build_A1(ocs.shape[1], ocs.device), build_A2(ocs.shape[0], ocs.device))}

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
                                         ds, dt, {})
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
