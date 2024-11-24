import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torch

from powersig.power_series import build_integration_gather_matrix_t, build_integration_gather_matrix_s, \
    MatrixPowerSeries
from powersig.util.series import torch_compute_dot_prod, torch_compute_derivative_batch, double_length


class FrontierParameters:
    def __init__(self, left_bc_ps: Optional[MatrixPowerSeries], bottom_bc_ps: Optional[MatrixPowerSeries], ic,
                 lb_samples=None, bb_samples=None):
        self.left_bc_ps = left_bc_ps
        self.bottom_bc_ps = bottom_bc_ps
        self.ic = ic
        self.lb_samples = lb_samples
        self.bb_samples = bb_samples

    def is_ready(self) -> bool:
        return self.left_bc_ps is not None and self.bottom_bc_ps is not None and self.ic is not None


class ParallelParameters:
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

    def compute_gram_matrix(self) -> torch.Tensor:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        gram_matrix = torch.zeros((self.dX.shape[0], self.dY.shape[0]), dtype=torch.float64)

        device_index = 0

        entries = []
        for i in range(self.dX.shape[0]):
            for j in range(self.dY.shape[0]):
                if len(devices) and self.dX.is_cuda > 0:
                    entries.append(ParallelParameters(i, j,
                                                      self.dX[i].to(device=devices[device_index]),
                                                      self.dY[j].to(device=devices[device_index])))
                    device_index = (device_index + 1) % len(devices)
                else:
                    entries.append(ParallelParameters(i, j, self.dX[i], self.dY[j]))

        for i, j, entry in self.executor.map(compute_gram_matrix_entry, entries):
            gram_matrix[i, j] = entry  # entry.item()

        return gram_matrix

    def compute_rho(self, i, j) -> torch.Tensor:
        return torch.matmul(self.dX[i], torch.t(self.dY[j]))


def compute_gram_matrix_entry(params: ParallelParameters) -> (int, int, float):
    return (
        params.i,
        params.j,
        compute_gram_entry(params.dX_i, params.dY_j)
    )


def build_tile_power_series(left_bc_ps: MatrixPowerSeries, bottom_bc_ps: MatrixPowerSeries, rho: float,
                            s_min: float, t_min: float, ic: float) -> MatrixPowerSeries:
    # Gather the constants into a new power series
    if torch.cuda.is_available():
        u = MatrixPowerSeries(left_bc_ps.coefficients.cuda()) + MatrixPowerSeries(bottom_bc_ps.coefficients.cuda()) - ic
    else:
        u = left_bc_ps + bottom_bc_ps - ic

    u_n = u.deep_clone()
    IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1],u_n.coefficients.device)
    IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0],u_n.coefficients.device)

    if torch.cuda.is_available():
        IminusG1 = IminusG1.cuda()
        IminusG2 = IminusG2.cuda()
    # print(f"u_0={u}")
    # Repeatedly integrate to generate the new power series
    # Truncate if necessary using tbd utility functions to eliminate terms with really small coefficients.

    while True:
        for step in range(1, 10):
            u_n.inplace_matrix_integrate(IminusG1, IminusG2)
            # u_n = u_n.integrate(s_base=s_min, t_base=t_min)
            u_n *= rho
            # print(f"u_{step}={u_n}")
            u += u_n
            # print(f"u={u}")

        if u_n.is_converged():
            break
        else:
            u_n = MatrixPowerSeries(double_length(u_n.coefficients))
            u = MatrixPowerSeries(double_length(u.coefficients))
            IminusG1 = build_integration_gather_matrix_s(s_min, u_n.coefficients.shape[1],u_n.coefficients.device)
            IminusG2 = build_integration_gather_matrix_t(t_min, u_n.coefficients.shape[0],u_n.coefficients.device)
            # print(f"Resized coefficient matrix to {u_n.coefficients.shape} for convergence.")
            if torch.cuda.is_available():
                IminusG1 = IminusG1.cuda()
                IminusG2 = IminusG2.cuda()

    print(f"Final size of coefficient matrix: {u.coefficients.shape}")
    # Return the resulting power series
    return u




def compute_gram_entry(dX_i, dY_j) -> float:
    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]

    # Initial boundary conditions
    ocs = torch.zeros([64, 64], device=dX_i.device, dtype=torch.float64)  # Hard coded truncation order
    ocs[0, 0] = 1
    one = MatrixPowerSeries(ocs)

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
            start = time.time()
            ps = build_tile_power_series(frontier.left_bc_ps, frontier.bottom_bc_ps,
                                         torch_compute_dot_prod(dX_i[s_i], dY_j[t_i]).item(),
                                         s_base, t_base, kg[s_i, t_i].item())
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
