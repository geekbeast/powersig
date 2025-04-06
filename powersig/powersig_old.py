import math
import os
import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from math import factorial
from typing import Optional

import torch

from power_series import MatrixPowerSeries, build_G1_s, build_G2_t, build_integration_gather_matrix_s, \
    build_integration_gather_matrix_t, TileSolutionPowerSeries
from powersig.util.series import torch_compute_derivative_batch, torch_compute_dot_prod


class VsParameters:
    def __init__(self, left_bc_ps: Optional[torch.Tensor], bottom_bc_ps: Optional[torch.Tensor], ic: torch.Tensor,
                 lb_samples=None, bb_samples=None):
        self.left_bc_ps = left_bc_ps
        self.bottom_bc_ps = bottom_bc_ps
        self.ic = ic
        self.lb_samples = lb_samples
        self.bb_samples = bb_samples

    def is_ready(self) -> bool:
        return self.left_bc_ps is not None and self.bottom_bc_ps is not None and self.ic is not None


class PowerSigGramEntryParams:
    def __init__(self, i, j, dX_i: torch.Tensor, dY_j: torch.Tensor):
        self.i = i
        self.j = j
        self.dX_i = dX_i
        self.dY_j = dY_j


class PowerSig:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.device == Y.device, "X and Y must be on same device."

        self.dX = torch_compute_derivative_batch(X)
        self.dY = torch_compute_derivative_batch(Y)

        # Hard coding for now.
        self.executor = ProcessPoolExecutor(2 * torch.cuda.device_count() if X.is_cuda else 8)

    def compute_gram_matrix(self) -> torch.Tensor:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        device_index = 0
        gram_matrix = torch.zeros((self.dX.shape[0], self.dY.shape[0]), dtype=torch.float64)
        entries = []
        for i in range(self.dX.shape[0]):
            for j in range(self.dY.shape[0]):
                entries.append(PowerSigGramEntryParams(i, j, self.dX[i].to(device=devices[device_index]),
                                                       self.dY[j].to(device=devices[device_index])))
                device_index = (device_index + 1) % len(devices)
        for i, j, entry in self.executor.map(compute_gram_matrix_entry, entries):
            gram_matrix[i, j] = entry  # entry.item()

        return gram_matrix

    def compute_rho(self, i, j) -> torch.Tensor:
        return torch.matmul(self.dX[i], torch.t(self.dY[j]))


def compute_gram_matrix_entry(params: PowerSigGramEntryParams) -> (int, int, float):
    return (
        params.i,
        params.j,
        compute_it(params.dX_i, params.dY_j)
    )

def compute_it(dX_i: torch.Tensor, dY_j: torch.Tensor) -> float:
    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]
    s = torch.arange(0,dX_i.shape[0], device=dX_i.device)/dX_i.shape[0]
    # print(f"s = {s}")
    grid_s, grid_t = torch.meshgrid( torch.arange(0,dX_i.shape[0], device=dX_i.device)/dX_i.shape[0], torch.arange(0,dY_j.shape[0], device=dX_i.device)/dY_j.shape[0])
    # print(f"grid_s: {(grid_s ).tolist()}")
    # print(f"grid_t: {grid_t}")
    rho = direct_compute_rho(dX_i, dY_j)
    term = 1
    total = term
    ell_contrib = 1.0
    for i in range(1,25):
        ell_contrib = ell(rho, ell_contrib, ds, dt, grid_s,grid_t, i).sum().item()
        # print(f"ell_contrib: {ell_contrib}")
        # term = ell_contrib
        total+=ell_contrib
    return total

def ell( rho: torch.Tensor, previous_ell: float, ds: float, dt:float, grid_s: torch.Tensor, grid_t: torch.Tensor, k: int):
    return rho*previous_ell*(torch.pow(grid_s + ds, k) - torch.pow(grid_s, k)) * (torch.pow(grid_t + dt, k) - torch.pow(grid_t, k)) * (k**-2)


def direct_compute_rho(dX_i: torch.Tensor, dY_j: torch.Tensor) -> torch.Tensor:
    return torch.matmul(dX_i, torch.t(dY_j))


def build_full_power_series(C: MatrixPowerSeries, dX_i: torch.Tensor, dY_j: torch.Tensor, ds: float,
                            dt: float) -> MatrixPowerSeries:
    print(f"C={C}")
    u = C.deep_clone()
    u_n = C.deep_clone()

    # Only do interior points
    for i in range(dX_i.shape[0] - 1):
        for j in range(dY_j.shape[0] - 1):
            rho = torch_compute_dot_prod(dX_i[i], dY_j[j])
            # u.coefficients[0,0] += rho *(ds*dt)
            print(f"rho = {rho}")
            G1_upper = build_G1_s(ds + (ds * j), u.coefficients.shape[1], u.coefficients.device)
            G1_diff = G1_upper - build_G1_s(ds * j, u.coefficients.shape[1], u.coefficients.device)
            G2_upper = build_G2_t(dt + (dt * i), u.coefficients.shape[0], u.coefficients.device)
            G2_diff = G2_upper - build_G2_t(dt * i, u.coefficients.shape[0], u.coefficients.device)
            G1 = torch.zeros([G1_diff.shape[0], G1_diff.shape[0]], dtype=torch.float64, device=u.coefficients.device)
            G2 = torch.zeros([G2_diff.shape[1], G2_diff.shape[1]], dtype=torch.float64, device=u.coefficients.device)
            G1[:, :1] = G1_diff
            G2[:1, :] = G2_diff
            # u_n.inplace_matrix_integrate(G1, G2)
            # u_n *= rho
            result = C.deep_clone().inplace_matrix_integrate(G1, G2) * rho  # u_n
            print(f"Result = {result}")
            u += C.deep_clone().inplace_matrix_integrate(G1, G2) * rho  # u_n
            # # Gather the constants into a new power series
    print(f"u_const={u}")
    j = dY_j.shape[0] - 1
    IminusG1 = build_integration_gather_matrix_s(j * ds, C.coefficients.shape[1], C.coefficients.device)
    for i in range(dX_i.shape[0] - 1):
        rho = torch_compute_dot_prod(dX_i[i], dY_j[j])
        print(f"rho = {rho}")
        G2_upper = build_G2_t(dt + (dt * i), u.coefficients.shape[0], u.coefficients.device)
        G2_diff = G2_upper - build_G2_t(dt * i, u.coefficients.shape[0], u.coefficients.device)
        G2 = torch.zeros([G2_diff.shape[1], G2_diff.shape[1]], dtype=torch.float64, device=u.coefficients.device)
        G2[:1, :] = G2_diff
        # u_n.inplace_matrix_integrate(G1, G2)
        # u_n *= rho
        u += C.deep_clone().inplace_matrix_integrate(IminusG1, G2) * rho  # u_n

    i = dX_i.shape[0] - 1
    IminusG2 = build_integration_gather_matrix_t(i * dt, C.coefficients.shape[0], C.coefficients.device)
    for j in range(dY_j.shape[0] - 1):
        rho = torch_compute_dot_prod(dX_i[i], dY_j[j])
        print(f"rho = {rho}")
        G1_upper = build_G1_s(ds + (ds * j), u.coefficients.shape[1], u.coefficients.device)
        G1_diff = G1_upper - build_G1_s(ds * j, u.coefficients.shape[1], u.coefficients.device)
        G1 = torch.zeros([G1_diff.shape[0], G1_diff.shape[0]], dtype=torch.float64, device=u.coefficients.device)
        G1[:, :1] = G1_diff
        # u_n.inplace_matrix_integrate(G1, G2)
        # u_n *= rho
        u += C.deep_clone().inplace_matrix_integrate(G1, IminusG2) * rho  # u_n

    j = dY_j.shape[0] - 1
    rho = torch_compute_dot_prod(dX_i[i], dY_j[j])
    IminusG1 = build_integration_gather_matrix_s(j * ds, C.coefficients.shape[1], C.coefficients.device)
    IminusG2 = build_integration_gather_matrix_t(i * dt, C.coefficients.shape[0], C.coefficients.device)

    # u_n.inplace_matrix_integrate(G1, G2)
    # u_n *= rho
    u += C.deep_clone().inplace_matrix_integrate(IminusG1, IminusG2) * rho  # u_n
    print(f"u_final={u}")
    return u

def build_tridiagonal_power_series(left_bcc: torch.Tensor, right_bcc: torch.Tensor, ic: torch.Tensor, rho: float, s_base: float, t_base: float):
    pass

def compute_gram_entry_full(dX_i, dY_j) -> float:
    ds = 1 / dX_i.shape[0]
    dt = 1 / dY_j.shape[0]

    # Initial boundary conditions
    ocs = torch.zeros([256, 256], device=dX_i.device, dtype=torch.float64)  # Hard coded truncation order
    ocs[0, 0] = 1
    one = torch.ones([1, 1], dtype=torch.float64, device=dX_i.device)
    zero = torch.zeros([1, 1], dtype=torch.float64, device=dX_i.device)
    initial_left_bc_ps = torch.clone(zero)
    initial_bottom_bc_ps = torch.clone(zero)

    frontiers = {}
    frontiers.setdefault((0, 0), VsParameters(initial_left_bc_ps, initial_bottom_bc_ps, one,
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
            ps = TileSolutionPowerSeries(torch_compute_dot_prod(dX_i[s_i], dY_j[t_i]).item(),s_base, t_base, kg[s_i, t_i].item(), frontier.left_bc_ps, frontier.bottom_bc_ps, dX_i.device)
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


