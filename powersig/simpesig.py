import random
from typing import Optional

import torch

from powersig.power_series import SimplePowerSeries
from powersig.util.series import torch_compute_derivative_batch


class FrontierParameters:
    def __init__(self, left_bc_ps: Optional[SimplePowerSeries], bottom_bc_ps: Optional[SimplePowerSeries], ic,
                 lb_samples=None, bb_samples=None):
        self.left_bc_ps = left_bc_ps
        self.bottom_bc_ps = bottom_bc_ps
        self.ic = ic
        self.lb_samples = lb_samples
        self.bb_samples = bb_samples

    def is_ready(self) -> bool:
        return self.left_bc_ps is not None and self.bottom_bc_ps is not None and self.ic is not None


class SimpleSig:
    def __init__(self, X, Y):
        self.dX = torch_compute_derivative_batch(X)
        self.dY = torch_compute_derivative_batch(Y)

    def compute_gram_matrix(self) -> torch.Tensor:
        gram_matrix = torch.zeros((self.dX.shape[0], self.dY.shape[0]), dtype=torch.float64).cuda()
        for i in range(self.dX.shape[0]):
            for j in range(self.dY.shape[0]):
                print(f"Computing entry for i = {i}, j = {j}")
                gram_matrix[i,j] = self.compute_gram_entry(i, j)
        return gram_matrix

    def compute_gram_entry(self, i: int, j: int) -> float:
        dX_i = self.dX[i]
        dY_j = self.dY[j]
        rho = self.compute_rho(i, j)
        ds = 1 / dX_i.shape[0]
        dt = 1 / dY_j.shape[0]

        # Initial boundary conditions
        one = SimplePowerSeries(torch.tensor([1], dtype=torch.float64).cuda(),
                                torch.zeros([1, 2], dtype=torch.int32).cuda())

        initial_left_bc_ps = one.deep_clone()
        initial_bottom_bc_ps = one.deep_clone()

        frontiers = {}
        frontiers.setdefault((0, 0), FrontierParameters(initial_left_bc_ps, initial_bottom_bc_ps, None,
                                                        lb_samples=(1 / (2 * dX_i.shape[0]), 1),
                                                        bb_samples=(1 / (2 * dY_j.shape[0]), 1)))
        next_frontiers = {}
        diagonal_count = dX_i.shape[0] + dY_j.shape[0] - 1
        kg = torch.zeros([dX_i.shape[0], dY_j.shape[0]], dtype=torch.float64).cuda()
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
                ps = self.build_tile_power_series(frontier.left_bc_ps, frontier.bottom_bc_ps, rho[s_i, t_i].item(),
                                                  s_base, t_base, kg[s_i, t_i].item())
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
                lb_sample_point = t_base + dt * random.uniform(0, 1)
                bb_sample_point = s_base + ds * random.uniform(0, 1)

                if next_s_i < dX_i.shape[0]:
                    # kg[next_s_i, t_i] = ps(next_s_base, t_base)
                    k_lb = ps(next_s_base, lb_sample_point)
                    nf = next_frontiers.setdefault((next_s_i, t_i), FrontierParameters(None, None, None))
                    if t_i == 0:
                        nf.bottom_bc_ps = one.deep_clone()
                    nf.left_bc_ps = ps.deep_clone().bind_s(next_s_base)
                    nf.lb_samples = (lb_sample_point, k_lb)

                if next_t_i < dY_j.shape[0]:
                    # kg[s_i, next_t_i] = ps(s_base, next_t_base)
                    k_bb = ps(bb_sample_point, next_t_base)
                    nf = next_frontiers.setdefault((s_i, next_t_i), FrontierParameters(None, None, None))
                    if s_i == 0:
                        nf.left_bc_ps = one.deep_clone()
                    nf.bottom_bc_ps = ps.bind_t(next_t_base)
                    nf.bb_samples = (bb_sample_point, k_bb)

                # Use most accurate value for ic
                if next_s_i < dX_i.shape[0] and next_t_i < dY_j.shape[0]:
                    # nf = next_frontiers.setdefault((next_s_i, next_t_i), FrontierParameters(None, None, None))
                    kg[next_s_i, next_t_i] = ps(next_s_base, next_t_base)

            frontiers = next_frontiers
            next_frontiers = {}

        raise RuntimeError("Unable to compute gram matrix")

    def compute_rho(self, i, j) -> torch.Tensor:
        return torch.matmul(self.dX[i], torch.t(self.dY[j]))


    def build_tile_power_series(self, left_bc_ps: SimplePowerSeries, bottom_bc_ps: SimplePowerSeries, rho: float,
                                s_min: float, t_min: float, ic: float) -> SimplePowerSeries:
        # Gather the constants into a new power series
        u = left_bc_ps + bottom_bc_ps - ic
        u_n = u.deep_clone()
        # print(f"u_0={u.human_readable()}")
        # Repeatedly integrate to generate the new power series
        # Truncate if necessary using tbd utility functions to eliminate terms with really small coefficients.
        for step in range(1,15):
            # u_n = u_n.integrate_in_place(s_base=s_min, t_base=t_min)
            u_n = u_n.integrate(s_base=s_min, t_base=t_min)
            u_n *= rho
            if u_n.coefficients.shape[0] == 0:
                return u
            # print(f"u_{step}={u_n.human_readable()}")
            u += u_n
            # print(f"u={u.human_readable()}")

        # Return the resulting power series
        return u
