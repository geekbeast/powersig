import math
from collections import defaultdict
from typing import Self, Tuple

import torch
from torch import Tensor

from powersig.util.series import resize


class TileSolutionPowerSeries:
    def __init__(self, rho: float, s_base: float, t_base: float, initial_condition: float,
                 left_boundary: torch.Tensor, bottom_boundary: torch.Tensor,
                 device: torch.device, truncation_order: int = 10):
        """

        :param initial_condition:
        :param device:
        :param truncation_order:
        """
        # Polynomial powers range: [0, truncation_order]
        # First row is central coefficients
        # Second row is s coefficients
        # Third row is t coefficients.
        self.rho = rho
        self.truncation_order = truncation_order
        self.left_boundary_series = left_boundary
        self.bottom_boundary_series = bottom_boundary
        # Polynomial order can be increase by truncation order each step.
        order = left_boundary.shape[1] + truncation_order
        self.initial_condition = torch.zeros([1, order], dtype=torch.float64, device=device)
        self.initial_condition[0, 0] = initial_condition
        bottom_integration_matrix = build_asymmetric_integration_matrix(rho, s_base, t_base, order, device)
        left_integration_matrix = build_asymmetric_integration_matrix(rho, t_base, s_base, order, device)
        symmetric_integration_matrix = build_symmetric_integration_matrix(rho, s_base, t_base, order, device)
        self.left_boundary_series = torch.matmul(left_boundary, left_integration_matrix)
        self.right_boundary_series = torch.matmul(bottom_boundary, bottom_integration_matrix)
        self.central_series = torch.matmul(self.initial_condition, symmetric_integration_matrix)

    def __call__(self, s: float, t: float):
        st = torch.pow(s, torch.arange(start=0, end=self.central_series.shape[1], dtype=torch.float64,
                                       device=self.central_series.device)) * torch.pow(t, torch.arange(start=0, end=
        self.central_series.shape[1], dtype=torch.float64, device=self.central_series.device))
        return torch.dot(self.central_series, st)

    def bind_s(self, s: float) -> torch.Tensor:
        s_from_left = self.left_boundary_series * torch.pow(s, torch.arange(start=0, end=self.central_series.shape[1],
                                                                            dtype=torch.float64,
                                                                            device=self.central_series.device))
        s_from_bottom = self.bottom_boundary_series * torch.pow(s,
                                                                torch.arange(start=0, end=self.central_series.shape[1],
                                                                             dtype=torch.float64,
                                                                             device=self.central_series.device))

    def bind_t(self, t: float) -> torch.Tensor:
        self.left_boundary_series * torch.pow(t, torch.arange(start=0, end=self.central_series.shape[1],
                                                              dtype=torch.float64, device=self.central_series.device))


def build_gather_matrices(g_1: float, g_2: float, order: int, device: torch.device) -> tuple[Tensor, Tensor]:
    G1 = torch.eye(order, dtype=torch.float64, device=device).to_sparse()
    G2 = torch.eye(order, dtype=torch.float64, device=device).to_sparse()
    G1[:, 0] -= torch.pow(g_1, torch.arange(start=0, end=order, dtype=torch.float64, device=device))
    G2[:, 0] -= torch.pow(g_2, torch.arange(start=0, end=order, dtype=torch.float64, device=device))
    return G1, G2


def build_asymmetric_integration_matrix(rho: float, g_1: float, g_2: float, order: int, device: torch.device):
    A_b = torch.diag_embed(rho / torch.arange(start=1, end=order, dtype=torch.float64, device=device),
                           offset=1).to_sparse()
    G1, G2 = build_gather_matrices(g_1, g_2, order, device)
    exp_arg = torch.matmul(torch.matmul(A_b, G1), G2)
    return torch.matrix_exp(exp_arg)


def build_symmetric_integration_matrix(rho: float, g_1: float, g_2: float, order: int, device: torch.device):
    A_c = torch.diag_embed(rho / (torch.arange(start=1, end=order, dtype=torch.float64, device=device) ** 2),
                           offset=1).to_sparse()
    D1 = torch.diag_embed(1 / (torch.arange(start=0, end=order, dtype=torch.float64, device=device) ** 2),
                          offset=0).to_sparse()
    G1, G2 = build_gather_matrices(g_1, g_2, order, device)
    series_arg = torch.matmul(A_c, torch.matmul(torch.matmul(G1, D1), G2))
    return matrix_geometric_series(series_arg, order)


def matrix_geometric_series(A: torch.Tensor, truncation_order: int = 10):
    """
    Computes the geometric sum of the matrix power series for the given truncation order
    :param A: The matrix for which to compute truncated geometric sum.
    :param truncation_order: The truncation order for the geometric sum.
    :return: The geometric sum of the matrix power series.
    """
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.matmul(torch.inverse(I - A), I - torch.matrix_power(A, truncation_order + 1))


class MatrixPowerSeries:
    """
    This class represents a general power series in two integer variables by storing the coefficients of the
    truncated power series for terms {s^i}{t^j} in a coefficient matrix $c_ij$
    """

    def __init__(self, coefficients: torch.Tensor):
        assert len(coefficients.shape) == 2, "Coefficients must be a matrix."
        self.coefficients = coefficients

    def __call__(self, s: float, t: float) -> torch.Tensor:
        return self.evaluate(self.build_gather_s(s), self.build_gather_t(t))

    def evaluate(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(g2, self.coefficients), g1)

    def build_gather_s(self, s: float) -> torch.Tensor:
        return build_G1_s(s, self.coefficients.shape[1], self.coefficients.device)

    def build_gather_t(self, t: float):
        return build_G2_t(t, self.coefficients.shape[0], self.coefficients.device)

    def inplace_matrix_integrate(self, IminusG1: torch.Tensor, IminusG2: torch.Tensor, A1, A2) -> Self:
        indefinite_integral = torch.matmul(A2, torch.matmul(self.coefficients, A1))
        self.coefficients = torch.matmul(torch.matmul(IminusG2, indefinite_integral), IminusG1)
        return self

    def inplace_integrate(self, s_base: float, t_base: float) -> Self:
        """
        This function is mainly used for testing integrate. When repeatedly integrating for the same domain, it is
        worth using matrix_integrate directly.
        :param s_base: The lower limit of integration for s
        :param t_base: The lower limit of integration for t
        :return: An integrated power series.
        """
        IminusG1 = build_integration_gather_matrix_s(s_base, self.coefficients.shape[1], self.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t_base, self.coefficients.shape[0], self.coefficients.device)
        return self.inplace_matrix_integrate(IminusG1, IminusG2)

    def bind_s(self, s: float) -> Self:
        return self.bind_s_with_matrix(self.build_gather_s(s))

    def bind_t(self, t: float) -> Self:
        return self.bind_t_with_matrix(self.build_gather_t(t))

    def bind_s_with_matrix(self, G1: torch.Tensor) -> Self:
        self.coefficients[:, :1] = torch.matmul(self.coefficients, G1)
        self.coefficients[:, 1:] = 0
        return self

    def bind_t_with_matrix(self, G2: torch.Tensor) -> Self:
        self.coefficients[:1, :] = torch.matmul(G2, self.coefficients)
        self.coefficients[1:, :] = 0
        return self

    def get_device(self):
        return self.coefficients.device

    def __str__(self):
        ps = ""
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                c = self.coefficients[i, j].item()
                if c > 0:
                    if ps == "":
                        ps += f"{c}*s^{i}*t^{j} "
                    else:
                        ps += f"+ {c}*s^{i}*t^{j} "

        return ps

    def __add__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            result = MatrixPowerSeries(torch.clone(self.coefficients))
            result.coefficients[0, 0] += other
            return result

        # Deliberately omitting check to make sure they are on the same device.
        if self.coefficients.shape == other.coefficients.shape:
            return MatrixPowerSeries(self.coefficients + other.coefficients)
        elif self.coefficients.shape > other.coefficients.shape:
            result = MatrixPowerSeries(resize(other.coefficients, self.coefficients.shape))
            return result.__iadd__(self)
        else:
            result = MatrixPowerSeries(resize(self.coefficients, other.coefficients.shape))
            return result.__iadd__(other)

    def __iadd__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            self.coefficients += other
        else:
            self.coefficients += other.coefficients

        return self

    def __sub__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            result = MatrixPowerSeries(torch.clone(self.coefficients))
            result.coefficients[0, 0] -= other
            return result

        return MatrixPowerSeries(self.coefficients - other.coefficients)

    def __isub__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            self.coefficients[0, 0] -= other
        else:
            self.coefficients -= other.coefficients

        return self

    def __mul__(self, other: int | float) -> Self:
        result = MatrixPowerSeries(torch.clone(self.coefficients))
        result.coefficients *= other

        return result

    def __imul__(self, other: int | float) -> Self:
        self.coefficients *= other

        return self

    def deep_clone(self) -> Self:
        return MatrixPowerSeries(torch.clone(self.coefficients))

    def is_converged(self, prev: float, g1: torch.Tensor, g2: torch.Tensor, tol: float = 1e-10) -> tuple[bool, float]:
        """
        This function checks to see if coefficients have dropped off far enough for it to be safe to truncate.
        """

        tile_sol = self.evaluate(g1, g2).item()
        l2d = (tile_sol - prev) ** 2

        if not math.isclose(0, l2d, rel_tol=0, abs_tol=tol):
            print(f"Sequence diff: {l2d}")
            return False, l2d
        else:
            return True, tile_sol


class SparseMatrixPowerSeries:
    """
    This class represents a general power series in two integer variables by storing the coefficients of the
    truncated power series for terms {s^i}{t^j} in a coefficient matrix $c_ij$
    """

    def __init__(self, coefficients: torch.Tensor):
        assert len(coefficients.shape) == 2, "Coefficients must be a matrix."
        self.coefficients = coefficients
        self.A1 = self.build_A1()
        self.A2 = self.build_A2()

    def __call__(self, s: float, t: float) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.build_gather_t(t), self.coefficients), self.build_gather_s(s))

    def build_gather_s(self, s: float) -> torch.Tensor:
        return build_G1_s(s, self.coefficients.shape[1], self.coefficients.device)

    def build_gather_t(self, t: float):
        return build_G2_t(t, self.coefficients.shape[0], self.coefficients.device)

    def inplace_matrix_integrate(self, IminusG1: torch.Tensor, IminusG2: torch.Tensor) -> Self:
        indefinite_integral = torch.matmul(self.A2, torch.matmul(self.coefficients, self.A1))
        self.coefficients = torch.matmul(torch.matmul(IminusG2, indefinite_integral), IminusG1)
        return self

    def inplace_integrate(self, s_base: float, t_base: float) -> Self:
        """
        This function is mainly used for testing integrate. When repeatedly integrating for the same domain, it is
        worth using matrix_integrate directly.
        :param s_base: The lower limit of integration for s
        :param t_base: The lower limit of integration for t
        :return: An integrated power series.
        """
        IminusG1 = build_integration_gather_matrix_s(s_base, self.coefficients.shape[1], self.coefficients.device)
        IminusG2 = build_integration_gather_matrix_t(t_base, self.coefficients.shape[0], self.coefficients.device)
        return self.inplace_matrix_integrate(IminusG1, IminusG2)

    def bind_s(self, s: float) -> Self:
        return self.bind_s_with_matrix(self.build_gather_s(s))

    def bind_t(self, t: float) -> Self:
        return self.bind_t_with_matrix(self.build_gather_t(t))

    def bind_s_with_matrix(self, G1: torch.Tensor) -> Self:

        # self.coefficients[:, :1] = torch.matmul(self.coefficients, G1)
        self.coefficients = torch.matmul(self.coefficients, G1).to_sparse().resize_as_sparse_(self.coefficients)
        # self.coefficients[:, 1:] = 0
        return self

    def bind_t_with_matrix(self, G2: torch.Tensor) -> Self:
        # self.coefficients[:1, :] = torch.matmul(G2, self.coefficients)
        self.coefficients = torch.matmul(G2, self.coefficients).to_sparse().resize_as_sparse_(
            self.coefficients).to_sparse()
        # self.coefficients[1:, :] = 0
        return self

    def build_A1(self):
        return torch.diag_embed(
            1 / torch.arange(start=1, end=self.coefficients.shape[1], dtype=torch.float64, device=self.get_device()),
            offset=1).to_sparse()

    def build_A2(self):
        return torch.diag_embed(
            1 / torch.arange(start=1, end=self.coefficients.shape[1], dtype=torch.float64, device=self.get_device()),
            offset=-1).to_sparse()

    def get_device(self):
        return self.coefficients.device

    def __str__(self):
        ps = ""
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                c = self.coefficients[i, j].item()
                if c > 0:
                    if ps == "":
                        ps += f"{c}*s^{i}*t^{j} "
                    else:
                        ps += f"+ {c}*s^{i}*t^{j} "

        return ps

    def __add__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            result = MatrixPowerSeries(torch.clone(self.coefficients))
            add_indices = torch.tensor([[0], [0]])  # Indices of values to subtract
            add_values = torch.tensor([other], dtype=torch.float)  # Values to subtract
            add_tensor = torch.sparse_coo_tensor(add_indices, add_values, size=self.coefficients.shape,
                                                 device=self.coefficients.device)
            result.coefficients += add_tensor
            return result
        # Deliberately omitting check to make sure they are on the same device.
        return MatrixPowerSeries(self.coefficients + other.coefficients)

    def __iadd__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            add_indices = torch.tensor([[0], [0]])  # Indices of values to subtract
            add_values = torch.tensor([other], dtype=torch.float)  # Values to subtract
            add_tensor = torch.sparse_coo_tensor(add_indices, add_values, size=self.coefficients.shape,
                                                 device=self.coefficients.device)
            self.coefficients += add_tensor
        else:
            self.coefficients += other.coefficients

        return self

    def __sub__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            result = MatrixPowerSeries(torch.clone(self.coefficients))
            # indices = torch.tensor([[0], [0]])  # Indices of non-zero elements
            # values = torch.tensor([other])  # Values of the non-zero elements
            # size = self.coefficients.shape
            # sparse_tensor = torch.sparse_coo_tensor(indices, values, size).cuda()
            # result.coefficients -= sparse_tensor

            # For example, subtract 3 from the entry at index [1, 2]
            subtract_indices = torch.tensor([[0], [0]])  # Indices of values to subtract
            subtract_values = torch.tensor([other], dtype=torch.float)  # Values to subtract
            subtract_tensor = torch.sparse_coo_tensor(subtract_indices, subtract_values, size=self.coefficients.shape,
                                                      device=self.coefficients.device)
            result.coefficients -= subtract_tensor
            return result

        return MatrixPowerSeries(self.coefficients - other.coefficients)

    def __isub__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            subtract_indices = torch.tensor([[0], [0]])  # Indices of values to subtract
            subtract_values = torch.tensor([other], dtype=torch.float)  # Values to subtract
            subtract_tensor = torch.sparse_coo_tensor(subtract_indices, subtract_values, size=self.coefficients.shape,
                                                      device=self.coefficients.device)
            self.coefficients -= subtract_tensor
        else:
            self.coefficients -= other.coefficients

        return self

    def __mul__(self, other: int | float) -> Self:
        result = MatrixPowerSeries(torch.clone(self.coefficients))
        result.coefficients *= other

        return result

    def __imul__(self, other: int | float) -> Self:
        self.coefficients *= other

        return self

    def deep_clone(self) -> Self:
        return MatrixPowerSeries(torch.clone(self.coefficients))


def build_G1_s(s: float, columns: int, device: torch.device):
    # if math.isclose(s, 0.0, rel_tol=0.0, abs_tol=1e-15):
    #     return torch.zeros((columns, 1), dtype=torch.float64, device=device).to_sparse()

    gs = torch.arange(columns, dtype=torch.float64, device=device).reshape(columns, 1)

    return torch.pow(s, gs)


def build_G2_t(t: float, rows: int, device: torch.device):
    # if math.isclose(t, 0.0, rel_tol=0.0, abs_tol=1e-15):
    #     return torch.zeros((1, rows), dtype=torch.float64, device=device).to_sparse()

    gt = torch.arange(rows, dtype=torch.float64, device=device).reshape(1, rows)

    return torch.pow(t, gt)


def build_integration_gather_matrix_s(s: float, columns: int, device: torch.device) -> torch.Tensor:
    IminusG1 = torch.eye(columns, dtype=torch.float64, device=device)
    IminusG1[:, :1] -= build_G1_s(s, columns, device)  # Subtract out the first column
    return IminusG1.to_sparse()


def build_integration_gather_matrix_t(t: float, rows: int, device: torch.device) -> torch.Tensor:
    IminusG2 = torch.eye(rows, device=device, dtype=torch.float64)
    IminusG2[:1, :] -= build_G2_t(t, rows, device=device)  # Subtract out the first row
    return IminusG2.to_sparse()


def build_integration_limit_matrix_s(s_min: float, s_max: float, columns: int, device: torch.device) -> torch.Tensor:
    IminusG1 = torch.eye(columns, dtype=torch.float64, device=device)
    IminusG1[:, :1] += (build_G1_s(s_max, columns, device) - build_G1_s(s_min, columns,
                                                                        device))  # Subtract out the first column
    return IminusG1.to_sparse()


def build_integration_limit_matrix_t(t_min: float, t_max: float, rows: int, device: torch.device) -> torch.Tensor:
    IminusG2 = torch.eye(rows, device=device, dtype=torch.float64)
    IminusG2[:1, :] += (build_G2_t(t_max, rows, device=device) - build_G2_t(t_min, rows,
                                                                            device=device))  # Subtract out the first row
    return IminusG2.to_sparse()


def build_A1(cols, device: torch.device):
    return torch.diag_embed(
        1 / torch.arange(start=1, end=cols, dtype=torch.float64, device=device),
        offset=1).to_sparse()


def build_A2(rows, device: torch.device):
    return torch.diag_embed(
        1 / torch.arange(start=1, end=rows, dtype=torch.float64, device=device),
        offset=-1).to_sparse()


class SimplePowerSeries:
    """
    This class represents a simple power series of the form $\\sum_{n=0}^{c}{s^(n+o_s)*t^(n+o_t)$

    """

    def __init__(self, coefficients: torch.Tensor, exponents: torch.Tensor):
        """
        :param coefficients: An array of coefficients with the index representing base power for that term in the power series
        :param s_exp: The offset for the s exponent
        :param t_exp: The offset for the t exponent
        """
        assert coefficients.shape[0] == exponents.shape[0], "Must have the same number of coefficients and exponents"
        assert len(coefficients.shape) == 1, "Coefficients must have 1 dimension"
        assert exponents.dtype in {torch.int16, torch.int32, torch.int64}, "Exponents must be integer data types."
        self.coefficients = coefficients
        self.exponents = exponents

    def __call__(self, s: float, t: float) -> float:
        """
        This function evaluates the power series at a specific point.
        """
        values = torch.pow(torch.tensor([t, s]).cuda(), self.exponents)
        return (self.coefficients * values[:, 0] * values[:, 1]).sum().item()

    def integrate_t(self, t_base) -> Self:
        # Integrate over t
        self.exponents[:, 0] += 1
        self.coefficients /= self.exponents[:, 0]
        base = self.deep_clone()
        base.coefficients *= -1
        base.bind_t(t_base)
        return self.__iadd__(base)

    def integrate_s(self, s_base) -> Self:
        # Integrate over t
        self.exponents[:, 1] += 1
        self.coefficients /= self.exponents[:, 1]
        base = self.deep_clone()
        base.coefficients *= -1
        base.bind_s(s_base)
        return self.__iadd__(base)

    def integrate_in_place_s(self, s_base) -> Self:
        self.exponents[:, 1] += 1
        self.coefficients /= self.exponents[:, 1]
        self.coefficients -= self.coefficients * torch.pow(s_base, self.exponents[:, 1])
        self.exponents[:, 1] = 0
        return self

    def integrate_in_place_t(self, t_base) -> Self:
        self.exponents[:, 0] += 1
        self.coefficients /= self.exponents[:, 0]
        self.coefficients -= self.coefficients * torch.pow(t_base, self.exponents[:, 0])
        self.exponents[:, 0] = 0
        return self

    def integrate_in_place(self, s_base: float, t_base: float) -> Self:
        return self.integrate_in_place_s(s_base).integrate_in_place_t(t_base)

    def indefinite_integrate_s(self) -> Self:
        indefinite_int_s = self.deep_clone()
        indefinite_int_s.exponents[:, 1] += 1
        indefinite_int_s.coefficients /= indefinite_int_s.exponents[:, 1]
        return indefinite_int_s

    def indefinite_integrate_t(self) -> Self:
        indefinite_int_t = self.deep_clone()
        indefinite_int_t.exponents[:, 0] += 1
        indefinite_int_t.coefficients /= indefinite_int_t.exponents[:, 0]
        return indefinite_int_t

    def integrate(self, s_base, t_base) -> Self:
        return self.integrate_s(s_base).integrate_t(t_base)

    def integrate_grid(self, s_points, t_points, dp):
        assert (len(s_points.shape) == 1)

        result = None
        # Need to perform piece wise integration one variable at a time.
        for s_i in range(s_points.shape[0] - 1):
            iis = self.indefinite_integrate_s()
            lbs = iis.deep_clone().bind_s(s_points[s_i])
            ubs = iis.bind_s(s_points[s_i + 1])
            eis = ubs - lbs
            for t_i in range(t_points.shape[0] - 1):
                iit = eis.indefinite_integrate_t()
                lbt = iit.deep_clone().bind_t(t_points[t_i])
                ubt = iit.bind_t(t_points[t_i + 1])
                eit = (ubt - lbt) * dp.get_c(s_i, t_i)
                if result is None:
                    result = eit
                else:
                    result += eit

        t_i = t_points.shape[0] - 1
        for s_i in range(s_points.shape[0] - 1):
            iis = self.indefinite_integrate_s()
            lbs = iis.deep_clone().bind_s(s_points[s_i])
            eis = iis - lbs
            eit = eis.integrate_t(t_points[t_i]) * dp.get_c(s_i, t_i)
            result += eit

        s_i = s_points.shape[0] - 1
        for t_i in range(t_points.shape[0] - 1):
            eis = self.integrate_s(s_points[s_i])
            iit = eis.indefinite_integrate_t()
            lbt = iit.deep_clone().bind_t(t_points[t_i])
            eit = (iit - lbt) * dp.get_c(s_i, t_i)
            result += eit

        if result is None:
            result = self.integrate(s_points[-1], t_points[-1]) * dp.get_c(s_points.shape[0] - 1, t_points.shape[0] - 1)
        else:
            result += self.integrate(s_points[-1], t_points[-1]) * dp.get_c(s_points.shape[0] - 1,
                                                                            t_points.shape[0] - 1)

        return result

    def bind_s(self, val) -> Self:
        self.coefficients *= torch.pow(val, self.exponents[:, 1])
        self.exponents[:, 1] = 0
        return self

    def bind_t(self, val) -> Self:
        self.coefficients *= torch.pow(val, self.exponents[:, 0])
        self.exponents[:, 0] = 0
        return self

    def deep_clone(self) -> Self:
        return SimplePowerSeries(torch.clone(self.coefficients), torch.clone(self.exponents))

    def derivative_s(self):
        self.coefficients *= self.exponents[:, 1]
        self.exponents[:, 1] -= 1

    def derivative_t(self):
        self.coefficients *= self.exponents[:, 0]
        self.exponents[:, 0] -= 1

    def extract_constants(self) -> float:
        """
        This function sums all coefficients where both exponents are zeros sum and returns the result.
        :return: The sum of constant terms in this power series.
        """
        constant = 0.0
        for i in range(self.exponents.shape[0]):
            if self.exponents[i][0] == 0 and self.exponents[i][1] == 0:
                constant += self.coefficients[i]
                self.coefficients[i] = 0
        return constant

    def __str__(self):
        return f"PowerSeries(coefficients = {self.coefficients}, exponents = {self.exponents})"

    def human_readable(self):
        ps = ""
        for i in range(self.coefficients.shape[0]):
            ps += f"{self.coefficients[i]}*s^{self.exponents[i, 1]}*t^{self.exponents[i, 0]}"
            if i < self.coefficients.shape[0] - 1:
                ps += " + "
        return ps

    def __mul__(self, other) -> Self:
        result = self.deep_clone()
        result *= other
        return result

    def __imul__(self, other) -> Self:
        self.coefficients *= other
        return self

    def __sub__(self, other):
        result = self.deep_clone()
        result -= other
        return result

    def __isub__(self, other) -> Self:
        return self.__iadd__(other * -1)

    def __add__(self, other: int | float | Self) -> Self:
        result = self.deep_clone()
        result += other
        return result

    def __iadd__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            for i in range(self.exponents.shape[0]):
                if self.exponents[i][0] == 0 and self.exponents[i][1] == 0:
                    self.coefficients[i] += other
                    return self
            torch.cat((torch.tensor([other], dtype=torch.float64).cuda(), self.coefficients), 0)
            torch.cat((torch.tensor([0, 0], dtype=torch.int32).cuda(), self.exponents), 0)
            return self

        c_map = {}

        # Sum up all the coefficients across the two series for unique exponent combinations
        for index in range(self.exponents.shape[0]):
            coeffs = c_map.setdefault(self.exponents[index][0].item(), defaultdict(float))
            coeffs[self.exponents[index][1].item()] += self.coefficients[index].item()

        for index in range(other.exponents.shape[0]):
            coeffs = c_map.setdefault(other.exponents[index][0].item(), defaultdict(float))
            coeffs[other.exponents[index][1].item()] += other.coefficients[index].item()

        # Now count the unique exponents
        unique_exponent_count = 0
        for t_exp in c_map:
            s_map = c_map[t_exp]
            # unique_exponent_count += len(c_map[t_exp])
            to_be_deleted = []
            for s_exp in s_map:
                c = s_map[s_exp]
                if not math.isclose(c, 0, rel_tol=0.0, abs_tol=1e-15):
                    unique_exponent_count += 1
                else:
                    to_be_deleted.append(s_exp)
            for tbd in to_be_deleted:
                del s_map[tbd]

        # Now allocate the new coefficient and exponent arrays
        new_coefficients = torch.zeros([unique_exponent_count], dtype=torch.float64).cuda()
        new_exponents = torch.zeros([unique_exponent_count, 2], dtype=torch.int32).cuda()

        # Probably not necessary can remove for performance improvement
        sorted_t_exponents = sorted(c_map.keys())

        coefficient_index = 0
        for t_exp in sorted_t_exponents:
            s_exponents = c_map[t_exp]
            sorted_s_exponents = sorted(s_exponents.keys())
            for s_exp in sorted_s_exponents:
                new_exponents[coefficient_index][0] = t_exp
                new_exponents[coefficient_index][1] = s_exp
                new_coefficients[coefficient_index] = s_exponents[s_exp]
                coefficient_index += 1

        self.coefficients = new_coefficients
        self.exponents = new_exponents
        return self


class MultivariablePowerSeries:
    """
    This class represents a power series of the form $\\sum_{n=0}^{k}{\\prod_{i,j=0}^{|X|-1,|Y|-1}{\\kappa_{ij}^{n+o_{k_ij}}*s^(n+o_s)*t^(n+o_t)$.
    The main difference between this and the simple power series is that it is intended to represent a power series as
    a function of the tiled derivative values, since we always evaluate the power series at s = 1, t = 1

    """

    def __init__(self, coefficients: torch.Tensor, exponents: torch.Tensor):
        """
        :param coefficients: An array of coefficients with the index representing base power for that term in the power series
        :param exponents: An array of exponents representing the exponents of each input to the power series.
        Will be len(coefficients) x 2 + (|X| - 1)(|Y|-1) in size.
        """
        assert coefficients.shape[0] == exponents.shape[0], "Must have the same number of coefficients and exponents"
        assert len(coefficients.shape) == 1, "Coefficients must have 1 dimension"
        assert exponents.dtype in {torch.int16, torch.int32, torch.int64}, "Exponents must be integer data types."
        self.coefficients = coefficients
        self.exponents = exponents

    def __call__(self, Rho: torch.Tensor) -> float:
        """
        This function evaluates the power series at a specific point.
        """
        values = torch.pow(Rho.cuda(), self.exponents)
        return (self.coefficients * values[:, 0] * values[:, 1]).sum().item()

    def integrate_t(self, t_base) -> Self:
        # Integrate over t
        self.exponents[:, 0] += 1
        self.coefficients /= self.exponents[:, 0]
        base = self.deep_clone()
        base.coefficients *= -1
        base.bind_t(t_base)
        return self.__add__(base)

    def integrate_s(self, s_base) -> Self:
        # Integrate over t
        self.exponents[:, 1] += 1
        self.coefficients /= self.exponents[:, 1]
        base = self.deep_clone()
        base.coefficients *= -1
        base.bind_s(s_base)
        return self.__add__(base)

    def indefinite_integrate_s(self) -> Self:
        indefinite_int_s = self.deep_clone()
        indefinite_int_s.exponents[:, 1] += 1
        indefinite_int_s.coefficients /= indefinite_int_s.exponents[:, 1]
        return indefinite_int_s

    def indefinite_integrate_t(self) -> Self:
        indefinite_int_t = self.deep_clone()
        indefinite_int_t.exponents[:, 0] += 1
        indefinite_int_t.coefficients /= indefinite_int_t.exponents[:, 0]
        return indefinite_int_t

    def integrate(self, s_base, t_base) -> Self:
        return self.integrate_s(s_base).integrate_t(t_base)

    def integrate_grid(self, s_points, t_points, dp):
        assert (len(s_points.shape) == 1)

        result = None
        # Need to perform piece wise integration one variable at a time.
        for s_i in range(s_points.shape[0] - 1):
            iis = self.indefinite_integrate_s()
            lbs = iis.deep_clone().bind_s(s_points[s_i])
            ubs = iis.bind_s(s_points[s_i + 1])
            eis = ubs - lbs
            for t_i in range(t_points.shape[0] - 1):
                iit = eis.indefinite_integrate_t()
                lbt = iit.deep_clone().bind_t(t_points[t_i])
                ubt = iit.bind_t(t_points[t_i + 1])
                eit = (ubt - lbt) * dp.get_c(s_i, t_i)
                if result is None:
                    result = eit
                else:
                    result += eit

        t_i = t_points.shape[0] - 1
        for s_i in range(s_points.shape[0] - 1):
            iis = self.indefinite_integrate_s()
            lbs = iis.deep_clone().bind_s(s_points[s_i])
            eis = iis - lbs
            eit = eis.integrate_t(t_points[t_i]) * dp.get_c(s_i, t_i)
            result += eit

        s_i = s_points.shape[0] - 1
        for t_i in range(t_points.shape[0] - 1):
            eis = self.integrate_s(s_points[s_i])
            iit = eis.indefinite_integrate_t()
            lbt = iit.deep_clone().bind_t(t_points[t_i])
            eit = (iit - lbt) * dp.get_c(s_i, t_i)
            result += eit

        if result is None:
            result = self.integrate(s_points[-1], t_points[-1]) * dp.get_c(s_points.shape[0] - 1, t_points.shape[0] - 1)
        else:
            result += self.integrate(s_points[-1], t_points[-1]) * dp.get_c(s_points.shape[0] - 1,
                                                                            t_points.shape[0] - 1)

        return result

    def bind_s(self, val) -> Self:
        self.coefficients *= torch.pow(val, self.exponents[:, 1])
        self.exponents[:, 1] = 0
        return self

    def bind_t(self, val) -> Self:
        self.coefficients *= torch.pow(val, self.exponents[:, 0])
        self.exponents[:, 0] = 0
        return self

    def deep_clone(self) -> Self:
        return SimplePowerSeries(torch.clone(self.coefficients), torch.clone(self.exponents))

    def derivative_s(self):
        self.coefficients *= self.exponents[:, 1]
        self.exponents[:, 1] -= 1

    def derivative_t(self):
        self.coefficients *= self.exponents[:, 0]
        self.exponents[:, 0] -= 1

    def extract_constants(self) -> float:
        """
        This function sums all coefficients where both exponents are zeros sum and returns the result.
        :return: The sum of constant terms in this power series.
        """
        constant = 0.0
        for i in range(self.exponents.shape[0]):
            if self.exponents[i][0] == 0 and self.exponents[i][1] == 0:
                constant += self.coefficients[i]
                self.coefficients[i] = 0
        return constant

    def __str__(self):
        return f"PowerSeries(coefficients = {self.coefficients}, exponents = {self.exponents})"

    def human_readable(self):
        ps = ""
        for i in range(self.coefficients.shape[0]):
            ps += f"{self.coefficients[i]}*s^{self.exponents[i, 1]}*t^{self.exponents[i, 0]}"
            if i < self.coefficients.shape[0] - 1:
                ps += " + "
        return ps

    def __mul__(self, other) -> Self:
        result = self.deep_clone()
        result *= other
        return result

    def __imul__(self, other) -> Self:
        self.coefficients *= other
        return self

    def __sub__(self, other):
        result = self.deep_clone()
        result -= other
        return result

    def __isub__(self, other) -> Self:
        return self.__iadd__(other * -1)

    def __add__(self, other: int | float | Self) -> Self:
        result = self.deep_clone()
        result += other
        return result

    def __iadd__(self, other: int | float | Self) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            for i in range(self.exponents.shape[0]):
                if self.exponents[i][0] == 0 and self.exponents[i][1] == 0:
                    self.coefficients[i] += other
                    return self
            torch.cat((torch.tensor([other], dtype=torch.float64).cuda(), self.coefficients), 0)
            torch.cat((torch.tensor([0, 0], dtype=torch.int32).cuda(), self.exponents), 0)
            return self

        c_map = defaultdict(float)

        # Sum up all the coefficients across the two series for unique exponent combinations
        for index in range(self.exponents.shape[0]):
            key = (self.exponents[index][0].item(), self.exponents[index][1].item())
            c_map[key] += self.coefficients[index].item()

        for index in range(other.exponents.shape[0]):
            key = (other.exponents[index][0].item(), other.exponents[index][1].item())
            c_map[key] += other.coefficients[index].item()

        # Now count the unique exponents
        to_be_deleted = []
        for key in c_map:
            c = c_map[key]
            if math.isclose(c, 0, rel_tol=0.0, abs_tol=1e-15):
                to_be_deleted.append(key)
        for tbd in to_be_deleted:
            del c_map[tbd]

        unique_exponent_count = len(c_map)

        # Now allocate the new coefficient and exponent arrays
        new_coefficients = torch.zeros([unique_exponent_count], dtype=torch.float64).cuda()
        new_exponents = torch.zeros([unique_exponent_count, 2], dtype=torch.int32).cuda()

        coefficient_index = 0
        for key in sorted(c_map.keys()):
            c = c_map[key]
            new_exponents[coefficient_index][0] = key[0]
            new_exponents[coefficient_index][1] = key[1]
            new_coefficients[coefficient_index] = c
            coefficient_index += 1

        self.coefficients = new_coefficients
        self.exponents = new_exponents
        return self
