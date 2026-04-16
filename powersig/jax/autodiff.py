"""Custom reverse-mode autodiff for the PowerSig diagonal sweep.

Implements checkpointed replay so that the backward pass uses O((T/k + k) * D * P)
memory instead of the O(T * D * P) tape that naive autograd would require.

The public entry points are:
    - compute_sig_kernel_fast_diff: single-pair differentiable kernel
    - PowerSigDiff.compute_gram_fast_diff: batched Gram matrix with gradients
"""

from math import ceil, sqrt
from typing import Optional
from functools import partial

from powersig.jax.jax_config import configure_jax
configure_jax()

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import toeplitz
from tqdm.auto import tqdm

from powersig.jax import static_kernels
from powersig.jax.algorithm import (
    PowerSigJax,
    compute_vandermonde_vectors,
    build_psi_stencil,
    build_stencil,
    estimate_bytes_per_pair,
    get_available_gpu_memory,
    _round_to_power_of_2,
    MAX_BLOCK_SIZE,
    JIT_BOUNDARY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Local adjoint: the smallest, most testable unit
# ---------------------------------------------------------------------------

def _map_diagonal_entry_fwd(rho, s, t, psi_s, psi_t, exponents):
    """Forward tile computation — identical to map_diagonal_entry_v2.

    Returns (s_next, t_next) and intermediates needed by the local adjoint.
    """
    r = rho ** exponents
    T_mat = psi_t * toeplitz(t, s)
    U = jnp.triu(T_mat, 1)
    L = jnp.tril(T_mat, -1)

    psi_s_t = psi_s @ t  # (P,)
    psi_s_s = psi_s @ s  # (P,)

    s_next = (r @ U) + (psi_s_t * r)  # r @ triu = U^T r for row-vector @ matrix
    t_next = (psi_s_s * r) + (L @ r)

    return s_next, t_next, r, U, L, psi_s_t, psi_s_s


def _map_diagonal_entry_bwd(rho, s, t, psi_s, psi_t, exponents,
                             bar_s_next, bar_t_next):
    """Exact local adjoint for one tile.

    Given incoming cotangents (bar_s_next, bar_t_next) for the *outputs*
    of the forward tile, returns:
        bar_s, bar_t   — cotangents for the *input* boundary vectors
        bar_rho        — scalar cotangent for rho
    """
    P = exponents.shape[0]
    dtype = s.dtype

    # --- recompute forward intermediates ---
    r = rho ** exponents
    T_mat = psi_t * toeplitz(t, s)
    U = jnp.triu(T_mat, 1)
    L = jnp.tril(T_mat, -1)
    psi_s_t = psi_s @ t
    psi_s_s = psi_s @ s

    # --- adjoint w.r.t. boundary vectors (s, t) ---
    # Forward:
    #   s_next = U^T r + diag(r) psi_s t
    #   t_next = diag(r) psi_s s + L r
    #
    # The Jacobian in block form is J = [[A, B], [B, A]] where:
    #   A maps s -> contribution from triu(toeplitz) part
    #   B = diag(r) @ psi_s
    #
    # But we can derive the adjoint directly from the forward expressions
    # by transposing each operation.
    #
    # For s_next = r @ U + (psi_s @ t) * r:
    #   - U depends on s via triu(psi_t * toeplitz(t, s))
    #     The columns of toeplitz(t,s) that land in triu come from s.
    #     Specifically, triu(toeplitz(t,s), 1)[i,j] = s[j-i] for j > i.
    #     After elementwise multiply with psi_t: U[i,j] = psi_t[i,j] * s[j-i].
    #     Then s_next[j] += sum_i r[i] * U[i,j] for j > i.
    #   - The (psi_s @ t) * r term depends on t through psi_s @ t.
    #
    # For t_next = (psi_s @ s) * r + L @ r:
    #   - L depends on t via tril(psi_t * toeplitz(t, s))
    #     tril(toeplitz(t,s), -1)[i,j] = t[i-j] for i > j.
    #   - The (psi_s @ s) * r term depends on s through psi_s @ s.
    #
    # Rather than materializing the P×P Jacobian blocks, we compute the
    # adjoint by transposing through the forward expressions directly.

    # --- bar_rho: sensitivity to rho ---
    # dr/drho = [0, 1, 2*rho, ..., (P-1)*rho^(P-2)]
    dr = jnp.concatenate([
        jnp.zeros((1,), dtype=dtype),
        jnp.arange(1, P, dtype=dtype) * r[:-1],
    ])

    # bar_rho = dr^T [ U @ bar_s_next + L^T @ bar_t_next
    #                 + psi_s_t * bar_s_next + psi_s_s * bar_t_next ]
    bar_rho = jnp.dot(dr,
                       U @ bar_s_next
                       + L.T @ bar_t_next
                       + psi_s_t * bar_s_next
                       + psi_s_s * bar_t_next)

    # --- bar_s: adjoint w.r.t. input s ---
    # s contributes to s_next through U = triu(psi_t * toeplitz(t, s), 1)
    # and to t_next through psi_s @ s.
    #
    # From s_next = r @ U:
    #   d(s_next)/ds is the transpose of "r @ triu(psi_t * toeplitz(t, ·), 1)"
    #   We can compute this as: for the toeplitz(t,s), the s-dependent part in
    #   the strict upper triangle is: col j of toeplitz has s[j-i] at row i < j.
    #   So U[i,j] = psi_t[i,j] * s[j-i] for j > i.
    #   (r @ U)[j] = sum_{i<j} r[i] * psi_t[i,j] * s[j-i]
    #   d/ds[k] of (r @ U)[j] = r[j-k] * psi_t[j-k, j] when j > j-k >= 0, i.e., k >= 1 and j >= k+1... wait
    #   Actually: s_next[j] = sum_{i: i<j} r[i] * psi_t[i,j] * s[j-i]
    #   So ds_next[j]/ds[m] = sum_{i: i<j, j-i=m} r[i] * psi_t[i,j]
    #                        = r[j-m] * psi_t[j-m, j]  when j-m >= 0 and j-m < j, i.e., m >= 1
    #   bar_s[m] from s_next = sum_j bar_s_next[j] * r[j-m] * psi_t[j-m, j] for j >= m+1
    #
    # This is exactly: bar_s from s_next = U^T_bar where
    #   U^T_bar is computed by transposing the toeplitz contraction.
    #   Equivalently: construct M_T = psi_t * toeplitz(s, t) (swapped args!)
    #   then tril(M_T, -1) @ (r * bar_s_next) ... not quite.
    #
    # Let's use the direct matrix approach. The transpose of "r @ U" w.r.t. s
    # can be computed by noting that U = triu(psi_t * toeplitz(t, s), 1).
    # The toeplitz matrix T = toeplitz(t, s) has T[i,j] = s[j-i] for j >= i
    # (upper triangle including diagonal) and T[i,j] = t[i-j] for i > j.
    # The strict upper triangle U[i,j] = psi_t[i,j] * s[j-i] for j > i.
    #
    # We need: bar_s[m] = sum_{i,j: j>i, j-i=m} r[i] * psi_t[i,j] * bar_s_next[j]
    # This is a convolution-like sum along diagonal m of the matrix.
    # For each m >= 1: bar_s[m] = sum_{i=0}^{P-1-m} r[i] * psi_t[i, i+m] * bar_s_next[i+m]
    # And bar_s[0] = 0 (the diagonal is excluded from strict upper triangle).
    #
    # Similarly, s contributes to t_next via (psi_s @ s) * r:
    #   bar_s from t_next = psi_s^T @ (bar_t_next * r)
    #
    # And t contributes to s_next via (psi_s @ t) * r:
    #   bar_t from s_next = psi_s^T @ (bar_s_next * r)
    #
    # And t contributes to t_next via L = tril(psi_t * toeplitz(t, s), -1):
    #   L[i,j] = psi_t[i,j] * t[i-j] for i > j.
    #   t_next[i] = sum_{j<i} L[i,j] * r[j] = sum_{j<i} psi_t[i,j] * t[i-j] * r[j]
    #   dt_next[i]/dt[m] = sum_{j: j<i, i-j=m} psi_t[i,j] * r[j]
    #                     = psi_t[i, i-m] * r[i-m] when i-m >= 0 and i > i-m, i.e., m >= 1
    #   bar_t[m] from t_next = sum_i bar_t_next[i] * psi_t[i, i-m] * r[i-m] for i >= m+1

    # Compute bar_s via the explicit diagonal contraction + psi_s^T term
    # bar_s[m] = sum_{i=0}^{P-1-m} r[i] * psi_t[i, i+m] * bar_s_next[i+m]  (from s_next, m >= 1)
    #          + (psi_s^T @ (bar_t_next * r))[m]                               (from t_next)
    # bar_s[0] = (psi_s^T @ (bar_t_next * r))[0]

    # Use the transposed Toeplitz structure:
    # toeplitz(t,s)^T = toeplitz(s,t).
    # triu(psi_t * toeplitz(t,s), 1)^T = tril((psi_t * toeplitz(t,s))^T, -1)
    #                                   = tril(psi_t^T * toeplitz(s,t), -1)
    # So bar_s from s_next = tril(psi_t^T * toeplitz(s,t), -1) @ (r * bar_s_next)
    # Wait, let me re-derive. s_next = r @ U where U = triu(psi_t * toeplitz(t,s), 1).
    # Using r as a row vector: s_next = r^T @ U (matrix multiply).
    # The adjoint: bar_r_from_s_next is not what we want.
    # Actually s_next[j] = sum_i r[i] * U[i,j].
    # This is (U^T @ r)[j]. So s_next = U^T r.
    # bar_s from the U^T r term: we need d(U^T r)/ds.
    # U[i,j] = psi_t[i,j] * s[j-i] for j > i, else 0.
    # (U^T r)[j] = sum_i U[i,j] * r[i] = sum_{i<j} psi_t[i,j] * s[j-i] * r[i].
    #
    # Hmm, let me just use the M^T approach:
    # Construct M_bwd = psi_t.T * toeplitz(s, t) and take its tril(-1).
    # Then (tril(M_bwd, -1) @ (r * bar_s_next)) should give bar_s from the U^T r term... no.
    #
    # Actually the simplest approach: reconstruct bar_s and bar_t by directly
    # using the fact that U^T = tril(M^T, -1) where M^T[j,i] = psi_t[i,j] * toeplitz(t,s)[i,j].
    # toeplitz(t,s)^T = toeplitz(s,t), so M^T = psi_t^T * toeplitz(s,t)? No—
    # (psi_t * toeplitz(t,s))^T = psi_t^T ⊙ toeplitz(t,s)^T = psi_t^T ⊙ toeplitz(s,t).
    # So triu(M, 1)^T = tril(psi_t^T ⊙ toeplitz(s,t), -1).
    # And tril(M, -1)^T = triu(psi_t^T ⊙ toeplitz(s,t), 1).
    #
    # For the U^T r term in s_next, the adjoint w.r.t. the *elements inside U that
    # depend on s* is what we need. But U^T r treats r and U as separate.
    # The dependency is: s -> toeplitz(t,s) -> U -> s_next.
    #
    # Let me take a step back and use JAX-style thinking.
    # s_next_j = sum_i r_i * psi_t_{i,j} * toeplitz(t,s)_{i,j} * [j > i]
    #          + (psi_s @ t)_j * r_j
    # toeplitz(t,s)_{i,j} = s_{j-i} if j >= i, t_{i-j} if i > j.
    # For j > i: toeplitz(t,s)_{i,j} = s_{j-i}.
    # So: s_next_j = sum_{i=0}^{j-1} r_i * psi_t_{i,j} * s_{j-i} + (psi_s @ t)_j * r_j
    # ds_next_j / ds_m = r_{j-m} * psi_t_{j-m, j}  if 1 <= m <= j-1... wait, j-i = m means i = j-m.
    # Need i < j and i >= 0, so 0 <= j-m < j, which means m >= 1 and j >= m+1... actually m >= 1 and j-m >= 0 so j >= m.
    # And j > j-m is always true for m >= 1.
    # So ds_next_j / ds_m = r_{j-m} * psi_t_{j-m, j} for m >= 1 and j >= m.
    #
    # bar_s_m = sum_j bar_s_next_j * r_{j-m} * psi_t_{j-m, j}  for j = m..P-1, when m >= 1
    # bar_s_0 = 0 (from this term)
    #
    # Similarly for t_next through L:
    # t_next_i = (psi_s @ s)_i * r_i + sum_{j=0}^{i-1} psi_t_{i,j} * t_{i-j} * r_j
    # dt_next_i / dt_m = psi_t_{i, i-m} * r_{i-m}  for m >= 1, i >= m+1... same logic, i-j=m means j=i-m, need j<i and j>=0.
    # bar_t_m = sum_i bar_t_next_i * psi_t_{i, i-m} * r_{i-m}  for i = m+1..P-1, when m >= 1... wait i >= m since j=i-m >= 0 and j < i means m >= 1.
    # bar_t_m (from L term) = sum_{i=m}^{P-1} bar_t_next_i * psi_t_{i, i-m} * r_{i-m} for m >= 1
    # Hmm actually i > j means i > i-m means m > 0. And j >= 0 means i >= m.
    # So: for m >= 1, bar_t_m = sum_{i=m}^{P-1} bar_t_next_i * psi_t_{i, i-m} * r_{i-m}

    # Now let's implement this efficiently using the transposed Toeplitz.
    #
    # For bar_s from the U^T r contribution:
    # We can write this as: construct the matrix with entry [m, j] = r_{j-m} * psi_t_{j-m, j}
    # for j >= m >= 1, which is exactly tril(psi_t^T * toeplitz(s, t), -1) evaluated at [m, j]... no.
    # Let me think again.
    #
    # Actually: the contribution to bar_s from s_next is:
    # bar_s = tril(M_T, -1)^T @ bar_s_next  where bar_s_next is weighted by... no.
    #
    # OK, let me just compute it properly using the transposed matrices.
    # M = psi_t ⊙ toeplitz(t, s)
    # s_next = U^T r + diag(r) (psi_s t)    where U = triu(M, 1)
    # t_next = diag(r) (psi_s s) + L r       where L = tril(M, -1)
    #
    # The full (2P x 2P) Jacobian J maps (ds, dt) -> (ds_next, dt_next).
    # We need J^T (bar_s_next, bar_t_next).
    #
    # The s-dependent part of M is in triu(toeplitz(t,s), 0) (including diagonal),
    # but only triu(M, 1) is used in U. The diagonal of toeplitz(t,s) is s[0] = t[0],
    # but wait — toeplitz(t,s)[i,i] = s[0] always (first element of s = first element
    # of the first row). Actually toeplitz(c, r) has r as first row and c as first column.
    # So toeplitz(t, s)[i, j] = s[j-i] for j >= i and t[i-j] for i > j.
    # toeplitz(t, s)[i, i] = s[0].
    # But s[0] is part of the boundary state we're differentiating through!
    # The triu(M, 1) excludes the diagonal, so U doesn't depend on s[0].
    # And the diagonal of M is psi_t[i,i] * s[0], which is excluded from both U and L.
    # The term (psi_s @ t) * r doesn't depend on s.
    # The term (psi_s @ s) * r depends on all of s through psi_s @ s.
    #
    # For s, contributions:
    #   1. s -> U (via triu of toeplitz) -> s_next: only s[1], ..., s[P-1]
    #   2. s -> psi_s @ s -> t_next: all of s
    #
    # For t, contributions:
    #   1. t -> L (via tril of toeplitz) -> t_next: only t[1], ..., t[P-1]
    #   2. t -> psi_s @ t -> s_next: all of t

    # Implementation: compute the two toeplitz-based contributions using matrix ops.
    # For contribution 1 (s -> U -> s_next):
    #   Recall U[i,j] = psi_t[i,j] * s[j-i] for j > i.
    #   s_next = U^T r. So s_next[j] = sum_{i<j} r[i] * psi_t[i,j] * s[j-i].
    #   This is a linear function of s. The adjoint is:
    #   bar_s[m] += sum_{j: j-i=m, i<j, i,j in range} r[i] * psi_t[i,j] * bar_s_next[j]
    #            = sum_{j >= m, j >= 1} r[j-m] * psi_t[j-m, j] * bar_s_next[j]  for m >= 1
    #
    # For contribution 2 (t -> L -> t_next):
    #   L[i,j] = psi_t[i,j] * t[i-j] for i > j.
    #   t_next = L r. So t_next[i] = sum_{j<i} psi_t[i,j] * t[i-j] * r[j].
    #   bar_t[m] += sum_{i: i-j=m, j<i} psi_t[i,j] * r[j] * bar_t_next[i]  for m >= 1
    #            = sum_{i >= m} psi_t[i, i-m] * r[i-m] * bar_t_next[i]

    # These are structured sums along sub-diagonals. We can compute them
    # efficiently using the transpose of the Toeplitz matrix.
    #
    # Key insight: the transposed Toeplitz matrix toeplitz(s, t) swaps rows/cols.
    # M^T = psi_t^T ⊙ toeplitz(s, t).
    # triu(M, 1)^T = tril(M^T, -1) (transpose swaps triu<->tril and shifts k -> -k).
    # tril(M, -1)^T = triu(M^T, 1).
    #
    # Contribution 1: bar_s from s_next via U^T r.
    #   s_next = U^T r where U = triu(M, 1).
    #   The linear map s -> s_next (through U) has adjoint:
    #   Think of it as: f(s) = triu(psi_t ⊙ toeplitz(t, s), 1)^T @ r
    #   For the adjoint, we need to figure out how s enters.
    #   toeplitz(t, s) has s in its first row (and upper triangle).
    #   Changing s[m] changes column entries toeplitz(t,s)[i, i+m] = s[m] for all valid i.
    #   Wait no: toeplitz(t,s)[i, j] = s[j-i] for j >= i. So s[m] appears at all positions
    #   (i, i+m) for i = 0, ..., P-1-m.
    #   In triu(·, 1): we need j > i, i.e., i+m > i, i.e., m > 0. ✓ (m >= 1)
    #   After psi_t masking: U[i, i+m] = psi_t[i, i+m] * s[m].
    #   Then (U^T r)[i+m] += U[i, i+m] * r[i] = psi_t[i, i+m] * s[m] * r[i].
    #   So s_next[j] = sum_{m=1}^{j} psi_t[j-m, j] * s[m] * r[j-m]  ... no.
    #   s_next[j] = sum_{i=0}^{j-1} r[i] * psi_t[i,j] * s[j-i].
    #   df/ds[m] gives bar_s[m] = sum_j bar_s_next[j] * r[j-m] * psi_t[j-m, j] for j >= m+1 (j-m >= 1 so j > j-m)
    #   Wait: j-i = m means i = j-m. Need i < j (m >= 1) and i >= 0 (j >= m). So j ranges from m to P-1.
    #   But also need j > i = j-m which gives m >= 1. And for m=0 we'd have i=j which violates strict upper.
    #   So bar_s[m] = sum_{j=m}^{P-1} bar_s_next[j] * r[j-m] * psi_t[j-m, j]  for m >= 1.
    #   bar_s[0] = 0 from this contribution.

    # For contribution from t_next via psi_s:
    #   t_next includes diag(r) @ (psi_s @ s). Adjoint w.r.t. s:
    #   bar_s += psi_s^T @ (r * bar_t_next)

    # Similarly for bar_t:
    # From t_next via L:
    #   t_next = L @ r. L[i,j] = psi_t[i,j] * t[i-j] for i > j.
    #   t_next[i] = sum_{j<i} psi_t[i,j] * t[i-j] * r[j].
    #   bar_t[m] = sum_{i >= m+1} bar_t_next[i] * psi_t[i, i-m] * r[i-m]  for m >= 1
    #   Wait: i-j = m, j = i-m, need j < i (m >= 1) and j >= 0 (i >= m). So i from m to P-1.
    #   But need i > j = i-m which gives m >= 1.
    #   bar_t[m] = sum_{i=m}^{P-1} bar_t_next[i] * psi_t[i, i-m] * r[i-m]  for m >= 1.
    #   bar_t[0] = 0 from this contribution.
    #
    # From s_next via psi_s:
    #   s_next includes diag(r) @ (psi_s @ t). Adjoint w.r.t. t:
    #   bar_t += psi_s^T @ (r * bar_s_next)

    # Now implement using matrix operations.
    # We can compute the toeplitz-based bar_s and bar_t using:
    # M_T = psi_t.T * toeplitz(s, t)   (this is M^T)
    # L_T = tril(M_T, -1) = triu(M, 1)^T
    # U_T = triu(M_T, 1) = tril(M, -1)^T
    #
    # bar_s from U^T r contribution:
    #   We showed bar_s[m] = sum_j (r * bar_s_next)[j] * psi_t[j-m, j] for sub-diagonal m.
    #   This is the same as: (tril(M^T, -1)^T @ (r * bar_s_next))[m]? Let's check.
    #   tril(M^T, -1)[j, m] = M^T[j, m] for j > m = psi_t[m, j] * toeplitz(s,t)[j, m].
    #   Hmm, toeplitz(s,t)[j,m] = s[m-j] for m >= j and t[j-m] for j > m.
    #   For j > m: toeplitz(s,t)[j,m] = t[j-m]. So tril(M^T, -1)[j,m] = psi_t[m,j] * t[j-m].
    #   That's not what we want — we want psi_t[j-m, j] not psi_t[m, j].
    #
    # Hmm, psi_t is symmetric! Let me check... build_stencil creates a symmetric matrix
    # (the test_symmetry test confirms stencil^T = stencil). So psi_t[m,j] = psi_t[j,m].
    # But wait, that means psi_t[j-m, j] vs psi_t[m, j]... these are not the same indices.
    # psi_t IS symmetric so psi_t[a,b] = psi_t[b,a]. So psi_t[j-m, j] = psi_t[j, j-m].
    # And psi_t[m, j] = psi_t[j, m]. These are different unless j-m = m, i.e., j = 2m.
    #
    # So the "M^T = psi_t^T ⊙ toeplitz(s,t)" trick doesn't directly give us what we want
    # because the toeplitz part transposes but psi_t is applied element-wise at the original
    # indices, not at the transposed indices.
    #
    # Let me reconsider. We have:
    # M = psi_t ⊙ toeplitz(t, s)
    # M^T[j, i] = M[i, j] = psi_t[i, j] * toeplitz(t, s)[i, j]
    # This is NOT the same as psi_t[j, i] * toeplitz(s, t)[j, i] in general
    # (unless psi_t is symmetric, in which case psi_t[i,j] = psi_t[j,i]).
    #
    # Actually from the tests, psi_t IS symmetric. build_stencil produces a symmetric matrix.
    # So psi_t[i,j] = psi_t[j,i] and:
    # M^T[j, i] = psi_t[i,j] * toeplitz(t,s)[i,j] = psi_t[j,i] * toeplitz(s,t)[j,i]
    #            = (psi_t ⊙ toeplitz(s, t))[j, i]
    # So M^T = psi_t ⊙ toeplitz(s, t). Great!
    #
    # Now: triu(M, 1)^T = tril(M^T, -1) = tril(psi_t ⊙ toeplitz(s, t), -1).
    # And: tril(M, -1)^T = triu(M^T, 1) = triu(psi_t ⊙ toeplitz(s, t), 1).
    #
    # For the adjoint of "s_next = U^T r" w.r.t. s:
    # The key issue is that U depends on s. So this isn't a simple linear-algebra transpose.
    # We need to differentiate through the toeplitz construction.
    #
    # Let me just implement it directly using the sub-diagonal formula and vectorize.

    # Efficient implementation: use the observation that the bar_s and bar_t
    # sub-diagonal sums can be computed via a single matrix-vector product
    # with an appropriately constructed matrix.
    #
    # For bar_s[m] (m >= 1) from s_next:
    #   bar_s[m] = sum_{j=m}^{P-1} r[j-m] * psi_t[j-m, j] * bar_s_next[j]
    #   Let w_s[j] = r[j] * bar_s_next[j+m]... this varies with m.
    #
    # Alternative: construct the P x P matrix Q where Q[m, j] = r[j-m] * psi_t[j-m, j]
    # for j >= m, m >= 1 (and 0 otherwise). Then bar_s = Q @ bar_s_next.
    # Q[m, j] = r[j-m] * psi_t[j-m, j] for j >= m >= 1.
    # Note that Q[m, j] depends on the offset j-m, with the psi_t factor at [j-m, j].
    # This is tril(something, -1)^T... it's the transpose of the lower triangular part
    # of the matrix with entry [j, m] = r[j-m] * psi_t[j-m, j].
    #
    # Actually, let's just build Q directly. For the s contribution from s_next:
    # Q_s[m, j] = r[j-m] * psi_t[j-m, j]  for j >= m >= 1, else 0.
    # For the t contribution from t_next:
    # Q_t[m, i] = r[i-m] * psi_t[i, i-m]  for i >= m >= 1, else 0.
    #
    # Since psi_t is symmetric, psi_t[j-m, j] = psi_t[j, j-m] and
    # psi_t[i, i-m] = psi_t[i-m, i]. So Q_s and Q_t have the same structure
    # (both use psi_t along sub-diagonal m).
    #
    # Rewrite: Q_s[m, j] = r[j-m] * psi_t[j-m, j] for j > m-1 and m >= 1.
    # Since psi_t is symmetric: Q_s[m, j] = r[j-m] * psi_t[j, j-m].
    #
    # Q_t[m, i] = r[i-m] * psi_t[i, i-m] for i >= m and m >= 1.
    # = r[i-m] * psi_t[i, i-m]  (same pattern as Q_s with different psi_t indexing).
    # Since psi_t symmetric: Q_t[m, i] = r[i-m] * psi_t[i-m, i] = Q_s[m, i]. Same matrix!
    #
    # So Q_s = Q_t. Let's call it Q.
    # Q[m, j] = r[j-m] * psi_t[j-m, j] for j >= m >= 1, else 0.
    # Q[0, :] = 0.

    # Implementation using the transposed Toeplitz approach:
    # Notice that the matrix with [i, j] = r[i] * psi_t[i, i + m] summed over... no.
    #
    # Simplest efficient approach: build the "weighted" transposed Toeplitz matrix.
    # bar_s_weighted = r * bar_s_next   (elementwise)
    # bar_t_weighted = r * bar_t_next
    # Then we need to "reverse-convolve" along sub-diagonals of psi_t.
    #
    # Actually the cleanest implementation:
    # Construct M_adj_s = psi_t * toeplitz(bar_s_weighted, zeros)
    # where bar_s_weighted plays the role of the "column" in the toeplitz.
    # Then bar_s[m] = sum of m-th sub-diagonal of M_adj_s... no, that's getting complicated.
    #
    # Let me just directly compute using matrix multiply with the transposed structure.
    #
    # FINAL APPROACH: Use the transposed Toeplitz directly.
    # bar_s from s_next:
    #   s_next[j] = (U^T r)[j] where U = triu(psi_t ⊙ toeplitz(t, s), 1).
    #   The dependence on s comes only through toeplitz(t, s).
    #   Define f(s) = triu(psi_t ⊙ toeplitz(t, s), 1)^T @ r.
    #   We showed: f(s)[j] = sum_{i<j} r[i] * psi_t[i,j] * s[j-i].
    #   This IS a convolution of (r * psi_t[·, j]) with s, restricted to j > i.
    #   The adjoint of a convolution is a correlation.
    #
    #   bar_s[m] = sum_j bar_s_next[j] * (df[j]/ds[m])
    #            = sum_{j >= m, m >= 1} bar_s_next[j] * r[j-m] * psi_t[j-m, j]
    #
    #   This can be written as a matrix-vector product with a lower-triangular Toeplitz-like matrix.
    #   Actually: define v[j] = bar_s_next[j] for j = 0..P-1.
    #   Then bar_s[m] = sum_{j=m}^{P-1} v[j] * r[j-m] * psi_t[j-m, j], m >= 1.
    #   Let k = j - m (k = 0..P-1-m):
    #   bar_s[m] = sum_{k=0}^{P-1-m} v[k+m] * r[k] * psi_t[k, k+m]
    #
    # This is exactly a dot product along the m-th super-diagonal of the matrix
    # with entry [k, k+m] = r[k] * psi_t[k, k+m], weighted by v[k+m].
    #
    # We can compute ALL these dot products at once using:
    # Construct matrix W with W[i, j] = r[i] * psi_t[i, j] for j > i (strict upper tri), else 0.
    # Note this is exactly U but with s factored out!
    # U[i,j] = psi_t[i,j] * s[j-i] * ... no, U = triu(psi_t * toeplitz(t,s), 1).
    # What we want: W[i,j] = r[i] * psi_t[i,j] for j > i.
    # Then bar_s[m] = sum_{k=0}^{P-1-m} W[k, k+m] * bar_s_next[k+m]
    #              = sum_j W[j-m, j] * bar_s_next[j]  for j >= m+... wait, k = j-m.
    #              = (W^T @ bar_s_next)[m] restricted to m >= 1... no.
    # W^T[m, k] = W[k, m] = r[k] * psi_t[k, m] for m > k (since W is strict upper).
    # (W^T @ bar_s_next)[m] = sum_k W^T[m, k] * bar_s_next[k]
    #                        = sum_{k < m} r[k] * psi_t[k, m] * bar_s_next[k]
    # That's not the same as bar_s[m] = sum_{k} r[k] * psi_t[k, k+m] * bar_s_next[k+m].
    # The psi_t indices differ: [k, m] vs [k, k+m]. Only the same when m = k+m, i.e., k=0.
    # So this approach doesn't work directly with a simple transpose.
    #
    # PRAGMATIC SOLUTION: Just build the full Jacobian blocks and multiply.
    # For order P (typically 8-32), materializing P×P matrices is fine.

    # Build the Jacobian blocks explicitly.
    # J_ss[m, j] = ds_next[j] / ds[m]
    # J_ts[m, i] = dt_next[i] / ds[m]
    # J_st[m, j] = ds_next[j] / dt[m]
    # J_tt[m, i] = dt_next[i] / dt[m]
    #
    # From the forward:
    #   s_next[j] = sum_{i<j} r[i] * psi_t[i,j] * s[j-i] + (psi_s @ t)[j] * r[j]
    #   t_next[i] = (psi_s @ s)[i] * r[i] + sum_{j<i} psi_t[i,j] * t[i-j] * r[j]
    #
    # J_ss[m, j] = r[j-m] * psi_t[j-m, j]  for 1 <= m, m <= j-1... no, j >= m and m >= 1.
    #   Actually for m >= 1 and j >= m (so j-m >= 0) and j-m < j (always for m >= 1):
    #   J_ss[m, j] = r[j-m] * psi_t[j-m, j]
    #   J_ss[0, j] = 0
    #
    # J_ts[m, i] = (d/ds[m]) ((psi_s @ s)[i] * r[i]) = psi_s[i, m] * r[i]
    #
    # J_st[m, j] = (d/dt[m]) ((psi_s @ t)[j] * r[j]) = psi_s[j, m] * r[j]
    #
    # J_tt[m, i] = psi_t[i, i-m] * r[i-m]  for m >= 1, i >= m.
    #   J_tt[0, i] = 0

    # bar_s = J_ss^T @ bar_s_next + J_ts^T @ bar_t_next
    # bar_t = J_st^T @ bar_s_next + J_tt^T @ bar_t_next

    # For the psi_s terms (J_ts and J_st):
    # J_ts^T @ bar_t_next: (J_ts^T)[m, i] = J_ts[i, m] = ... wait, J_ts is the Jacobian
    # dt_next/ds, so J_ts[m, i] maps ds[m] -> dt_next[i]. The adjoint is:
    # (J_ts^T)[i, m] = J_ts[m, i]. Hmm, let me be careful with the convention.
    # We want bar_s[m] = sum_i (dt_next/ds[m])[i] * bar_t_next[i] = sum_i J_ts[m, i] * bar_t_next[i].
    # Wait no — J_ts[m, i] is the sensitivity: how much does t_next[i] change when s[m] changes.
    # But the standard Jacobian convention: J[output_idx, input_idx].
    # So let's define properly:
    # (ds_next/ds)[j, m] = J^{s_next, s}[j, m] = ds_next_j / ds_m
    # Then bar_s = (ds_next/ds)^T @ bar_s_next + (dt_next/ds)^T @ bar_t_next
    #            = sum_j (ds_next_j/ds_m) * bar_s_next_j + sum_i (dt_next_i/ds_m) * bar_t_next_i

    # (ds_next/ds)[j, m] = r[j-m] * psi_t[j-m, j]  for m >= 1, j >= m; else 0.
    # (dt_next/ds)[i, m] = psi_s[i, m] * r[i]       for all i, m.
    # (ds_next/dt)[j, m] = psi_s[j, m] * r[j]       for all j, m.
    # (dt_next/dt)[i, m] = psi_t[i, i-m] * r[i-m]   for m >= 1, i >= m; else 0.

    # Now implement bar_s and bar_t:
    # bar_s[m] = sum_j (ds_next/ds)[j,m] * bar_s_next[j] + sum_i (dt_next/ds)[i,m] * bar_t_next[i]
    #          = (for m >= 1) sum_{j=m}^{P-1} r[j-m]*psi_t[j-m,j]*bar_s_next[j]
    #            + sum_i psi_s[i,m]*r[i]*bar_t_next[i]
    #          = (for m >= 1) sum_{j=m}^{P-1} r[j-m]*psi_t[j-m,j]*bar_s_next[j]
    #            + (psi_s^T @ (r * bar_t_next))[m]
    # bar_s[0] = (psi_s^T @ (r * bar_t_next))[0]  (the triu sum is empty for m=0 since m >= 1 is required)
    #
    # bar_t[m] = sum_j (ds_next/dt)[j,m] * bar_s_next[j] + sum_i (dt_next/dt)[i,m] * bar_t_next[i]
    #          = sum_j psi_s[j,m]*r[j]*bar_s_next[j] + (for m >= 1) sum_{i=m}^{P-1} psi_t[i,i-m]*r[i-m]*bar_t_next[i]
    #          = (psi_s^T @ (r * bar_s_next))[m]
    #            + (for m >= 1) sum_{i=m}^{P-1} psi_t[i,i-m]*r[i-m]*bar_t_next[i]
    # bar_t[0] = (psi_s^T @ (r * bar_s_next))[0]

    # For the toeplitz-structured sums, build the Jacobian matrices efficiently.
    # J_ss_block[j, m] = r[j-m] * psi_t[j-m, j] for j >= m >= 1, else 0.
    # J_tt_block[i, m] = r[i-m] * psi_t[i, i-m] for i >= m >= 1, else 0.
    #
    # These are both strict-lower-triangular P×P matrices (nonzero for j > m-1 and m >= 1,
    # equivalently j >= m >= 1, equivalently j > m-1 >= 0).
    # Actually J_ss_block[m, m] = r[0] * psi_t[0, m] for m >= 1. So the diagonal is included for m >= 1.
    # Let me recheck: j >= m and m >= 1. For j = m: r[0] * psi_t[0, m]. So diagonal entries exist for m >= 1.
    # But column m=0 is all zeros. So it's a matrix that's zero in column 0 and has entries from the
    # diagonal down in columns 1..P-1.

    # Efficient construction using index arrays:
    j_idx = jnp.arange(P, dtype=jnp.int32)[:, None]  # (P, 1) -- output index
    m_idx = jnp.arange(P, dtype=jnp.int32)[None, :]  # (1, P) -- input index

    # J_ss[j, m] = r[j-m] * psi_t[j-m, j] where j >= m >= 1
    diff_jm = j_idx - m_idx  # (P, P), j - m
    mask_ss = (m_idx >= 1) & (j_idx >= m_idx)  # j >= m >= 1
    # Safe indexing: clamp diff to valid range
    safe_diff = jnp.clip(diff_jm, 0, P - 1)
    J_ss = jnp.where(mask_ss, r[safe_diff] * psi_t[safe_diff, j_idx], 0.0)

    # J_tt[i, m] = r[i-m] * psi_t[i, i-m] where i >= m >= 1
    # diff_im = i - m (same as diff_jm with i playing the role of j)
    mask_tt = mask_ss  # Same mask: i >= m >= 1
    J_tt = jnp.where(mask_tt, r[safe_diff] * psi_t[j_idx, safe_diff], 0.0)

    # bar_s and bar_t from all contributions:
    r_bar_s_next = r * bar_s_next
    r_bar_t_next = r * bar_t_next

    bar_s = J_ss.T @ bar_s_next + psi_s.T @ r_bar_t_next
    bar_t = J_tt.T @ bar_t_next + psi_s.T @ r_bar_s_next

    return bar_s, bar_t, bar_rho


# ---------------------------------------------------------------------------
# Forward sweep (plain, no checkpoints — used as the primal)
# ---------------------------------------------------------------------------

def _forward_sweep(X_i, Y_j, v_s, v_t, psi_s, psi_t, ic, exponents,
                   static_kernel):
    """Run the diagonal sweep for a single (X_i, Y_j) pair. Returns scalar."""
    P = exponents.shape[0]
    M = X_i.shape[0]
    N = Y_j.shape[0]
    rows = M - 1
    cols = N - 1
    diagonal_count = rows + cols - 1
    longest_diagonal = min(rows, cols)
    dtype = X_i.dtype

    indices = jnp.arange(longest_diagonal, dtype=jnp.int32)

    S_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    T_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    S_buf = S_buf.at[:, 0].set(1.0)
    T_buf = T_buf.at[:, 0].set(1.0)

    def compute_diagonal(d, carry):
        S_buf, T_buf = carry
        t_start = jnp.where(d < cols, 0, d - cols + 1)
        s_start = jnp.where(d < cols, d, cols - 1)
        is_before_wrap = d < rows

        def next_entry(diagonal_index, S, T):
            s_index = diagonal_index - is_before_wrap
            t_index = diagonal_index + (1 - is_before_wrap)
            s = jnp.where(t_start + diagonal_index == 0, ic, S[s_index])
            t = jnp.where(s_start - diagonal_index == 0, ic, T[t_index])
            dX_idx = (s_start - diagonal_index) * ((s_start - diagonal_index) < rows)
            dY_idx = (t_start + diagonal_index) * ((t_start + diagonal_index) < cols)
            rho = static_kernel(X_i[dX_idx + 1], X_i[dX_idx],
                                Y_j[dY_idx + 1], Y_j[dY_idx])
            r = rho ** exponents
            T_mat = psi_t * toeplitz(t, s)
            s_next = (r @ jnp.triu(T_mat, 1)) + ((psi_s @ t) * r)
            t_next = ((psi_s @ s) * r) + (jnp.tril(T_mat, -1) @ r)
            return s_next, t_next

        S_next, T_next = vmap(next_entry, in_axes=(0, None, None))(
            indices, S_buf, T_buf)
        return S_next, T_next

    S_buf, T_buf = jax.lax.fori_loop(0, diagonal_count, compute_diagonal,
                                      (S_buf, T_buf))
    return S_buf[0] @ v_s


# ---------------------------------------------------------------------------
# Forward sweep with checkpoints
# ---------------------------------------------------------------------------

def _forward_sweep_with_checkpoints(X_i, Y_j, v_s, v_t, psi_s, psi_t,
                                     ic, exponents, static_kernel,
                                     checkpoint_interval):
    """Forward sweep that saves (S_buf, T_buf) every checkpoint_interval diagonals.

    Returns (scalar_output, S_checkpoints, T_checkpoints).
    """
    P = exponents.shape[0]
    M = X_i.shape[0]
    N = Y_j.shape[0]
    rows = M - 1
    cols = N - 1
    diagonal_count = rows + cols - 1
    longest_diagonal = min(rows, cols)
    dtype = X_i.dtype
    k = checkpoint_interval
    num_checkpoints = ceil(diagonal_count / k)

    indices = jnp.arange(longest_diagonal, dtype=jnp.int32)

    S_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    T_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    S_buf = S_buf.at[:, 0].set(1.0)
    T_buf = T_buf.at[:, 0].set(1.0)

    S_checkpoints = jnp.zeros((num_checkpoints, longest_diagonal, P), dtype=dtype)
    T_checkpoints = jnp.zeros((num_checkpoints, longest_diagonal, P), dtype=dtype)

    def compute_diagonal(d, carry):
        S_buf, T_buf, S_ckpt, T_ckpt = carry

        # Save checkpoint at the start of each block
        ckpt_idx = d // k
        is_checkpoint = (d % k == 0)
        S_ckpt = jnp.where(is_checkpoint, S_ckpt.at[ckpt_idx].set(S_buf), S_ckpt)
        T_ckpt = jnp.where(is_checkpoint, T_ckpt.at[ckpt_idx].set(T_buf), T_ckpt)

        t_start = jnp.where(d < cols, 0, d - cols + 1)
        s_start = jnp.where(d < cols, d, cols - 1)
        is_before_wrap = d < rows

        def next_entry(diagonal_index, S, T):
            s_index = diagonal_index - is_before_wrap
            t_index = diagonal_index + (1 - is_before_wrap)
            s = jnp.where(t_start + diagonal_index == 0, ic, S[s_index])
            t = jnp.where(s_start - diagonal_index == 0, ic, T[t_index])
            dX_idx = (s_start - diagonal_index) * ((s_start - diagonal_index) < rows)
            dY_idx = (t_start + diagonal_index) * ((t_start + diagonal_index) < cols)
            rho = static_kernel(X_i[dX_idx + 1], X_i[dX_idx],
                                Y_j[dY_idx + 1], Y_j[dY_idx])
            r = rho ** exponents
            T_mat = psi_t * toeplitz(t, s)
            s_next = (r @ jnp.triu(T_mat, 1)) + ((psi_s @ t) * r)
            t_next = ((psi_s @ s) * r) + (jnp.tril(T_mat, -1) @ r)
            return s_next, t_next

        S_next, T_next = vmap(next_entry, in_axes=(0, None, None))(
            indices, S_buf, T_buf)
        return S_next, T_next, S_ckpt, T_ckpt

    S_buf, T_buf, S_checkpoints, T_checkpoints = jax.lax.fori_loop(
        0, diagonal_count, compute_diagonal,
        (S_buf, T_buf, S_checkpoints, T_checkpoints))

    out = S_buf[0] @ v_s
    return out, S_checkpoints, T_checkpoints


# ---------------------------------------------------------------------------
# Reverse sweep with replay
# ---------------------------------------------------------------------------

def _reverse_sweep_with_replay(X_i, Y_j, S_checkpoints, T_checkpoints,
                                g, v_s, v_t, psi_s, psi_t, ic, exponents,
                                static_kernel, checkpoint_interval):
    """Backward pass using checkpointed replay.

    For each block of `checkpoint_interval` diagonals (processed in reverse):
    1. Load the checkpoint at the block start.
    2. Replay the block forward, saving per-diagonal (S_buf, T_buf) into a tape.
    3. Reverse through the block, accumulating adjoint boundary states and bar_rho.
    4. Scatter bar_rho into grad_X and grad_Y.

    Returns (grad_X, grad_Y) with the same shapes as X_i and Y_j.
    """
    P = exponents.shape[0]
    M = X_i.shape[0]
    N = Y_j.shape[0]
    d_dim = X_i.shape[1]
    rows = M - 1
    cols = N - 1
    diagonal_count = rows + cols - 1
    longest_diagonal = min(rows, cols)
    dtype = X_i.dtype
    k = checkpoint_interval
    num_checkpoints = ceil(diagonal_count / k)

    indices = jnp.arange(longest_diagonal, dtype=jnp.int32)

    # Gradient accumulators
    grad_X = jnp.zeros_like(X_i)
    grad_Y = jnp.zeros_like(Y_j)

    # Adjoint boundary buffers — start from the final output
    # The output is S_buf[0] @ v_s, so the initial adjoint is:
    # bar_S_buf[0] = g * v_s, bar_S_buf[i>0] = 0, bar_T_buf = 0.
    bar_S_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    bar_T_buf = jnp.zeros((longest_diagonal, P), dtype=dtype)
    bar_S_buf = bar_S_buf.at[0].set(g * v_s)

    # Replay tape for one block
    S_tape = jnp.zeros((k, longest_diagonal, P), dtype=dtype)
    T_tape = jnp.zeros((k, longest_diagonal, P), dtype=dtype)

    # Process blocks in reverse
    # Block b covers diagonals [b*k, min((b+1)*k, diagonal_count))
    # We iterate b from num_checkpoints-1 down to 0.

    def process_block(b, carry):
        bar_S_buf, bar_T_buf, grad_X, grad_Y, S_tape, T_tape = carry

        block_start = b * k
        block_end = jnp.minimum(block_start + k, diagonal_count)
        block_len = block_end - block_start  # actual number of diagonals in this block

        # Step 1: Load checkpoint
        S_buf_replay = S_checkpoints[b]
        T_buf_replay = T_checkpoints[b]

        # Step 2: Replay forward, saving into tape
        def replay_diagonal(step, replay_carry):
            S_buf_r, T_buf_r, S_tape_r, T_tape_r = replay_carry
            d = block_start + step

            # Save current state into tape
            S_tape_r = S_tape_r.at[step].set(S_buf_r)
            T_tape_r = T_tape_r.at[step].set(T_buf_r)

            t_start = jnp.where(d < cols, 0, d - cols + 1)
            s_start = jnp.where(d < cols, d, cols - 1)
            is_before_wrap = d < rows

            def next_entry(diagonal_index, S, T):
                s_index = diagonal_index - is_before_wrap
                t_index = diagonal_index + (1 - is_before_wrap)
                s = jnp.where(t_start + diagonal_index == 0, ic, S[s_index])
                t = jnp.where(s_start - diagonal_index == 0, ic, T[t_index])
                dX_idx = (s_start - diagonal_index) * ((s_start - diagonal_index) < rows)
                dY_idx = (t_start + diagonal_index) * ((t_start + diagonal_index) < cols)
                rho = static_kernel(X_i[dX_idx + 1], X_i[dX_idx],
                                    Y_j[dY_idx + 1], Y_j[dY_idx])
                r = rho ** exponents
                T_mat = psi_t * toeplitz(t, s)
                s_next = (r @ jnp.triu(T_mat, 1)) + ((psi_s @ t) * r)
                t_next = ((psi_s @ s) * r) + (jnp.tril(T_mat, -1) @ r)
                return s_next, t_next

            S_next, T_next = vmap(next_entry, in_axes=(0, None, None))(
                indices, S_buf_r, T_buf_r)
            return S_next, T_next, S_tape_r, T_tape_r

        _, _, S_tape, T_tape = jax.lax.fori_loop(
            0, block_end - block_start, replay_diagonal,
            (S_buf_replay, T_buf_replay, S_tape, T_tape))

        # Step 3: Reverse through the block
        def reverse_diagonal(step, rev_carry):
            bar_S, bar_T, gX, gY = rev_carry
            # step counts from 0; actual diagonal index within block is (block_len - 1 - step)
            local_idx = (block_end - block_start) - 1 - step
            d = block_start + local_idx

            # Load the saved (S_buf, T_buf) at the START of diagonal d
            S_saved = S_tape[local_idx]
            T_saved = T_tape[local_idx]

            t_start = jnp.where(d < cols, 0, d - cols + 1)
            s_start = jnp.where(d < cols, d, cols - 1)
            is_before_wrap = d < rows

            def adjoint_entry(diagonal_index, S_saved, T_saved, bar_S, bar_T):
                """Compute local adjoint for one tile on diagonal d."""
                s_index = diagonal_index - is_before_wrap
                t_index = diagonal_index + (1 - is_before_wrap)
                s = jnp.where(t_start + diagonal_index == 0, ic, S_saved[s_index])
                t = jnp.where(s_start - diagonal_index == 0, ic, T_saved[t_index])
                dX_idx = (s_start - diagonal_index) * ((s_start - diagonal_index) < rows)
                dY_idx = (t_start + diagonal_index) * ((t_start + diagonal_index) < cols)
                rho = static_kernel(X_i[dX_idx + 1], X_i[dX_idx],
                                    Y_j[dY_idx + 1], Y_j[dY_idx])

                # Incoming adjoint for this tile's outputs
                bar_s_next = bar_S[diagonal_index]
                bar_t_next = bar_T[diagonal_index]

                # Local adjoint
                bar_s, bar_t, bar_rho = _map_diagonal_entry_bwd(
                    rho, s, t, psi_s, psi_t, exponents,
                    bar_s_next, bar_t_next)

                return bar_s, bar_t, bar_rho, dX_idx, dY_idx

            # Vmap the adjoint over the diagonal
            bar_s_all, bar_t_all, bar_rho_all, dX_idx_all, dY_idx_all = vmap(
                adjoint_entry, in_axes=(0, None, None, None, None))(
                indices, S_saved, T_saved, bar_S, bar_T)

            # bar_s_all: (longest_diagonal, P) — these need to be scattered back
            # into bar_S and bar_T at the positions they came from.
            #
            # The scatter pattern mirrors the forward gather:
            # In the forward, tile `diagonal_index` on diagonal `d` reads:
            #   s from S_buf[s_index] (or ic if at boundary)
            #   t from T_buf[t_index] (or ic if at boundary)
            # In the backward, bar_s must be accumulated into bar_S[s_index]
            # and bar_t into bar_T[t_index].
            #
            # But there's a subtlety: the forward reads from the PREVIOUS diagonal's
            # S_buf/T_buf, and writes to the CURRENT diagonal's S_buf/T_buf.
            # The adjoint of the current diagonal's write is what bar_S/bar_T already holds.
            # The adjoint of the previous diagonal's read scatters into the NEW bar_S/bar_T
            # for the previous diagonal.
            #
            # So: we need to construct the new bar_S_prev, bar_T_prev by scattering
            # bar_s_all into the s_index positions and bar_t_all into the t_index positions.

            # Compute the scatter indices
            s_indices = indices - is_before_wrap
            t_indices = indices + (1 - is_before_wrap)

            # Boundary flags: if the tile reads ic instead of S_buf/T_buf,
            # the adjoint w.r.t. ic is discarded (ic is a constant).
            is_s_boundary = (t_start + indices == 0)
            is_t_boundary = (s_start - indices == 0)

            # Zero out adjoints that correspond to boundary reads
            bar_s_scattered = jnp.where(is_s_boundary[:, None], 0.0, bar_s_all)
            bar_t_scattered = jnp.where(is_t_boundary[:, None], 0.0, bar_t_all)

            # Build new bar_S_prev, bar_T_prev by scattering
            # Multiple tiles may scatter into the same slot, so we use segment_sum-like logic.
            # However, in the diagonal structure, each s_index and t_index should be unique
            # (each position in the previous S_buf/T_buf is read by at most one tile).
            new_bar_S = jnp.zeros_like(bar_S)
            new_bar_T = jnp.zeros_like(bar_T)

            # Clamp indices to valid range for the .at[].add() calls
            safe_s_indices = jnp.clip(s_indices, 0, longest_diagonal - 1)
            safe_t_indices = jnp.clip(t_indices, 0, longest_diagonal - 1)

            new_bar_S = new_bar_S.at[safe_s_indices].add(bar_s_scattered)
            new_bar_T = new_bar_T.at[safe_t_indices].add(bar_t_scattered)

            # Scatter bar_rho into grad_X and grad_Y
            # For linear kernel: rho = <X[i+1] - X[i], Y[j+1] - Y[j]>
            # bar_rho scatters as:
            #   grad_X[i+1] += bar_rho * (Y[j+1] - Y[j])
            #   grad_X[i]   -= bar_rho * (Y[j+1] - Y[j])
            #   grad_Y[j+1] += bar_rho * (X[i+1] - X[i])
            #   grad_Y[j]   -= bar_rho * (X[i+1] - X[i])
            #
            # For generality, we use JAX autodiff through the static_kernel for the
            # bar_rho -> path gradient scatter. This handles both linear and RBF.

            def scatter_one(bar_rho, dX_idx, dY_idx):
                """Compute path gradients from one tile's bar_rho."""
                x2, x1 = X_i[dX_idx + 1], X_i[dX_idx]
                y2, y1 = Y_j[dY_idx + 1], Y_j[dY_idx]

                # Use JAX to differentiate the static kernel w.r.t. its 4 arguments
                _, vjp_fn = jax.vjp(
                    lambda a, b, c, d_arg: static_kernel(a, b, c, d_arg),
                    x2, x1, y2, y1)
                g_x2, g_x1, g_y2, g_y1 = vjp_fn(bar_rho)

                return dX_idx, g_x2, g_x1, dY_idx, g_y2, g_y1

            dX_idxs, g_x2s, g_x1s, dY_idxs, g_y2s, g_y1s = vmap(scatter_one)(
                bar_rho_all, dX_idx_all, dY_idx_all)

            # Scatter into grad_X and grad_Y
            gX = gX.at[dX_idxs + 1].add(g_x2s)
            gX = gX.at[dX_idxs].add(g_x1s)
            gY = gY.at[dY_idxs + 1].add(g_y2s)
            gY = gY.at[dY_idxs].add(g_y1s)

            return new_bar_S, new_bar_T, gX, gY

        bar_S_buf, bar_T_buf, grad_X, grad_Y = jax.lax.fori_loop(
            0, block_end - block_start, reverse_diagonal,
            (bar_S_buf, bar_T_buf, grad_X, grad_Y))

        return bar_S_buf, bar_T_buf, grad_X, grad_Y, S_tape, T_tape

    bar_S_buf, bar_T_buf, grad_X, grad_Y, _, _ = jax.lax.fori_loop(
        0, num_checkpoints, lambda b, carry: process_block(
            num_checkpoints - 1 - b, carry),
        (bar_S_buf, bar_T_buf, grad_X, grad_Y, S_tape, T_tape))

    return grad_X, grad_Y


# ---------------------------------------------------------------------------
# custom_vjp wiring
# ---------------------------------------------------------------------------

def _make_sig_kernel_diff(static_kernel, checkpoint_interval,
                          v_s, v_t, psi_s, psi_t, ic, exponents):
    """Factory that returns a (custom_vjp-wrapped, jittable) differentiable
    signature kernel for a given static kernel and structural constants.

    We use a closure rather than nondiff_argnums because JAX custom_vjp
    requires nondiff args to be hashable, and jnp arrays are not.
    """

    @jax.custom_vjp
    def sig_kernel(X_i, Y_j):
        return _forward_sweep(X_i, Y_j, v_s, v_t, psi_s, psi_t, ic,
                              exponents, static_kernel)

    def sig_kernel_fwd(X_i, Y_j):
        out, S_ckpt, T_ckpt = _forward_sweep_with_checkpoints(
            X_i, Y_j, v_s, v_t, psi_s, psi_t, ic, exponents,
            static_kernel, checkpoint_interval)
        return out, (X_i, Y_j, S_ckpt, T_ckpt)

    def sig_kernel_bwd(residuals, g):
        X_i, Y_j, S_ckpt, T_ckpt = residuals
        grad_X, grad_Y = _reverse_sweep_with_replay(
            X_i, Y_j, S_ckpt, T_ckpt, g,
            v_s, v_t, psi_s, psi_t, ic, exponents,
            static_kernel, checkpoint_interval)
        return grad_X, grad_Y

    sig_kernel.defvjp(sig_kernel_fwd, sig_kernel_bwd)
    return sig_kernel


# ---------------------------------------------------------------------------
# Block sizing for differentiable path
# ---------------------------------------------------------------------------

def _compute_checkpoint_interval(diagonal_count):
    """Pick k ~= sqrt(T), clamped to [1, T]."""
    return max(1, min(int(sqrt(diagonal_count)), diagonal_count))


def estimate_diff_bytes_per_pair(longest_diagonal, order, dtype,
                                 checkpoint_interval, diagonal_count):
    """Estimate peak memory for one differentiable (X_i, Y_j) pair."""
    elem = jnp.dtype(dtype).itemsize
    D, P, k_val = longest_diagonal, order, checkpoint_interval
    C = ceil(diagonal_count / k_val)

    forward_buffers = 2 * D * P
    toeplitz_intermediates = 3 * D * P * P
    checkpoint_storage = 2 * C * D * P
    replay_tape = 2 * k_val * D * P
    adjoint_buffers = 2 * D * P
    aux = 5 * D * P

    return elem * (forward_buffers + toeplitz_intermediates +
                   checkpoint_storage + replay_tape + adjoint_buffers + aux)


def compute_diff_block_size(longest_diagonal, order, dtype, device,
                            total_pairs, checkpoint_interval,
                            diagonal_count, safety_factor=0.5):
    """Pick the largest vmap block_size that fits in GPU memory for the diff path."""
    per_pair = estimate_diff_bytes_per_pair(
        longest_diagonal, order, dtype, checkpoint_interval, diagonal_count)
    available = get_available_gpu_memory(device)
    budget = int(available * safety_factor)

    if per_pair > 0:
        raw = max(1, budget // per_pair)
    else:
        raw = MAX_BLOCK_SIZE

    raw = min(raw, total_pairs, MAX_BLOCK_SIZE)
    return _round_to_power_of_2(raw)


# ---------------------------------------------------------------------------
# Public API: compute_gram_fast_diff (on PowerSigJax via mixin)
# ---------------------------------------------------------------------------

def compute_sig_kernel_fast_diff(ps, X_i, Y_j, checkpoint_interval=None):
    """Compute a single differentiable signature kernel value.

    Args:
        ps: PowerSigJax instance (provides order, psi_s, psi_t, etc.)
        X_i: (M, d) single path
        Y_j: (N, d) single path
        checkpoint_interval: int or None (auto-tuned from path lengths)

    Returns:
        Scalar kernel value (differentiable w.r.t. X_i and Y_j).
    """
    M, N = X_i.shape[0], Y_j.shape[0]
    rows, cols = M - 1, N - 1
    diagonal_count = rows + cols - 1

    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(diagonal_count)

    v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, ps.order, X_i.dtype)

    sig_fn = _make_sig_kernel_diff(
        ps.static_kernel, checkpoint_interval,
        v_s, v_t, ps.psi_s, ps.psi_t, ps.ic, ps.exponents)

    return sig_fn(X_i, Y_j)


def compute_gram_fast_diff(ps, X, Y, symmetric=False, block_size=None,
                           checkpoint_interval=None, show_progress=True):
    """Compute the Gram matrix with custom VJP support.

    This is the differentiable counterpart to PowerSigJax.compute_gram_matrix.

    Args:
        ps: PowerSigJax instance
        X: (batch_X, M, d) paths
        Y: (batch_Y, N, d) paths
        symmetric: if True, only compute upper triangle and mirror
        block_size: vmap batch size (auto-tuned if None)
        checkpoint_interval: diagonals between checkpoints (auto-tuned if None)

    Returns:
        (batch_X, batch_Y) Gram matrix, differentiable w.r.t. X and Y.
    """
    if not isinstance(X, jnp.ndarray):
        X = jnp.array(X, device=ps.device)
    if not isinstance(Y, jnp.ndarray):
        Y = jnp.array(Y, device=ps.device)

    M, N = X.shape[1], Y.shape[1]
    rows, cols = M - 1, N - 1
    diagonal_count = rows + cols - 1
    longest_diagonal = min(rows, cols)

    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(diagonal_count)

    v_s, v_t = compute_vandermonde_vectors(1.0, 1.0, ps.order, X.dtype,
                                            device=ps.device)

    # Ensure structural constants are on the right device
    exponents = jax.device_put(ps.exponents, ps.device)
    psi_s = jax.device_put(ps.psi_s, ps.device)
    psi_t = jax.device_put(ps.psi_t, ps.device)
    ic = jax.device_put(ps.ic, ps.device)

    sig_fn = _make_sig_kernel_diff(
        ps.static_kernel, checkpoint_interval,
        v_s, v_t, psi_s, psi_t, ic, exponents)

    # Build pairs
    pairs_i, pairs_j = [], []
    for i in range(X.shape[0]):
        for j in range(i if symmetric else 0, Y.shape[0]):
            pairs_i.append(i)
            pairs_j.append(j)

    total_pairs = len(pairs_i)
    if total_pairs == 0:
        return jnp.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype,
                          device=ps.device)

    i_all = jnp.array(pairs_i, dtype=jnp.int32)
    j_all = jnp.array(pairs_j, dtype=jnp.int32)

    # Block sizing
    if block_size is None:
        block_size = compute_diff_block_size(
            longest_diagonal, ps.order, X.dtype, ps.device,
            total_pairs, checkpoint_interval, diagonal_count)
    else:
        block_size = _round_to_power_of_2(min(block_size, total_pairs))

    batched_sig = vmap(sig_fn, in_axes=(0, 0))

    gram = jnp.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype, device=ps.device)

    pbar = tqdm(total=total_pairs, desc="Computing Gram (diff)", disable=not show_progress)
    offset = 0
    while offset < total_pairs:
        end = min(offset + block_size, total_pairs)
        actual = end - offset

        bi = i_all[offset:end]
        bj = j_all[offset:end]

        if actual < block_size:
            pad = block_size - actual
            bi = jnp.concatenate([bi, jnp.full(pad, bi[-1], dtype=jnp.int32)])
            bj = jnp.concatenate([bj, jnp.full(pad, bj[-1], dtype=jnp.int32)])

        results = batched_sig(X[bi], Y[bj])

        ai = i_all[offset:end]
        aj = j_all[offset:end]
        gram = gram.at[ai, aj].set(results[:actual])
        if symmetric:
            gram = gram.at[aj, ai].set(results[:actual])

        pbar.update(actual)
        offset = end

    pbar.close()
    return gram
