"""Reduced-order reaction-diffusion model and its numerical bistability analysis.

    dA_i/dt = P_i + beta H(A_i) + D sum_j w_ij (A_j - A_i) - delta A_i

Integration is explicit Euler with enough substeps to guarantee positivity and
stability. Explicit steps are also what keeps the model decentralised: every update
needs only a facility's own state and its neighbours' last transmitted values.
"""
import math

import numpy as np
from scipy.optimize import brentq

from network.facility_network import laplacian
from quorum.hill_activation import hill, hill_max_slope


def stable_substeps(W, params, safety=0.5):
    """Substeps per day so that h * (D * max_degree + delta + beta * max H') <= safety.
    Because h * (D * degree + delta) <= 1, each update is a nonnegative combination of
    current concentrations, which keeps A nonnegative. Since lambda_max(L) <= 2 * degree,
    the linear part also has |1 - h * rate| <= 1."""
    rate = params.D * W.sum(axis=1).max() + params.delta + params.beta * hill_max_slope(params.K, params.n)
    return max(1, math.ceil(rate / safety))


def simulate_signals(production, W, params, n_sub=None, A0=None):
    """Integrate one day at a time. `production` has shape (..., T, N); batched runs share
    W. Returns concentrations of the same shape, recorded at the end of each day."""
    if n_sub is None:
        n_sub = stable_substeps(W, params)
    h = 1.0 / n_sub
    L = laplacian(W)
    T = production.shape[-2]
    A = np.zeros(production.shape[:-2] + production.shape[-1:]) if A0 is None else np.array(A0, float)
    out = np.zeros(production.shape, dtype=np.float32)
    for t in range(T):
        P = production[..., t, :]
        for _ in range(n_sub):
            dA = P - params.delta * A
            if params.beta:
                dA = dA + params.beta * hill(A, params.K, params.n)
            if params.D:
                dA = dA - params.D * (A @ L)
            A = np.maximum(A + h * dA, 0.0)
        out[..., t, :] = A
    return out


def equilibria(P, params):
    """Single-node equilibria of P + beta H(A) - delta A, each classified by the sign of
    the derivative of the right-hand side."""
    def f(A):
        return P + params.beta * hill(A, params.K, params.n) - params.delta * A

    upper = (P + params.beta) / params.delta + 1.0
    grid = np.concatenate([[0.0], np.geomspace(1e-5, upper, 6000)])
    values = f(grid)
    roots = []
    if P == 0:
        roots.append(0.0)
    sign_change = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    for k in sign_change:
        roots.append(brentq(f, grid[k], grid[k + 1], xtol=1e-11))
    result = []
    for root in roots:
        eps = max(1e-7, abs(root) * 1e-6)
        slope = (f(root + eps) - f(max(root - eps, 0.0))) / (root + eps - max(root - eps, 0.0))
        result.append({"A": float(root), "stable": bool(slope < 0)})
    return result


def n_stable(P, params):
    return sum(e["stable"] for e in equilibria(P, params))


def network_bistability(W, params, P, days=4000):
    """Integrate the full network to steady state from low and high initial conditions.
    Different stable end points demonstrate multistability at this production level."""
    N = len(W)
    production = np.full((days, N), float(P))
    low = simulate_signals(production, W, params, A0=np.zeros(N))[-1]
    high = simulate_signals(production, W, params, A0=np.full(N, (P + params.beta) / params.delta))[-1]
    return {"P": float(P), "A_low_start": float(low.mean()), "A_high_start": float(high.mean()),
            "multistable": bool(abs(high.mean() - low.mean()) > 1e-3)}


def hysteresis_sweep(W, params, P_max, steps=41, days_per_step=150):
    """Quasi-static up and down sweep of uniform production. Returns the mean A on each branch."""
    N = len(W)
    levels = np.linspace(0, P_max, steps)
    rows = []
    A = np.zeros(N)
    for branch, seq in (("up", levels), ("down", levels[::-1])):
        for P in seq:
            A = simulate_signals(np.full((days_per_step, N), P), W, params, A0=A)[-1].astype(float)
            rows.append({"branch": branch, "P": float(P), "mean_A": float(A.mean())})
    return rows
