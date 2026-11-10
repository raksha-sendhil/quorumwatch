"""Facility-level normalisation and leak-free rolling features.

Each facility fits its own expected-count profile from its own history, so the
normalisation needs no data from other sites.
"""
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ["z_current", "rolling_mean_z", "rolling_std_z", "rate_of_change_z",
                 "dow_sin", "dow_cos", "season_sin", "season_cos", "long_window_z"]


def _design(days):
    days = np.asarray(days)
    angle = 2 * np.pi * days / 365.0
    dow = np.eye(7)[days % 7][:, 1:]
    return np.column_stack([np.ones(len(days)), np.sin(angle), np.cos(angle), dow])


def _poisson_irls(X, y, n_iter=25):
    """Log-link Poisson regression by iteratively reweighted least squares."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))
    for _ in range(n_iter):
        mu = np.exp(np.clip(X @ beta, -10, 12))
        step = np.linalg.solve((X * mu[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]),
                               X.T @ (y - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-7:
            break
    return beta


@dataclass
class FacilityProfiles:
    """Per-facility count model: mean mu(day) and variance mu + phi * mu^2."""
    beta: np.ndarray
    phi: np.ndarray

    def expected(self, days):
        return np.exp(_design(days) @ self.beta.T)

    def standardise(self, counts, days, clip=8.0):
        mu = self.expected(days)
        z = (counts - mu) / np.sqrt(mu + self.phi * mu ** 2)
        return np.clip(z, -clip, clip)


def fit_profiles(reported, days):
    """Fit one profile per facility. One robust refit drops gross artefacts."""
    X = _design(days)
    betas, phis = [], []
    for i in range(reported.shape[1]):
        y = reported[:, i]
        ok = ~np.isnan(y)
        beta = _poisson_irls(X[ok], y[ok])
        mu = np.exp(X[ok] @ beta)
        residual = (y[ok] - mu) / np.sqrt(mu)
        keep = np.abs(residual) < 4 * max(np.median(np.abs(residual)) / 0.6745, 1e-6)
        beta = _poisson_irls(X[ok][keep], y[ok][keep])
        mu = np.exp(X[ok][keep] @ beta)
        phi = max(0.0, np.mean(((y[ok][keep] - mu) ** 2 - mu) / mu ** 2))
        betas.append(beta)
        phis.append(phi)
    return FacilityProfiles(np.array(betas), np.array(phis))


def rolling_stats(z, arrival, window):
    """Rolling statistics of each report, using only reports already known when that
    report arrived. All outputs have shape (T, N); values for absent reports are NaN."""
    T, N = z.shape
    pad = window - 1
    z_pad = np.vstack([np.full((pad, N), np.nan), z])
    arr_pad = np.vstack([np.full((pad, N), -1, dtype=arrival.dtype), arrival])
    z_win = np.lib.stride_tricks.sliding_window_view(z_pad, window, axis=0)       # T,N,w
    arr_win = np.lib.stride_tricks.sliding_window_view(arr_pad, window, axis=0)
    known = (arr_win >= 0) & (arr_win <= arrival[:, :, None])
    vals = np.where(known, z_win, 0.0)
    n = known.sum(axis=2)
    safe_n = np.maximum(n, 1)
    mean = vals.sum(axis=2) / safe_n
    var = ((vals - mean[:, :, None]) ** 2 * known).sum(axis=2) / safe_n
    prior_n = np.maximum(n - 1, 1)
    prior_mean = (vals.sum(axis=2) - np.nan_to_num(z)) / prior_n
    rate = np.where(n > 1, np.nan_to_num(z) - prior_mean, 0.0)
    absent = arrival < 0
    mean = np.where(absent, np.nan, mean)
    return {"mean": mean, "std": np.where(absent, np.nan, np.sqrt(var)),
            "rate": np.where(absent, np.nan, rate),
            "n": np.where(absent, 0, n)}


def build_features(z, arrival, days, window, long_window=0):
    """Return X (n_reports, n_features), the (s, i) index of each row, and rolling stats.
    With long_window > 0, adds a long-window z-test that accumulates weak sustained evidence."""
    stats = rolling_stats(z, arrival, window)
    s, i = np.nonzero(arrival >= 0)
    d = np.asarray(days)[s]
    columns = [
        z[s, i], stats["mean"][s, i], stats["std"][s, i], stats["rate"][s, i],
        np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
        np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)]
    if long_window:
        long_stats = rolling_stats(z, arrival, long_window)
        stats["long_z"] = long_stats["mean"] * np.sqrt(np.maximum(long_stats["n"], 1))
        columns.append(stats["long_z"][s, i])
    return np.column_stack(columns), (s, i), stats


def direction_excess(z, stats):
    """Isolation Forest is two-sided. Only excess, never deficit, may produce signal."""
    excess = np.maximum(np.nan_to_num(z, nan=-np.inf), np.nan_to_num(stats["mean"], nan=-np.inf))
    if "long_z" in stats:
        excess = np.maximum(excess, np.nan_to_num(stats["long_z"], nan=-np.inf))
    return excess
