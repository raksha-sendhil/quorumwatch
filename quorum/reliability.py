"""Adaptive quorum quenching: each facility tracks a local reliability score.

r_i(t) = timeliness_i(t) * (1 - min(1, penalty * outlier_rate_i(t)))

Timeliness is an EWMA of on-time reporting, with part credit for late reports.
Outlier rate is an EWMA of implausible reports and measures historical consistency.

A report counts as implausible only when it is very large (|z| > outlier_z) and nothing backs it
up: neither the site's neighbours (mean z of the neighbour reports already received for that day)
nor the site's own previous week (prior rolling mean z). An isolated spike is treated as a likely
reporting error; a rise shared with neighbours, or sustained over time, is treated as real.

Signal production is multiplied by r ** gamma, a simplified analogue of quorum-quenching enzymes
that degrade signal. Each site uses only its own reports and its neighbours' z-scores.
"""
import numpy as np

from data.reporting import count_by_arrival, sum_by_arrival
from ml.feature_engineering import rolling_stats


def implausible_reports(z, arrival, cfg, neighbourhoods=None):
    """Boolean (T, N): very large reports that neither neighbours nor the site's recent past support."""
    T, N = z.shape
    present = (arrival >= 0) & ~np.isnan(z)
    zf = np.where(present, z, 0.0)
    large = present & (np.abs(zf) > cfg.outlier_z)

    stats = rolling_stats(z, arrival, 7)
    n = stats["n"]
    prior_mean = np.where(n > 1, (np.nan_to_num(stats["mean"]) * n - zf) / np.maximum(n - 1, 1), 0.0)
    supported = prior_mean >= cfg.sustained_z

    if neighbourhoods is not None:
        others = neighbourhoods & ~np.eye(N, dtype=bool)
        # Neighbour j's report for the same day counts if it had arrived by the time i's report did.
        known = present[:, None, :] & (arrival[:, None, :] <= arrival[:, :, None]) & others[None]
        count = known.sum(axis=2)
        mean = (known * zf[:, None, :]).sum(axis=2) / np.maximum(count, 1)
        supported |= (count > 0) & (mean >= cfg.corroborate_z)
    return large & ~supported


def compute_reliability(z, arrival, cfg, neighbourhoods=None):
    """Online reliability, shape (T, N). r[t] uses only reports that arrived by day t."""
    T, N = arrival.shape
    days = np.arange(T)[:, None]
    on_time = arrival == days
    arrivals = count_by_arrival(arrival)
    late = arrivals - on_time
    outliers_arrived = sum_by_arrival(implausible_reports(z, arrival, cfg, neighbourhoods).astype(float), arrival)

    timeliness = np.full(N, cfg.prior)
    outlier_rate = np.zeros(N)
    out = np.zeros((T, N))
    for t in range(T):
        event = np.clip(on_time[t] + cfg.late_credit * late[t], 0, 1)
        timeliness += cfg.ewma_alpha * (event - timeliness)
        seen = arrivals[t] > 0
        rate = outliers_arrived[t][seen] / arrivals[t][seen]
        outlier_rate[seen] += cfg.ewma_alpha * (rate - outlier_rate[seen])
        r = timeliness * (1 - np.minimum(1.0, cfg.outlier_penalty * outlier_rate))
        out[t] = np.clip(r, cfg.min_reliability, 1.0)
    return out


def quench_weight(reliability, cfg):
    return reliability ** cfg.quench_gamma
