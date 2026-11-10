"""Synthetic surveillance data: facilities, Poisson counts, weak outbreak injection."""
import numpy as np
import pandas as pd


def make_facilities(cfg):
    """Facility table with coordinates, volume and nominal reporting reliability."""
    rng = np.random.default_rng(cfg.seed)
    coords = []
    while len(coords) < cfg.n_facilities:
        c = rng.uniform(0, cfg.region_km, 2)
        if all(np.hypot(*(c - o)) >= cfg.min_separation_km for o in coords):
            coords.append(c)
    coords = np.array(coords)
    baseline = rng.lognormal(np.log(cfg.baseline_median), cfg.baseline_log_sigma, cfg.n_facilities)
    return pd.DataFrame({
        "facility_id": [f"F{i:02d}" for i in range(cfg.n_facilities)],
        "x_km": coords[:, 0],
        "y_km": coords[:, 1],
        "baseline_daily_count": np.clip(baseline, cfg.baseline_min, cfg.baseline_max),
        "nominal_reliability": np.clip(rng.beta(30, 1.5, cfg.n_facilities), 0.85, 0.995),
    })


def expected_mean(baseline, days, cfg):
    """True noiseless mean, shape (T, N): volume x season x day of week."""
    days = np.asarray(days)
    season = 1 + cfg.season_amplitude * np.cos(2 * np.pi * (days - cfg.season_peak_day) / 365.0)
    dow = np.asarray(cfg.dow_factors)[days % 7]
    return np.outer(season * dow, baseline)


def simulate_rates(baseline, days, rng, cfg):
    """Poisson rates with facility noise and a shared AR(1) regional fluctuation."""
    mean = expected_mean(baseline, days, cfg)
    T, N = mean.shape
    innovation = cfg.regional_noise_sigma * np.sqrt(1 - cfg.regional_noise_rho ** 2)
    regional = np.zeros(T)
    regional[0] = rng.normal(0, cfg.regional_noise_sigma)
    for t in range(1, T):
        regional[t] = cfg.regional_noise_rho * regional[t - 1] + rng.normal(0, innovation)
    local = rng.normal(0, cfg.facility_noise_sigma, (T, N))
    return mean * np.exp(regional[:, None] + local)


def select_cluster(seed_idx, W, size):
    """Grow a connected cluster from a seed by repeatedly adding the strongest link."""
    cluster = [int(seed_idx)]
    while len(cluster) < size:
        link = W[cluster].max(axis=0).copy()
        link[cluster] = 0
        if link.max() <= 0:
            break
        cluster.append(int(link.argmax()))
    return np.array(sorted(cluster))


def outbreak_profile(T, t0, cfg):
    """Gradual relative growth g(t): 0 before t0, linear ramp, then plateau until end."""
    g = np.zeros(T)
    days = np.arange(T)
    active = (days >= t0) & (days < t0 + cfg.duration_days)
    g[active] = np.minimum(1.0, (days[active] - t0 + 1) / cfg.ramp_days)
    return g


def inject_outbreak(counts, rates, cluster, t0, peak, rng, cfg):
    """Add Poisson excess cases on top of the same null counts (common random numbers)."""
    g = outbreak_profile(len(counts), t0, cfg)
    out = counts.copy()
    excess = rates[:, cluster] * peak * g[:, None]
    out[:, cluster] += rng.poisson(excess)
    return out
