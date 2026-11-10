"""Builds the fixed world (facilities, network, trained detector) and paired evaluation runs."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import spearmanr

from data.generator import inject_outbreak, make_facilities, select_cluster, simulate_rates
from data.reporting import apply_reporting, draw_reporting, make_profile, sum_by_arrival
from ml.feature_engineering import build_features, direction_excess, fit_profiles
from ml.isolation_forest import ScoreCalibrator, fit_forest, raw_scores
from network.facility_network import build_weights, closed_neighbourhood
from quorum.reliability import compute_reliability, quench_weight


@dataclass
class World:
    cfg: object
    facilities: object
    W: np.ndarray
    dist: np.ndarray
    neighbourhoods: np.ndarray
    profiles: object
    forest: object
    calibrator_ml: ScoreCalibrator
    forest_site_mean: np.ndarray
    forest_site_sd: np.ndarray
    history: dict
    model_card: dict


def _archive(baseline, nominal, days, rng, cfg):
    """Normal data with only the sites' nominal reporting problems and no outbreak."""
    T = len(days)
    profile = make_profile(nominal, 0.0, rng, cfg.reporting)
    rates = simulate_rates(baseline, days, rng, cfg.world)
    counts = rng.poisson(rates).astype(float)
    arrival, factor = draw_reporting(profile, T, rng, cfg.reporting)
    return counts, apply_reporting(counts, factor, arrival), arrival


def build_world(cfg):
    w = cfg.world
    facilities = make_facilities(w)
    coords = facilities[["x_km", "y_km"]].to_numpy()
    baseline = facilities["baseline_daily_count"].to_numpy()
    nominal = facilities["nominal_reliability"].to_numpy()
    W, dist = build_weights(coords, w.knn, w.length_scale_km)
    neighbourhoods = closed_neighbourhood(W)

    # Training period: facility profiles and the pooled forest. Outbreaks are never injected.
    rng = np.random.default_rng([w.seed, 1])
    days = np.arange(w.history_days)
    counts, reported, arrival = _archive(baseline, nominal, days, rng, cfg)
    profiles = fit_profiles(reported, days)
    z = profiles.standardise(reported, days, cfg.features.z_clip)
    X_train, _, _ = build_features(z, arrival, days, cfg.features.window, cfg.features.long_window)
    forest = fit_forest(X_train, cfg.forest)

    # Separate normal period: score calibration only.
    rng = np.random.default_rng([w.seed, 2])
    cal_days = w.history_days + np.arange(w.score_calibration_days)
    _, cal_reported, cal_arrival = _archive(baseline, nominal, cal_days, rng, cfg)
    cal_z = profiles.standardise(cal_reported, cal_days, cfg.features.z_clip)
    X_cal, (s, i), stats = build_features(cal_z, cal_arrival, cal_days, cfg.features.window, cfg.features.long_window)
    excess = direction_excess(cal_z, stats)[s, i]
    tail = cfg.features.production_tail
    cal_raw = raw_scores(forest, X_cal)
    calibrator_ml = ScoreCalibrator.fit(cal_raw, excess, i, w.n_facilities, tail)
    site_mean_score = np.bincount(i, cal_raw, w.n_facilities) / np.bincount(i, minlength=w.n_facilities)
    site_sd_score = np.array([max(cal_raw[i == k].std(), 1e-6) for k in range(w.n_facilities)])

    model_card = {
        "training_rows": int(len(X_train)),
        "calibration_rows": int(len(X_cal)),
        "training_days": int(w.history_days),
        "calibration_days": int(w.score_calibration_days),
        "training_contains_outbreaks": False,
        "z_mean_training": float(np.nanmean(z)),
        "z_std_training": float(np.nanstd(z)),
        "volume_vs_z_std_correlation": float(np.corrcoef(baseline, np.nanstd(z, axis=0))[0, 1]),
        "volume_vs_mean_raw_score_spearman": float(spearmanr(baseline, site_mean_score).statistic),
    }
    history = {"days": days, "counts": counts, "reported": reported, "arrival": arrival}
    return World(cfg, facilities, W, dist, neighbourhoods, profiles, forest, calibrator_ml,
                 site_mean_score, site_sd_score, history, model_card)


def generate_run(world, run_seed, scenario_idx, unreliable_fraction):
    """One paired run. Counts and outbreak depend only on the seed, so reliability scenarios
    are also paired. The null version uses exactly the same noise and reporting draws."""
    cfg = world.cfg
    base = cfg.world.seed
    exp, ob = cfg.experiment, cfg.outbreak
    baseline = world.facilities["baseline_daily_count"].to_numpy()
    nominal = world.facilities["nominal_reliability"].to_numpy()
    T, N = exp.n_days, cfg.world.n_facilities

    rng_count = np.random.default_rng([base, run_seed, 10])
    start = cfg.world.history_days + cfg.world.score_calibration_days + int(rng_count.integers(365))
    days = start + np.arange(T)
    rates = simulate_rates(baseline, days, rng_count, cfg.world)
    null_counts = rng_count.poisson(rates).astype(float)

    rng_ob = np.random.default_rng([base, run_seed, 11])
    cluster_seed = rng_ob.integers(N)
    t0 = exp.burn_in_days + int(rng_ob.integers(ob.start_min, ob.start_max + 1))
    peak = float(rng_ob.uniform(ob.peak_excess_low, ob.peak_excess_high))
    size = ob.cluster_size
    if ob.general:
        ob = replace(ob, ramp_days=int(rng_ob.choice(ob.general_ramp_days)),
                     duration_days=int(rng_ob.choice(ob.general_duration_days)))
        size = int(rng_ob.choice(ob.general_cluster_sizes))
        peak = float(rng_ob.uniform(ob.general_peak_low, ob.general_peak_high))
    cluster = select_cluster(cluster_seed, world.W, size)
    outbreak_counts = inject_outbreak(null_counts, rates, cluster, t0, peak, rng_ob, ob)

    rng_rep = np.random.default_rng([base, run_seed, 12, scenario_idx])
    profile = make_profile(nominal, unreliable_fraction, rng_rep, cfg.reporting)
    arrival, factor = draw_reporting(profile, T, rng_rep, cfg.reporting)
    return {
        "seed": run_seed, "days": days, "arrival": arrival, "unreliable": profile.unreliable,
        "cluster": cluster, "t0": t0, "t_end": t0 + ob.duration_days + ob.grace_days, "peak": peak,
        "ramp": ob.ramp_days, "duration": ob.duration_days, "size": size,
        "null_counts": null_counts, "outbreak_counts": outbreak_counts,
        "null_reported": apply_reporting(null_counts, factor, arrival),
        "outbreak_reported": apply_reporting(outbreak_counts, factor, arrival),
    }


@dataclass
class Part:
    """Arrays with shape (R, T, N), for either the null or the outbreak version of the runs."""
    reported: np.ndarray
    z: np.ndarray
    score_ml: np.ndarray
    production_ml: np.ndarray
    reliability: np.ndarray
    weight: np.ndarray
    excess: np.ndarray


@dataclass
class Bundle:
    seeds: np.ndarray
    scenarios: np.ndarray
    days: np.ndarray
    arrival: np.ndarray
    unreliable: np.ndarray
    cluster_mask: np.ndarray
    t0: np.ndarray
    t_end: np.ndarray
    peak: np.ndarray
    ramp: np.ndarray
    duration: np.ndarray
    cluster_size: np.ndarray
    true_counts: np.ndarray
    null: Part
    outbreak: Part


def _process(world, runs, key):
    """Convert received reports to all detector inputs. The forest scores one large batch."""
    cfg = world.cfg
    fc = cfg.features
    prepared, blocks = [], []
    for run in runs:
        z = world.profiles.standardise(run[f"{key}_reported"], run["days"], fc.z_clip)
        X, idx, stats = build_features(z, run["arrival"], run["days"], fc.window, fc.long_window)
        prepared.append((z, idx, direction_excess(z, stats)))
        blocks.append(X)
    raw = raw_scores(world.forest, np.vstack(blocks))
    offsets = np.cumsum([0] + [len(b) for b in blocks])

    fields = {name: [] for name in Part.__dataclass_fields__}
    for r, run in enumerate(runs):
        z, (s, i), excess = prepared[r]
        arrival = run["arrival"]
        score_ml = np.full(z.shape, np.nan)
        score_ml[s, i] = raw[offsets[r]:offsets[r + 1]]
        p_ml = np.zeros(z.shape)
        p_ml[s, i] = world.calibrator_ml.production(score_ml[s, i], excess[s, i], i)
        reliability = compute_reliability(z, arrival, cfg.reliability, world.neighbourhoods)
        for name, value in (("excess", excess), ("reported", run[f"{key}_reported"]), ("z", z), ("score_ml", score_ml),
                            ("production_ml", sum_by_arrival(p_ml, arrival)), ("reliability", reliability),
                            ("weight", quench_weight(reliability, cfg.reliability))):
            fields[name].append(value.astype(np.float32))
    return Part(**{name: np.stack(v) for name, v in fields.items()})


def build_bundle(world, seed_base, n_seeds):
    """Generate seeds x scenarios paired runs and compute every detector's inputs."""
    cfg = world.cfg
    runs, labels = [], []
    for k, (name, fraction) in enumerate(cfg.experiment.scenarios.items()):
        for offset in range(n_seeds):
            runs.append(generate_run(world, seed_base + offset, k, fraction))
            labels.append(name)
    N = cfg.world.n_facilities
    cluster_mask = np.zeros((len(runs), N), bool)
    for r, run in enumerate(runs):
        cluster_mask[r, run["cluster"]] = True
    return Bundle(
        seeds=np.array([r["seed"] for r in runs]), scenarios=np.array(labels),
        days=np.stack([r["days"] for r in runs]), arrival=np.stack([r["arrival"] for r in runs]),
        unreliable=np.stack([r["unreliable"] for r in runs]), cluster_mask=cluster_mask,
        t0=np.array([r["t0"] for r in runs]), t_end=np.array([r["t_end"] for r in runs]),
        peak=np.array([r["peak"] for r in runs]), ramp=np.array([r["ramp"] for r in runs]),
        duration=np.array([r["duration"] for r in runs]), cluster_size=np.array([r["size"] for r in runs]),
        true_counts=np.stack([r["outbreak_counts"] for r in runs]).astype(np.float32),
        null=_process(world, runs, "null"), outbreak=_process(world, runs, "outbreak"))
