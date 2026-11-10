"""Primary study: Isolation Forest alone versus Isolation Forest + bacterial layer, on held-out test seeds."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data.reporting import count_by_arrival
from experiments.calibration import qw_mask, simulate_full
from experiments.evaluation import attributable_detection_day, bootstrap_metrics, evaluate_mask, paired_difference
from ml import forest_alone
from network.facility_network import n_directed_edges

FULL = "full"
FOREST = "forest_alone"
METHOD_LABELS = {FULL: "Isolation Forest + bacterial layer", FOREST: "Isolation Forest alone"}


def _rows(method, mask_null, mask_out, null_res, out_res, bundle, messages, cfg):
    """One row per run (seed x scenario) for a method."""
    exp = cfg.experiment
    eval_days = exp.eval_days
    chance = _chance(mask_null, bundle, exp)
    attributable = attributable_detection_day(mask_out, mask_null, bundle.t0, bundle.t_end, bundle.cluster_mask,
                                              exp.attribution_tolerance_days)
    alert_days = mask_out.any(axis=2).sum(axis=1)
    alert_size = np.where(alert_days > 0, mask_out.sum(axis=(1, 2)) / np.maximum(alert_days, 1), np.nan)
    cap = bundle.t_end - bundle.t0
    detected = out_res["detection_day"] >= 0
    delay = np.where(detected, out_res["detection_day"] - bundle.t0, np.nan)
    return pd.DataFrame({
        "chance_detection": chance,
        "attributable_detection_rate": (attributable >= 0).astype(float),
        "attributable_restricted_delay_days": np.where(attributable >= 0, attributable - bundle.t0, cap),
        "mean_alert_size": alert_size,
        "seed": bundle.seeds, "scenario": bundle.scenarios, "method": method,
        "detected": detected.astype(int), "detection_rate": detected.astype(float), "recall": detected.astype(float),
        "detection_delay_days": delay, "restricted_delay_days": np.where(detected, delay, cap),
        "false_alerts_null_run": null_res["n_false"], "false_alerts_outbreak_run": out_res["n_false"],
        "false_alerts_per_year": null_res["n_false"] / eval_days * 365.0,
        "tp": detected.astype(int), "fp": null_res["n_false"] + out_res["n_false"],
        "alert_burden_null": null_res["burden"], "messages_per_day": messages,
        "peak_excess": bundle.peak, "outbreak_start_day": bundle.t0,
        "ramp_days": bundle.ramp, "duration_days": bundle.duration, "cluster_size": bundle.cluster_size})


def _chance(mask_null, bundle, exp):
    """Would this run count as detected if the same window and cluster were scored on the
    outbreak-free twin? Measures how often detection credit arises by coincidence."""
    result = evaluate_mask(mask_null, exp.burn_in_days, exp.refractory_days, bundle.t0, bundle.t_end, bundle.cluster_mask)
    return (result["detection_day"] >= 0).astype(float)


def signal_messages(active, world, n_sub):
    """Directed scalar transmissions per day: diffused signal plus state-change bits."""
    exp = world.cfg.experiment
    signal = n_sub * n_directed_edges(world.W)
    degree = (world.W > 0).sum(axis=1).astype(float)
    changes = (active[:, exp.burn_in_days:] != active[:, exp.burn_in_days - 1:-1]).astype(float)
    return signal + (changes @ degree).sum(axis=1) / exp.eval_days


def run_primary(world, bundle, params, theta_on, forest_h):
    """Evaluate both methods. Returns raw rows and arrays for exports and threshold curves."""
    cfg, exp = world.cfg, world.cfg.experiment
    nb = world.neighbourhoods
    tables, keep = [], {}

    A_null, H_null, _ = simulate_full(bundle.null, world.W, params)
    A_out, H_out, n_sub = simulate_full(bundle.outbreak, world.W, params)
    mask_null, _ = qw_mask(H_null, theta_on, exp, nb, params.quorum)
    mask_out, active_out = qw_mask(H_out, theta_on, exp, nb, params.quorum)
    null_res = evaluate_mask(mask_null, exp.burn_in_days, exp.refractory_days)
    out_res = evaluate_mask(mask_out, exp.burn_in_days, exp.refractory_days, bundle.t0, bundle.t_end, bundle.cluster_mask)
    tables.append(_rows(FULL, mask_null, mask_out, null_res, out_res, bundle, signal_messages(active_out, world, n_sub), cfg))
    keep[FULL] = {"H_null": H_null, "H_out": H_out, "A_null": A_null, "A_out": A_out, "mask_out": mask_out,
                  "active_out": active_out, "mask_null": mask_null}

    # Isolation Forest alone: each facility receives its neighbours' scores for its own neighbourhood.
    degree = (world.W > 0).sum(axis=1).astype(float)
    messages = np.stack([count_by_arrival(a)[exp.burn_in_days:].sum(axis=0) @ degree
                         for a in bundle.arrival]) / exp.eval_days
    pooled = {k: forest_alone.neighbourhood_scores(
        forest_alone.site_scores(getattr(bundle, k).score_ml, getattr(bundle, k).excess, bundle.arrival,
                                 world.forest_site_mean, world.forest_site_sd), bundle.arrival, nb)
              for k in ("null", "outbreak")}
    mask_null = forest_alone.neighbourhood_mask(pooled["null"], forest_h, nb)
    mask_out = forest_alone.neighbourhood_mask(pooled["outbreak"], forest_h, nb)
    null_res = evaluate_mask(mask_null, exp.burn_in_days, exp.refractory_days)
    out_res = evaluate_mask(mask_out, exp.burn_in_days, exp.refractory_days, bundle.t0, bundle.t_end, bundle.cluster_mask)
    tables.append(_rows(FOREST, mask_null, mask_out, null_res, out_res, bundle, messages, cfg))
    keep[FOREST] = {"null": pooled["null"], "out": pooled["outbreak"], "mask_out": mask_out}
    return pd.concat(tables, ignore_index=True), keep


def aggregate(raw, cfg):
    """Means and 95% bootstrap CIs per scenario, plus all scenarios pooled by seed."""
    exp = cfg.experiment
    tables = []
    for method, group in raw.groupby("method", sort=False):
        for scenario, frame in list(group.groupby("scenario", sort=False)) + [("all_scenarios", group)]:
            stats = bootstrap_metrics(frame, exp.n_bootstrap, exp.bootstrap_seed)
            stats.insert(0, "method", method)
            stats.insert(0, "scenario", scenario)
            tables.append(stats)
    return pd.concat(tables, ignore_index=True)


def paired_comparisons(raw, cfg):
    """Differences on identical runs: Isolation Forest + bacterial layer minus Isolation Forest alone."""
    exp = cfg.experiment
    metrics = ["detection_rate", "restricted_delay_days", "attributable_detection_rate",
               "attributable_restricted_delay_days", "false_alerts_per_year", "alert_burden_null", "chance_detection"]
    rows = []
    for scenario in list(raw.scenario.unique()) + ["all_scenarios"]:
        frame = raw if scenario == "all_scenarios" else raw[raw.scenario == scenario]
        order = ["scenario", "seed"]
        left = frame[frame.method == FULL].sort_values(order)
        right = frame[frame.method == FOREST].sort_values(order)
        for metric in metrics:
            rows.append({"method": FULL, "versus": FOREST, "scenario": scenario, "metric": metric,
                         **paired_difference(left, right, metric, exp.n_bootstrap, exp.bootstrap_seed)})
    return pd.DataFrame(rows)


def tradeoff_curves(world, bundle, keep, params, n_points=36):
    """False alerts versus detection across thresholds on test runs. Used only to describe
    behaviour, never to set a parameter. It compares methods at equal false-alert rates in each
    scenario, removing differences caused by using a single pooled threshold.

    Below some threshold, alerts stay on continuously, so onsets (and apparent false alerts) fall
    again. Only thresholds at or above the false-alert peak are used for matching.
    Returns (curves, matched comparison with bootstrap CIs)."""
    exp = world.cfg.experiment
    cap = bundle.t_end - bundle.t0
    per_year = 365.0 / exp.eval_days
    store = {}

    def record(method, threshold, mask_null, mask_out):
        null_res = evaluate_mask(mask_null, exp.burn_in_days, exp.refractory_days)
        out_res = evaluate_mask(mask_out, exp.burn_in_days, exp.refractory_days,
                                bundle.t0, bundle.t_end, bundle.cluster_mask)
        attr = attributable_detection_day(mask_out, mask_null, bundle.t0, bundle.t_end, bundle.cluster_mask,
                                          exp.attribution_tolerance_days)
        detected = out_res["detection_day"] >= 0
        store.setdefault(method, []).append((
            float(threshold), null_res["n_false"] * per_year, detected.astype(float),
            np.where(detected, out_res["detection_day"] - bundle.t0, cap).astype(float),
            (attr >= 0).astype(float), np.where(attr >= 0, attr - bundle.t0, cap).astype(float)))

    H_null, H_out = keep[FULL]["H_null"], keep[FULL]["H_out"]
    sample = H_null[:, exp.burn_in_days:, :].ravel()[::5]
    for theta in np.unique(np.quantile(sample, 1 - np.geomspace(0.5, 1e-5, n_points))):
        record(FULL, theta, qw_mask(H_null, theta, exp, world.neighbourhoods, params.quorum)[0],
               qw_mask(H_out, theta, exp, world.neighbourhoods, params.quorum)[0])
    scores_null, scores_out = keep[FOREST]["null"], keep[FOREST]["out"]
    sample = scores_null[:, exp.burn_in_days:]
    sample = sample[np.isfinite(sample)]
    for h in np.unique(np.quantile(sample, 1 - np.geomspace(0.5, 1e-6, n_points))):
        record(FOREST, h, forest_alone.neighbourhood_mask(scores_null, h, world.neighbourhoods),
               forest_alone.neighbourhood_mask(scores_out, h, world.neighbourhoods))

    rng = np.random.default_rng(exp.bootstrap_seed)
    names = ["detection_rate", "restricted_delay_days", "attributable_detection_rate", "attributable_restricted_delay_days"]
    curves, matched = [], []
    scenarios = list(pd.unique(bundle.scenarios)) + ["all_scenarios"]

    def at_rate(rate, fa, values):
        order = np.argsort(fa, kind="stable")
        return np.interp(rate, fa[order], values[order])

    for method, entries in store.items():
        thresholds = np.array([e[0] for e in entries])
        F = np.stack([e[1] for e in entries])
        V = [np.stack([e[i] for e in entries]) for i in (2, 3, 4, 5)]
        for scenario in scenarios:
            idx = np.arange(len(bundle.seeds)) if scenario == "all_scenarios" else np.nonzero(bundle.scenarios == scenario)[0]
            fa = F[:, idx].mean(axis=1)
            start = int(np.argmax(fa))                     # usable branch: thresholds above the peak
            curves.append(pd.DataFrame({"method": method, "scenario": scenario, "threshold": thresholds,
                                        "usable_for_matching": np.arange(len(fa)) >= start, "false_alerts_per_year": fa,
                                        **{n: v[:, idx].mean(axis=1) for n, v in zip(names, V)}}))
            draws = [rng.choice(idx, len(idx)) for _ in range(exp.n_bootstrap // 4)]
            for rate in exp.matched_false_alert_rates:
                row = {"method": method, "scenario": scenario, "false_alerts_per_year": rate,
                       "lowest_rate_reached": float(fa[start:].min())}
                for name, values in zip(names, V):
                    point = at_rate(rate, fa[start:], values[start:, idx].mean(axis=1))
                    boot = [at_rate(rate, F[start:, d].mean(axis=1), values[start:, d].mean(axis=1)) for d in draws]
                    low, high = np.percentile(boot, [2.5, 97.5])
                    row.update({name: point, f"{name}_ci_low": low, f"{name}_ci_high": high})
                matched.append(row)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(matched)


def by_outbreak_type(raw):
    """Results for each value of each outbreak property."""
    data = raw.copy()
    data["peak_band"] = pd.cut(data.peak_excess, [0.15, 0.30, 0.45, 0.60, 0.75], include_lowest=True).astype(str)
    tables = []
    for dimension in ("ramp_days", "duration_days", "cluster_size", "peak_band"):
        table = (data.groupby(["method", dimension], sort=True)
                 .agg(n_runs=("detected", "size"), detection_rate=("detection_rate", "mean"),
                      attributable_detection_rate=("attributable_detection_rate", "mean"),
                      chance_detection=("chance_detection", "mean"),
                      attributable_restricted_delay_days=("attributable_restricted_delay_days", "mean")).reset_index())
        tables.append(table.rename(columns={dimension: "value"}).assign(dimension=dimension))
    return pd.concat(tables, ignore_index=True)


def outbreak_diagnostic(world, bundle):
    """How strong are outbreaks at a single site? And how well do forest scores separate
    plateau outbreak reports from the paired null reports?"""
    plateau = np.zeros(bundle.arrival.shape, bool)
    for r in range(len(bundle.seeds)):
        plateau[r, bundle.t0[r] + bundle.ramp[r]:bundle.t0[r] + bundle.duration[r]] = bundle.cluster_mask[r]
    z_out, z_null = bundle.outbreak.z[plateau], bundle.null.z[plateau]
    present = ~np.isnan(z_out) & ~np.isnan(z_null)
    z_out, z_null = z_out[present], z_null[present]
    ml_out, ml_null = bundle.outbreak.score_ml[plateau][present], bundle.null.score_ml[plateau][present]
    labels = np.r_[np.ones(len(ml_out)), np.zeros(len(ml_null))]
    pooled = bundle.outbreak.z[bundle.arrival >= 0]
    return pd.DataFrame([{
        "plateau_report_count": int(len(z_out)),
        "median_z_outbreak_plateau": float(np.median(z_out)),
        "median_z_null_same_reports": float(np.median(z_null)),
        "share_plateau_reports_z_ge_3": float((z_out >= 3).mean()),
        "share_null_reports_z_ge_3": float((z_null >= 3).mean()),
        "share_all_reports_z_ge_3": float((pooled >= 3).mean()),
        "local_auc_isolation_forest": float(roc_auc_score(labels, np.r_[ml_out, ml_null])),
        "volume_vs_null_production_spearman": float(spearmanr(
            world.facilities.baseline_daily_count, bundle.null.production_ml.mean(axis=(0, 1))).statistic),
        "mean_null_production_ml": float(bundle.null.production_ml.mean()),
        "mean_plateau_production_ml": float(bundle.outbreak.production_ml[plateau].mean()),
    }])


def quenching_diagnostic(bundle, cfg):
    """Does the reliability score actually separate unreliable from reliable facilities?"""
    burn = cfg.experiment.burn_in_days
    rows = []
    for scenario in pd.unique(bundle.scenarios):
        idx = bundle.scenarios == scenario
        rel = bundle.null.reliability[idx][:, burn:].mean(axis=1)
        unreliable = bundle.unreliable[idx]
        rows.append({"scenario": scenario,
                     "mean_reliability_reliable_sites": float(rel[~unreliable].mean()),
                     "mean_reliability_unreliable_sites": float(rel[unreliable].mean()) if unreliable.any() else np.nan})
    return pd.DataFrame(rows)
