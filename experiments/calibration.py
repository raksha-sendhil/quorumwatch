"""Validation-only parameter selection: a random search, then one-at-a-time refinement.

For each candidate, the activation threshold is set to meet the false-alert target on
validation null runs. Candidates are then ranked by restricted detection delay on validation
outbreak runs. No test seed is used here.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from config import SignalParams
from experiments.evaluation import evaluate_mask
from ml import forest_alone
from quorum.hill_activation import hill
from quorum.hysteresis import apply_hysteresis, quorum_alert
from quorum.signal_model import equilibria, n_stable, simulate_signals, stable_substeps


def simulate_full(part, W, params):
    """Return (A, H, substeps per day) for reliability-weighted forest signal production."""
    n_sub = stable_substeps(W, params)
    A = simulate_signals(part.production_ml * part.weight, W, params, n_sub=n_sub)
    return A, hill(A, params.K, params.n).astype(np.float32), n_sub


def qw_mask(H, theta_on, exp, neighbourhoods, quorum):
    active = apply_hysteresis(H, theta_on, exp.theta_off_ratio * theta_on)
    return quorum_alert(active, neighbourhoods, quorum, exp.min_quorum_count), active


def false_rate(result, exp):
    return result["n_false"].sum() / (len(result["n_false"]) * exp.eval_days) * 365.0


def theta_grid(H_null, exp, n_grid):
    sample = H_null[:, exp.burn_in_days:, :].ravel()[::3]
    levels = 1 - np.geomspace(0.5, 2e-5, n_grid)
    return np.unique(np.quantile(sample, levels))


def calibrate_theta(H_null, exp, neighbourhoods, quorum, n_grid=64):
    """Smallest threshold meeting the false-alert and alert-burden targets on null runs,
    found by bisection on a fine grid (false alerts fall as the threshold rises)."""
    grid = theta_grid(H_null, exp, n_grid)

    def check(theta):
        mask, _ = qw_mask(H_null, theta, exp, neighbourhoods, quorum)
        res = evaluate_mask(mask, exp.burn_in_days, exp.refractory_days)
        rate, burden = false_rate(res, exp), res["burden"].mean()
        ok = rate <= exp.target_false_alerts_per_year and burden <= exp.max_alert_burden
        return {"theta_on": float(theta), "val_false_per_year": rate, "val_burden": burden, "feasible": ok}

    best = None
    low, high = 0, len(grid) - 1
    while low <= high:
        mid = (low + high) // 2
        result = check(grid[mid])
        if result["feasible"]:
            best, high = result, mid - 1
        else:
            low = mid + 1
    return best if best is not None else check(grid[-1])


def outbreak_summary(result, bundle, duration_cap):
    detected = result["detection_day"] >= 0
    delays = np.where(detected, result["detection_day"] - bundle.t0, bundle.t_end - bundle.t0)
    return {"val_detection_rate": detected.mean(), "val_restricted_delay": delays.mean(),
            "val_delay_detected": delays[detected].mean() if detected.any() else np.nan}


def evaluate_candidate(params, world, bundle, n_grid=64):
    exp = world.cfg.experiment
    cap = world.cfg.outbreak.duration_days + world.cfg.outbreak.grace_days
    _, H_null, n_sub = simulate_full(bundle.null, world.W, params)
    cal = calibrate_theta(H_null, exp, world.neighbourhoods, params.quorum, n_grid)
    _, H_out, _ = simulate_full(bundle.outbreak, world.W, params)
    mask, _ = qw_mask(H_out, cal["theta_on"], exp, world.neighbourhoods, params.quorum)
    res = evaluate_mask(mask, exp.burn_in_days, exp.refractory_days, bundle.t0, bundle.t_end, bundle.cluster_mask)
    return {"D": params.D, "delta": params.delta, "beta": params.beta, "K": params.K, "quorum": params.quorum,
            "substeps_per_day": n_sub, "single_node_stable_states_P0": n_stable(0.0, params), **cal,
            **outbreak_summary(res, bundle, cap)}


def _rank(frame):
    return frame.sort_values(["feasible", "val_restricted_delay", "val_detection_rate"],
                             ascending=[False, True, False])


def sweep_parameters(world, bundle, verbose=print):
    """Lightweight search. Returns (all candidates, selected SignalParams)."""
    exp = world.cfg.experiment
    rng = np.random.default_rng(exp.sweep_seed)
    grids = {"D": exp.grid_D, "delta": exp.grid_delta, "beta": exp.grid_beta, "K": exp.grid_K,
             "quorum": exp.grid_quorum}
    seen, rows = set(), []

    def consider(params, phase, varied):
        key = (params.D, params.delta, params.beta, params.K, params.quorum)
        if key in seen:
            return
        seen.add(key)
        rows.append({"phase": phase, "varied_parameter": varied, **evaluate_candidate(params, world, bundle)})

    consider(SignalParams(), "default", "none")
    for _ in range(exp.n_random_configs):
        consider(SignalParams(**{k: float(rng.choice(g)) for k, g in grids.items()}), "random", "all")
    verbose(f"  random search done: {len(rows)} candidates")

    best = _rank(pd.DataFrame(rows)).iloc[0]
    centre = SignalParams(**{k: float(best[k]) for k in grids})
    for name, grid in grids.items():
        for value in grid:
            consider(replace(centre, **{name: float(value)}), "one_at_a_time", name)
    frame = pd.DataFrame(rows)
    selected = _rank(frame).iloc[0]
    params = SignalParams(**{k: float(selected[k]) for k in grids})
    frame["selected"] = (frame[list(grids)] == selected[list(grids)].astype(float)).all(axis=1)
    frame.attrs["oat_centre"] = {k: float(best[k]) for k in grids}
    verbose(f"  one-at-a-time done: {len(frame)} candidates; selected {params}")
    return frame, params


def calibrate_forest_alone(world, bundle, n_grid=80):
    """Lowest forest-alone threshold meeting the same targets on validation null runs."""
    exp = world.cfg.experiment
    nb = world.neighbourhoods
    site = forest_alone.site_scores(bundle.null.score_ml, bundle.null.excess, bundle.arrival,
                                    world.forest_site_mean, world.forest_site_sd)
    pooled = forest_alone.neighbourhood_scores(site, bundle.arrival, nb)
    sample = pooled[:, exp.burn_in_days:]
    sample = sample[np.isfinite(sample)]
    for h in np.unique(np.quantile(sample, 1 - np.geomspace(0.2, 1e-6, n_grid))):
        res = evaluate_mask(forest_alone.neighbourhood_mask(pooled, h, nb), exp.burn_in_days, exp.refractory_days)
        rate, burden = false_rate(res, exp), res["burden"].mean()
        feasible = rate <= exp.target_false_alerts_per_year and burden <= exp.max_alert_burden
        if feasible:
            break
    return {"h": float(h), "val_false_per_year": rate, "val_burden": burden, "feasible": bool(feasible)}


def calibrate_full(world, bundle, params):
    """Fine activation-threshold calibration at the selected parameters."""
    exp = world.cfg.experiment
    _, H_null, _ = simulate_full(bundle.null, world.W, params)
    return calibrate_theta(H_null, exp, world.neighbourhoods, params.quorum, n_grid=256)


def bistability_report(world, params, null_P, outbreak_P):
    """Single-node equilibria at representative production levels."""
    rows = []
    for label, P in (("zero_production", 0.0), ("mean_null_production", null_P), ("mean_outbreak_production", outbreak_P)):
        eq = equilibria(P, params)
        rows.append({"production_level": label, "P": P, "n_equilibria": len(eq),
                     "n_stable": sum(e["stable"] for e in eq),
                     "A_stable": ";".join(f"{e['A']:.4f}" for e in eq if e["stable"]),
                     "A_unstable": ";".join(f"{e['A']:.4f}" for e in eq if not e["stable"])})
    return rows
