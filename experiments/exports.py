"""Writes the data files and JSON that a later React frontend can consume."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, cls=_Encoder, allow_nan=False), encoding="utf8")


def _clean(array, digits=4):
    rounded = np.round(np.asarray(array, float), digits)
    return [[None if np.isnan(v) else float(v) for v in row] for row in rounded]


def export_network(world, data_dir):
    facilities = world.facilities.copy()
    W = world.W
    facilities["degree"] = (W > 0).sum(axis=1)
    facilities["weighted_degree"] = W.sum(axis=1)
    facilities.to_csv(data_dir / "facility_network_metadata.csv", index=False)
    ids = facilities.facility_id.tolist()
    pd.DataFrame(W, index=ids, columns=ids).to_csv(data_dir / "adjacency_matrix.csv")
    i, j = np.nonzero(np.triu(W > 0))
    edges = pd.DataFrame({"source": [ids[k] for k in i], "target": [ids[k] for k in j],
                          "weight": W[i, j], "distance_km": world.dist[i, j]})
    edges.to_csv(data_dir / "network_edges.csv", index=False)
    write_json({"facilities": facilities.to_dict("records"), "edges": edges.to_dict("records")},
               data_dir / "network.json")


def export_history(world, data_dir):
    h = world.history
    ids = world.facilities.facility_id.to_numpy()
    T, N = h["counts"].shape
    pd.DataFrame({
        "facility_id": np.tile(ids, T), "day": np.repeat(h["days"], N),
        "true_count": h["counts"].ravel().astype(int), "reported_count": h["reported"].ravel(),
        "arrival_day": h["arrival"].ravel()}).to_csv(data_dir / "historical_surveillance_normal.csv", index=False)


def export_showcase(world, bundle, keep, params, theta_on, forest_h, data_dir):
    """The first test seed in each scenario, chosen in advance and not for how it looks.
    Long-format CSVs are concatenated across scenarios; each scenario also gets a state JSON."""
    ids = world.facilities.facility_id.to_numpy()
    T, N = bundle.arrival.shape[1:]
    frames = {k: [] for k in ("surveillance", "scores", "signals", "activation", "alerts")}
    for scenario in pd.unique(bundle.scenarios):
        r = int(np.nonzero(bundle.scenarios == scenario)[0][0])
        o = bundle.outbreak
        A, H = keep["full"]["A_out"][r], keep["full"]["H_out"][r]
        alert, active = keep["full"]["mask_out"][r], keep["full"]["active_out"][r]
        forest = keep["forest_alone"]["mask_out"][r]
        day = np.repeat(np.arange(T), N)
        base = {"scenario": scenario, "seed": bundle.seeds[r], "facility_id": np.tile(ids, T), "day": day}
        frames["surveillance"].append(pd.DataFrame({
            **base, "true_count": bundle.true_counts[r].ravel().astype(int), "reported_count": o.reported[r].ravel(),
            "arrival_day": bundle.arrival[r].ravel(), "in_outbreak_cluster": np.tile(bundle.cluster_mask[r], T),
            "facility_unreliable": np.tile(bundle.unreliable[r], T)}))
        frames["scores"].append(pd.DataFrame({
            **base, "z": o.z[r].ravel(), "isolation_forest_score": o.score_ml[r].ravel(),
            "signal_production_unquenched": o.production_ml[r].ravel(),
            "reliability": o.reliability[r].ravel(),
            "signal_production": (o.production_ml[r] * o.weight[r]).ravel()}))
        frames["signals"].append(pd.DataFrame({**base, "signal_concentration": A.ravel()}))
        frames["activation"].append(pd.DataFrame({
            **base, "hill_activation": H.ravel(), "active": active.ravel().astype(int),
            "regional_alert_declared": alert.ravel().astype(int)}))
        for name, mask in (("full", alert), ("forest_alone", forest)):
            frames["alerts"].append(pd.DataFrame({
                "scenario": scenario, "seed": bundle.seeds[r], "method": name, "day": np.arange(T),
                "alert": mask.any(axis=1).astype(int), "n_facilities_in_alert": mask.sum(axis=1),
                "outbreak_active": ((np.arange(T) >= bundle.t0[r]) & (np.arange(T) < bundle.t_end[r])).astype(int)}))
        write_json({
            "scenario": scenario, "seed": bundle.seeds[r], "days": np.arange(T),
            "facility_ids": ids, "burn_in_days": world.cfg.experiment.burn_in_days,
            "outbreak": {"start_day": bundle.t0[r], "end_day_with_grace": bundle.t_end[r], "peak_excess": bundle.peak[r],
                         "cluster": ids[bundle.cluster_mask[r]]},
            "unreliable_facilities": ids[bundle.unreliable[r]],
            "parameters": {**vars(params), "theta_on": theta_on,
                           "theta_off": theta_on * world.cfg.experiment.theta_off_ratio,
                           "forest_alone_threshold": forest_h},
            "series": {
                "reported_count": _clean(o.reported[r], 0), "anomaly_score": _clean(o.score_ml[r]),
                "signal_production": _clean(o.production_ml[r] * o.weight[r]),
                "signal_concentration": _clean(A), "hill_activation": _clean(H),
                "reliability": _clean(o.reliability[r]), "active": active.astype(int),
                "regional_alert": alert.astype(int), "forest_alone_alert": forest.astype(int)}},
            data_dir / f"simulation_state_{scenario}.json")
    names = {"surveillance": "synthetic_surveillance_dataset", "scores": "ml_anomaly_scores",
             "signals": "signal_concentration_timeseries", "activation": "facility_activation_states",
             "alerts": "regional_alert_timeline"}
    for key, name in names.items():
        pd.concat(frames[key], ignore_index=True).to_csv(data_dir / f"{name}.csv", index=False)
