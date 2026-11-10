"""Run the complete QuorumWatch pipeline: calibrate on validation data, evaluate once on test
data, and write all data, results and figures.

    python run_experiments.py                                  # full study
    python run_experiments.py --world-seed 777 --out outputs_world777   # second geography
    python run_experiments.py --quick --out outputs_quick      # small smoke run on separate seeds
"""
import argparse
import dataclasses
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from experiments import calibration, exports, plots, primary_experiment as pe
from experiments.world import build_bundle, build_world
from ml.feature_engineering import FEATURE_NAMES
from network.facility_network import n_components, n_directed_edges
from quorum import signal_model as sm


def analyse_bistability(world, params, null_P, outbreak_P, sweep):
    """Numerical search for bistability. Multiple stable states count only when they are found."""
    exp = world.cfg.experiment
    node_rows = calibration.bistability_report(world, params, null_P, outbreak_P)
    network_rows = [sm.network_bistability(world.W, params, P) for P in (0.0, null_P, outbreak_P)]
    map_rows = [{"beta": b, "delta": d, "K": params.K,
                 "n_stable": sm.n_stable(0.0, replace(params, beta=float(b), delta=float(d)))}
                for b in exp.grid_beta for d in exp.grid_delta]
    map_frame = pd.DataFrame(map_rows)
    # A comparison parameter set: at the selected K, the grid point closest to the selected
    # dynamics that is bistable as an isolated node (highest delta, then lowest beta), if any.
    candidates = map_frame[map_frame.n_stable >= 2].sort_values(["delta", "beta"], ascending=[False, True])
    comparison = (replace(params, beta=float(candidates.beta.iloc[0]), delta=float(candidates.delta.iloc[0]))
                  if len(candidates) else None)
    P_max = max(2.0 * outbreak_P, 1.0)
    sweeps = {f"Selected parameters (β = {params.beta:g})": pd.DataFrame(sm.hysteresis_sweep(world.W, params, P_max))}
    summary = {
        "selected_parameters": dataclasses.asdict(params),
        "single_node": node_rows,
        "network_two_initial_conditions": network_rows,
        "selected_parameters_single_node_bistable_at_any_tested_level": any(r["n_stable"] >= 2 for r in node_rows),
        "selected_parameters_network_multistable_at_any_tested_level": any(r["multistable"] for r in network_rows),
        "selected_parameters_bistable_under_mean_null_production":
            next(r for r in node_rows if r["production_level"] == "mean_null_production")["n_stable"] >= 2,
    }
    by_state = sweep.groupby("single_node_stable_states_P0").val_restricted_delay
    summary["validation_restricted_delay_by_isolated_node_stable_states"] = {
        str(k): {"n_candidates": int(v.count()), "mean": float(v.mean()), "best": float(v.min())} for k, v in by_state}
    if comparison is not None:
        sweeps[f"Comparison (β = {comparison.beta:g}, δ = {comparison.delta:g})"] = pd.DataFrame(sm.hysteresis_sweep(world.W, comparison, P_max))
        summary["comparison_parameters"] = dataclasses.asdict(comparison)
        summary["comparison_single_node"] = calibration.bistability_report(world, comparison, null_P, outbreak_P)
        summary["comparison_network_two_initial_conditions"] = [
            sm.network_bistability(world.W, comparison, P) for P in (0.0, null_P, outbreak_P)]
    return summary, map_frame, sweeps


def markdown_table(agg, cfg):
    names = {"detection_rate": "Detection", "chance_detection": "Chance detection", "detection_delay_days": "Delay (detected)", "restricted_delay_days": "Restricted delay",
             "attributable_detection_rate": "Attributable detection", "attributable_restricted_delay_days": "Attributable delay",
             "mean_alert_size": "Alert size",
             "false_alerts_per_year": "False/yr", "precision": "Precision", "alert_burden_null": "Null alert burden",
             "messages_per_day": "Messages/day"}
    lines = []
    for scenario in plots.SCENARIOS:
        lines += [f"\n### {plots.SCENARIOS[scenario]}\n", "| Method | " + " | ".join(names.values()) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for method in plots.ORDER:
            sub = agg[(agg.scenario == scenario) & (agg.method == method)].set_index("metric")
            cells = [f"{sub.loc[m, 'mean']:.3g} [{sub.loc[m, 'ci_low']:.3g}, {sub.loc[m, 'ci_high']:.3g}]" for m in names]
            lines.append(f"| {pe.METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-seed", type=int, default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.world_seed is not None:
        cfg = replace(cfg, world=replace(cfg.world, seed=args.world_seed))
    if args.quick:
        cfg = replace(cfg, experiment=replace(cfg.experiment, n_validation_seeds=8, n_test_seeds=10,
                                              n_random_configs=4, n_bootstrap=200,
                                              test_seed_base=900000))
    exp = cfg.experiment
    root = Path(__file__).resolve().parent / args.out
    data_dir, results_dir, fig_dir = root / "data", root / "results", root / "figures"
    for d in (data_dir, results_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("[1/6] Building world, training Isolation Forest on normal data")
    world = build_world(cfg)
    assert n_components(world.W) == 1, "facility network must be connected"
    exports.export_network(world, data_dir)
    exports.export_history(world, data_dir)

    print("[2/6] Generating validation runs")
    val = build_bundle(world, exp.validation_seed_base, exp.n_validation_seeds)
    print(f"      {len(val.seeds)} paired runs, {time.time() - start:.0f}s")

    print("[3/6] Calibrating on validation data only")
    sweep, params = calibration.sweep_parameters(world, val)
    forest_cal = calibration.calibrate_forest_alone(world, val)
    theta_cal = calibration.calibrate_full(world, val, params)
    theta_on, forest_h = theta_cal["theta_on"], forest_cal["h"]
    sweep.to_csv(results_dir / "parameter_calibration.csv", index=False)
    print(f"      theta_on={theta_on:.4f}; forest threshold={forest_h:.3f}")
    exports.write_json({
        "selected_signal_parameters": dataclasses.asdict(params),
        "activation_threshold": theta_cal,
        "theta_off_ratio": exp.theta_off_ratio,
        "forest_alone_threshold": forest_cal,
        "procedure": ("Random search over the parameter grids, then one-at-a-time refinement around the best "
                      "candidate. Each candidate's threshold meets the false-alert and alert-burden targets on "
                      "validation null runs. Candidates are ranked by restricted delay on validation outbreak runs."),
        "validation_seed_range": [exp.validation_seed_base, exp.validation_seed_base + exp.n_validation_seeds - 1],
        "test_seed_range": [exp.test_seed_base, exp.test_seed_base + exp.n_test_seeds - 1],
        "full_config": cfg.to_dict()}, results_dir / "selected_parameters.json")

    print("[4/6] Evaluating on held-out test runs")
    test = build_bundle(world, exp.test_seed_base, exp.n_test_seeds)
    raw, keep = pe.run_primary(world, test, params, theta_on, forest_h)
    raw.to_csv(results_dir / "run_metrics_raw.csv", index=False)
    agg = pe.aggregate(raw, cfg)
    agg.to_csv(results_dir / "aggregated_metrics.csv", index=False)
    pe.paired_comparisons(raw, cfg).to_csv(results_dir / "paired_differences.csv", index=False)
    curves, matched = pe.tradeoff_curves(world, test, keep, params)
    curves.to_csv(results_dir / "tradeoff_curves.csv", index=False)
    matched.to_csv(results_dir / "false_alert_matched_comparison.csv", index=False)
    pe.by_outbreak_type(raw).to_csv(results_dir / "results_by_outbreak_type.csv", index=False)
    diagnostic = pe.outbreak_diagnostic(world, test)
    diagnostic.to_csv(results_dir / "outbreak_diagnostic.csv", index=False)
    pe.quenching_diagnostic(test, cfg).to_csv(results_dir / "reliability_diagnostic.csv", index=False)

    print("[5/6] Bistability analysis")
    summary, map_frame, sweeps = analyse_bistability(
        world, params, float(diagnostic.mean_null_production_ml[0]), float(diagnostic.mean_plateau_production_ml[0]), sweep)
    exports.write_json(summary, results_dir / "bistability_summary.json")
    map_frame.to_csv(results_dir / "bistability_map.csv", index=False)
    pd.concat([f.assign(parameters=name) for name, f in sweeps.items()]).to_csv(results_dir / "hysteresis_sweep.csv", index=False)

    card = {**world.model_card, "algorithm": "sklearn.ensemble.IsolationForest", "features": FEATURE_NAMES,
            "long_window": cfg.features.long_window, "quench_gamma": cfg.reliability.quench_gamma,
            "hyperparameters": dataclasses.asdict(cfg.forest), "score_calibration":
            "P = max(0, ln(tail / p)), with p the upper-tail probability under that facility's own separate normal data",
            "production_tail": cfg.features.production_tail,
            "statement": "Scores unusual local reporting patterns. It does not predict outbreaks."}
    exports.write_json(card, results_dir / "ml_model_card.json")

    print("[6/6] Writing exports and figures")
    showcase = "medium_reliability"
    exports.export_showcase(world, test, keep, params, theta_on, forest_h, data_dir)
    target = exp.target_false_alerts_per_year
    plots.set_style()
    plots.network_topology(world, fig_dir / "network_topology.png")
    plots.signal_propagation(world, test, keep, showcase, fig_dir / "signal_propagation.png")
    plots.hill_curve(params, fig_dir / "hill_activation_curve.png")
    plots.signal_over_time(test, keep, showcase, theta_on, params, fig_dir / "signal_concentration_over_time.png")
    plots.network_activation(test, keep, showcase, fig_dir / "network_activation_over_time.png")
    plots.comparison(agg, "attributable_detection_rate", "Detection rate",
                     "Outbreak detection rate on test runs, coincidental alerts excluded, mean and 95% CI", fig_dir / "detection_rate_comparison.png")
    plots.comparison(agg, "attributable_restricted_delay_days", "Days",
                     "Detection delay on test runs, coincidental alerts excluded, misses count as full window, mean and 95% CI", fig_dir / "detection_delay_comparison.png")
    plots.comparison(agg, "false_alerts_per_year", "False alerts per year", "False regional alerts on test null runs, mean and 95% CI",
                     fig_dir / "false_alarm_comparison.png", reference=target)
    plots.parameter_sensitivity(sweep, fig_dir / "parameter_sensitivity.png")
    plots.bistability_figure(sweeps, map_frame, params.K, fig_dir / "bistability_analysis.png")
    plots.tradeoff_figure(curves, target, fig_dir / "false_alert_detection_tradeoff.png")

    manifest = {str(p.relative_to(root)).replace("\\", "/"): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}
    exports.write_json({"elapsed_seconds": round(time.time() - start, 1), "n_directed_edges": n_directed_edges(world.W),
                        "files": manifest}, root / "manifest.json")
    (results_dir / "results_tables.md").write_text("# Test results (mean [95% CI])\n" + markdown_table(agg, cfg), encoding="utf8")
    print(markdown_table(agg, cfg))
    print(f"\nDone in {time.time() - start:.0f}s. {len(manifest)} files in {root}")


if __name__ == "__main__":
    main()
