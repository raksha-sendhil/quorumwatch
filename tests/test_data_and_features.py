from dataclasses import replace

import numpy as np

from config import Config, OutbreakConfig
from data.generator import inject_outbreak, make_facilities, outbreak_profile, select_cluster, simulate_rates
from data.reporting import apply_reporting, count_by_arrival, draw_reporting, make_profile, sum_by_arrival
from ml.feature_engineering import build_features, direction_excess, fit_profiles, rolling_stats
from ml.isolation_forest import ScoreCalibrator
from network.facility_network import build_weights, closed_neighbourhood, laplacian, n_components

CFG = Config()


def test_facilities_are_reproducible():
    a, b = make_facilities(CFG.world), make_facilities(CFG.world)
    assert a.equals(b) and len(a) == 30 and a.facility_id.is_unique


def test_network_is_connected_symmetric_and_normalised():
    fac = make_facilities(CFG.world)
    W, _ = build_weights(fac[["x_km", "y_km"]].to_numpy(), CFG.world.knn, CFG.world.length_scale_km)
    assert n_components(W) == 1
    assert np.allclose(W, W.T) and np.allclose(np.diag(W), 0)
    assert np.isclose(W.sum(axis=1).max(), 1.0)
    assert np.allclose(laplacian(W).sum(axis=1), 0)
    assert closed_neighbourhood(W).diagonal().all()


def test_outbreak_grows_gradually_and_ends():
    cfg = OutbreakConfig(ramp_days=10, duration_days=20)
    g = outbreak_profile(50, 5, cfg)
    assert (g[:5] == 0).all() and (np.diff(g[5:15]) > 0).all()
    assert g[14] == 1.0 and (g[25:] == 0).all()


def test_outbreak_only_changes_cluster_after_start():
    fac = make_facilities(CFG.world)
    rng = np.random.default_rng(0)
    rates = simulate_rates(fac.baseline_daily_count.to_numpy(), np.arange(100), rng, CFG.world)
    counts = rng.poisson(rates).astype(float)
    out = inject_outbreak(counts, rates, np.array([1, 2]), 40, 0.3, rng, CFG.outbreak)
    diff = out - counts
    assert (diff[:40] == 0).all() and (diff >= 0).all()
    assert (np.delete(diff, [1, 2], axis=1) == 0).all() and diff[40:, [1, 2]].sum() > 0


def test_cluster_is_connected():
    fac = make_facilities(CFG.world)
    W, _ = build_weights(fac[["x_km", "y_km"]].to_numpy(), CFG.world.knn, CFG.world.length_scale_km)
    sub = W[np.ix_(c := select_cluster(5, W, 6), c)]
    assert len(c) == 6 and n_components(sub) == 1


def test_reporting_respects_delay_and_missingness():
    rng = np.random.default_rng(1)
    rep = CFG.reporting
    profile = make_profile(np.full(30, 0.95), 0.5, rng, rep)
    assert profile.unreliable.sum() == 15
    arrival, factor = draw_reporting(profile, 400, rng, rep)
    days = np.arange(400)[:, None]
    seen = arrival >= 0
    assert (arrival[seen] >= np.broadcast_to(days, arrival.shape)[seen]).all()
    assert ((arrival - days)[seen] <= rep.unreliable_max_delay).all()
    assert (~seen[:, profile.unreliable]).mean() > (~seen[:, ~profile.unreliable]).mean()
    reported = apply_reporting(np.full((400, 30), 10.0), factor, arrival)
    assert np.isnan(reported[~seen]).all()


def test_sum_by_arrival_moves_report_to_arrival_day():
    arrival = np.array([[2], [-1], [2], [3]], dtype=np.int16)
    values = np.array([[1.0], [100.0], [3.0], [5.0]])
    assert sum_by_arrival(values, arrival).ravel().tolist() == [0, 0, 4, 5]
    assert count_by_arrival(arrival).ravel().tolist() == [0, 0, 2, 1]


def test_rolling_features_never_use_reports_not_yet_arrived():
    z = np.array([[1.0], [100.0], [3.0]])
    arrival = np.array([[0], [10], [2]], dtype=np.int16)    # day 1 arrives after day 2
    stats = rolling_stats(z, arrival, window=3)
    assert stats["mean"][2, 0] == 2.0 and stats["n"][2, 0] == 2


def test_missing_reports_yield_no_feature_row():
    z = np.array([[1.0], [np.nan], [3.0]])
    arrival = np.array([[0], [-1], [2]], dtype=np.int16)
    X, (s, _), _ = build_features(z, arrival, np.arange(3), 7)
    assert s.tolist() == [0, 2] and not np.isnan(X).any()


def test_profiles_recover_mean_and_normalise_volume():
    fac = make_facilities(CFG.world)
    rng = np.random.default_rng(2)
    days = np.arange(730)
    counts = rng.poisson(simulate_rates(fac.baseline_daily_count.to_numpy(), days, rng, CFG.world)).astype(float)
    profiles = fit_profiles(counts, days)
    z = profiles.standardise(counts, days)
    assert abs(z.mean()) < 0.02
    assert (np.abs(z.std(axis=0) - 1) < 0.15).all()


def test_calibrator_null_production_budget_and_direction_gate():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=20000)
    excess = np.ones_like(scores)
    sites = rng.integers(0, 4, len(scores))
    scores = scores + sites                                # sites differ in score level
    cal = ScoreCalibrator.fit(scores, excess, sites, 4, 0.1)
    fresh_sites = rng.integers(0, 4, 20000)
    fresh = rng.normal(size=20000) + fresh_sites
    production = cal.production(fresh, np.ones_like(fresh), fresh_sites)
    for k in range(4):
        assert abs((production[fresh_sites == k] > 0).mean() - 0.1) < 0.02
    assert (production >= 0).all()
    assert abs((production > 0).mean() - 0.1) < 0.01
    assert abs(production.mean() - 0.1) < 0.02
    assert (cal.production(fresh, -np.ones_like(fresh), fresh_sites) == 0).all()


def test_direction_excess_blocks_deficits():
    z = np.array([[-3.0]])
    stats = {"mean": np.array([[-2.0]])}
    assert direction_excess(z, stats)[0, 0] < 0


def test_general_outbreaks_vary_and_stay_inside_the_evaluation_window():
    from dataclasses import dataclass

    from experiments.world import generate_run

    @dataclass
    class FakeWorld:
        cfg: object
        facilities: object
        W: object

    cfg = replace(CFG, outbreak=replace(CFG.outbreak, general=True))
    fac = make_facilities(cfg.world)
    W, _ = build_weights(fac[["x_km", "y_km"]].to_numpy(), cfg.world.knn, cfg.world.length_scale_km)
    world = FakeWorld(cfg, fac, W)
    runs = [generate_run(world, seed, 0, 0.1) for seed in range(60)]
    assert {r["ramp"] for r in runs} == {1, 7, 21}
    assert {r["duration"] for r in runs} == {14, 28, 42}
    assert {r["size"] for r in runs} == {3, 6, 10}
    assert all(0.15 <= r["peak"] <= 0.75 for r in runs)
    assert all(r["t_end"] <= cfg.experiment.n_days for r in runs)
    assert all(len(r["cluster"]) == r["size"] for r in runs)


def test_fixed_outbreak_option_uses_configured_values():
    from dataclasses import dataclass

    from experiments.world import generate_run

    @dataclass
    class FakeWorld:
        cfg: object
        facilities: object
        W: object

    cfg = replace(CFG, outbreak=replace(CFG.outbreak, general=False))
    fac = make_facilities(cfg.world)
    W, _ = build_weights(fac[["x_km", "y_km"]].to_numpy(), cfg.world.knn, cfg.world.length_scale_km)
    run = generate_run(FakeWorld(cfg, fac, W), 5, 0, 0.1)
    assert run["ramp"] == 21 and run["duration"] == 42 and run["size"] == 6
    assert 0.15 <= run["peak"] <= 0.30
