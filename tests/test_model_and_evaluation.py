from dataclasses import replace

import numpy as np

from config import Config, ReliabilityConfig, SignalParams
from experiments.evaluation import bootstrap_metrics, evaluate_mask
from network.facility_network import closed_neighbourhood, laplacian
from quorum.hill_activation import hill, hill_max_slope
from quorum.hysteresis import apply_hysteresis, quorum_alert
from quorum.reliability import compute_reliability
from quorum.signal_model import equilibria, n_stable, network_bistability, simulate_signals, stable_substeps

CFG = Config()
W = np.array([[0, .5, 0, 0], [.5, 0, .5, 0], [0, .5, 0, .5], [0, 0, .5, 0]], float)
P_ZERO = np.zeros((1, 4))


def test_hill_properties():
    assert hill(np.array([0.0]), 2, 2.2)[0] == 0
    assert np.isclose(hill(np.array([2.0]), 2, 2.2)[0], 0.5)
    assert (np.diff(hill(np.linspace(0, 50, 200), 2, 2.2)) > 0).all()
    grid = np.linspace(0, 20, 200001)
    numeric = np.gradient(hill(grid, 2, 2.2), grid).max()
    assert np.isclose(numeric, hill_max_slope(2, 2.2), rtol=1e-3)


def test_diffusion_conserves_mass_and_equalises():
    p = SignalParams(D=1.0, delta=0.0, beta=0.0)
    A0 = np.array([8.0, 0, 0, 0])
    A = simulate_signals(np.zeros((400, 4)), W, p, A0=A0)
    assert np.isclose(A[0].sum(), 8, atol=1e-4) and np.isclose(A[-1].sum(), 8, atol=1e-4)
    assert np.ptp(A[-1]) < 1e-3


def test_decay_matches_analytic_solution():
    p = SignalParams(D=0.0, delta=0.2, beta=0.0)
    A = simulate_signals(np.zeros((10, 4)), W, p, n_sub=200, A0=np.ones(4))
    assert np.isclose(A[-1, 0], np.exp(-2.0), rtol=1e-2)


def test_concentration_is_nonnegative_and_bounded_under_extreme_parameters():
    rng = np.random.default_rng(0)
    production = rng.exponential(3.0, (500, 4))
    for D, delta, beta, K in [(0.8, 0.5, 3, 1), (0.8, 0.1, 3, 1), (0.05, 0.5, 0.25, 5)]:
        p = SignalParams(D=D, delta=delta, beta=beta, K=K)
        A = simulate_signals(production, W, p)
        assert (A >= 0).all() and np.isfinite(A).all()
        assert A.max() <= (production.max() + beta) / delta + 1e-3


def test_time_step_converges():
    rng = np.random.default_rng(1)
    production = rng.exponential(1.0, (120, 4))
    p = SignalParams(D=0.4, delta=0.2, beta=1.0, K=2.0)
    coarse = simulate_signals(production, W, p)
    fine = simulate_signals(production, W, p, n_sub=64)
    assert np.abs(coarse - fine).max() / fine.max() < 0.15
    assert stable_substeps(W, p) >= 1


def test_missing_reports_decay_without_new_signal():
    p = SignalParams(D=0.0, delta=0.2, beta=0.0)
    A = simulate_signals(np.zeros((5, 4)), W, p, A0=np.full(4, 2.0))
    assert (np.diff(A, axis=0) < 0).all() and (A > 0).all()


def test_bistability_is_detected_only_when_present():
    bistable = SignalParams(beta=1.0, delta=0.1, K=2.0)
    eq = equilibria(0.0, bistable)
    assert sum(e["stable"] for e in eq) == 2 and sum(not e["stable"] for e in eq) == 1
    assert network_bistability(W, replace(bistable, D=0.2), 0.0, days=1500)["multistable"]
    monostable = SignalParams(beta=0.25, delta=0.5, K=5.0)
    assert n_stable(0.0, monostable) == 1
    assert not network_bistability(W, replace(monostable, D=0.2), 0.0, days=1500)["multistable"]


def test_hysteresis_holds_between_thresholds():
    H = np.array([.1, .9, .7, .6, .39, .5, .79, .8])[:, None]
    assert apply_hysteresis(H, 0.8, 0.4).ravel().astype(int).tolist() == [0, 1, 1, 1, 0, 0, 0, 1]


def test_quorum_is_local_and_requires_two_sites():
    closed = closed_neighbourhood(W)
    alone = np.array([[True, False, False, False]])
    assert not quorum_alert(alone, closed, 0.25, 2).any()
    pair = np.array([[True, True, False, False]])
    assert quorum_alert(pair, closed, 0.5, 2).tolist() == [[True, True, False, False]]
    assert quorum_alert(pair, closed, 0.67, 2).tolist() == [[True, False, False, False]]


def test_reliability_penalises_gaps_delays_and_outliers():
    T = 300
    z = np.zeros((T, 4))
    days = np.arange(T)
    arrival = np.tile(days[:, None], (1, 4)).astype(np.int16)
    z[:, 1] = np.nan
    arrival[:, 1] = -1                                            # never reports
    arrival[:, 2] = np.minimum(days + 3, T + 10)                   # always late
    arrival[arrival >= T] = -1
    z[::10, 3] = 7.0                                               # frequent implausible values
    r = compute_reliability(z, arrival, ReliabilityConfig())[-1]
    assert r[0] > 0.95
    assert r[1] == ReliabilityConfig().min_reliability
    assert r[3] < r[2] < r[0]


def _mask(days, sites, T=100, N=4):
    mask = np.zeros((1, T, N), bool)
    for d in days:
        mask[0, d, sites] = True
    return mask


def test_evaluation_rules():
    cluster = np.array([[True, True, False, False]])
    t0, t_end = np.array([50]), np.array([80])
    kw = dict(burn_in=10, refractory=14, t0=t0, t_end=t_end, cluster=cluster)
    # A true onset is credited with the correct day.
    res = evaluate_mask(_mask([60], [0]), **kw)
    assert res["detection_day"][0] == 60 and res["n_false"][0] == 0
    # An alert elsewhere during the outbreak is false.
    res = evaluate_mask(_mask([60], [3]), **kw)
    assert res["detection_day"][0] == -1 and res["n_false"][0] == 1
    # An alert already on before the outbreak earns no credit for it.
    res = evaluate_mask(_mask(list(range(45, 70)), [0]), **kw)
    assert res["detection_day"][0] == -1 and res["n_false"][0] == 1
    # Onsets during burn-in are ignored; nearby onsets merge within the refractory period.
    assert evaluate_mask(_mask([5], [0]), **kw)["n_events"][0] == 0
    assert evaluate_mask(_mask([20, 30], [3]), burn_in=10, refractory=14)["n_events"][0] == 1
    assert evaluate_mask(_mask([20, 40], [3]), burn_in=10, refractory=14)["n_events"][0] == 2


def test_validation_and_test_seeds_are_disjoint():
    exp = CFG.experiment
    val = set(range(exp.validation_seed_base, exp.validation_seed_base + exp.n_validation_seeds))
    test = set(range(exp.test_seed_base, exp.test_seed_base + exp.n_test_seeds))
    assert not val & test


def test_long_window_feature_is_added_and_leak_free():
    from ml.feature_engineering import build_features
    T = 40
    z = np.zeros((T, 1))
    z[20:] = 1.0
    arrival = np.arange(T, dtype=np.int16)[:, None].copy()
    arrival[30, 0] = 39                        # arrives late: unseen by days 31-38
    X, _, stats = build_features(z, arrival, np.arange(T), 7, 14)
    assert X.shape[1] == 9
    seen = stats["long_z"][35, 0]
    z2 = z.copy()
    z2[30, 0] = 50.0
    assert build_features(z2, arrival, np.arange(T), 7, 14)[2]["long_z"][35, 0] == seen


def test_forest_alone_gates_deficits_and_pools_only_neighbours():
    from ml.forest_alone import neighbourhood_mask, neighbourhood_scores, site_scores
    nb = closed_neighbourhood(W)
    T = 3
    arrival = np.tile(np.arange(T)[:, None], (1, 4)).astype(np.int16)[None]
    raw = np.array([[[1.0, 2.0, 3.0, 4.0]] * T])
    excess = np.ones_like(raw)
    excess[0, 1, 0] = -1.0                     # a deficit report may not score
    raw[0, 2, 3] = np.nan                      # a missing report may not score
    arrival[0, 2, 3] = -1
    site = site_scores(raw, excess, arrival, np.zeros(4), np.ones(4))
    assert site[0, 0, 0] == 1.0 and site[0, 1, 0] == -np.inf and site[0, 2, 3] == -np.inf
    pooled = neighbourhood_scores(site, arrival, nb)
    assert np.isclose(pooled[0, 0, 0], (1 + 2) / np.sqrt(2))   # facility 0 pools only itself and 1
    assert np.isclose(pooled[0, 2, 2], (3 + 2) / np.sqrt(2))   # silent facility 3 is rescaled away
    mask = neighbourhood_mask(pooled, pooled[0, 0, 0] + 1e-3, nb)
    assert mask.any() and mask.shape == pooled.shape


def test_only_unsupported_spikes_count_as_implausible():
    from quorum.reliability import implausible_reports
    cfg = ReliabilityConfig()
    nb = closed_neighbourhood(W)
    T = 30
    arrival = np.tile(np.arange(T)[:, None], (1, 4)).astype(np.int16)
    z = np.zeros((T, 4))
    z[10, 0] = 7.0                                   # isolated one-day spike: implausible
    z[20, 1] = 7.0
    z[20, [0, 2]] = 3.0                              # neighbours also elevated that day: plausible
    z[22:29, 3] = 3.0
    z[29, 3] = 7.0                                   # follows a week of elevated reports: plausible
    flagged = implausible_reports(z, arrival, cfg, nb)
    assert flagged[10, 0]
    assert not flagged[20, 1]
    assert not flagged[29, 3]
    assert implausible_reports(z, arrival, cfg)[20, 1]          # without neighbours, only time can support it


def test_large_shared_outbreak_keeps_reliability_high():
    cfg = ReliabilityConfig()
    nb = closed_neighbourhood(W)
    T = 200
    arrival = np.tile(np.arange(T)[:, None], (1, 4)).astype(np.int16)
    z = np.zeros((T, 4))
    z[100:160] = 7.0                                 # every site far above normal for two months
    assert compute_reliability(z, arrival, cfg, nb)[159].min() > 0.9
    assert compute_reliability(z, arrival, cfg)[159].min() > 0.9   # sustained, so also supported over time
