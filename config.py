"""Central configuration. Every value used by the experiment is defined here and
is exported with the results so a run can be fully reproduced."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 20260401
    n_facilities: int = 30
    region_km: float = 100.0
    min_separation_km: float = 7.0
    baseline_median: float = 25.0
    baseline_log_sigma: float = 0.7
    baseline_min: float = 5.0
    baseline_max: float = 150.0
    season_amplitude: float = 0.20
    season_peak_day: int = 15
    dow_factors: tuple = (1.10, 1.08, 1.04, 1.02, 1.00, 0.76, 0.60)
    facility_noise_sigma: float = 0.08      # per-facility, per-day log noise
    regional_noise_sigma: float = 0.03      # shared AR(1) log noise, stationary sd
    regional_noise_rho: float = 0.7
    history_days: int = 730                 # normal data: profiles and IF training
    score_calibration_days: int = 365       # separate normal data: score calibration
    knn: int = 4
    length_scale_km: float = 20.0


@dataclass(frozen=True)
class ReportingConfig:
    # Reliable facilities derive missing/delay rates from nominal reliability.
    reliable_delay_share: float = 0.5
    reliable_max_delay: int = 2
    reliable_spike_prob: float = 0.002
    # Unreliable facilities are drawn at random in each run.
    unreliable_missing_prob: float = 0.20
    unreliable_delay_prob: float = 0.30
    unreliable_max_delay: int = 5
    unreliable_spike_prob: float = 0.04
    spike_factor_low: float = 1.5           # batch-dump / data-entry artefacts
    spike_factor_high: float = 3.5


@dataclass(frozen=True)
class OutbreakConfig:
    cluster_size: int = 6                   # connected facilities (20% of network)
    peak_excess_low: float = 0.15           # relative rise at plateau
    peak_excess_high: float = 0.30
    ramp_days: int = 21                     # gradual linear growth
    duration_days: int = 42
    grace_days: int = 7                     # alert accepted until duration + grace
    start_min: int = 28                     # relative to start of evaluation window
    start_max: int = 84
    # When True, each run draws its outbreak type from these values.
    general: bool = True
    general_ramp_days: tuple = (1, 7, 21)
    general_duration_days: tuple = (14, 28, 42)
    general_cluster_sizes: tuple = (3, 6, 10)
    general_peak_low: float = 0.15
    general_peak_high: float = 0.75


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 7
    long_window: int = 14                   # adds a long-window z-test feature
    z_clip: float = 8.0
    production_tail: float = 0.50           # top 50% of null scores produce signal


@dataclass(frozen=True)
class IsolationForestConfig:
    n_estimators: int = 200
    max_samples: int = 512
    max_features: float = 1.0
    random_state: int = 7


@dataclass(frozen=True)
class SignalParams:
    D: float = 0.2                          # diffusion coefficient
    delta: float = 0.2                      # degradation rate, per day
    beta: float = 0.5                       # positive feedback strength
    K: float = 2.0                          # half-activation concentration
    n: float = 2.2                          # Hill coefficient (held fixed)
    quorum: float = 0.34                    # neighbourhood fraction needed for alert


@dataclass(frozen=True)
class ReliabilityConfig:
    ewma_alpha: float = 0.05
    prior: float = 0.9
    late_credit: float = 0.5
    outlier_z: float = 4.0
    outlier_penalty: float = 10.0
    corroborate_z: float = 1.5              # neighbours' mean z that makes a high report plausible
    sustained_z: float = 2.0                # the site's own prior-week mean z with the same effect
    min_reliability: float = 0.05
    quench_gamma: float = 1.0               # influence = reliability ** gamma


@dataclass(frozen=True)
class ExperimentConfig:
    burn_in_days: int = 56
    eval_days: int = 154
    refractory_days: int = 14               # onsets closer than this are one event
    theta_off_ratio: float = 0.6            # hysteresis: theta_off = ratio * theta_on
    min_quorum_count: int = 2               # a "regional" alert needs two facilities
    target_false_alerts_per_year: float = 2.0
    matched_false_alert_rates: tuple = (1.0, 2.0, 4.0)
    attribution_tolerance_days: int = 3
    max_alert_burden: float = 0.05          # share of null days spent in alert
    scenarios: dict = field(default_factory=lambda: {
        "high_reliability": 0.10, "medium_reliability": 0.30, "low_reliability": 0.50})
    n_validation_seeds: int = 40
    n_test_seeds: int = 100
    validation_seed_base: int = 100000
    test_seed_base: int = 810000
    n_random_configs: int = 40
    sweep_seed: int = 11
    n_bootstrap: int = 2000
    bootstrap_seed: int = 3
    grid_D: tuple = (0.05, 0.1, 0.2, 0.4, 0.8)
    grid_delta: tuple = (0.1, 0.2, 0.35, 0.5)
    grid_beta: tuple = (0.25, 0.5, 1.0, 2.0, 3.0)
    grid_K: tuple = (1.0, 2.0, 3.0, 5.0)
    grid_quorum: tuple = (0.25, 0.34, 0.5, 0.67)

    @property
    def n_days(self):
        return self.burn_in_days + self.eval_days


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    outbreak: OutbreakConfig = field(default_factory=OutbreakConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self):
        return asdict(self)
