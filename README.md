# QuorumWatch: research backend

QuorumWatch adds a bacterial quorum-sensing-inspired signalling layer to local Isolation Forest anomaly detection, for decentralized outbreak early warning. This repository holds the simulation, the detectors, the experiment, all generated results, and a demo app that runs the pipeline on real Karnataka dengue surveillance data.

**The question this repository answers:** what does adding the bacterial layer change, compared with using Isolation Forest without it?

**Scope.** This repository evaluates only these two configurations. It makes no claim about how QuorumWatch performs relative to other surveillance methods, and its results should not be read as showing it is better than them.

## Main result

Across a mixture of simulated outbreak types, adding the bacterial layer improved detection and shortened delay at an equal rate of 2 false alerts per year. Values are held-out test runs across all reliability scenarios, mean [95% CI].

| Configuration | Detection rate | Second geography | Delay, days | Second geography | Messages/day |
|---|---|---|---|---|---|
| **Isolation Forest + bacterial layer** | **0.91** [0.88, 0.94] | 0.94 [0.91, 0.96] | **11.4** [10.3, 12.2] | 9.0 [8.3, 9.8] | 439 |
| **Isolation Forest alone** | **0.75** [0.69, 0.79] | 0.81 [0.76, 0.85] | **16.0** [14.7, 17.4] | 13.6 [12.5, 15.1] | 135 |

Detection and delay count only alerts attributable to the outbreak, and a miss counts as that run's full detection window. In paired runs at each configuration's calibrated threshold, the bacterial layer adds 0.15 [0.11, 0.20] to detection and changes delay by -4.5 [-5.6, -3.3] days. In the second geography, the figures are 0.14 [0.09, 0.18] and -5.0 [-6.1, -3.9] days.

**Main caveats** (details under [Results](#results)):

- The bacterial layer sends about 3-4 times more messages.
- Results come from synthetic data with assumptions chosen for this study.
- All reported results are from simulation. The Karnataka data in the demo app is for demonstration only.

## Run it

```
pip install -r requirements.txt
python run_experiments.py                                           # main study, about 15 minutes on 8 threads
python run_experiments.py --world-seed 777 --out outputs_world777   # second geography
python -m pytest tests
python run_experiments.py --quick --out outputs_quick               # smoke run on separate seeds
```

Runs are deterministic. All settings are in [config.py](config.py) and are saved with the results.

## Demo app

```
python run_experiments.py      # once, if outputs/ is missing: the app uses its calibrated model
python app/server.py           # opens http://127.0.0.1:8050
```

The first launch trains the models and takes about a minute; later launches load cached copies from `app/.cache/`. The app needs no extra packages and no internet connection.

- **Explore:** real dengue surveillance on a map of Karnataka's 30 districts, weekly. Pick one of five example periods, or upload a case file. Click any week on the timeline to see, for that week, each district's Isolation Forest score (left map) and the bacterial layer's signal (right map), with alerting districts outlined. Click a district for its history.
- **Results:** the study's simulated results: the detection / false-alert trade-off, robustness to unreliable reporting, outbreak types, and the second geography.
- **Upload:** a CSV with `date`, `facility_id` and `cases` columns (blank = missing report). Weekly Karnataka files use district names, dated Mondays, and need 14 warm-up weeks. Daily synthetic files use hospitals `F00`-`F29` and need 56 warm-up days (examples in [app/demo_data/synthetic/](app/demo_data/synthetic/)).

[docs/UI_GUIDE.md](docs/UI_GUIDE.md) explains every map, colour, number and state shown in the app, and how to walk through a demo.

For each Karnataka file the app rebuilds the model from earlier years only. The examples were chosen as clear illustrations; the app reports no performance on real data.

## Architecture

```
counts -> reporting process -> facility z-scores -> Isolation Forest -> calibrated signal production
                                                                                 |
              reliability weighting -> reaction-diffusion signal -> Hill activation -> hysteresis -> quorum vote
```

The forest only scores local anomalies and never decides regional alerts. In the full model, regional alerts come from the signal dynamics and the quorum vote.

| Path | Contents |
|---|---|
| `data/` | Facilities, Poisson counts, outbreak injection, reporting process |
| `ml/` | Facility profiles, features, Isolation Forest, score calibration, Isolation Forest alone |
| `network/` | Weighted adjacency matrix and Laplacian (NumPy only) |
| `quorum/` | Signal model, Hill function, hysteresis and quorum vote, reliability, bistability analysis |
| `experiments/` | World and run generation, calibration, evaluation, plots, exports |
| `real_data/` | Karnataka dengue data, district map and per-year models (demo app) |
| `app/` | Demo app (server, frontend, demo files) |
| `tests/` | 41 unit tests |

## Biological foundations and how they were adapted

The mapping is an abstraction. Facilities are treated as cells, and "signal" is a number with no chemical meaning.

| Biology | Implementation |
|---|---|
| LuxI autoinducer synthesis | Calibrated anomaly score becomes signal production `P_i` |
| Autoinducer diffusion | Graph Laplacian exchange between connected facilities |
| Signal degradation | First-order decay, `-delta A_i` |
| LuxR cooperative binding | Hill function with `n = 2.2` |
| Positive feedback (autoinduction) | `beta H(A_i)` term |
| Switch-like, stable response | Separate activation and deactivation thresholds (hysteresis) |
| Quorum quenching enzymes | Production is scaled by a local reliability score |

## Mathematical model

```
dA_i/dt = P_i + beta H(A_i) + D sum_j w_ij (A_j - A_i) - delta A_i
H(A)    = A^n / (K^n + A^n),  n = 2.2
```

**Numerics.** Explicit Euler with `s` substeps per day, where `s = ceil((D max_degree + delta + beta max H') / 0.5)`.

- Each update is a nonnegative combination of current concentrations, so `A` stays nonnegative.
- The linear part is stable because `lambda_max(L) <= 2 max_degree`.
- The explicit form means each facility needs only its own state and its neighbours' last values, with no global solve.
- Tests check mass conservation under pure diffusion, agreement with the analytic decay solution, nonnegativity and boundedness under extreme parameters, and time-step convergence.

**Network.** 30 facilities in a 100 x 100 km region. Each connects to its 4 nearest neighbours (symmetrized), plus a minimum spanning tree that guarantees connectivity. Weights are `exp(-distance / 20 km)`, scaled so that the largest weighted degree is 1.

**Local decision.** Facility `i` is active if `H(A_i) >= theta_on`, and it stays active until `H(A_i) < 0.6 theta_on`. It declares a regional alert when it is active and at least a fraction `q` of its closed neighbourhood (and at least 2 sites) is active. Every step uses local information. Counting "any facility declared" happens only in the evaluation code. Facility nodes are simulated in one process, so this is not a working distributed system.

**Reliability (quenching).** `r_i = timeliness_i x (1 - min(1, 10 x outlier_rate_i))`, clipped to [0.05, 1]. Both terms are EWMAs (alpha = 0.05) of locally known events: on-time reporting (half credit for late reports) and implausible reports. Production is multiplied by `r_i`.

A report counts as implausible only when it is very large (`|z| > 4`) **and nothing backs it up**. A rise is backed up when the neighbours' mean z that day is at least 1.5, or when the site's own previous week averaged at least z = 2. Only neighbour reports already received count. So an isolated one-day spike is treated as a probable reporting error, while a rise shared with neighbours, or sustained over time, is treated as real. This rule replaced an earlier one that penalised every `|z| > 4` report. That rule made large real outbreaks lower their own sites' reliability and suppress their own signal: see [Development history](#development-history).

**Missing and delayed reports.** A missing report produces no signal, while existing signal keeps diffusing and decaying. A delayed report enters on its arrival day, and its rolling features use only reports known by that day.

## Synthetic data

- **Counts:** Poisson, with rate = volume x seasonal cycle x day-of-week factor x facility log-noise x shared AR(1) regional log-noise. Facility volume is lognormal (median about 26 reports/day, range 7-145).
- **Normal reporting:** reliable sites have small missing and delay rates, and very rare artefacts.
- **Unreliable sites:** in each run, a random 10%, 30% or 50% of sites are unreliable: 20% missing, 30% delayed by up to 5 days, and 4% of reports are 1.5-3.5x artefacts.
- **Outbreaks:** connected facilities share a simultaneous rise. Each run independently draws its onset (ramp of 1, 7 or 21 days), peak rise (15-75% above normal), duration (14, 28 or 42 days) and size (3, 6 or 10 sites).
- **Pairing:** each outbreak run has an outbreak-free twin with identical noise and reporting. Reliability scenarios also share the same counts.

**Outbreak strength.** On test runs, the median plateau report at an outbreak facility has z = 1.90 (1.85 in the second geography), and 29% (26%) of those reports reach z >= 3, compared with about 2% across all reports.

## Machine learning component

Isolation Forest flags unusual local reporting patterns. **It does not predict disease outbreaks.**

| Item | Value |
|---|---|
| Training data | 730 normal days, 30 facilities, pooled (21,464 reports). No outbreaks |
| Calibration data | Separate 365 normal days (10,739 reports) |
| Normalisation | Per-facility log-link Poisson regression (season, day of week), variance `mu + phi mu^2`, then z-score |
| Features | Current z; rolling 7-day mean, SD and rate of change of z; 14-day window z-test; sin/cos of day of week and season |
| Hyperparameters | 200 trees, `max_samples=512`, `max_features=1.0`, `random_state=7`. No contamination threshold is used |

**Score calibration.** `P = max(0, ln(0.50 / p))`, where `p` is the upper-tail probability of the score under that facility's own calibration data.

**Direction gate.** Isolation Forest is two-sided, so an unusually low count would also score as anomalous. Production is set to zero unless the current, rolling or long-window z is positive.

**Limitations of unsupervised anomaly detection here:**

- It responds to anything unusual, including reporting artefacts. It has no concept of disease.
- This study does not measure how much the forest itself contributes. It only measures what is added on top of it.
- Raw scores rise with facility volume in the primary geography (Spearman 0.64; 0.04 in the second), partly because artefacts produce larger z-scores at large sites. Per-site calibration reduces this but does not remove it: the correlation between volume and signal production on null runs was -0.22, and 0.34 in the second geography.

## The two configurations

Both use the same reports, the same trained forest, and the same evaluation rules.

**Isolation Forest + bacterial layer.** The full model described above.

**Isolation Forest alone.** Each score is standardised against that site's own normal data, and only excess reports can score. A forest score at one site still needs some rule to become a regional alert, so each centre pools its closed neighbourhood's scores that day, as sum / sqrt(n reporting). The neighbourhood alerts when this reaches a threshold. There is no signal decay, accumulation, Hill activation, hysteresis, reliability weighting or quorum vote, though the forest's rolling features still carry some memory. Neighbours exchange scores, so this configuration also needs no central party.

## Experimental method

**Data split.** Validation: 40 seeds x 3 reliability scenarios (120 paired runs). Test: 100 seeds x 3 scenarios (300 paired runs, seeds 810000-810099). Each run has 56 burn-in days and 154 evaluation days. One fixed world (geography and trained forest) is used per study, with noise, reporting, and outbreak type, location and timing varying by seed. The whole study was repeated on a second geography.

**Calibration (validation data only).** Random search of 40 configurations, then one-at-a-time refinement, for 56 candidates in total. For each candidate, `theta_on` is set to meet a false-alert target of 2 per year and an alert burden of at most 5% on null runs. Candidates are ranked by restricted delay. The Isolation Forest alone threshold is calibrated to the same targets.

| Selected parameter | Primary geography | Second geography | Note |
|---|---|---|---|
| D | 0.8 | 0.8 | Upper edge of grid |
| delta | 0.5 | 0.5 | Upper edge of grid |
| beta | 1.0 | 2.0 | |
| K | 5.0 | 5.0 | Upper edge of grid |
| q | 0.25 | 0.25 | Lower edge of grid |
| n | 2.2 | 2.2 | Fixed, not swept |
| theta_on | 0.39 (A about 4.1) | 0.42 (A about 4.3) | |

**Development history.**

- **Earlier development runs informed the design.** The 14-day feature, production tail, quenching strength and outbreak mixture were all chosen after results from those runs.
- **The reliability rule changed after a check on real surveillance data.** The check found that the old rule suppressed large real outbreaks, so the rule was changed and the full study was rerun on the same test seeds.
  - **Isolation Forest alone:** its results are identical before and after the change, because it does not use reliability.
  - **Full model:** detection at 2 false alerts per year rose from 0.85 to 0.91 (0.89 to 0.94 in the second geography), and the earlier weakness on the largest outbreaks disappeared.
  - **Re-used seeds:** these test seeds had already been evaluated once before the change.
- **Fixed in advance:** the Isolation Forest alone rule was fixed before its results were seen, and no parameter was tuned on test results.

**Metrics (identical rules for both configurations).**

- **Onset:** a facility enters the alert set. Onsets within 14 days form one alert event.
- **Detection / recall:** an onset at an outbreak facility, from outbreak start until 7 days after its end. An alert already on before the outbreak earns no credit.
- **Attributable detection / delay:** first qualifying onset with no twin onset at an outbreak site within 3 days. This removes coincidences. A miss counts as that run's full window.
- **Chance detection:** how often the same scoring "detects" on the outbreak-free twin.
- **False alert rate:** alert events per year on null runs.
- **Precision:** true events / all events, pooled over outbreak and null runs. It depends on simulated prevalence of about one outbreak per 300 days.
- **Communication:** directed scalar transmissions per day.

**Confidence intervals.** 95% percentile bootstrap over seeds (2,000 resamples, or 500 for matched values). Paired differences use the same runs. Intervals are not adjusted for multiple comparisons.

**Equal-false-alert comparison.** A single pooled threshold gives different false-alert rates per scenario, so test-set threshold curves are also interpolated to exactly 1, 2 and 4 false alerts per year. This describes behaviour only and sets no parameters.

## Results

### At calibrated thresholds, all scenarios (mean [95% CI])

| Metric | + bacterial layer | Forest alone | + bacterial layer, second geography | Forest alone, second geography |
|---|---|---|---|---|
| Detection | 0.92 [0.88, 0.95] | 0.77 [0.72, 0.81] | 0.94 [0.91, 0.97] | 0.80 [0.76, 0.85] |
| Attributable detection | 0.91 [0.88, 0.94] | 0.76 [0.71, 0.81] | 0.94 [0.91, 0.96] | 0.80 [0.76, 0.84] |
| Attributable delay (days) | 11.1 [10.2, 12.1] | 15.6 [14.3, 16.9] | 8.9 [8.2, 9.7] | 13.9 [12.7, 15.3] |
| Chance detection | 0.07 [0.04, 0.10] | 0.09 [0.06, 0.12] | 0.04 [0.02, 0.07] | 0.09 [0.06, 0.12] |
| False alerts per year | 2.26 [2.02, 2.49] | 2.12 [1.82, 2.43] | 1.93 [1.71, 2.15] | 1.88 [1.60, 2.19] |
| Precision | 0.35 [0.33, 0.38] | 0.33 [0.30, 0.36] | 0.38 [0.36, 0.41] | 0.37 [0.34, 0.41] |
| Sites per alert | 4.35 [4.10, 4.61] | 8.37 [8.11, 8.67] | 4.79 [4.54, 5.03] | 7.59 [7.39, 7.80] |
| Days in alert, null runs | 0.02 [0.02, 0.02] | 0.01 [0.01, 0.01] | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.01] |
| Messages per day | 439 | 135 | 577 | 133 |

The full model slightly exceeded the 2-per-year false-alert target on test data in the primary geography. Per-scenario tables are in `outputs/results/results_tables.md`.

### Robustness to reporting reliability

Detection rate, bacterial layer vs forest alone, at exactly 2 false alerts per year:

| Unreliable sites | Primary geography | Second geography |
|---|---|---|
| 10% | 0.93 vs 0.90 | 0.94 vs 0.88 |
| 30% | 0.89 vs 0.81 | 0.94 vs 0.86 |
| 50% | 0.91 vs 0.69 | 0.92 vs 0.66 |

- **The bacterial layer holds up as reliability falls; the forest alone does not.** The largest gain is at 50% unreliable sites. At 10% the two are close.
- **Calibrated-threshold false-alert rates differ by scenario.** With one pooled threshold, false alerts per year at 10% / 30% / 50% unreliable are 2.75 / 2.30 / 1.73 for the bacterial layer, but 0.28 / 1.61 / 4.46 for forest alone. The matched table above is the fair per-scenario comparison.
- **Reliability score:** it separates site types, with mean reliability about 0.94 at reliable sites and 0.43 at unreliable ones. The unreliable-site behaviour, including its false spikes, was designed for this study, so part of the low-reliability gain follows from that assumption.

### By outbreak type

Attributable detection at calibrated thresholds, bacterial layer vs forest alone (60-123 runs per group):

| Outbreak property | Value | Primary geography | Second geography |
|---|---|---|---|
| Peak rise | 15-30% | 0.87 vs 0.61 | 0.79 vs 0.62 |
| | 30-45% | 0.88 vs 0.71 | 1.00 vs 0.77 |
| | 45-60% | 0.95 vs 0.78 | 0.98 vs 0.85 |
| | 60-75% | 0.96 vs 0.95 | 0.94 vs 0.93 |
| Onset ramp | 1 day | 0.90 vs 0.72 | 0.97 vs 0.92 |
| | 7 days | 0.94 vs 0.90 | 0.95 vs 0.78 |
| | 21 days | 0.90 vs 0.68 | 0.86 vs 0.67 |
| Duration | 14 days | 0.82 vs 0.62 | 0.80 vs 0.61 |
| | 28 days | 0.94 vs 0.77 | 1.00 vs 0.87 |
| | 42 days | 0.97 vs 0.88 | 0.99 vs 0.90 |
| Sites affected | 3 | 0.85 vs 0.60 | 0.89 vs 0.59 |
| | 6 | 0.97 vs 0.84 | 0.96 vs 0.93 |
| | 10 | 0.94 vs 0.86 | 0.97 vs 0.89 |

The bacterial layer scores higher in every group in both geographies. The gain is largest on weak, gradual, short and small outbreaks, and close to zero for the largest outbreaks, which both configurations catch.

### Communication

The bacterial layer needs about 3-4 times more messages than forest alone (439 vs 135 per day in the primary geography). That is 3 signal exchanges per day across 146 directed links, plus activation-state changes; the second geography uses 4 exchanges per day, because its larger beta needs more substeps. The reliability rule also needs neighbours' z-scores. Each message is one scalar. These are simulated counts, not a networking measurement, and this is not production-grade decentralized networking.

The simulation exchanges abstract signals and scores rather than patient-level records. **This is not a formal privacy guarantee, no differential privacy is claimed, and how much either configuration reveals was not measured.**

### Bistability

**The selected parameters are not bistable.** A single-node equilibrium analysis found one stable state at zero, mean-null and mean-outbreak production. A full-network simulation from low and high starting points reached the same state at each level. An up-and-down production sweep showed no hysteresis loop (`bistability_analysis.png`).

**Bistability does exist elsewhere in the model.** At `beta = 2, delta = 0.2, K = 5, D = 0.8` there are two stable states at zero production (A = 0 and A = 5.90), confirmed at network level. However, this is lost by the system's mean normal production level, where only one high state remains, so the network would stay latched on. Validation data also disfavoured this regime: candidates with two isolated-node stable states averaged a restricted delay of 14.6 days (best 11.3), against 10.4 days (best 8.0) for those with one.

**Consequence for the project abstract.** The abstract's claim that bistability produces sharp, stable alerts is not supported by this study. Alert stability here comes from the explicit hysteresis thresholds.

## Real data (demo only)

The demo app runs the same pipeline on Karnataka's dengue surveillance reports (weekly lab-confirmed cases in 30 districts, 2012-2025). The forest and district profiles are rebuilt for each year from preceding non-epidemic years; the signal parameters and threshold are the study's. No performance figures are reported on this data. Sources and cleaning are documented in [real_data/README.md](real_data/README.md).

## Limitations

- **Synthetic study.** All results are on synthetic data. Performance on real surveillance data has not been established.
- **Scope.** Only two configurations are evaluated. See the scope statement at the top.
- **Effect sizes depend on assumptions** about noise, artefacts and the outbreak mixture.
- **Forest-alone design.** "Alone" here means the neighbourhood pooling rule described above. Other pooling rules would give different scores.
- **Parameter edges.** Four selected values sit at the edge of their search ranges, so better values may lie outside them.
- **Parameters were tuned for the full model,** while forest alone has only its alert threshold to calibrate.
- **Re-used test seeds.** The reliability fix was evaluated on test seeds that had been used before the change (see [Development history](#development-history)).
- **CI scope.** Intervals reflect seed variation within a geography and are not adjusted for multiple comparisons. Subgroup results have small samples.
- **`n = 2.2`** was given as a literature-grounded starting value in the project brief and held fixed. A specific source for it still needs to be cited in the paper.

## Output files

`outputs/` holds the primary geography and `outputs_world777/` the second. Each has the same layout.

| Location | Files |
|---|---|
| `data/` | `synthetic_surveillance_dataset.csv`, `historical_surveillance_normal.csv`, `facility_network_metadata.csv`, `adjacency_matrix.csv`, `network_edges.csv`, `ml_anomaly_scores.csv`, `signal_concentration_timeseries.csv`, `facility_activation_states.csv`, `regional_alert_timeline.csv` |
| `data/` (JSON) | `network.json`, `simulation_state_<scenario>.json` |
| `results/` | `run_metrics_raw.csv`, `aggregated_metrics.csv`, `paired_differences.csv`, `false_alert_matched_comparison.csv`, `tradeoff_curves.csv`, `results_by_outbreak_type.csv`, `outbreak_diagnostic.csv`, `reliability_diagnostic.csv`, `parameter_calibration.csv`, `selected_parameters.json`, `bistability_summary.json`, `bistability_map.csv`, `hysteresis_sweep.csv`, `ml_model_card.json`, `results_tables.md` |
| `figures/` | Network topology, signal propagation, Hill curve, signal over time, network activation, detection rate, detection delay, false alarms, parameter sensitivity, bistability, false-alert trade-off |
| `manifest.json` | Every generated file and its size |

The per-day data files and state JSONs hold the first test seed of each scenario, chosen in advance and not for how it looks.

## References

Citations were written from memory without being checked against the sources. Verify the details before using them in a paper.

1. Fuqua WC, Winans SC, Greenberg EP. Quorum sensing in bacteria: the LuxR-LuxI family of cell density-responsive transcriptional regulators. *J Bacteriol*. 1994;176(2):269-275.
2. Waters CM, Bassler BL. Quorum sensing: cell-to-cell communication in bacteria. *Annu Rev Cell Dev Biol*. 2005;21:319-346.
3. Dong YH, et al. Quenching quorum-sensing-dependent bacterial infection by an N-acyl homoserine lactonase. *Nature*. 2001;411:813-817.
4. James S, Nilsson P, James G, Kjelleberg S, Fagerström T. Luminescence control in the marine bacterium *Vibrio fischeri*: an analysis of the dynamics of lux regulation. *J Mol Biol*. 2000;296:1127-1137.
5. Dockery JD, Keener JP. A mathematical model for quorum sensing in *Pseudomonas aeruginosa*. *Bull Math Biol*. 2001;63:95-116.
6. Ward JP, King JR, Koerber AJ, Williams P, Croft JM, Sockett RE. Mathematical modelling of quorum sensing in bacteria. *IMA J Math Appl Med Biol*. 2001;18:263-292.
7. Hill AV. The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. *J Physiol*. 1910;40:iv-vii.
8. Liu FT, Ting KM, Zhou ZH. Isolation Forest. *Proc IEEE ICDM*. 2008:413-422.
9. Karnataka Dengue Daily Surveillance Reports, state vector-borne disease control programme, Karnataka (structured release `0015-karnataka-dengue-summaries`; exact source and licence to be confirmed).
