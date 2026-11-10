# Real data (demo only)

Karnataka dengue surveillance data, used to demonstrate the QuorumWatch pipeline in the demo app. It is not part of the study, and no results on it are reported.

## Karnataka dengue surveillance

**Source:** Karnataka Dengue Daily Surveillance Reports, the state vector-borne disease control programme's daily district reports. They were used in the structured `0015-karnataka-dengue-summaries` release (one row per district per day: suspected, tested and lab-confirmed dengue cases, 2012 to January 2025). The same reports are archived as PDFs in the ICTS dengue data repository. The exact download location and licence of the structured release still need to be confirmed for citation.

**Files in this folder:**

- `karnataka_dengue_weekly.csv`: cleaned weekly lab-confirmed cases, 30 districts, 2012 to 2025.
- `karnataka_map.json`: simplified district outlines and centroids, projected for the app, from the Karnataka district GeoJSON used in the EpiSentinel project.

`python -c "from real_data import karnataka as k; k.build_weekly_file(); k.build_map_file()"` rebuilds both from the raw release. Set `QW_KARNATAKA_RAW` and `QW_KARNATAKA_GEOJSON` if the raw files are not in `~/Downloads`.

### Data problems and how they are handled

| Problem found | Handling |
|---|---|
| District names in mixed case (`BALLARI` / `Ballari`) and dates in two formats | Normalised |
| Vijayanagara split from Ballari in 2021; Bengaluru reported separately as BBMP (city corporation) and revenue district | Merged into Ballari and Bengaluru Urban, giving the same 30 districts in every year, matching the map |
| Days without reports (only about 250 of 365 days reported in 2012-2017) | Weekly totals. A district-week with no report at all is missing, not zero |
| Batch reporting: in 70% of district-weeks with 20 or more positives, one day holds at least 70% of the week (for example 1,070 BBMP positives logged on 3 July 2019 after two weeks of zeros) | The weekly value is the plain sum of reported positives. Remaining dumps are left to the reliability layer |
| Strong growth over time (about 5,000 cases a year in 2012 to over 15,000 by 2023), as testing and reporting expanded | The model for each test year is rebuilt from the preceding years |
| Earlier epidemic years would make "normal" look too high | Endemic-channel practice: of the candidate training years Y-5 to Y-2, years with a statewide total above 1.5 times the candidates' median are excluded. Year Y-1 calibrates the scores |
| Incomplete years: 2012 (40% of district-weeks missing), 2017 (25%), 2024 (57%, almost all of July-September) | Used as available; the 2024 example stops at June |

The model code, signal parameters and activation threshold are the study's, unchanged. One time step is one week.

### How the cleaning was arrived at

A first version (weekly value = mean daily positives × 7, no epidemic-year exclusion) was replaced by the current rules after its output was inspected: it inflated batch dumps, and earlier epidemic years made the baseline far too noisy (variance up to `mu + 12 mu^2`). A variant that also excluded epidemic years from calibration was tried and dropped.

### What this data is used for

The Karnataka data drives the demo app only. **No performance figures are reported on it.** The study's results come from the synthetic experiment. The five demo examples (listed in `app/demo_data/README.md`) were chosen after inspecting the model's output, as clear illustrations of how it behaves; the model does not alert on every real dengue season.

### Limitations

- The threshold is calibrated on synthetic data. Nothing was re-tuned for Karnataka beyond the data-cleaning rules above.
- Weekly district totals are coarse, and batch reporting makes timing uncertain by up to a week or more.
- Lab-confirmed positives depend on testing volume, which varies across districts and years.
- Whether a given alert is "true" can only be judged against statewide totals here; there is no independent outbreak record.
