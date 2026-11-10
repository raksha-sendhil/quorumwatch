# Demo case files

Upload any of these with the app's *Upload CSV* button; the Karnataka examples are also in the data menu at the top of the Explore view.

## Real data: Karnataka dengue surveillance

Weekly lab-confirmed dengue cases (IgM + NS1) per district, from the Karnataka Dengue Daily Surveillance
Reports, cleaned as described in `real_data/karnataka.py` and exported without further changes. Each file
starts with 14 warm-up weeks. For each file the app rebuilds the model from earlier years only (see the
model column). These periods were chosen after checking how the model behaves on every year from 2016
to 2024; `real_data/README.md` lists them all.

| File | Weeks | What it shows | Model | First regional alert |
|---|---|---|---|---|
| `karnataka_2022_monsoon.csv` | 2021-09-20 to 2022-12-26 | 2022 monsoon season: Dengue rises from July | Model trained on 2017, 2018, 2020 and calibrated on 2021 | 2022-07-04: Chikkamagaluru, Chitradurga (+2 later alerts) |
| `karnataka_2016_monsoon.csv` | 2015-09-21 to 2016-12-26 | 2016 monsoon season: Malnad and coastal districts | Model trained on 2012, 2014 and calibrated on 2015 | 2016-08-01: Chikkamagaluru, Udupi (+3 later alerts) |
| `karnataka_2024_early_surge.csv` | 2023-09-25 to 2024-06-24 | Early 2024 surge: Cases rise from February | Model trained on 2020, 2021, 2022 and calibrated on 2023 | 2024-03-18: Chitradurga, Tumakuru (+1 later alert) |
| `karnataka_2020_epidemic_tail.csv` | 2019-09-23 to 2020-06-29 | Early 2020: End of the 2019 epidemic | Model trained on 2015, 2016, 2018 and calibrated on 2019 | 2020-02-17: Ballari, Gadag |
| `karnataka_2018_quiet_year.csv` | 2017-09-25 to 2018-12-31 | 2018, a quiet year: For comparison | Model trained on 2013, 2014, 2015, 2016 and calibrated on 2017 | No alert |

## Synthetic data (in `synthetic/`)

Generated with the study's data generator. The answer key is what was simulated; the last column is
what the model did when the file was uploaded (uploaded reports count as on time).

| File | Scenario | Outbreak starts | Hospitals affected | Model result |
|---|---|---|---|---|
| `synthetic/01_clear_outbreak.csv` | Clear outbreak | 2028-02-28 | F00, F10, F14, F18, F22, F28 | Alert on 2028-03-03, 4 days after onset (+2 false alerts) |
| `synthetic/02_messy_reporting.csv` | Outbreak with half the hospitals reporting unreliably | 2028-04-17 | F00, F10, F12, F14, F16, F18, F21, F22, F28, F29 | Alert on 2028-04-20, 3 days after onset |
