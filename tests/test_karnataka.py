import numpy as np
import pandas as pd

from real_data import karnataka as k

COLUMNS = ["metadata.recordDate", "location.admin.hierarchy", "location.admin2.name", "daily.positive.total"]


def _raw(tmp_path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(tmp_path / "2019.csv", index=False)


def test_districts_match_the_map_one_to_one():
    assert len(k.DISTRICTS) == 30 == len(set(k.MAP_NAMES.values()))
    assert k.site_id("BENGALURU URBAN") == "bengaluru_urban"


def test_weekly_file_sums_reports_merges_units_and_keeps_gaps_missing(tmp_path):
    _raw(tmp_path, [
        ["2019-07-01T00:00:00Z", "REVENUE", "BENGALURU URBAN", 2],
        ["2019-07-01T00:00:00Z", "ULB", "BENGALURU URBAN", 10],           # BBMP joins Bengaluru Urban
        ["2019-07-03T00:00:00", "REVENUE", "Bengaluru Urban", 5],         # mixed case and date format
        ["2019-07-02T00:00:00Z", "REVENUE", "VIJAYANAGAR", 4],            # merged into Ballari
        ["2019-07-02T00:00:00Z", "REVENUE", "BALLARI", 1],
        ["2019-07-02T00:00:00Z", "admin_0", "KARNATAKA", 999],            # state total dropped
        ["2019-07-09T00:00:00Z", "REVENUE", "BALLARI", np.nan],           # a row without a value
    ])
    out = tmp_path / "weekly.csv"
    k.build_weekly_file(tmp_path, out)
    weekly = pd.read_csv(out, parse_dates=["week_start"]).set_index(["district", "week_start"])
    monday = pd.Timestamp("2019-07-01")
    assert weekly.loc[("BENGALURU URBAN", monday), "cases"] == 17
    assert weekly.loc[("BALLARI", monday), "cases"] == 5
    assert ("BALLARI", monday + pd.Timedelta(weeks=1)) not in weekly.index   # no usable report: missing, not zero
    assert "KARNATAKA" not in weekly.index.get_level_values(0)


def test_epidemic_years_are_left_out_of_the_baseline():
    weeks = pd.date_range("2012-01-02", "2020-12-28", freq="7D")
    matrix = pd.DataFrame(1.0, index=weeks, columns=k.DISTRICTS)
    matrix.loc["2016-01-01":"2016-12-31"] = 5.0                           # an epidemic year
    train, calibration = k.baseline_years(matrix, 2019)
    assert 2016 not in train and calibration == 2018
    assert train == [2014, 2015, 2017]
