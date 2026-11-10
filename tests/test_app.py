import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from model_service import EPOCH, parse_cases_csv, to_date, to_day

IDS = ["F00", "F01"]


def test_dates_map_weekday_to_profile_index():
    assert EPOCH.weekday() == 0                      # synthetic day 0 is a Monday
    for day in (0, 6, 400, -3):
        assert to_day(to_date(day)) == day
        assert to_date(day).weekday() == day % 7


def test_parse_fills_gaps_and_blank_cases_as_missing():
    text = "date,facility_id,cases\n2027-01-01,F00,5\n2027-01-01,F01,\n2027-01-03,F01,7\n"
    days, reported = parse_cases_csv(text, IDS)
    assert len(days) == 3 and days[0] == to_day(to_date(days[0]))
    assert reported[0, 0] == 5 and np.isnan(reported[0, 1])
    assert np.isnan(reported[1]).all()               # a day with no rows is missing everywhere
    assert reported[2, 1] == 7


@pytest.mark.parametrize("text, message", [
    ("", "empty"),
    ("a,b\n1,2\n", "Columns needed"),
    ("date,facility_id,cases\n2027-01-01,F99,3\n", "unknown facility"),
    ("date,facility_id,cases\nnot-a-date,F00,3\n", "could not read"),
    ("date,facility_id,cases\n2027-01-01,F00,-1\n", "negative"),
    ("day,facility_id,cases\n0,F00,1\n500,F00,1\n", "at most"),
])
def test_parse_rejects_bad_files_with_readable_messages(text, message):
    with pytest.raises(ValueError, match=message):
        parse_cases_csv(text, IDS)
