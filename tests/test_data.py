import io
from pathlib import Path

import pandas as pd
import pytest

from uberopt.data import (
    build_planning_data,
    estimate_q,
    load_requests,
    parse_manual_table,
    planning_to_manual_table,
)

CSV = Path(__file__).resolve().parents[1] / "Uber Request Data.csv"


@pytest.fixture(scope="module")
def requests():
    return load_requests(CSV)


def test_load_full_dataset(requests):
    assert len(requests) == 6745
    assert (requests["Status"] == "Trip Completed").sum() == 2831
    assert requests["Request timestamp"].isna().sum() == 0


def test_estimate_q(requests):
    assert abs(estimate_q(requests) - 60 / 52.0833) < 0.01


def test_planning_totals(requests):
    data = build_planning_data(requests)
    p = data.periods
    assert data.source == "csv"
    assert p["D_C"].sum() == 3507
    assert p["D_A"].sum() == 3238
    assert p["hist_C"].sum() + p["hist_A"].sum() == 2831
    assert len(p) <= 120


def test_missing_columns_raises():
    bad = io.StringIO("Request id,Pickup point,Driver id,Request timestamp,Drop timestamp\n1,City,1,11/7/2016 9:00,\n")
    with pytest.raises(ValueError, match="Status"):
        load_requests(bad)


def test_no_valid_rows_raises():
    bad = io.StringIO("Request id,Pickup point,Driver id,Status,Request timestamp,Drop timestamp\n"
                      "1,City,1,Trip Completed,not-a-date,also-bad\n")
    with pytest.raises(ValueError, match="valid Request timestamp"):
        load_requests(bad)


def test_manual_parse_rejects_bad_rows():
    df = pd.DataFrame(
        [("P1", 10, 5, 4), ("P2", -1, 5, 4), ("P3", "x", 5, 4), ("P4", None, 5, 4)],
        columns=["Period", "City demand", "Airport demand", "Active drivers"],
    )
    data, errors = parse_manual_table(df, q=1.15)
    assert len(data.periods) == 1
    assert len(errors) == 3
    assert data.source == "manual"
    assert "hist_C" not in data.periods.columns


def test_manual_roundtrip(requests):
    data = build_planning_data(requests)
    back, errors = parse_manual_table(planning_to_manual_table(data), data.q)
    assert errors == []
    for col in ["D_C", "D_A", "N"]:
        assert back.periods[col].tolist() == data.periods[col].tolist()
