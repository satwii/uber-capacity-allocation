from pathlib import Path

import plotly.graph_objects as go
import pytest

from uberopt import stats
from uberopt.data import build_planning_data, load_requests

CSV = Path(__file__).resolve().parents[1] / "Uber Request Data.csv"


@pytest.fixture(scope="module")
def requests():
    return load_requests(CSV)


def test_overview(requests):
    o = stats.overview(requests)
    assert o["rows"] == 6745
    assert o["pickup_counts"]["City"] == 3507
    assert o["status_counts"]["Trip Completed"] == 2831


def test_status_by_pickup(requests):
    assert stats.status_by_pickup(requests).loc["City"].sum() == 3507


@pytest.mark.parametrize("name", ["fig_status_by_pickup", "fig_hourly_demand", "fig_gap_heatmap", "fig_duration_hist"])
def test_request_figures(requests, name):
    fig = getattr(stats, name)(requests)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_drivers_vs_demand(requests):
    fig = stats.fig_drivers_vs_demand(build_planning_data(requests))
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1
