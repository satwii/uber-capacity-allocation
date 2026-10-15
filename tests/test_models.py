from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uberopt.data import PlanningData, build_planning_data, load_requests
from uberopt.models import historical_result, shadow_prices, solve_gp, solve_ilp

CSV = Path(__file__).resolve().parents[1] / "Uber Request Data.csv"


def toy(rows, q=1.0):
    periods = pd.DataFrame(rows, columns=["D_C", "D_A", "N"])
    periods.insert(0, "period", [f"P{i + 1}" for i in range(len(rows))])
    return PlanningData(periods=periods, q=q, source="manual")


def test_ilp_toy():
    r = solve_ilp(toy([(5, 5, 6), (2, 0, 4)]))
    assert r.status == "Optimal"
    assert r.kpis["unmet"] == 4
    t = r.table
    assert ((t["Y_C"] + t["Y_A"]) <= t["N_adj"]).all()
    for col in ["Y_C", "Y_A", "X_C", "X_A"]:
        assert np.allclose(t[col], t[col].round())


def test_ilp_weights_shift():
    r = solve_ilp(toy([(5, 5, 6)]), w_A=2.0)
    row = r.table.iloc[0]
    assert row["X_A"] == 5 and row["X_C"] == 1


def test_ilp_integer_drivers():
    r = solve_ilp(toy([(3, 3, 3)], q=1.5))
    row = r.table.iloc[0]
    assert row["Y_C"] + row["Y_A"] <= 3
    assert row["X_C"] <= np.floor(1.5 * row["Y_C"]) and row["X_A"] <= np.floor(1.5 * row["Y_A"])
    assert r.kpis["unmet"] == 2  # LP relaxation would give 1.5


def test_zero_drivers():
    r = solve_ilp(toy([(4, 3, 0)]))
    assert r.status == "Optimal"
    assert r.kpis["served"] == 0
    assert np.isnan(r.table.iloc[0]["utilisation"])


def test_gp_meets_targets_when_ample():
    r = solve_gp(toy([(10, 8, 100), (3, 4, 100)]))
    assert r.status == "Optimal"
    assert (r.table[["dminus_C", "dminus_A"]].abs() < 1e-6).all().all()


def test_gp_balance_reduces_gap():
    data = toy([(10, 2, 6)])
    with_balance = solve_gp(data, P=(10, 3, 5))
    without = solve_gp(data, P=(10, 3, 0))
    assert with_balance.kpis["rate_gap"] <= without.kpis["rate_gap"] + 1e-9
    assert with_balance.kpis["rate_gap"] < 0.01  # 5 of 10 vs 1 of 2


def test_gp_zero_demand_location():
    r = solve_gp(toy([(5, 0, 3)]))
    assert r.status == "Optimal"
    assert np.isnan(r.table.iloc[0]["rate_A"])


@pytest.mark.parametrize("rows", [[(5, 5, 6), (2, 0, 4)], [(10, 2, 6), (7, 9, 5)]])
def test_gp_preemptive_matches_ilp_unmet(rows):
    data = toy(rows)
    assert solve_gp(data, mode="preemptive").kpis["unmet"] == solve_ilp(data).kpis["unmet"]


def test_shadow_prices():
    sp = shadow_prices(toy([(2, 1, 10), (8, 8, 5)])).set_index("period")
    assert sp.loc["P1", "shadow_price"] == pytest.approx(0)
    assert sp.loc["P2", "shadow_price"] > 0


def test_empty_data():
    r = solve_ilp(toy([]))
    assert r.status != "Optimal"
    assert r.table.empty


@pytest.fixture(scope="module")
def real():
    return build_planning_data(load_requests(CSV))


def test_real_data_solves(real):
    hist = historical_result(real)
    ilp = solve_ilp(real)
    gp = solve_gp(real)
    assert ilp.status == "Optimal" and gp.status == "Optimal"
    assert ilp.kpis["served"] >= hist.kpis["served"] == 2831
    assert gp.kpis["unmet"] >= ilp.kpis["unmet"]


def test_historical_none_for_manual():
    assert historical_result(toy([(1, 1, 1)])) is None
