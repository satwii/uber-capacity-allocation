import pandas as pd

from uberopt.data import PlanningData
from uberopt.sensitivity import capacity_sweep, productivity_sweep, target_sweep


def toy(q=1.0):
    periods = pd.DataFrame({"period": ["P1", "P2", "P3"], "D_C": [10, 4, 7], "D_A": [3, 9, 6], "N": [8, 6, 5]})
    return PlanningData(periods=periods, q=q, source="manual")


def non_increasing(s):
    return all(a >= b for a, b in zip(s, s[1:]))


def test_capacity_sweep_monotone():
    for model in ["ilp", "gp"]:
        df = capacity_sweep(toy(), model, [-0.2, -0.1, 0, 0.1, 0.2])
        assert df["alpha"].tolist() == [-0.2, -0.1, 0, 0.1, 0.2]
        assert (df["status"] == "Optimal").all()
        assert non_increasing(df["unmet"].tolist())


def test_capacity_minus_100_percent():
    df = capacity_sweep(toy(), "ilp", [-1.0])
    assert df.iloc[0]["status"] == "Optimal"
    assert df.iloc[0]["served"] == 0


def test_productivity_sweep_monotone():
    df = productivity_sweep(toy(), "ilp", [0.8, 1.0, 1.2, 1.5])
    assert df["q"].tolist() == [0.8, 1.0, 1.2, 1.5]
    assert non_increasing(df["unmet"].tolist())


def test_target_sweep():
    df = target_sweep(toy(), [0.5, 0.6, 0.7, 0.8, 0.9])
    assert len(df) == 5
    assert {"target", "shortfall", "unmet", "rate_gap"} <= set(df.columns)
    assert df["shortfall"].is_monotonic_increasing
